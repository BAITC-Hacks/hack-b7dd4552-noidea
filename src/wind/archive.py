"""As-issued NOAA GFS forecasts, with object-level publication evidence.

Dynamical supplies a fixed, spatially subsetted copy of the operational GRIBs.
NOAA S3, not model initialization or a dissemination schedule, supplies the
availability evidence. See docs/weather-availability.md for its precise scope.
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd

from wind.storage import data_dir, get_turbine

PROVIDER = "noaa"
MODEL = "noaa_gfs_025"
POLICY = "noaa_s3_last_modified"
POLICY_VERSION = 1
DATASET_VERSION = "0.2.7"
DATASET_SNAPSHOT = "4BCAFQ3TN2NDBZNDDZR0"
DATASET_URL = (
    "https://dynamical-noaa-gfs.s3.us-west-2.amazonaws.com/noaa-gfs-forecast/v0.2.7.icechunk"
)
NOAA_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
CATALOG_URL = "https://dynamical.org/catalog/noaa-gfs-forecast/"
STAC_URL = "https://stac.dynamical.org/noaa-gfs-forecast/collection.json"
SOURCE_CODE_URL = (
    "https://github.com/dynamical-org/reformatters/blob/main/"
    "src/reformatters/noaa/gfs/region_job.py"
)
S3_METADATA_URL = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingMetadata.html"
S3_ETAG_URL = (
    "https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html"
)
VARIABLES = ("temperature_2m", "wind_speed_10m", "wind_speed_100m", "wind_direction_100m")
RAW_VARIABLES = ("temperature_2m", "wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m")
MAX_LEAD = 96


class ArchiveUnavailableError(ValueError):
    """No complete forecast with adequate publication evidence exists as of issue."""


def _utc(value: datetime | str, name: str = "time") -> datetime:
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise ValueError(f"{name}: требуется время с часовым поясом")
    return result.astimezone(UTC)


def _issue(value: datetime | str, horizon: int) -> datetime:
    value = _utc(value, "issue_at")
    if value.minute or value.second or value.microsecond:
        raise ValueError("issue_at должен начинаться на границе часа")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 24 <= horizon <= 48:
        raise ValueError("Горизонт должен быть целым числом от 24 до 48 часов")
    if value < datetime(2021, 5, 1, tzinfo=UTC) or value > datetime.now(UTC):
        raise ValueError("issue_at вне доступного исторического периода")
    return value


def _bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _root() -> Path:
    root = data_dir() / "nwp_archive" / MODEL
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        name = file.name
        file.write(content)
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _save_blob(content: bytes, suffix: str) -> str:
    relative = f"blobs/{_sha(content)}.{suffix}"
    path = _root() / relative
    if path.exists():
        if path.read_bytes() != content:
            raise ArchiveUnavailableError("Повреждён неизменяемый погодный кэш")
    else:
        _atomic(path, content)
    return relative


def _blob(relative: str, sha256: str) -> bytes:
    path = (_root() / relative).resolve()
    if not path.is_relative_to((_root() / "blobs").resolve()):
        raise ArchiveUnavailableError("Недопустимый путь погодного кэша")
    content = path.read_bytes()
    if _sha(content) != sha256:
        raise ArchiveUnavailableError("Контрольная сумма погодного кэша не совпала")
    return content


def _prefix(run: datetime) -> str:
    return f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f"


def parse_publication_listing(content: bytes, run: datetime) -> dict[int, dict]:
    """Read actual object timestamps. Missing, duplicated or backfilled files fail closed."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise ArchiveUnavailableError("Некорректный XML публикаций NOAA") from error
    namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if root.findtext("s:IsTruncated", namespaces=namespace) != "false":
        raise ArchiveUnavailableError("Неполный список публикаций NOAA")
    prefix = _prefix(run)
    result = {}
    for entry in root.findall("s:Contents", namespace):
        values = {child.tag.split("}")[-1]: child.text for child in entry}
        key = values.get("Key") or ""
        suffix = key.removeprefix(prefix)
        if not key.startswith(prefix) or len(suffix) != 3 or not suffix.isdecimal():
            continue  # Index files are not forecast files.
        lead = int(suffix)
        try:
            available = _utc(values["LastModified"], "LastModified")
            size = int(values["Size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArchiveUnavailableError("Неполные метаданные исходного NOAA GRIB") from error
        if lead in result or available < run or not values.get("ETag"):
            raise ArchiveUnavailableError("Некорректное свидетельство публикации NOAA")
        # Multipart LastModified can mark upload *initiation*, not availability.
        # AWS documents the '-partcount' ETag suffix for multipart objects.
        if not re.fullmatch(r'"[0-9a-fA-F]{32}"', values["ETag"]):
            raise ArchiveUnavailableError(
                "Multipart/неизвестный ETag не доказывает время публикации"
            )
        if size <= 0:
            raise ArchiveUnavailableError("Пустой исходный NOAA GRIB")
        result[lead] = {
            "key": key,
            "url": NOAA_URL + key,
            "available_at": available.isoformat(),
            "etag": values["ETag"],
            "size": size,
            "upload_evidence": "single_part_etag",
        }
    if not result:
        raise ArchiveUnavailableError("Выпуск отсутствует в открытом архиве NOAA")
    return result


def _publication(run: datetime, *, offline: bool = False) -> tuple[dict, dict]:
    path = _root() / "publication" / f"{run:%Y%m%dT%H}.json"
    if path.exists():
        metadata = json.loads(path.read_bytes())
        content = _blob(metadata["raw_path"], metadata["sha256"])
    else:
        if offline:
            raise ArchiveUnavailableError("Свидетельство публикации ещё не загружено")
        with httpx.Client(timeout=45, follow_redirects=False) as client:
            response = client.get(
                NOAA_URL, params={"list-type": "2", "prefix": _prefix(run), "max-keys": 1000}
            )
            response.raise_for_status()
        content = response.content
        parse_publication_listing(content, run)
        metadata = {
            "run": run.isoformat(),
            "url": str(response.url),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": _sha(content),
            "raw_path": _save_blob(content, "xml"),
        }
        _atomic(path, _bytes(metadata))
    if metadata["run"] != run.isoformat():
        raise ArchiveUnavailableError("Свидетельство относится к другому выпуску")
    return parse_publication_listing(content, run), metadata


@lru_cache(maxsize=1)
def _dataset():
    # Lazy imports keep offline cache reading independent of optional reader startup.
    import icechunk
    import xarray as xr

    repo = icechunk.Repository.open(icechunk.http_storage(DATASET_URL))
    session = repo.readonly_session(snapshot_id=DATASET_SNAPSHOT)
    return xr.open_zarr(session.store, chunks=None, decode_timedelta=True)


def _grid(latitude: float, longitude: float) -> tuple[float, float]:
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Недопустимая широта")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Недопустимая долгота")
    lon = (round(longitude * 4) / 4 + 180) % 360 - 180
    return round(latitude * 4) / 4, lon


def _download_point(run: datetime, latitude: float, longitude: float) -> dict:
    import numpy as np

    ds = _dataset()
    for variable in RAW_VARIABLES:
        units = "degree_Celsius" if variable == "temperature_2m" else "m s-1"
        if ds[variable].attrs.get("units") != units:
            raise ArchiveUnavailableError(f"Неожиданные единицы {variable}")
    try:
        # Never nearest-match init_time: a missing cycle cannot become another cycle.
        point = ds[list(RAW_VARIABLES)].sel(init_time=run.replace(tzinfo=None))
        point = point.sel(latitude=latitude, longitude=longitude)
        point = point.sel(lead_time=np.arange(MAX_LEAD + 1).astype("timedelta64[h]"))
        point.load()
    except KeyError as error:
        raise ArchiveUnavailableError(
            "Выпуск/ячейка отсутствует в зафиксированном архиве"
        ) from error
    points = []
    for lead in range(MAX_LEAD + 1):
        values = {v: float(point[v].values[lead]) for v in RAW_VARIABLES}
        points.append(
            {
                "time": (run + timedelta(hours=lead)).isoformat(),
                "lead_hour": lead,
                **{key: value if math.isfinite(value) else None for key, value in values.items()},
            }
        )
    return {
        "run": run.isoformat(),
        "grid_latitude": float(point.latitude),
        "grid_longitude": float(point.longitude),
        "dataset_snapshot": DATASET_SNAPSHOT,
        "points": points,
    }


def _point_cache(
    run: datetime,
    latitude: float,
    longitude: float,
    *,
    offline: bool = False,
) -> tuple[dict, dict]:
    key = _sha(
        _bytes(
            {
                "run": run.isoformat(),
                "latitude": latitude,
                "longitude": longitude,
                "snapshot": DATASET_SNAPSHOT,
                "max_lead": MAX_LEAD,
            }
        )
    )
    path = _root() / "points" / f"{key}.json"
    if path.exists():
        metadata = json.loads(path.read_bytes())
        payload = json.loads(_blob(metadata["raw_path"], metadata["sha256"]))
    else:
        if offline:
            raise ArchiveUnavailableError("Погодная ячейка ещё не загружена")
        payload = _download_point(run, latitude, longitude)
        content = _bytes(payload)
        metadata = {
            "sha256": _sha(content),
            "raw_path": _save_blob(content, "json"),
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
        _atomic(path, _bytes(metadata))
    if (
        payload["run"] != run.isoformat()
        or payload["dataset_snapshot"] != DATASET_SNAPSHOT
        or payload["grid_latitude"] != latitude
        or payload["grid_longitude"] != longitude
    ):
        raise ArchiveUnavailableError("Погодный кэш относится к другому выпуску/сетке/версии")
    expected = [(run + timedelta(hours=h)).isoformat() for h in range(MAX_LEAD + 1)]
    if [point["time"] for point in payload["points"]] != expected:
        raise ArchiveUnavailableError("Нарушена последовательность часов в погодном кэше")
    return payload, metadata


def _provenance(turbine: dict) -> dict:
    latitude, longitude = _grid(turbine["latitude"], turbine["longitude"])
    return {
        "provider": PROVIDER,
        "model": MODEL,
        "availability_policy": POLICY,
        "policy_version": POLICY_VERSION,
        "latitude": turbine["latitude"],
        "longitude": turbine["longitude"],
        "grid_latitude": latitude,
        "grid_longitude": longitude,
        "dataset_snapshot": DATASET_SNAPSHOT,
        "dataset_version": DATASET_VERSION,
        "source": CATALOG_URL,
        "dataset_url": DATASET_URL,
        "stac_url": STAC_URL,
        "availability_evidence": [
            NOAA_URL,
            CATALOG_URL,
            SOURCE_CODE_URL,
            S3_METADATA_URL,
            S3_ETAG_URL,
        ],
        "availability_scope": "original_noaa_grib_objects_not_dynamical_api",
        "selection": "fixed_00_utc_nearest_0.25_degree_cell",
        "transformations": "rounded GFS U/V; speed=hypot(U,V); from-direction=atan2(-U,-V)",
    }


def _normalized(raw: dict, publication: dict, run: datetime) -> dict:
    if any(raw.get(key) is None or not math.isfinite(raw[key]) for key in RAW_VARIABLES):
        raise ArchiveUnavailableError("В выбранном горизонте отсутствуют погодные значения")
    u, v = raw["wind_u_100m"], raw["wind_v_100m"]
    return {
        "time": raw["time"],
        "run": run.isoformat(),
        "available_at": publication["available_at"],
        "historical_eligibility": "verified",
        "temperature_2m": raw["temperature_2m"],
        "wind_speed_10m": math.hypot(raw["wind_u_10m"], raw["wind_v_10m"]),
        "wind_speed_100m": math.hypot(u, v),
        "wind_direction_100m": math.degrees(math.atan2(-u, -v)) % 360 if u or v else 0.0,
    }


def fetch_issue_weather(
    turbine_id: int,
    issue_at: datetime | str,
    horizon: int = 48,
    previous_run: bool = False,
    *,
    offline: bool = False,
) -> dict:
    """Return exactly issue+1..horizon, never future-publication weather.

    Fixed 00 UTC run selection is shared by training/replay. If today's run was
    unavailable at issue time, yesterday's 00 UTC is tried. No model fallback.
    """
    issue = _issue(issue_at, horizon)
    turbine = get_turbine(turbine_id)
    provenance = _provenance(turbine)
    today = issue.replace(hour=0)
    candidates = [today - timedelta(days=1)] if previous_run else [today, today - timedelta(days=1)]
    failures = []
    for run in candidates:
        try:
            publications, publication_metadata = _publication(run, offline=offline)
            leads = [int((issue - run).total_seconds() / 3600) + h for h in range(1, horizon + 1)]
            if not all(lead in publications and lead <= MAX_LEAD for lead in leads):
                raise ArchiveUnavailableError("Нет свидетельств публикации всех нужных часов")
            available = max(_utc(publications[h]["available_at"]) for h in leads)
            if available > issue:
                raise ArchiveUnavailableError("Погодные файлы опубликованы позже момента прогноза")
            payload, point_metadata = _point_cache(
                run,
                provenance["grid_latitude"],
                provenance["grid_longitude"],
                offline=offline,
            )
            points = [_normalized(payload["points"][h], publications[h], run) for h in leads]
            evidence = {
                **provenance,
                "point_sha256": point_metadata["sha256"],
                "publication_sha256": publication_metadata["sha256"],
                "publication_url": publication_metadata["url"],
                "source_objects": [publications[h] for h in leads],
            }
            return {
                "turbine_id": turbine_id,
                "issue_at": issue.isoformat(),
                "horizon": horizon,
                "hours": len(points),
                "run": run.isoformat(),
                "available_at": available.isoformat(),
                "source": "NOAA GFS / dynamical.org",
                "provider": PROVIDER,
                "model": MODEL,
                "availability_policy": POLICY,
                "historical_eligibility": "verified",
                "eligible": True,
                "provenance": evidence,
                "points": points,
                "sha256": point_metadata["sha256"],
            }
        except (ArchiveUnavailableError, httpx.HTTPError, OSError) as error:
            failures.append(f"{run.isoformat()}: {error}")
    raise ArchiveUnavailableError("Нет исторически доступного выпуска: " + "; ".join(failures))


def load_archive_frame(turbine_id: int) -> pd.DataFrame:
    """Load all locally cached 00 UTC forecasts for an active turbine, without network."""
    turbine = get_turbine(turbine_id)
    provenance = _provenance(turbine)
    rows, hashes, seen_runs = [], [], set()
    for path in sorted((_root() / "publication").glob("*.json")):
        metadata = json.loads(path.read_bytes())
        run = _utc(metadata["run"])
        if run in seen_runs:
            raise ArchiveUnavailableError("Дублирующийся выпуск в погодном кэше")
        seen_runs.add(run)
        publications, pubmeta = _publication(run, offline=True)
        try:
            payload, pointmeta = _point_cache(
                run,
                provenance["grid_latitude"],
                provenance["grid_longitude"],
                offline=True,
            )
        except ArchiveUnavailableError as error:
            if "ещё не загружена" in str(error):
                continue
            raise
        for lead, raw in enumerate(payload["points"]):
            if lead == 0 or lead not in publications:
                continue  # Forecasts only; analysis (lead=0) is never a feature.
            try:
                rows.append(_normalized(raw, publications[lead], run))
            except ArchiveUnavailableError:
                continue  # Missing values remain absent; never impute from future runs.
        hashes.append(
            {
                "run": run.isoformat(),
                "point_sha256": pointmeta["sha256"],
                "publication_sha256": pubmeta["sha256"],
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "time",
            "run",
            "available_at",
            "historical_eligibility",
            *VARIABLES,
        ],
    )
    if not result.empty:
        for column in ("time", "run", "available_at"):
            result[column] = pd.to_datetime(result[column], utc=True)
        result = result.sort_values(["run", "time"]).reset_index(drop=True)
    result.attrs["provenance"] = {**provenance, "archive_hashes": hashes}
    return result


def download_archive(
    turbine_ids: list[int],
    start: datetime | str,
    end: datetime | str,
    *,
    workers: int = 4,
    progress=None,
) -> dict:
    """Inclusive daily 12 UTC issues, bounded to 450 runs; cached downloads resume."""
    start, end = _issue(start, 48), _issue(end, 48)
    if start.hour != 12 or end.hour != 12 or end < start:
        raise ValueError("Границы пакетной загрузки: 12 UTC, конец не раньше начала")
    days = (end - start).days + 1
    if days > 450 or not 1 <= workers <= 4 or not 1 <= len(turbine_ids) <= 20:
        raise ValueError("Лимит: 450 дней, 1–4 workers, 1–20 турбин")
    # One fetch per actual model cell. Public archive numerics are shared, turbine IDs are not.
    grid_groups = {}
    for turbine_id in dict.fromkeys(turbine_ids):
        turbine = get_turbine(turbine_id)
        grid_groups.setdefault(_grid(turbine["latitude"], turbine["longitude"]), turbine_id)
    tasks = [(t, start + timedelta(days=d)) for t in grid_groups.values() for d in range(days)]
    summary = {
        "completed": 0,
        "failed": [],
        "tasks": len(tasks),
        "unique_grid_cells": len(grid_groups),
    }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_issue_weather, t, issue): (t, issue) for t, issue in tasks}
        for future in as_completed(futures):
            turbine_id, issue = futures[future]
            try:
                result = future.result()
                # A silent fallback would make a purported daily archive incomplete.
                if _utc(result["run"]).date() != issue.date():
                    raise ArchiveUnavailableError(
                        "Дневной выпуск отсутствует; доступен лишь предыдущий"
                    )
                summary["completed"] += 1
            except Exception as error:
                summary["failed"].append(
                    {"issue_at": issue.isoformat(), "turbine_id": turbine_id, "error": str(error)}
                )
            if progress:
                progress(summary)
    _atomic(_root() / "last_download.json", _bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbine-id", type=int, action="append", required=True)
    parser.add_argument("--start", required=True, help="Inclusive issue at 12 UTC")
    parser.add_argument("--end", required=True, help="Inclusive issue at 12 UTC")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()

    def progress(summary):
        done = summary["completed"] + len(summary["failed"])
        if done % 10 == 0 or done == summary["tasks"]:
            print(f"{done}/{summary['tasks']}; errors={len(summary['failed'])}", flush=True)

    result = download_archive(
        args.turbine_id, args.start, args.end, workers=args.workers, progress=progress
    )
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for turbine_id in args.turbine_id:
            frame = load_archive_frame(turbine_id)
            path = args.export_dir / f"{turbine_id}-{MODEL}.parquet"
            frame.to_parquet(path, index=False)
            _atomic(path.with_suffix(".metadata.json"), _bytes(frame.attrs["provenance"]))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
