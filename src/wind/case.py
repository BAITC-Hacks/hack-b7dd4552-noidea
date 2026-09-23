"""One explicit command for the historical weather -> model -> February replay case."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import pandas as pd

from wind.agent import run_agent, run_tools
from wind.archive import download_archive, fetch_issue_weather, load_archive_frame
from wind.nwp import _contract, artifact_directory, model_readiness
from wind.replay import ReplayRequest, create_job, export_csv, folder, run_job, write_json
from wind.storage import all_metadata, data_dir, get_turbine
from wind_ml.nwp import train_nwp
from wind_ml.telemetry import hourly_telemetry

ARCHIVE_START = "2025-01-01T12:00:00Z"
ARCHIVE_END = "2026-02-28T12:00:00Z"
TRAIN_END = "2025-10-01T00:00:00Z"
VALIDATION_END = "2026-01-01T00:00:00Z"
FIRST_ISSUE = "2026-01-31T12:00:00Z"


def _timestamp():
    return datetime.now(UTC).isoformat()


def _local_file(relative, suffix):
    path = (data_dir() / relative).resolve()
    if not path.is_relative_to(data_dir()) or path.suffix != suffix or not path.is_file():
        raise ValueError(f"Не найден локальный файл данных {relative}")
    return path


def validate_measurements(turbine_ids, timezone, semantics):
    """Validate every input before any archive download or paid LLM request."""
    datasets = {dataset["id"]: dataset for dataset in all_metadata("datasets")}
    inputs = []
    for turbine_id in turbine_ids:
        turbine = get_turbine(turbine_id)
        dataset = datasets.get(turbine_id)
        if not dataset or not dataset.get("hourly_path") or not dataset.get("raw_path"):
            raise ValueError(f"Турбина {turbine_id}: сначала импортируйте полный исходный CSV")
        source = _local_file(dataset["raw_path"], ".csv")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != dataset.get("sha256"):
            raise ValueError(f"Турбина {turbine_id}: исходный CSV изменился после импорта")
        raw_path = _local_file(
            dataset["hourly_path"].replace("-hourly.parquet", "-10min.parquet"), ".parquet"
        )
        raw = pd.read_parquet(raw_path)
        hourly = hourly_telemetry(raw, timezone, semantics)
        complete = hourly.loc[hourly.quality.eq("complete")].copy()
        if complete.empty:
            raise ValueError(f"Турбина {turbine_id}: нет полных часов измерений")
        start, final = pd.Timestamp(ARCHIVE_START), pd.Timestamp(FIRST_ISSUE)
        train_end, validation_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VALIDATION_END)
        counts = {
            "train": int(
                (
                    complete.time.ge(start)
                    & complete.time.lt(train_end)
                    & complete.available_at.le(train_end)
                ).sum()
            ),
            "validation": int(
                (
                    complete.time.ge(train_end)
                    & complete.time.lt(validation_end)
                    & complete.available_at.le(validation_end)
                ).sum()
            ),
            "control": int(
                (
                    complete.time.ge(validation_end)
                    & complete.time.lt(final)
                    & complete.available_at.le(final)
                ).sum()
            ),
        }
        if (
            dataset.get("dataset_kind") == "demo"
            or complete.time.min() > start
            or complete.available_at.max() < final
            or counts["train"] < 720
            or counts["validation"] < 168
            or counts["control"] < 168
        ):
            raise ValueError(
                f"Турбина {turbine_id}: недостаточно полной истории для кейса. "
                "Нужны измерения от начала 2025 года до 31.01.2026 12:00 UTC, "
                "не менее 720 полных часов обучения и по 168 часов валидации/контроля. "
                "Семидневный demo CSV не подходит."
            )
        inputs.append(
            {
                "turbine_id": turbine_id,
                "name": turbine["name"],
                "source_sha256": digest,
                "latitude": turbine["latitude"],
                "longitude": turbine["longitude"],
                "first_complete_hour": complete.time.min().isoformat(),
                "last_available_hour": complete.available_at.max().isoformat(),
                "complete_hours_by_split": counts,
                "dataset": dataset,
                "raw": raw,
            }
        )
    return inputs


def _unchanged(item):
    turbine = get_turbine(item["turbine_id"])
    dataset = next((d for d in all_metadata("datasets") if d["id"] == item["turbine_id"]), None)
    if (
        not dataset
        or dataset.get("sha256") != item["source_sha256"]
        or turbine["latitude"] != item["latitude"]
        or turbine["longitude"] != item["longitude"]
    ):
        raise ValueError(
            f"Турбина {item['turbine_id']}: входные данные изменились; начните новый кейс"
        )


def _activate_model(staging: Path, destination: Path, backup: Path):
    """Keep previous generated artifacts; never overwrite imported measurements."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)
    replaced = destination.exists()
    if replaced:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except OSError:
        if replaced:
            os.replace(backup, destination)
        raise
    return str(backup) if replaced else None


def run_case(
    *,
    turbine_ids=None,
    all_turbines=False,
    measurement_timezone,
    timestamp_semantics,
    execution_mode="tools",
    offline=False,
    skip_train=False,
    workers=4,
    progress=None,
):
    if bool(turbine_ids) == bool(all_turbines):
        raise ValueError("Выберите --all либо --turbine-ids")
    if not 1 <= workers <= 4:
        raise ValueError("Допустимо от 1 до 4 работников загрузки")
    if offline and execution_mode == "llm":
        raise ValueError("--offline поддерживает только tools: LLM требует сетевого OpenAI API")
    ids = [t["id"] for t in all_metadata("turbines")] if all_turbines else list(turbine_ids)
    if not ids:
        raise ValueError("Нет активных турбин. Создайте турбины и импортируйте полные CSV")
    if not offline and len(ids) > 20:
        raise ValueError("Один пакет загрузки поддерживает до 20 турбин")
    request = ReplayRequest(
        turbine_ids=ids,
        measurement_timezone=measurement_timezone,
        timestamp_semantics=timestamp_semantics,
        execution_mode=execution_mode,
    )
    report_dir = data_dir() / "case-runs" / uuid4().hex
    report = {
        "status": "running",
        "created_at": _timestamp(),
        "request": request.model_dump(mode="json"),
        "offline": offline,
        "skip_train": skip_train,
        "phases": [],
        "models": [],
        "archive_start": ARCHIVE_START,
        "archive_end": ARCHIVE_END,
        "report_path": str(report_dir / "report.json"),
        "february_accuracy_evaluated": False,
    }

    def phase(name, status, **values):
        event = {"phase": name, "status": status, "at": _timestamp(), **values}
        report["phases"].append(event)
        report["updated_at"] = event["at"]
        write_json(report_dir / "report.json", report)
        if progress:
            progress(event)

    try:
        phase("validate_inputs", "running")
        inputs = validate_measurements(ids, measurement_timezone, timestamp_semantics)
        report["inputs"] = [
            {k: v for k, v in item.items() if k not in {"dataset", "raw"}} for item in inputs
        ]
        if skip_train:
            for item in inputs:
                ready = model_readiness(
                    item["dataset"],
                    timezone=measurement_timezone,
                    semantics=timestamp_semantics,
                    issue_at=FIRST_ISSUE,
                    horizon=request.horizon,
                )
                if not ready["ok"]:
                    raise ValueError(
                        f"Турбина {item['turbine_id']}: --skip-train: {ready.get('reason')}"
                    )
        phase("validate_inputs", "completed", turbine_ids=ids)

        if offline:
            phase("download_archive", "skipped", reason="offline: только существующий кеш")
        else:
            phase("download_archive", "running")

            def download_progress(value):
                if progress:
                    progress(
                        {
                            "phase": "download_archive",
                            "status": "running",
                            "completed": value["completed"],
                            "failed": len(value["failed"]),
                            "total": value["tasks"],
                        }
                    )

            summary = download_archive(
                ids, ARCHIVE_START, ARCHIVE_END, workers=workers, progress=download_progress
            )
            report["archive"] = summary
            phase("download_archive", "partial" if summary["failed"] else "completed")

        phase("load_weather", "running")
        weather_frames = {}
        for item in inputs:
            _unchanged(item)
            weather = load_archive_frame(item["turbine_id"])
            if weather.empty:
                raise ValueError(
                    f"Турбина {item['turbine_id']}: погодный кеш пуст; сначала загрузите архив"
                )
            _contract(weather.attrs.get("provenance"))
            weather_frames[item["turbine_id"]] = weather
        phase("load_weather", "completed", rows={str(k): len(v) for k, v in weather_frames.items()})

        if skip_train:
            phase("train_models", "skipped", reason="проверенные существующие артефакты")
        else:
            phase("train_models", "running")
            staged = []
            for item in inputs:
                turbine_id = item["turbine_id"]
                weather = weather_frames[turbine_id]
                staging = report_dir / "models" / str(turbine_id)
                manifest = train_nwp(
                    item["raw"],
                    weather,
                    turbine_id=turbine_id,
                    source_sha256=item["source_sha256"],
                    timezone=measurement_timezone,
                    semantics=timestamp_semantics,
                    directory=staging,
                    weather_provenance=weather.attrs["provenance"],
                    train_end=TRAIN_END,
                    validation_end=VALIDATION_END,
                    control_end=FIRST_ISSUE,
                    final_issue_at=FIRST_ISSUE,
                )
                report["models"].append(manifest)
                if manifest.get("promoted") is not True:
                    raise ValueError(
                        f"Турбина {turbine_id}: погодная модель не прошла валидацию; "
                        "replay не запущен"
                    )
                _unchanged(item)
                staged.append((item, staging))
            # Publish only after every requested model passed selection.
            for item, staging in staged:
                _unchanged(item)
                destination = artifact_directory(
                    item["turbine_id"], measurement_timezone, timestamp_semantics
                )
                backup = report_dir / "previous-models" / str(item["turbine_id"])
                backup_path = _activate_model(staging, destination, backup)
                report.setdefault("model_paths", []).append(
                    {
                        "turbine_id": item["turbine_id"],
                        "active": str(destination),
                        "previous": backup_path,
                    }
                )
            phase("train_models", "completed")

        phase("replay", "running")
        for item in inputs:
            _unchanged(item)
        job = create_job(request)
        report["replay_id"] = job["id"]
        write_json(report_dir / "report.json", report)
        runner = run_agent if execution_mode == "llm" else run_tools
        if offline:
            runner = partial(
                run_tools, issue_weather_provider=partial(fetch_issue_weather, offline=True)
            )
        job = run_job(job["id"], runner=runner)
        report["replay_status"] = job["status"]
        report["summary"] = job.get("summary")
        phase("replay", job["status"], progress=job["progress"])

        phase("export", "running")
        exports = {}
        kinds = ["forecasts", "coverage"]
        if all(
            (job.get("input_snapshots", {}).get(str(turbine_id), {}).get("rated_power_kw") or 0) > 0
            for turbine_id in request.turbine_ids
        ):
            kinds.append("plant")
        for kind in kinds:
            path = report_dir / f"{kind}.csv"
            path.write_text(export_csv(job, kind), encoding="utf-8")
            exports[kind] = str(path)
        write_json(report_dir / "replay.json", job)
        report["exports"] = {
            **exports,
            "replay": str(report_dir / "replay.json"),
            "job": str(folder(job["id"]) / "job.json"),
        }
        report["status"] = job["status"]
        phase("export", "completed")
        return report
    except (OSError, ValueError, KeyError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)[:1000]
        phase("case", "failed", detail=report["error"])
        raise ValueError(f"{exc}. Отчёт: {report_dir / 'report.json'}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", dest="all_turbines", action="store_true")
    selection.add_argument("--turbine-ids", type=int, nargs="+")
    parser.add_argument("--measurement-timezone", required=True)
    parser.add_argument(
        "--timestamp-semantics", choices=["interval_start", "interval_end"], required=True
    )
    parser.add_argument("--execution-mode", choices=["tools", "llm"], default="tools")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Только локальный погодный кеш и tools, без сетевых запросов",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Использовать существующие совместимые артефакты"
    )
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)

    def progress(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        report = run_case(**vars(args), progress=progress)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
