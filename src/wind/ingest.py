"""Strict import, immutable raw files and explicit missing hourly intervals."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wind.storage import data_dir, get_turbine, save_metadata

COLUMNS = {
    "Статистическое время": "time",
    "Средняя скорость ветра(m/s)": "wind_speed",
    "Нормализованная активная мощность": "power",
    "Средняя температура окружающей среды(°C)": "temperature",
}
METRICS = ["wind_speed", "power", "temperature"]


def normalize(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(source)
    absent = set(COLUMNS) - set(raw.columns)
    if absent:
        raise ValueError(f"Отсутствуют столбцы: {', '.join(sorted(absent))}")
    if raw.empty:
        raise ValueError("CSV пуст")
    df = raw.rename(columns=COLUMNS)[["time", *METRICS]].copy()
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    invalid_times = int(df.time.isna().sum())
    off_grid = int(((df.time.dt.minute % 10 != 0) | (df.time.dt.second != 0)).sum())
    if invalid_times or off_grid:
        raise ValueError(f"Некорректное время: {invalid_times}; вне сетки 10 минут: {off_grid}")
    duplicate_count = int(df.time.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Повторяющиеся отметки времени: {duplicate_count}. Импорт отменён")
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan
    invalid_values = {col: int(df[col].isna().sum()) for col in METRICS}
    out_of_range = (df.power.lt(0) | df.power.gt(1)) | df.wind_speed.lt(0)
    if out_of_range.any():
        raise ValueError(f"Мощность вне [0,1] или отрицательный ветер: {int(out_of_range.sum())}")
    df = df.sort_values("time").set_index("time")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    missing = expected.difference(df.index)
    hours = df[METRICS].resample("1h").mean()
    counts = df[METRICS].resample("1h").count()
    hours["sample_count"] = df.resample("1h").size()
    hours["valid_count"] = counts.min(axis=1)
    hours["completeness"] = hours.valid_count / 6
    hours["quality"] = np.select(
        [hours.valid_count.eq(6), hours.valid_count.eq(0)],
        ["complete", "missing"],
        default="partial",
    )
    # Only complete hours have a trustworthy hourly average. Partial means remain in raw data.
    hours.loc[hours.quality.ne("complete"), METRICS] = np.nan
    delta = df.index.to_series().diff()
    gaps = []
    for end, duration in delta[delta.gt(pd.Timedelta(minutes=10))].items():
        gaps.append(
            {
                "start": (end - duration + pd.Timedelta(minutes=10)).isoformat(),
                "end": (end - pd.Timedelta(minutes=10)).isoformat(),
                "missing_slots": int(duration.total_seconds() // 600 - 1),
            }
        )
    summary = {
        "rows": len(df),
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
        "expected_slots": len(expected),
        "missing_slots": len(missing),
        "coverage": round(len(df) / len(expected), 6),
        "duplicate_times": 0,
        "invalid_values": invalid_values,
        "timezone": "unconfirmed",
        "hours": len(hours),
        "complete_hours": int(hours.quality.eq("complete").sum()),
        "partial_hours": int(hours.quality.eq("partial").sum()),
        "missing_hours": int(hours.quality.eq("missing").sum()),
        "gap_count": len(gaps),
        "largest_gaps": sorted(gaps, key=lambda x: x["missing_slots"], reverse=True)[:10],
        "statistics": {
            col: {
                "min": float(df[col].min()) if df[col].notna().any() else None,
                "max": float(df[col].max()) if df[col].notna().any() else None,
            }
            for col in METRICS
        },
    }
    return df.reset_index(), hours.reset_index(), summary


def import_csv(source: Path, turbine_id: int, kind: str = "full") -> dict:
    turbine = get_turbine(turbine_id)
    raw, hourly, summary = normalize(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    root = data_dir()
    raw_dir = root / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw_path = raw_dir / f"{digest}.csv"
    if not raw_path.exists():
        shutil.copyfile(source, raw_path)
    # Versioned files: catalog is updated only after both files have been written.
    raw.to_parquet(root / f"{turbine_id}-{digest}-10min.parquet", index=False)
    hourly.to_parquet(root / f"{turbine_id}-{digest}-hourly.parquet", index=False)
    summary.update(
        {
            **turbine,
            "sha256": digest,
            "source_name": source.name,
            "dataset_kind": kind,
            "imported_at": datetime.now(UTC).isoformat(),
            "raw_path": str(raw_path.relative_to(root)),
            "hourly_path": f"{turbine_id}-{digest}-hourly.parquet",
        }
    )
    save_metadata("datasets", turbine_id, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Импорт исходных CSV ВЭС")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    from wind.storage import all_metadata

    registered = all_metadata("turbines")
    if not registered:
        parser.error("Сначала создайте турбину через веб-интерфейс")
    for turbine in registered:
        turbine_id = turbine["id"]
        candidates = sorted(args.source_dir.glob(f"*turbine {turbine_id}.csv"))
        if len(candidates) != 1:
            parser.error(f"Ожидается ровно один '*turbine {turbine_id}.csv' в {args.source_dir}")
        summary = import_csv(candidates[0], turbine_id, "demo" if args.demo else "full")
        print(json.dumps({k: summary[k] for k in ["name", "rows", "missing_slots", "sha256"]}))


if __name__ == "__main__":
    main()
