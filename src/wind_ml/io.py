"""Адаптеры CSV/JSON. Они не интерполируют пропуски измерений."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import (
    MeasurementTimeConfig,
    SchemaError,
    validate_hourly_measurements,
    validate_weather_forecasts,
)

RAW_TIME = "Статистическое время"
RAW_WIND = "Средняя скорость ветра(m/s)"
RAW_POWER = "Нормализованная активная мощность"
RAW_TEMPERATURE = "Средняя температура окружающей среды(°C)"
RAW_COLUMNS = (RAW_TIME, RAW_WIND, RAW_POWER, RAW_TEMPERATURE)


def hourly_from_scada_csv(
    path: str | Path, *, config: MeasurementTimeConfig | None = None
) -> pd.DataFrame:
    """Агрегировать 10-минутный CSV строго по контракту первого этапа.

    Час считается ``complete`` только при шести уникальных отметках каждые
    десять минут и при валидных значениях всех трёх измерений. Длинные разрывы
    попадают в результат как ``missing`` с null-числами, а не заполняются.
    """

    raw = pd.read_csv(path)
    absent = [column for column in RAW_COLUMNS if column not in raw.columns]
    if absent:
        raise SchemaError(f"SCADA CSV: нет столбцов: {', '.join(absent)}")
    raw = raw.loc[:, RAW_COLUMNS].copy()
    raw["_time"] = pd.to_datetime(raw.pop(RAW_TIME), errors="coerce")
    if raw["_time"].isna().any():
        raise SchemaError("SCADA CSV: неразбираемое Статистическое время")
    raw["wind_speed"] = pd.to_numeric(raw.pop(RAW_WIND), errors="coerce")
    raw["power"] = pd.to_numeric(raw.pop(RAW_POWER), errors="coerce")
    raw["temperature"] = pd.to_numeric(raw.pop(RAW_TEMPERATURE), errors="coerce")
    if raw.empty:
        raise SchemaError("SCADA CSV пуст")
    if config is not None:
        time = raw["_time"]
        if not isinstance(time.dtype, pd.DatetimeTZDtype):
            time = time.dt.tz_localize(
                config.source_timezone, ambiguous="raise", nonexistent="raise"
            )
        raw["_time"] = time.dt.tz_convert("UTC")
        if config.timestamp_semantics == "interval_end":
            raw["_time"] -= pd.Timedelta(minutes=10)
    raw["_hour"] = raw["_time"].dt.floor("h")

    rows: list[dict[str, object]] = []
    index = pd.date_range(raw["_hour"].min(), raw["_hour"].max(), freq="h")
    groups = dict(tuple(raw.groupby("_hour")))
    for hour in index:
        group = groups.get(hour, raw.iloc[:0])
        sample_count = len(group)
        valid = (
            pd.Series(
                np.isfinite(group[["wind_speed", "power", "temperature"]].to_numpy()).all(axis=1),
                index=group.index,
            )
            & group["power"].between(0, 1)
            & group["wind_speed"].ge(0)
        )
        valid_count = int(valid.sum())
        expected_marks = {hour + pd.Timedelta(10 * offset, unit="min") for offset in range(6)}
        observed_marks = set(group.loc[valid, "_time"])
        complete = (
            sample_count == 6
            and valid_count == 6
            and observed_marks == expected_marks
            and group["_time"].nunique() == 6
        )
        row: dict[str, object] = {
            "time": hour,
            "sample_count": sample_count,
            "valid_count": valid_count,
            "completeness": valid_count / 6,
            "quality": "complete" if complete else ("partial" if sample_count else "missing"),
            "wind_speed": np.nan,
            "power": np.nan,
            "temperature": np.nan,
        }
        if complete:
            row.update(group[["wind_speed", "power", "temperature"]].mean().to_dict())
        rows.append(row)
    result = pd.DataFrame(rows)
    if config is not None:
        result["time_utc"] = result["time"]
        result["available_at"] = result["time_utc"] + pd.Timedelta(hours=1)
        result["source_timezone"] = config.source_timezone
        result["source_timestamp_semantics"] = config.timestamp_semantics
    return validate_hourly_measurements(result)


def read_hourly_csv(path: str | Path) -> pd.DataFrame:
    return validate_hourly_measurements(pd.read_csv(path))


def write_hourly_csv(frame: pd.DataFrame, path: str | Path) -> None:
    result = validate_hourly_measurements(frame)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


def weather_from_open_meteo_json(
    path: str | Path,
    *,
    run: str,
    available_at: str | None,
    historical_eligibility: str,
) -> pd.DataFrame:
    """Нормализовать один сохранённый ответ Single Runs API в погодный контракт.

    ``available_at`` намеренно не выводится из ``run``. Публикация и старт
    численной модели — разные события, поэтому пока время публикации неизвестно
    архив помечается ``unverified``.
    """

    response = json.loads(Path(path).read_text(encoding="utf-8"))
    hourly = response.get("hourly")
    if not isinstance(hourly, dict):
        raise SchemaError("Open-Meteo JSON не содержит hourly")
    columns = [
        "time",
        "temperature_2m",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_100m",
    ]
    absent = [column for column in columns if column not in hourly]
    if absent:
        raise SchemaError(f"Open-Meteo JSON: нет hourly-полей: {', '.join(absent)}")
    result = pd.DataFrame({column: hourly[column] for column in columns})
    timezone = response.get("timezone")
    if timezone != "GMT":
        raise SchemaError("Нормализатор принимает только ответ, явно запрошенный с timezone=GMT")
    result["time"] = pd.to_datetime(result["time"], utc=True)
    run_time = pd.Timestamp(run)
    if run_time.tzinfo is None:
        raise SchemaError("run должен содержать UTC-смещение, например 2026-01-31T00:00:00Z")
    result["run"] = run_time.tz_convert("UTC")
    available_time = pd.NaT if available_at is None else pd.Timestamp(available_at)
    if available_at is not None and available_time.tzinfo is None:
        raise SchemaError("available_at должен содержать UTC-смещение")
    if available_at is not None:
        available_time = available_time.tz_convert("UTC")
    result["available_at"] = available_time
    result["historical_eligibility"] = historical_eligibility
    return validate_weather_forecasts(result)


def read_weather_csv(path: str | Path, *, require_verified: bool = False) -> pd.DataFrame:
    return validate_weather_forecasts(pd.read_csv(path), require_verified=require_verified)


def write_weather_csv(frame: pd.DataFrame, path: str | Path) -> None:
    result = validate_weather_forecasts(frame)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
