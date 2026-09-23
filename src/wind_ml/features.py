"""Построение выборок в точке выпуска прогноза (issue time)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import (
    MeasurementTimeConfig,
    measurement_time_to_utc,
    validate_weather_forecasts,
)

FEATURE_COLUMNS = (
    "wind_speed_10m",
    "wind_speed_100m",
    "temperature_2m",
    "wind_direction_sin",
    "wind_direction_cos",
    "horizon_hours",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "last_power",
    "last_wind_speed",
    "last_temperature",
    "lag_age_hours",
)


@dataclass
class BuildResult:
    """Строки признаков и журнал причин, почему часть запросов исключена."""

    examples: pd.DataFrame
    audit: pd.DataFrame


def parse_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Время должно включать UTC-смещение, например 2026-01-01T00:00:00Z")
    return timestamp.tz_convert("UTC")


def hourly_issue_grid(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Вернуть часы [start, end), явно в UTC."""

    start_utc, end_utc = parse_utc(start), parse_utc(end)
    if end_utc <= start_utc:
        raise ValueError("Конец диапазона выпуска должен быть позже начала")
    return pd.date_range(start_utc, end_utc, inclusive="left", freq="h")


def _request_grid(issue_times: Iterable[pd.Timestamp], horizons: Iterable[int]) -> pd.DataFrame:
    valid_horizons = sorted({int(value) for value in horizons})
    if not valid_horizons or valid_horizons[0] < 1 or valid_horizons[-1] > 48:
        raise ValueError("Горизонты должны быть целыми часами от 1 до 48")
    issues = [parse_utc(value) for value in issue_times]
    result = pd.MultiIndex.from_product(
        [issues, valid_horizons], names=["issue_time", "horizon_hours"]
    ).to_frame(index=False)
    result["target_time"] = result["issue_time"] + pd.to_timedelta(
        result["horizon_hours"], unit="h"
    )
    result["request_id"] = np.arange(len(result))
    return result


def _attach_measurement_context(
    requests: pd.DataFrame, measurements: pd.DataFrame, config: MeasurementTimeConfig
) -> pd.DataFrame:
    observations = measurement_time_to_utc(measurements, config)
    complete = observations.loc[
        observations["quality"].eq("complete") & observations["time_utc"].notna(),
        ["time_utc", "available_at", "power", "wind_speed", "temperature"],
    ].copy()
    targets = complete.rename(
        columns={
            "time_utc": "target_time",
            "power": "target_power",
            "available_at": "target_available_at",
        }
    )[["target_time", "target_power", "target_available_at"]]
    result = requests.merge(targets, on="target_time", how="left", validate="many_to_one")
    asof_source = complete.rename(
        columns={
            "available_at": "measurement_available_at",
            "power": "last_power",
            "wind_speed": "last_wind_speed",
            "temperature": "last_temperature",
        }
    )[
        ["measurement_available_at", "last_power", "last_wind_speed", "last_temperature"]
    ].sort_values("measurement_available_at")
    result = pd.merge_asof(
        result.sort_values("issue_time"),
        asof_source,
        left_on="issue_time",
        right_on="measurement_available_at",
        direction="backward",
    ).sort_values("request_id")
    result["lag_age_hours"] = (
        result["issue_time"] - result["measurement_available_at"]
    ).dt.total_seconds() / 3600
    return result


def _attach_weather_context(requests: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    forecasts = validate_weather_forecasts(weather)
    eligible = forecasts.loc[
        forecasts["historical_eligibility"].eq("verified") & forecasts["available_at"].notna()
    ].copy()
    eligible = eligible.sort_values(["available_at", "run", "time"])
    if eligible.empty:
        result = requests.copy()
        for column in (
            "temperature_2m",
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_100m",
        ):
            result[column] = np.nan
        result["weather_available_at"] = pd.NaT
        result["weather_run"] = pd.NaT
        return result
    weather_columns = eligible.rename(
        columns={"available_at": "weather_available_at", "run": "weather_run"}
    )[
        [
            "time",
            "weather_available_at",
            "weather_run",
            "temperature_2m",
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_100m",
        ]
    ].sort_values("weather_available_at")
    # merge_asof выбирает самый поздний выпуск, опубликованный к issue_time,
    # и только для той же forecast-valid time.
    return pd.merge_asof(
        requests.sort_values("issue_time"),
        weather_columns,
        left_on="issue_time",
        right_on="weather_available_at",
        left_by="target_time",
        right_by="time",
        direction="backward",
    ).sort_values("request_id")


def _feature_calendar(frame: pd.DataFrame, calendar_timezone: str) -> pd.DataFrame:
    result = frame.copy()
    local = result["target_time"].dt.tz_convert(calendar_timezone)
    direction = np.deg2rad(result["wind_direction_100m"])
    result["wind_direction_sin"] = np.sin(direction)
    result["wind_direction_cos"] = np.cos(direction)
    result["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    result["day_of_year_sin"] = np.sin(2 * np.pi * local.dt.dayofyear / 366)
    result["day_of_year_cos"] = np.cos(2 * np.pi * local.dt.dayofyear / 366)
    return result


def build_persistence_requests(
    measurements: pd.DataFrame,
    config: MeasurementTimeConfig,
    issue_times: Iterable[pd.Timestamp],
    horizons: Iterable[int] = range(1, 49),
    *,
    require_targets: bool = True,
) -> BuildResult:
    """Persistence: последняя *уже известная* мощность на весь горизонт."""

    result = _attach_measurement_context(_request_grid(issue_times, horizons), measurements, config)
    result["reason"] = pd.NA
    if require_targets:
        result.loc[result["target_power"].isna(), "reason"] = "missing_target"
    result.loc[result["last_power"].isna(), "reason"] = "missing_lag"
    result.loc[result["lag_age_hours"].gt(config.max_lag_hours), "reason"] = "stale_telemetry"
    result["prediction"] = np.where(result["reason"].isna(), result["last_power"], np.nan)
    return BuildResult(
        examples=result.loc[result["reason"].isna()].copy(),
        audit=result,
    )


def build_forecast_requests(
    measurements: pd.DataFrame,
    weather: pd.DataFrame,
    config: MeasurementTimeConfig,
    issue_times: Iterable[pd.Timestamp],
    horizons: Iterable[int] = range(1, 49),
    *,
    require_targets: bool = True,
) -> BuildResult:
    """Собрать прогнозные признаки, доступные строго в момент выпуска."""

    result = _attach_measurement_context(_request_grid(issue_times, horizons), measurements, config)
    result = _attach_weather_context(result, weather)
    result["reason"] = pd.NA
    if require_targets:
        result.loc[result["target_power"].isna(), "reason"] = "missing_target"
    result.loc[result["last_power"].isna(), "reason"] = "missing_lag"
    result.loc[result["lag_age_hours"].gt(config.max_lag_hours), "reason"] = "stale_telemetry"
    weather_missing = result["weather_available_at"].isna()
    all_weather = validate_weather_forecasts(weather)
    unverified_targets = set(
        all_weather.loc[~all_weather["historical_eligibility"].eq("verified"), "time"].tolist()
    )
    result.loc[
        weather_missing & result["target_time"].isin(unverified_targets) & result["reason"].isna(),
        "reason",
    ] = "weather_unverified"
    result.loc[weather_missing & result["reason"].isna(), "reason"] = "weather_not_available"
    result = _feature_calendar(result, config.source_timezone)
    return BuildResult(
        examples=result.loc[result["reason"].isna()].copy(),
        audit=result,
    )


def feature_matrix(examples: pd.DataFrame) -> pd.DataFrame:
    absent = [column for column in FEATURE_COLUMNS if column not in examples.columns]
    if absent:
        raise ValueError(f"Не построены признаки: {', '.join(absent)}")
    result = examples.loc[:, FEATURE_COLUMNS].copy()
    if result.isna().any(axis=None):
        raise ValueError("В признаках есть пропуски; строка должна быть исключена из оценки")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("В признаках есть бесконечные значения")
    return result


def verified_issue_times(
    weather: pd.DataFrame, *, before: pd.Timestamp | None = None
) -> pd.DatetimeIndex:
    """Моменты публикации, а не инициализации численной модели."""

    forecasts = validate_weather_forecasts(weather)
    values = (
        forecasts.loc[forecasts["historical_eligibility"].eq("verified"), "available_at"]
        .dropna()
        .drop_duplicates()
    )
    if before is not None:
        values = values.loc[values < parse_utc(before)]
    return pd.DatetimeIndex(values.sort_values())
