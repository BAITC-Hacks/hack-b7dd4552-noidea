"""Point-in-time features from telemetry; future weather/observations are never features."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

METRICS = ("power", "wind_speed", "temperature")
FEATURE_COLUMNS = (
    "last_power",
    "last_wind_speed",
    "last_temperature",
    "lag_age_hours",
    *(f"{metric}_lag_{lag}h" for metric in METRICS for lag in (1, 6, 24)),
    *(
        f"{metric}_{stat}_{window}h"
        for metric in METRICS
        for window in (6, 24)
        for stat in ("mean", "std")
    ),
    "complete_hours_6h",
    "complete_hours_24h",
    "horizon_hours",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)


@dataclass
class BuildResult:
    examples: pd.DataFrame
    audit: pd.DataFrame


def _utc(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Время выпуска должно содержать явный UTC offset")
    value = value.tz_convert("UTC")
    if value != value.floor("h"):
        raise ValueError("Время выпуска должно быть на часовой сетке")
    return value


def hourly_telemetry(raw: pd.DataFrame, timezone: str, semantics: str) -> pd.DataFrame:
    """UTC interval-start hours, available only after all six intervals finish.

    Interval-end CSV labels move back ten minutes BEFORE aggregation. Ambiguous or
    nonexistent local labels are dropped; their hours consequently stay incomplete.
    No guesses about the source timezone and no interpolation are made.
    """
    if not timezone or timezone == "unknown":
        raise ValueError("Нужен явный часовой пояс CSV")
    ZoneInfo(timezone)
    if semantics not in {"interval_start", "interval_end"}:
        raise ValueError("Неизвестная семантика времени измерения")
    if not {"time", *METRICS}.issubset(raw.columns) or raw.empty:
        raise ValueError("Нет исходных измерений time, power, wind_speed, temperature")
    data = raw.loc[:, ["time", *METRICS]].copy()
    labels = pd.to_datetime(data["time"], errors="raise")
    if labels.isna().any() or labels.duplicated().any():
        raise ValueError("Пустые или повторяющиеся отметки измерений")
    if (
        (labels.dt.minute % 10 != 0)
        | labels.dt.second.ne(0)
        | labels.dt.microsecond.ne(0)
        | labels.dt.nanosecond.ne(0)
    ).any():
        raise ValueError("Измерения должны находиться на сетке 10 минут")
    if labels.dt.tz is None:
        labels = labels.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    labels = labels.dt.tz_convert("UTC")
    if semantics == "interval_end":
        labels -= pd.Timedelta(minutes=10)
    invalid_local_times = int(labels.isna().sum())
    if ((labels.dt.minute % 10 != 0) | labels.dt.second.ne(0)).loc[labels.notna()].any():
        raise ValueError("Интервалы после перевода в UTC не совпадают с сеткой 10 минут")
    data["time"] = labels
    data = data.dropna(subset=["time"]).set_index("time").sort_index()
    if data.empty or data.index.duplicated().any():
        raise ValueError("После преобразования в UTC нет однозначных измерений")
    for column in METRICS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[~np.isfinite(data[column]), column] = np.nan
    data.loc[~data.power.between(0, 1), "power"] = np.nan
    data.loc[data.wind_speed.lt(0), "wind_speed"] = np.nan
    hourly = data.resample("h").mean()
    hourly["sample_count"] = data.resample("h").size()
    hourly["valid_count"] = data[list(METRICS)].resample("h").count().min(axis=1)
    complete = hourly.valid_count.eq(6) & hourly.sample_count.eq(6)
    hourly["quality"] = np.where(complete, "complete", "incomplete")
    hourly.loc[~complete, list(METRICS)] = np.nan
    hourly = hourly.reset_index()
    hourly["available_at"] = hourly["time"] + pd.Timedelta(hours=1)
    hourly.attrs["dropped_ambiguous_or_nonexistent_labels"] = invalid_local_times
    return hourly


def _contexts(hourly: pd.DataFrame, issues: pd.DatetimeIndex) -> pd.DataFrame:
    required = {"time", "available_at", "quality", *METRICS}
    if not required.issubset(hourly.columns) or hourly.empty:
        raise ValueError("Нужны нормализованные почасовые измерения")
    data = hourly.copy()
    for column in ("time", "available_at"):
        if not isinstance(data[column].dtype, pd.DatetimeTZDtype):
            raise ValueError("Почасовые измерения должны иметь явный UTC offset")
        data[column] = data[column].dt.tz_convert("UTC")
    if (
        data.time.duplicated().any()
        or not data.available_at.eq(data.time + pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Неверные времена доступности почасовых измерений")
    if not data.time.eq(data.time.dt.floor("h")).all():
        raise ValueError("Измерения вне часовой сетки")
    complete = data.quality.eq("complete") & np.isfinite(data[list(METRICS)]).all(axis=1)
    complete &= data.power.between(0, 1) & data.wind_speed.ge(0)
    valid = data.loc[complete].sort_values("available_at")
    source = valid.rename(
        columns={
            "available_at": "measurement_available_at",
            **{column: f"last_{column}" for column in METRICS},
        }
    )[["measurement_available_at", *(f"last_{column}" for column in METRICS)]]
    contexts = pd.merge_asof(
        pd.DataFrame({"issue_time": issues}).sort_values("issue_time"),
        source,
        left_on="issue_time",
        right_on="measurement_available_at",
        direction="backward",
    ).set_index("issue_time")
    contexts["lag_age_hours"] = (
        contexts.index.to_series() - contexts.measurement_available_at
    ).dt.total_seconds() / 3600
    # Calendar-aligned reindexing preserves gaps; shifting cannot cross a missing hour.
    data.loc[~complete, list(METRICS)] = np.nan
    observations = data.set_index("available_at")[list(METRICS)].sort_index()
    grid = pd.date_range(
        min(observations.index.min(), issues.min() - pd.Timedelta(hours=24)),
        max(observations.index.max(), issues.max()),
        freq="h",
    )
    observations = observations.reindex(grid)
    for column in METRICS:
        for lag in (1, 6, 24):
            contexts[f"{column}_lag_{lag}h"] = observations[column].shift(lag).reindex(issues)
        for window in (6, 24):
            rolling = observations[column].rolling(window, min_periods=1)
            for stat in ("mean", "std"):
                contexts[f"{column}_{stat}_{window}h"] = getattr(rolling, stat)().reindex(issues)
    for window in (6, 24):
        contexts[f"complete_hours_{window}h"] = (
            observations.power.rolling(window, min_periods=1).count().reindex(issues)
        )
    return contexts.reset_index()


def build_telemetry_requests(
    hourly: pd.DataFrame,
    issue_times: Iterable,
    horizons: Iterable[int] = range(1, 49),
    require_targets: bool = True,
) -> BuildResult:
    issues = pd.DatetimeIndex(sorted({_utc(value) for value in issue_times}))
    horizon_list = list(horizons)
    if not horizon_list or any(
        isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 48
        for value in horizon_list
    ):
        raise ValueError("Горизонт должен содержать целые часы 1–48")
    if issues.empty:
        raise ValueError("Не заданы моменты выпуска")
    result = pd.MultiIndex.from_product(
        [issues, sorted(set(horizon_list))], names=["issue_time", "horizon_hours"]
    ).to_frame(index=False)
    result["target_time"] = result.issue_time + pd.to_timedelta(result.horizon_hours, unit="h")
    result["target_available_at"] = result.target_time + pd.Timedelta(hours=1)
    result = result.merge(_contexts(hourly, issues), on="issue_time", validate="many_to_one")
    target = result.target_time.dt
    result["hour_sin"] = np.sin(2 * np.pi * target.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * target.hour / 24)
    result["day_of_year_sin"] = np.sin(2 * np.pi * target.dayofyear / 366)
    result["day_of_year_cos"] = np.cos(2 * np.pi * target.dayofyear / 366)
    result["target_power"] = np.nan
    if require_targets:
        targets = hourly.loc[hourly.quality.eq("complete")].set_index("time").power
        targets = targets.where(targets.between(0, 1))
        result["target_power"] = result.target_time.map(targets)
    result["reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if require_targets:
        result.loc[result.target_power.isna(), "reason"] = "missing_target"
    result.loc[result.lag_age_hours.gt(2), "reason"] = "stale_measurements"
    result.loc[result.last_power.isna(), "reason"] = "missing_history"
    assert not result.measurement_available_at.gt(result.issue_time).any()
    return BuildResult(result.loc[result.reason.isna()].copy(), result)


def predict_telemetry(model, hourly: pd.DataFrame, issue_at, horizon: int) -> pd.DataFrame:
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or not 1 <= horizon <= 48:
        raise ValueError("Горизонт должен быть целым числом часов 1–48")
    built = build_telemetry_requests(
        hourly,
        [issue_at],
        range(1, horizon + 1),
        require_targets=False,
    )
    if len(built.examples) != horizon:
        reasons = built.audit.reason.dropna().unique().tolist()
        raise ValueError(f"Недостаточно свежей телеметрии: {reasons}")
    with threadpool_limits(limits=1):
        values = np.asarray(model.predict(built.examples.loc[:, FEATURE_COLUMNS]), dtype=float)
    if values.shape != (horizon,) or not np.isfinite(values).all():
        raise ValueError("ML-модель вернула некорректный прогноз")
    return pd.DataFrame({"time": built.examples.target_time, "power": np.clip(values, 0, 1)})
