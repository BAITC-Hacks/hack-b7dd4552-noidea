"""Явные контракты входных таблиц и правила интерпретации времени."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HOURLY_MEASUREMENT_COLUMNS = (
    "time",
    "wind_speed",
    "power",
    "temperature",
    "sample_count",
    "valid_count",
    "completeness",
    "quality",
)
WEATHER_COLUMNS = (
    "time",
    "run",
    "available_at",
    "historical_eligibility",
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
)
QUALITY_VALUES = frozenset({"complete", "partial", "missing"})
ELIGIBILITY_VALUES = frozenset({"verified", "unverified", "ineligible"})
CANONICAL_TIME_COLUMNS = (
    "time_utc",
    "available_at",
    "source_timezone",
    "source_timestamp_semantics",
)


class SchemaError(ValueError):
    """Таблица не удовлетворяет публичному контракту модуля."""


@dataclass(frozen=True)
class MeasurementTimeConfig:
    """Как интерпретировать стенное время SCADA.

    ``source_timezone`` нельзя угадывать: вызывающий код обязан передать его
    явно. Почасовой интервал [10:00, 11:00) становится доступен в 11:00.
    Семантика относится к исходным 10-минутным меткам: interval_end требует
    сдвига на 10 минут до агрегации, а не переобозначения готового среднего.
    """

    source_timezone: str
    timestamp_semantics: Literal["interval_start", "interval_end"]
    max_lag_hours: float = 2.0

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.source_timezone)
        except Exception as exc:  # pragma: no cover - детали зависят от ОС
            raise SchemaError(f"Неизвестный часовой пояс: {self.source_timezone}") from exc
        if self.timestamp_semantics not in {"interval_start", "interval_end"}:
            raise SchemaError("timestamp_semantics: interval_start или interval_end")
        if not isfinite(self.max_lag_hours) or self.max_lag_hours < 0:
            raise SchemaError("max_lag_hours должен быть конечным неотрицательным числом")

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def _required(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"{label}: нет обязательных столбцов: {', '.join(missing)}")


def _to_utc(values: pd.Series, field: str, *, allow_na: bool = False) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if not allow_na and parsed.isna().any():
        raise SchemaError(f"{field} содержит пустое или неразбираемое время")
    if allow_na and parsed.isna().all():
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{field} должно содержать время с явным UTC-смещением")
    return parsed.dt.tz_convert("UTC")


def validate_hourly_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Проверить предварительный контракт первого этапа без преобразования TZ."""

    _required(frame, HOURLY_MEASUREMENT_COLUMNS, "Почасовые измерения")
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"], errors="coerce")
    if result["time"].isna().any():
        raise SchemaError("time содержит пустое или неразбираемое значение")
    if result["time"].duplicated().any():
        raise SchemaError("time содержит дублирующиеся часовые отметки")
    for column in ("wind_speed", "power", "temperature", "completeness"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("sample_count", "valid_count"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or (result[column] < 0).any():
            raise SchemaError(f"{column} должен быть неотрицательным числом")
    result["quality"] = result["quality"].astype(str)
    if not set(result["quality"]).issubset(QUALITY_VALUES):
        raise SchemaError("quality: допустимы complete / partial / missing")
    if (result["valid_count"] > result["sample_count"]).any():
        raise SchemaError("valid_count не может быть больше sample_count")
    complete = result["quality"].eq("complete")
    complete_fields = ["wind_speed", "power", "temperature"]
    if result.loc[complete, complete_fields].isna().any(axis=None):
        raise SchemaError("complete-час обязан содержать все численные средние")
    if not np.isfinite(result.loc[complete, complete_fields].to_numpy(dtype=float)).all():
        raise SchemaError("complete-час не может содержать бесконечные значения")
    if not result.loc[complete, "power"].between(0, 1).all():
        raise SchemaError("Нормализованная мощность должна быть в диапазоне 0..1")
    if result.loc[complete, "wind_speed"].lt(0).any():
        raise SchemaError("Скорость ветра не может быть отрицательной")
    if (
        result.loc[complete, ["sample_count", "valid_count"]].ne(6).any(axis=None)
        or result.loc[complete, "completeness"].ne(1.0).any()
    ):
        raise SchemaError("complete-час требует шесть валидных 10-минутных измерений")
    incomplete = ~complete
    if result.loc[incomplete, complete_fields].notna().any(axis=None):
        raise SchemaError("partial/missing-час не должен содержать численные средние")
    return result.sort_values("time").reset_index(drop=True)


def measurement_time_to_utc(
    measurements: pd.DataFrame, config: MeasurementTimeConfig
) -> pd.DataFrame:
    """Добавить UTC-время и момент, когда измерение действительно стало известно."""

    result = validate_hourly_measurements(measurements)
    canonical = set(CANONICAL_TIME_COLUMNS) & set(result.columns)
    if canonical:
        _required(result, CANONICAL_TIME_COLUMNS, "Почасовые UTC-интервалы")
        if (
            not result["source_timezone"].eq(config.source_timezone).all()
            or not result["source_timestamp_semantics"].eq(config.timestamp_semantics).all()
        ):
            raise SchemaError("Временная конфигурация не совпадает с агрегацией исходных данных")
        result["time_utc"] = _to_utc(result["time_utc"], "time_utc")
        result["available_at"] = _to_utc(result["available_at"], "available_at")
        if result["time_utc"].duplicated().any():
            raise SchemaError("Дублирующиеся UTC-интервалы")
        if (result["available_at"] < result["time_utc"] + pd.Timedelta(hours=1)).any():
            raise SchemaError("Почасовое измерение не может быть доступно раньше конца часа")
        result["time_status"] = "resolved"
        return result
    if config.timestamp_semantics == "interval_end":
        raise SchemaError(
            "Legacy hourly не поддерживает interval_end: сначала агрегируйте raw 10-min "
            "с временной конфигурацией; нужны time_utc/available_at и происхождение агрегации"
        )
    time = result["time"]
    if isinstance(time.dtype, pd.DatetimeTZDtype):
        utc = time.dt.tz_convert("UTC")
    else:
        localized = time.dt.tz_localize(config.source_timezone, ambiguous="NaT", nonexistent="NaT")
        utc = localized.dt.tz_convert("UTC")
    result["time_utc"] = utc
    # При исторической смене смещения один wall-clock час может быть
    # неоднозначным. Его нельзя молча привязывать к одному из двух UTC-часов:
    # последующий feature builder просто не использует такую запись.
    result["time_status"] = "resolved"
    result.loc[result["time_utc"].isna(), "time_status"] = "unresolved_local_time"
    interval = (
        pd.Timedelta(1, unit="h")
        if config.timestamp_semantics == "interval_start"
        else pd.Timedelta(0, unit="h")
    )
    result["available_at"] = result["time_utc"] + interval
    return result


def validate_weather_forecasts(
    frame: pd.DataFrame, *, require_verified: bool = False
) -> pd.DataFrame:
    """Проверить архив выпусков, не подменяя фактическую погоду прогнозом."""

    _required(frame, WEATHER_COLUMNS, "Архив прогнозов погоды")
    result = frame.copy()
    result["time"] = _to_utc(result["time"], "weather.time")
    result["run"] = _to_utc(result["run"], "weather.run")
    result["available_at"] = _to_utc(result["available_at"], "weather.available_at", allow_na=True)
    result["historical_eligibility"] = result["historical_eligibility"].astype(str)
    if not set(result["historical_eligibility"]).issubset(ELIGIBILITY_VALUES):
        raise SchemaError("historical_eligibility: verified / unverified / ineligible")
    weather_numbers = ["temperature_2m", "wind_speed_10m", "wind_speed_100m", "wind_direction_100m"]
    for column in weather_numbers:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[weather_numbers].isna().any(axis=None):
        raise SchemaError("Архив погоды содержит пропуски обязательных переменных")
    if not np.isfinite(result[weather_numbers].to_numpy(dtype=float)).all():
        raise SchemaError("Архив погоды содержит бесконечные значения")
    if result.duplicated(["run", "time"]).any():
        raise SchemaError("Архив содержит дубликат пары run/time")
    published = result["available_at"].notna()
    if (result.loc[published, "available_at"] < result.loc[published, "run"]).any():
        raise SchemaError("available_at не может быть раньше инициализации run")
    if require_verified and (
        result["available_at"].isna().any()
        or ~result["historical_eligibility"].eq("verified").all()
    ):
        raise SchemaError("Для честного обучения нужны verified-прогнозы с известным available_at")
    return result.sort_values(["available_at", "time", "run"], na_position="last").reset_index(
        drop=True
    )
