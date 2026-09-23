"""Сохранение модели вместе с достаточным контекстом для воспроизведения."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from .features import FEATURE_COLUMNS, parse_utc
from .schemas import MeasurementTimeConfig


def save_artifact(
    directory: str | Path,
    *,
    model_type: str,
    payload: Any,
    turbine_id: str,
    time_config: MeasurementTimeConfig,
    training_bounds: dict[str, str | None],
    metrics: dict[str, Any] | None = None,
    weather_contract: dict[str, Any] | None = None,
    trained_horizon_hours: int = 48,
) -> Path:
    """Записать бинарный payload и читаемый manifest.json в новый каталог."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target / "model.joblib")
    manifest = {
        "artifact_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_type": model_type,
        "turbine_id": str(turbine_id),
        "measurement_time": time_config.to_dict(),
        "training_bounds": training_bounds,
        "feature_schema": list(FEATURE_COLUMNS) if model_type == "boosting" else [],
        "metrics": metrics or {},
        "weather_contract": weather_contract or {},
        "trained_horizon_hours": trained_horizon_hours,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_artifact(directory: str | Path) -> tuple[dict[str, Any], Any]:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != 1:
        raise ValueError("Неподдерживаемая версия ML-артефакта")
    return manifest, joblib.load(source / "model.joblib")


def validate_artifact_request(
    manifest: dict[str, Any],
    *,
    turbine_id: str | int,
    time_config: MeasurementTimeConfig,
    issue_at: str,
    horizon: int,
) -> None:
    """Refuse retrospective leakage or incompatible interpretation at inference."""
    if str(manifest.get("turbine_id")) != str(turbine_id):
        raise ValueError("Артефакт обучен для другой турбины")
    saved = manifest.get("measurement_time", {})
    if (
        saved.get("source_timezone") != time_config.source_timezone
        or saved.get("timestamp_semantics") != time_config.timestamp_semantics
        or saved.get("max_lag_hours", 2.0) != time_config.max_lag_hours
    ):
        raise ValueError("Временная конфигурация артефакта отличается; переобучите модель")
    bounds = manifest.get("training_bounds", {})
    cutoff = bounds.get("target_time_max_exclusive_utc")
    if not cutoff:
        raise ValueError("В артефакте нет границы обучения; переобучите модель с train-end")
    issue = parse_utc(issue_at)
    if issue < parse_utc(cutoff):
        raise ValueError("Нельзя прогнозировать раньше границы обучения артефакта")
    known_at = bounds.get("target_available_at_max_utc")
    if known_at and issue < parse_utc(known_at):
        raise ValueError("Обучающие цели ещё не были доступны в момент прогноза")
    maximum = manifest.get("trained_horizon_hours")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 48:
        raise ValueError("В артефакте не указан проверяемый горизонт; переобучите модель")
    if not 1 <= horizon <= maximum:
        raise ValueError(f"Горизонт превышает обученный диапазон 1..{maximum}")
