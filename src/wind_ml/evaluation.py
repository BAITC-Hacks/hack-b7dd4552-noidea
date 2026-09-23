"""Замороженная walk-forward проверка и метрики по диапазонам горизонта."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .features import (
    build_forecast_requests,
    build_persistence_requests,
    parse_utc,
    verified_issue_times,
)
from .models import PowerCurve, fit_boosting, persistence_predict
from .schemas import MeasurementTimeConfig, measurement_time_to_utc

ModelType = Literal["persistence", "curve", "boosting"]


@dataclass(frozen=True)
class ValidationFold:
    """[issue_start, issue_end): модель получает цели строго раньше issue_start."""

    issue_start: pd.Timestamp
    issue_end: pd.Timestamp

    @classmethod
    def create(cls, issue_start: str, issue_end: str) -> ValidationFold:
        start, end = parse_utc(issue_start), parse_utc(issue_end)
        if end <= start:
            raise ValueError("Конец fold должен быть позже начала")
        return cls(start, end)


def metrics_from_audit(
    audit: pd.DataFrame, *, turbine_id: str, model_type: str
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total = len(audit)
    for label, mask in (
        ("1-24h", audit["horizon_hours"].between(1, 24)),
        ("25-48h", audit["horizon_hours"].between(25, 48)),
    ):
        bucket = audit.loc[mask]
        usable = bucket.loc[bucket["reason"].isna() & bucket["prediction"].notna()]
        if usable.empty:
            mae = rmse = None
        else:
            error = usable["prediction"] - usable["target_power"]
            mae = float(np.abs(error).mean())
            rmse = float(np.sqrt(np.mean(np.square(error))))
        requested = len(bucket)
        rows.append(
            {
                "turbine_id": str(turbine_id),
                "model": model_type,
                "horizon": label,
                "mae": mae,
                "rmse": rmse,
                "evaluated_hours": len(usable),
                "excluded_hours": requested - len(usable),
                "excluded_share": None if requested == 0 else float(1 - len(usable) / requested),
                "requested_hours": int(requested),
            }
        )
    return {"requested_hours": total, "by_horizon": rows}


def _evaluation_issues(
    model_type: ModelType,
    weather: pd.DataFrame | None,
    fold: ValidationFold,
) -> pd.DatetimeIndex:
    if model_type == "persistence":
        return pd.date_range(fold.issue_start, fold.issue_end, inclusive="left", freq="h")
    if weather is None:
        raise ValueError("curve и boosting требуют архив прогнозов погоды")
    issues = verified_issue_times(weather)
    return issues[(issues >= fold.issue_start) & (issues < fold.issue_end)]


def _training_measurements_before(
    measurements: pd.DataFrame, config: MeasurementTimeConfig, boundary: pd.Timestamp
) -> pd.DataFrame:
    converted = measurement_time_to_utc(measurements, config)
    # Цель с меткой boundary и позднее не может участвовать в замороженной модели.
    keep_times = converted.loc[
        (converted["time_utc"] < boundary) & (converted["available_at"] <= boundary), "time"
    ]
    return measurements.loc[measurements["time"].isin(keep_times)].copy()


def _limit_validation_targets(audit: pd.DataFrame, fold: ValidationFold) -> None:
    outside = (
        audit["target_time"].lt(fold.issue_start)
        | audit["target_time"].ge(fold.issue_end)
        | audit["target_available_at"].gt(fold.issue_end)
    )
    audit.loc[outside, "reason"] = "target_outside_fold"


def validate_walk_forward(
    *,
    model_type: ModelType,
    measurements: pd.DataFrame,
    time_config: MeasurementTimeConfig,
    folds: list[ValidationFold],
    turbine_id: str,
    weather: pd.DataFrame | None = None,
    horizons: range = range(1, 49),
) -> tuple[dict[str, object], pd.DataFrame]:
    """Оценить folds, каждый раз обучая новый экземпляр только на прошлом.

    Валидационные цели никогда не входят в fit: даже выпуск до границы fold
    отбрасывается, если его горизонт пересёк ``issue_start``.
    """

    if not folds:
        raise ValueError("Нужен хотя бы один validation fold")
    folds = sorted(folds, key=lambda fold: fold.issue_start)
    for index, fold in enumerate(folds):
        if fold.issue_end <= fold.issue_start:
            raise ValueError("Конец fold должен быть позже начала")
        if index and fold.issue_start < folds[index - 1].issue_end:
            raise ValueError("Validation folds не должны пересекаться или дублироваться")
    all_audits: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(folds, start=1):
        eval_issues = _evaluation_issues(model_type, weather, fold)
        if model_type == "persistence":
            result = build_persistence_requests(
                measurements, time_config, eval_issues, horizons, require_targets=True
            )
            _limit_validation_targets(result.audit, fold)
            result.audit["prediction"] = np.where(
                result.audit["reason"].isna(),
                persistence_predict(result.audit["last_power"]),
                np.nan,
            )
        elif model_type == "curve":
            assert weather is not None
            curve = PowerCurve.fit(
                _training_measurements_before(measurements, time_config, fold.issue_start)
            )
            result = build_forecast_requests(
                measurements, weather, time_config, eval_issues, horizons, require_targets=True
            )
            _limit_validation_targets(result.audit, fold)
            result.audit["prediction"] = np.where(
                result.audit["reason"].isna(),
                curve.predict(result.audit["wind_speed_100m"].fillna(0)),
                np.nan,
            )
        else:
            assert weather is not None
            train_issues = verified_issue_times(weather, before=fold.issue_start)
            train = build_forecast_requests(
                measurements, weather, time_config, train_issues, horizons, require_targets=True
            ).examples
            # Заморозка по target_time устраняет даже пересекающиеся горизонты.
            train = train.loc[
                (train["target_time"] < fold.issue_start)
                & (train["target_available_at"] <= fold.issue_start)
            ].copy()
            model = fit_boosting(train)
            result = build_forecast_requests(
                measurements, weather, time_config, eval_issues, horizons, require_targets=True
            )
            _limit_validation_targets(result.audit, fold)
            result.audit["prediction"] = np.nan
            valid_positions = result.audit["reason"].isna()
            if valid_positions.any():
                result.audit.loc[valid_positions, "prediction"] = model.predict(
                    result.audit.loc[valid_positions]
                )
        result.audit["fold"] = fold_number
        result.audit["fold_issue_start"] = fold.issue_start
        all_audits.append(result.audit)
    audit = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame()
    return metrics_from_audit(audit, turbine_id=turbine_id, model_type=model_type), audit
