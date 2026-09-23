"""Weather-only forecasting: no measured SCADA is needed after the training cutoff."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .schemas import validate_weather_forecasts
from .telemetry import _utc, hourly_telemetry

MODEL_TYPE = "nwp_hist_gradient_boosting"
WEATHER_COLUMNS = ("wind_speed_10m", "wind_speed_100m", "temperature_2m", "wind_direction_100m")
FEATURE_COLUMNS = (
    "wind_speed_10m",
    "wind_speed_100m",
    "temperature_2m",
    "direction_sin",
    "direction_cos",
    "horizon_hours",
    "nwp_lead_hours",
    "run_age_hours",
    "hour_sin",
    "hour_cos",
    "year_sin",
    "year_cos",
)
PARAMETERS = {
    "loss": "absolute_error",
    "learning_rate": 0.06,
    "max_iter": 140,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 35,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


@dataclass
class BuildResult:
    examples: pd.DataFrame
    audit: pd.DataFrame


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _weather(frame: pd.DataFrame) -> pd.DataFrame:
    weather = validate_weather_forecasts(frame)
    if weather.empty:
        raise ValueError("Пустой архив прогнозов погоды")
    if (
        weather.time.lt(weather.run).any()
        or weather.wind_speed_10m.lt(0).any()
        or weather.wind_speed_100m.lt(0).any()
        or ~weather.wind_direction_100m.between(0, 360).all()
    ):
        raise ValueError("Некорректные значения или горизонт погодного выпуска")
    for column in ("time", "run"):
        if not weather[column].eq(weather[column].dt.floor("h")).all():
            raise ValueError("Погодные отметки должны лежать на часовой сетке UTC")
    # Different lead files of the same run may be published at different times.
    # Availability is checked per target, never inferred from initialization alone.
    return weather


def build_nwp_requests(
    hourly_targets: pd.DataFrame | None,
    weather: pd.DataFrame,
    issue_times,
    horizons=range(1, 49),
    require_targets=True,
) -> BuildResult:
    """Latest complete eligible 00 UTC run; all features exist at issue time.

    Inference neither needs nor reads ``hourly_targets``. Target labels are UTC
    interval starts. No reanalysis, future measured wind, or forward filling.
    """
    issues = pd.DatetimeIndex(sorted({_utc(value) for value in issue_times}))
    horizons = list(horizons)
    if (
        issues.empty
        or not horizons
        or any(
            isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 48
            for value in horizons
        )
    ):
        raise ValueError("Нужны моменты выпуска и целые горизонты 1–48 часов")
    source = _weather(weather)
    result = pd.MultiIndex.from_product(
        [issues, sorted(set(horizons))],
        names=["issue_time", "horizon_hours"],
    ).to_frame(index=False)
    result["target_time"] = result.issue_time + pd.to_timedelta(result.horizon_hours, unit="h")
    result["target_available_at"] = result.target_time + pd.Timedelta(hours=1)
    result["request_id"] = np.arange(len(result))
    eligible = source.loc[
        source.historical_eligibility.eq("verified") & source.available_at.notna()
    ].rename(
        columns={
            "time": "target_time",
            "run": "weather_run",
            "available_at": "weather_available_at",
        }
    )
    columns = ["target_time", "weather_run", "weather_available_at", *WEATHER_COLUMNS]
    if eligible.empty:
        for name in ("weather_run", "weather_available_at"):
            result[name] = pd.Series(pd.NaT, dtype="datetime64[ns, UTC]", index=result.index)
        for name in WEATHER_COLUMNS:
            result[name] = np.nan
    else:
        candidates = result[["issue_time", "target_time"]].merge(
            eligible.loc[:, columns],
            on="target_time",
            how="inner",
        )
        # Match the runtime archive policy exactly: today's or yesterday's 00 UTC
        # cycle, with one complete cycle for the whole requested horizon.
        candidates = candidates.loc[
            candidates.weather_available_at.le(candidates.issue_time)
            & candidates.weather_run.dt.hour.eq(0)
            & candidates.weather_run.ge(candidates.issue_time.dt.normalize() - pd.Timedelta(days=1))
        ]
        counts = candidates.groupby(["issue_time", "weather_run"]).target_time.nunique()
        complete_runs = counts.loc[counts.eq(len(set(horizons)))].reset_index()
        latest = complete_runs.groupby("issue_time", as_index=False).weather_run.max()
        chosen = candidates.merge(latest, on=["issue_time", "weather_run"], how="inner")
        result = result.merge(
            chosen,
            on=["issue_time", "target_time"],
            how="left",
            validate="one_to_one",
        ).sort_values("request_id")
    result["nwp_lead_hours"] = (result.target_time - result.weather_run).dt.total_seconds() / 3600
    result["run_age_hours"] = (result.issue_time - result.weather_run).dt.total_seconds() / 3600
    direction = np.deg2rad(result.wind_direction_100m)
    result["direction_sin"], result["direction_cos"] = np.sin(direction), np.cos(direction)
    target = result.target_time.dt
    result["hour_sin"], result["hour_cos"] = (
        np.sin(2 * np.pi * target.hour / 24),
        np.cos(2 * np.pi * target.hour / 24),
    )
    result["year_sin"], result["year_cos"] = (
        np.sin(2 * np.pi * target.dayofyear / 366),
        np.cos(2 * np.pi * target.dayofyear / 366),
    )
    result["target_power"] = np.nan
    if require_targets:
        if hourly_targets is None or hourly_targets.time.duplicated().any():
            raise ValueError("Нужны уникальные почасовые целевые измерения")
        targets = hourly_targets.loc[hourly_targets.quality.eq("complete")].set_index("time").power
        result["target_power"] = result.target_time.map(targets.where(targets.between(0, 1)))
    result["reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if require_targets:
        result.loc[result.target_power.isna(), "reason"] = "missing_target"
    unavailable = result.weather_available_at.isna()
    result.loc[unavailable, "reason"] = "no_as_issued_weather"
    if result.weather_available_at.gt(result.issue_time).any():
        raise ValueError("Погода опубликована после момента прогноза")
    if result.weather_run.gt(result.weather_available_at).any():
        raise ValueError("Погодный выпуск ещё не существовал")
    return BuildResult(result.loc[result.reason.isna()].copy(), result)


def _metrics(frame, prediction, train_mean, frozen_power):
    result = {}
    truth = frame.target_power.to_numpy()
    for bucket, mask in {
        "all": np.ones(len(frame), dtype=bool),
        "1_24h": frame.horizon_hours.le(24).to_numpy(),
        "25_48h": frame.horizon_hours.gt(24).to_numpy(),
    }.items():
        value = {"rows": int(mask.sum())}
        for name, pred in {
            "model": prediction,
            "training_mean": np.repeat(train_mean, len(frame)),
            "frozen_persistence": np.repeat(frozen_power, len(frame)),
        }.items():
            errors = pred[mask] - truth[mask]
            value[name] = (
                {
                    "mae": float(np.abs(errors).mean()),
                    "rmse": float(np.sqrt(np.square(errors).mean())),
                }
                if mask.any()
                else None
            )
        result[bucket] = value
    return result


def _frozen_power(hourly, origin):
    known = hourly.loc[hourly.quality.eq("complete") & hourly.available_at.le(origin)]
    if known.empty:
        raise ValueError("Нет мощности для frozen-origin baseline")
    return float(known.sort_values("available_at").power.iloc[-1])


def _frame_digest(frame):
    columns = [
        "issue_time",
        "target_time",
        "target_available_at",
        "weather_run",
        "weather_available_at",
        *FEATURE_COLUMNS,
        "target_power",
    ]
    return _digest(frame.loc[:, columns].to_csv(index=False, float_format="%.17g").encode())


def _summary(frame):
    return {
        "rows": len(frame),
        "unique_targets": int(frame.target_time.nunique()),
        "unique_issues": int(frame.issue_time.nunique()),
        "target_min": frame.target_time.min().isoformat(),
        "target_max": frame.target_time.max().isoformat(),
        "target_available_at_max": frame.target_available_at.max().isoformat(),
        "issue_min": frame.issue_time.min().isoformat(),
        "issue_max": frame.issue_time.max().isoformat(),
        "rows_sha256": _frame_digest(frame),
    }


def train_nwp(
    raw: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    turbine_id: int,
    source_sha256: str,
    timezone: str,
    semantics: str,
    directory: Path,
    weather_provenance: dict,
    train_end="2025-10-01T00:00:00Z",
    validation_end="2026-01-01T00:00:00Z",
    control_end="2026-01-31T12:00:00Z",
    final_issue_at="2026-01-31T12:00:00Z",
) -> dict:
    """Evaluate frozen fit, select only on validation, then separately fit final weights."""
    if not re.fullmatch("[a-f0-9]{64}", source_sha256):
        raise ValueError("Нужен SHA256 исходных измерений")
    if not weather_provenance or not weather_provenance.get("availability_evidence"):
        raise ValueError("Нужно доказательство доступности архива погоды")
    train_cut, validation_cut, control_cut, final_cut = map(
        _utc,
        (train_end, validation_end, control_end, final_issue_at),
    )
    if not train_cut < validation_cut < control_cut <= final_cut:
        raise ValueError("Неверные границы train / validation / control / final")
    hourly = hourly_telemetry(raw, timezone, semantics)
    verified = _weather(weather)
    verified = verified.loc[verified.historical_eligibility.eq("verified")]
    if verified.empty:
        raise ValueError("Нет подтверждённого архива прогнозов погоды")
    first = max(hourly.available_at.min(), verified.available_at.min()).normalize()
    first += pd.Timedelta(hours=12)
    issues = pd.date_range(first, final_cut, freq="24h")
    built = build_nwp_requests(hourly, weather, issues)
    data = built.examples
    masks = {
        "train": data.target_available_at.le(train_cut) & data.target_time.lt(train_cut),
        "validation": data.issue_time.ge(train_cut)
        & data.target_available_at.le(validation_cut)
        & data.target_time.lt(validation_cut),
        "control": data.issue_time.ge(validation_cut)
        & data.target_available_at.le(control_cut)
        & data.target_time.lt(control_cut),
    }
    splits = {name: data.loc[mask].copy() for name, mask in masks.items()}
    for name, frame in splits.items():
        minimum = 30 * 24 if name == "train" else 7 * 24
        if frame.target_time.nunique() < minimum:
            raise ValueError(f"Недостаточно целевых часов {name}: {frame.target_time.nunique()}")
    target_sets = {name: set(frame.target_time) for name, frame in splits.items()}
    intersections = {
        f"{left}_{right}": len(target_sets[left] & target_sets[right])
        for left, right in (
            ("train", "validation"),
            ("train", "control"),
            ("validation", "control"),
        )
    }
    if any(intersections.values()):
        raise ValueError("Пересекаются цели обучающей и проверочных выборок")
    train, validation, control = (splits[name] for name in ("train", "validation", "control"))
    fit_mean = float(train.target_power.mean())
    model = HistGradientBoostingRegressor(**PARAMETERS)
    with threadpool_limits(limits=1):
        model.fit(train.loc[:, FEATURE_COLUMNS], train.target_power)
        val_prediction = np.clip(model.predict(validation.loc[:, FEATURE_COLUMNS]), 0, 1)
    val_metrics = _metrics(validation, val_prediction, fit_mean, _frozen_power(hourly, train_cut))
    promoted = val_metrics["all"]["model"]["mae"] < min(
        val_metrics["all"]["training_mean"]["mae"],
        val_metrics["all"]["frozen_persistence"]["mae"],
    )
    # January labels are used only for this report, not model/parameter selection.
    with threadpool_limits(limits=1):
        control_prediction = np.clip(model.predict(control.loc[:, FEATURE_COLUMNS]), 0, 1)
    control_metrics = _metrics(
        control,
        control_prediction,
        fit_mean,
        _frozen_power(hourly, validation_cut),
    )
    final_rows = data.loc[
        data.target_available_at.le(final_cut) & data.target_time.lt(final_cut)
    ].copy()
    final = HistGradientBoostingRegressor(**PARAMETERS)
    with threadpool_limits(limits=1):
        final.fit(final_rows.loc[:, FEATURE_COLUMNS], final_rows.target_power)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    joblib.dump(model, directory / "evaluation-model.joblib", compress=3)
    joblib.dump(final, directory / "model.joblib", compress=3)
    final_rows.to_parquet(directory / "fit_rows.parquet", index=False)
    eval_sha = _digest((directory / "evaluation-model.joblib").read_bytes())
    report = {
        "model_type": MODEL_TYPE,
        "parameters": PARAMETERS,
        "splits": {name: _summary(frame) for name, frame in splits.items()},
        "target_overlap_counts": intersections,
        "validation": val_metrics,
        "control": control_metrics,
        "validation_model_sha256": eval_sha,
        "promotion": {
            "promoted": promoted,
            "frozen_before_control": True,
            "criterion": "validation MAE below training mean and frozen persistence",
        },
        "final_fit": _summary(final_rows),
        "final_fit_is_separate_from_evaluation": True,
        "validation_control_telemetry_features_used": False,
        "prediction_requires_fresh_scada": False,
        "february_metrics": None,
        "january_is_untouched_test": False,
        "exclusions": {
            "requested": len(built.audit),
            "reasons": {
                str(k): int(v) for k, v in built.audit.reason.dropna().value_counts().items()
            },
            "crossing_split_boundaries": int(
                (~(masks["train"] | masks["validation"] | masks["control"])).sum()
            ),
        },
        "weather_provenance": weather_provenance,
    }
    for name, (start, end) in {
        "train": (None, train_cut),
        "validation": (train_cut, validation_cut),
        "control": (validation_cut, control_cut),
    }.items():
        requested = built.audit.loc[
            built.audit.target_time.lt(end)
            & built.audit.target_available_at.le(end)
            & (True if start is None else built.audit.issue_time.ge(start))
        ]
        report["splits"][name].update(
            {
                "requested_rows": len(requested),
                "usable_fraction": len(splits[name]) / len(requested) if len(requested) else 0.0,
                "excluded_reasons": {
                    str(key): int(value)
                    for key, value in requested.reason.dropna().value_counts().items()
                },
            }
        )
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "artifact_version": 1,
        "model_type": MODEL_TYPE,
        "turbine_id": turbine_id,
        "source_sha256": source_sha256,
        "timezone": timezone,
        "timestamp_semantics": semantics,
        "promoted": promoted,
        "feature_columns": list(FEATURE_COLUMNS),
        "horizon_max": 48,
        "usable_from": final_cut.isoformat(),
        "training_end": final_cut.isoformat(),
        "fit_target_available_at_max": final_rows.target_available_at.max().isoformat(),
        "evaluation_training_end": train_cut.isoformat(),
        "validation_end": validation_cut.isoformat(),
        "control_end": control_cut.isoformat(),
        "model_sha256": _digest((directory / "model.joblib").read_bytes()),
        "evaluation_model_sha256": eval_sha,
        "report_sha256": _digest((directory / "report.json").read_bytes()),
        "fit_rows_sha256": _digest((directory / "fit_rows.parquet").read_bytes()),
        "sklearn_version": sklearn.__version__,
        "weather_used": True,
        "requires_current_scada": False,
        "forecast_units": "normalized_power",
        "target_time_semantics": "interval_start",
        "issue_hour_utc": 12,
        "validation_mae": val_metrics["all"]["model"]["mae"],
        "mean_validation_mae": val_metrics["all"]["training_mean"]["mae"],
        "frozen_persistence_validation_mae": val_metrics["all"]["frozen_persistence"]["mae"],
        "weather_provenance": weather_provenance,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def predict_nwp_model(model, weather, issue_at, horizon=48):
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or not 1 <= horizon <= 48:
        raise ValueError("Горизонт должен быть целым числом 1–48")
    built = build_nwp_requests(None, weather, [issue_at], range(1, horizon + 1), False)
    if len(built.examples) != horizon:
        raise ValueError("Архив as-issued погоды не покрывает весь горизонт")
    with threadpool_limits(limits=1):
        values = np.asarray(model.predict(built.examples.loc[:, FEATURE_COLUMNS]), dtype=float)
    if values.shape != (horizon,) or not np.isfinite(values).all():
        raise ValueError("Некорректный результат погодной модели")
    frame = built.examples.loc[:, ["target_time", "weather_run", "weather_available_at"]].copy()
    frame["power"] = np.clip(values, 0, 1)
    return frame.rename(columns={"target_time": "time"})
