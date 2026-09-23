"""Reproducible chronological training; validation selects, control data only reports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .telemetry import FEATURE_COLUMNS, _utc, build_telemetry_requests, hourly_telemetry

MODEL_TYPE = "telemetry_hist_gradient_boosting"
PARAMETERS = {
    "loss": "absolute_error",
    "learning_rate": 0.06,
    "max_iter": 120,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    result = {}
    truth = frame.target_power.to_numpy()
    baseline = frame.last_power.to_numpy()
    for label, mask in {
        "all": np.ones(len(frame), dtype=bool),
        "1_24h": frame.horizon_hours.le(24).to_numpy(),
        "25_48h": frame.horizon_hours.gt(24).to_numpy(),
    }.items():
        if not mask.any():
            result[label] = {"rows": 0, "model": None, "persistence": None}
            continue
        scores = {"rows": int(mask.sum())}
        for name, values in {"model": predicted, "persistence": baseline}.items():
            error = values[mask] - truth[mask]
            scores[name] = {
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
            }
        result[label] = scores
    return result


def _split_summary(frame: pd.DataFrame) -> dict:
    unique_targets = frame.target_time.drop_duplicates().sort_values()
    return {
        "rows": len(frame),
        "unique_target_hours": len(unique_targets),
        "unique_issue_times": int(frame.issue_time.nunique()),
        "issue_min": frame.issue_time.min().isoformat(),
        "issue_max": frame.issue_time.max().isoformat(),
        "target_min": unique_targets.min().isoformat(),
        "target_max": unique_targets.max().isoformat(),
        "target_available_at_max": frame.target_available_at.max().isoformat(),
        "feature_available_at_max": frame.measurement_available_at.max().isoformat(),
        "unique_target_sha256": _digest("\n".join(unique_targets.astype(str)).encode()),
    }


def _row_hashes(frame: pd.DataFrame) -> pd.Series:
    columns = [
        "issue_time",
        "target_time",
        "target_available_at",
        "measurement_available_at",
        *FEATURE_COLUMNS,
        "target_power",
    ]

    # Hex-encoded float values preserve exact training values across JSON/CSV formatting.
    def encode(row):
        return _digest(
            "|".join(
                value.isoformat() if isinstance(value, pd.Timestamp) else float(value).hex()
                for value in row
            ).encode()
        )

    return pd.Series(
        (encode(row) for row in frame.loc[:, columns].itertuples(index=False, name=None)),
        index=frame.index,
    )


def train_telemetry(
    raw: pd.DataFrame,
    *,
    turbine_id: int,
    source_sha256: str,
    timezone: str,
    semantics: str,
    directory: Path,
    train_end="2025-10-01T00:00:00Z",
    validation_end="2026-01-01T00:00:00Z",
    test_end="2026-02-01T00:00:00Z",
) -> dict:
    """Train only on pre-cutoff labels and select on a separate validation interval.

    Fixed hyperparameters, no random split, no preprocessing learned from control
    data, no refit on validation, no weather, and no hidden early-stopping split.
    The January period is a retrospective control: timezone exploration previously
    inspected it, so it is explicitly NOT advertised as an untouched holdout.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("Нужен SHA256 исходного CSV")
    if isinstance(turbine_id, bool) or not isinstance(turbine_id, int) or turbine_id < 1:
        raise ValueError("Некорректный идентификатор турбины")
    cut_train, cut_validation, cut_test = map(_utc, (train_end, validation_end, test_end))
    if not cut_train < cut_validation < cut_test:
        raise ValueError("Границы train / validation / control должны возрастать")
    hourly = hourly_telemetry(raw, timezone, semantics)
    start = hourly.available_at.min().ceil("6h")
    end = min(hourly.time.max(), cut_test)
    if start >= cut_train or end <= cut_validation:
        raise ValueError("История не покрывает фиксированные train / validation / control периоды")
    issues = pd.date_range(start, end, freq="6h", inclusive="left")
    built = build_telemetry_requests(hourly, issues)
    data = built.examples
    masks = {
        "train": data.target_time.lt(cut_train) & data.target_available_at.le(cut_train),
        "validation": data.issue_time.ge(cut_train)
        & data.target_time.lt(cut_validation)
        & data.target_available_at.le(cut_validation),
        "control": data.issue_time.ge(cut_validation)
        & data.target_time.lt(cut_test)
        & data.target_available_at.le(cut_test),
    }
    splits = {name: data.loc[mask].copy() for name, mask in masks.items()}
    minimums = {"train": 30 * 24, "validation": 7 * 24, "control": 7 * 24}
    for name, frame in splits.items():
        if frame.target_time.nunique() < minimums[name]:
            raise ValueError(
                f"Недостаточно полных target-часов в {name}: "
                f"{frame.target_time.nunique()} < {minimums[name]}"
            )
    # A target may repeat across issue times inside one split, but never between splits.
    target_sets = {name: set(frame.target_time) for name, frame in splits.items()}
    overlaps = {
        f"{left}_{right}": len(target_sets[left] & target_sets[right])
        for left, right in (
            ("train", "validation"),
            ("train", "control"),
            ("validation", "control"),
        )
    }
    if any(overlaps.values()) or data.measurement_available_at.gt(data.issue_time).any():
        raise ValueError("Нарушено хронологическое разделение признаков или целевых часов")
    train, validation, control = (splits[name] for name in ("train", "validation", "control"))
    model = HistGradientBoostingRegressor(**PARAMETERS)
    with threadpool_limits(limits=1):
        model.fit(train.loc[:, FEATURE_COLUMNS], train.target_power)
        validation_predictions = np.clip(model.predict(validation.loc[:, FEATURE_COLUMNS]), 0, 1)
    validation_metrics = _metrics(validation, validation_predictions)
    # Freeze this decision BEFORE requesting any predictions or metrics on control rows.
    promoted = (
        validation_metrics["all"]["model"]["mae"] < validation_metrics["all"]["persistence"]["mae"]
    )
    with threadpool_limits(limits=1):
        control_predictions = np.clip(model.predict(control.loc[:, FEATURE_COLUMNS]), 0, 1)
    control_metrics = _metrics(control, control_predictions)
    report = {
        "model_type": MODEL_TYPE,
        "parameters": PARAMETERS,
        "split_boundaries": {
            "train_end": cut_train.isoformat(),
            "validation_end": cut_validation.isoformat(),
            "control_end": cut_test.isoformat(),
        },
        "splits": {name: _split_summary(frame) for name, frame in splits.items()},
        "target_overlap_counts": overlaps,
        "feature_availability_violations": int(
            data.measurement_available_at.gt(data.issue_time).sum()
        ),
        "fit_labels_available_at_max": train.target_available_at.max().isoformat(),
        "fit_labels_available_by_train_end": bool(train.target_available_at.le(cut_train).all()),
        "validation": validation_metrics,
        "control": control_metrics,
        "test": control_metrics,
        "promotion": {
            "promoted": promoted,
            "criterion": "validation MAE strictly below persistence MAE",
            "frozen_before_control_metrics": True,
            "refit_after_selection": False,
        },
        "exclusions": {
            "requested_rows": len(built.audit),
            "reasons": {
                str(key): int(value)
                for key, value in built.audit.reason.dropna().value_counts().items()
            },
            "crossing_split_boundary_or_outside_window": int(
                (~(masks["train"] | masks["validation"] | masks["control"])).sum()
            ),
            "incomplete_hours": int(hourly.quality.ne("complete").sum()),
            "dropped_ambiguous_or_nonexistent_labels": hourly.attrs.get(
                "dropped_ambiguous_or_nonexistent_labels", 0
            ),
        },
        "data_assessment": {
            "weather_used": False,
            "telemetry_only": True,
            "timezone_is_user_assumption": True,
            "timestamps_assumed_instantly_available": True,
            "source_period_start": hourly.time.min().isoformat(),
            "source_period_end": hourly.time.max().isoformat(),
            "complete_hours": int(hourly.quality.eq("complete").sum()),
            "complete_hour_fraction": float(hourly.quality.eq("complete").mean()),
            "unique_turbines": 1,
            "forecast_horizons": [1, 48],
            "issue_stride_hours": 6,
            "independent_untouched_test_available": False,
            "control_warning": (
                "January 2026 was inspected during timezone analysis. This is a retrospective "
                "control, not an untouched final test. February-or-later labels are needed "
                "for an independent final test. Temporal row leakage is checked separately."
            ),
        },
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate the old manifest before replacing files so partially saved artifacts fail closed.
    (directory / "manifest.json").unlink(missing_ok=True)
    joblib.dump(model, directory / "model.joblib", compress=3)
    fit_rows = train.loc[
        :,
        [
            "issue_time",
            "target_time",
            "target_available_at",
            "measurement_available_at",
        ],
    ].copy()
    fit_rows["row_sha256"] = _row_hashes(train)
    fit_rows.to_parquet(directory / "fit_rows.parquet", index=False)
    report["fit_rows_sha256"] = _digest((directory / "fit_rows.parquet").read_bytes())
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "artifact_version": 1,
        "model_type": MODEL_TYPE,
        "promoted": promoted,
        "turbine_id": turbine_id,
        "source_sha256": source_sha256,
        "timezone": timezone,
        "timestamp_semantics": semantics,
        "feature_columns": list(FEATURE_COLUMNS),
        "horizons": [1, 48],
        "horizon_max": 48,
        "usable_from": cut_validation.isoformat(),
        "trained_at": datetime.now(UTC).isoformat(),
        "training_end": cut_train.isoformat(),
        "validation_end": cut_validation.isoformat(),
        "control_end": cut_test.isoformat(),
        "test_end": cut_test.isoformat(),
        "model_sha256": _digest((directory / "model.joblib").read_bytes()),
        "report_sha256": _digest((directory / "report.json").read_bytes()),
        "fit_rows_sha256": report["fit_rows_sha256"],
        "sklearn_version": sklearn.__version__,
        "validation_mae": validation_metrics["all"]["model"]["mae"],
        "persistence_validation_mae": validation_metrics["all"]["persistence"]["mae"],
        "weather_used": False,
        "independent_untouched_test_available": False,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest
