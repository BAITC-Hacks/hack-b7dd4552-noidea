"""Causality, clock semantics and chronological train/validation/control isolation."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_ml.telemetry import (
    FEATURE_COLUMNS,
    build_telemetry_requests,
    hourly_telemetry,
    predict_telemetry,
)
from wind_ml.training import train_telemetry


def raw_data(start="2025-01-01", days=4):
    times = pd.date_range(start, periods=days * 24 * 6, freq="10min")
    hours = np.arange(len(times)) / 6
    return pd.DataFrame(
        {
            "time": times,
            "power": np.clip(0.45 + 0.3 * np.sin(hours / 10) + 0.1 * np.cos(hours / 19), 0, 1),
            "wind_speed": 7 + 4 * np.sin(hours / 10),
            "temperature": 15 + 5 * np.cos(hours / 24),
        }
    )


def test_interval_end_shift_precedes_aggregation():
    start_labels = raw_data(days=2)
    end_labels = start_labels.assign(time=start_labels.time + pd.Timedelta(minutes=10))
    starts = hourly_telemetry(start_labels, "Etc/GMT-6", "interval_start")
    ends = hourly_telemetry(end_labels, "Etc/GMT-6", "interval_end")
    assert_frame_equal(starts, ends)
    assert starts.time.iloc[0] == pd.Timestamp("2024-12-31T18:00:00Z")
    assert starts.available_at.iloc[0] == pd.Timestamp("2024-12-31T19:00:00Z")
    assert starts.quality.eq("complete").all()


def test_incomplete_hours_and_unknown_times_fail_closed():
    raw = raw_data(days=2).drop(index=7)
    raw.loc[14, "power"] = np.nan
    hourly = hourly_telemetry(raw, "UTC", "interval_start")
    assert hourly.iloc[1:3].quality.eq("incomplete").all()
    assert hourly.iloc[1:3][["power", "wind_speed", "temperature"]].isna().all().all()
    with pytest.raises(ValueError, match="часовой пояс"):
        hourly_telemetry(raw, "unknown", "interval_start")
    with pytest.raises(ValueError, match="семантика"):
        hourly_telemetry(raw, "UTC", "unknown")
    raw.loc[0, "time"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="сетке"):
        hourly_telemetry(raw, "UTC", "interval_start")


def test_ambiguous_local_hour_is_not_inferred_from_future():
    raw = raw_data("2024-02-29", days=3)
    hourly = hourly_telemetry(raw, "Asia/Almaty", "interval_start")
    assert hourly.attrs["dropped_ambiguous_or_nonexistent_labels"] == 6
    assert hourly.quality.eq("incomplete").sum() >= 1


def test_offset_must_not_make_an_hour_available_before_last_interval_ends():
    # 00:00 local becomes 18:15 UTC. Averaging the six UTC xx:05..xx:55 starts
    # into xx:00 would incorrectly make their final interval available 5min early.
    with pytest.raises(ValueError, match="UTC не совпадают"):
        hourly_telemetry(raw_data(), "Asia/Kathmandu", "interval_start")


@pytest.mark.parametrize("semantics", ["interval_start", "interval_end"])
def test_all_features_and_predictions_ignore_future_observations(semantics):
    raw = raw_data(days=4)
    if semantics == "interval_end":
        raw["time"] += pd.Timedelta(minutes=10)
    issue = pd.Timestamp("2025-01-02T12:00:00Z")
    shifted = raw.copy()
    future = (
        raw.time.gt(issue.tz_localize(None))
        if semantics == "interval_end"
        else (raw.time.ge(issue.tz_localize(None)))
    )
    shifted.loc[future, ["power", "wind_speed", "temperature"]] = [0.999, 60, -30]
    before = hourly_telemetry(raw, "UTC", semantics)
    after = hourly_telemetry(shifted, "UTC", semantics)
    original = build_telemetry_requests(before, [issue], require_targets=False).examples
    altered = build_telemetry_requests(after, [issue], require_targets=False).examples
    assert_frame_equal(original, altered)
    assert original.target_power.isna().all()
    assert original.measurement_available_at.le(original.issue_time).all()
    assert "target_power" not in FEATURE_COLUMNS

    class PastOnlyModel:
        def predict(self, features):
            assert tuple(features.columns) == FEATURE_COLUMNS
            return features.last_power.to_numpy() * 0.5 + features.power_mean_24h.to_numpy() * 0.5

    assert_frame_equal(
        predict_telemetry(PastOnlyModel(), before, issue, 48),
        predict_telemetry(PastOnlyModel(), after, issue, 48),
    )
    # Also works when labels for all target hours are absent entirely.
    truncated = before.loc[before.available_at.le(issue)]
    no_targets = build_telemetry_requests(truncated, [issue], require_targets=False).examples
    assert_frame_equal(original, no_targets)


def test_last_hour_availability_staleness_and_lags_preserve_gaps():
    hourly = hourly_telemetry(raw_data(days=4), "UTC", "interval_start")
    issue = pd.Timestamp("2025-01-02T12:00:00Z")
    row = build_telemetry_requests(hourly, [issue], [1]).examples.iloc[0]
    last = hourly.loc[hourly.time.eq(issue - pd.Timedelta(hours=1))].iloc[0]
    assert row.last_power == last.power
    assert row.measurement_available_at == issue
    assert row.target_available_at == issue + pd.Timedelta(hours=2)
    truncated = hourly.loc[hourly.available_at.le(issue - pd.Timedelta(hours=3))]
    built = build_telemetry_requests(truncated, [issue], require_targets=False)
    assert built.examples.empty
    assert set(built.audit.reason) == {"stale_measurements"}
    gapped = hourly.loc[~hourly.available_at.eq(issue - pd.Timedelta(hours=1))]
    row = build_telemetry_requests(gapped, [issue], [1]).examples.iloc[0]
    assert pd.isna(row.power_lag_1h)
    assert row.complete_hours_6h == 5


def test_request_validation():
    hourly = hourly_telemetry(raw_data(), "UTC", "interval_start")
    for issue in ("2025-01-02", "2025-01-02T00:01:00Z"):
        with pytest.raises(ValueError):
            build_telemetry_requests(hourly, [issue])
    for horizons in ([1.5], [True], [0], [49], []):
        with pytest.raises(ValueError):
            build_telemetry_requests(hourly, ["2025-01-02T00:00:00Z"], horizons)


def test_training_has_disjoint_targets_and_control_cannot_change_fitted_model(tmp_path):
    raw = raw_data(days=75)
    common = {
        "turbine_id": 1,
        "source_sha256": "a" * 64,
        "timezone": "UTC",
        "semantics": "interval_start",
        "train_end": "2025-02-15T00:00:00Z",
        "validation_end": "2025-03-01T00:00:00Z",
        "test_end": "2025-03-16T00:00:00Z",
    }
    first = train_telemetry(raw, directory=tmp_path / "first", **common)
    changed = raw.copy()
    changed.loc[changed.time.ge("2025-03-01"), ["power", "wind_speed", "temperature"]] = (
        0.99,
        28,
        -10,
    )
    second = train_telemetry(changed, directory=tmp_path / "second", **common)
    assert first["model_sha256"] == second["model_sha256"]
    assert first["fit_rows_sha256"] == second["fit_rows_sha256"]
    assert first["validation_mae"] == second["validation_mae"]
    assert first["promoted"] == second["promoted"]
    # Changing every label beyond the fit cutoff may alter model selection, never the fit.
    changed.loc[changed.time.ge("2025-02-15"), ["power", "wind_speed", "temperature"]] = (
        0.01,
        1,
        35,
    )
    third = train_telemetry(changed, directory=tmp_path / "third", **common)
    assert first["model_sha256"] == third["model_sha256"]
    assert first["fit_rows_sha256"] == third["fit_rows_sha256"]
    assert first["usable_from"] == "2025-03-01T00:00:00+00:00"
    report = json.loads((tmp_path / "first/report.json").read_text())
    assert set(report["target_overlap_counts"].values()) == {0}
    assert report["fit_labels_available_by_train_end"] is True
    assert report["feature_availability_violations"] == 0
    assert report["promotion"]["frozen_before_control_metrics"] is True
    assert report["data_assessment"]["independent_untouched_test_available"] is False
    for horizon in ("all", "1_24h", "25_48h"):
        assert report["test"][horizon]["rows"] > 0
        assert report["test"][horizon]["model"]["mae"] >= 0
    rows = pd.read_parquet(tmp_path / "first/fit_rows.parquet")
    assert rows.target_available_at.le(pd.Timestamp(common["train_end"])).all()
    assert rows.measurement_available_at.le(rows.issue_time).all()
    assert rows.row_sha256.str.fullmatch("[a-f0-9]{64}").all()
    assert (
        first["model_sha256"]
        == hashlib.sha256((tmp_path / "first/model.joblib").read_bytes()).hexdigest()
    )
    model = joblib.load(tmp_path / "first/model.joblib")
    hourly = hourly_telemetry(raw, "UTC", "interval_start")
    values = predict_telemetry(model, hourly, "2025-03-10T12:00:00Z", 48)
    assert values.power.between(0, 1).all()
    assert len(values) == 48


def test_demo_data_cannot_be_misreported_as_trained_model(tmp_path):
    with pytest.raises(ValueError, match="не покрывает"):
        train_telemetry(
            raw_data("2026-01-25", 7),
            turbine_id=1,
            source_sha256="a" * 64,
            timezone="UTC",
            semantics="interval_start",
            directory=tmp_path,
        )
    assert not (tmp_path / "manifest.json").exists()
