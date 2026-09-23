"""NWP availability, monthly operation without SCADA, and separate final refit."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind import nwp as runtime
from wind_ml.nwp import FEATURE_COLUMNS, build_nwp_requests, predict_nwp_model, train_nwp
from wind_ml.telemetry import hourly_telemetry

PROVENANCE = {
    "provider": "synthetic_fixture",
    "model": "synthetic_operational",
    "availability_policy": "fixture_published_after_4h",
    "policy_version": 1,
    "latitude": 43.64,
    "longitude": 78.53,
    "grid_latitude": 43.75,
    "grid_longitude": 78.5,
    "dataset_snapshot": "synthetic_fixture_v1",
    "availability_evidence": "Synthetic fixture only: run + 4h. Not a real archive.",
}


def signal(time):
    hours = (time - pd.Timestamp("2025-01-01T00:00:00Z")).total_seconds() / 3600
    wind = 8 + 5 * np.sin(hours / 11) + np.cos(hours / 23)
    return wind, 15 + 5 * np.cos(hours / 24)


def raw_data(days=75):
    times = pd.date_range("2025-01-01", periods=days * 24 * 6, freq="10min")
    wind, temperature = signal(times.tz_localize("UTC").floor("h"))
    return pd.DataFrame(
        {
            "time": times,
            "wind_speed": wind,
            "temperature": temperature,
            "power": np.clip((wind - 3) / 11, 0, 1) ** 2,
        }
    )


def weather_data(start="2025-01-01T00:00:00Z", days=77):
    rows = []
    for run in pd.date_range(start, periods=days, freq="24h"):
        times = pd.date_range(run, periods=73, freq="h")
        wind, temperature = signal(times)
        rows.append(
            pd.DataFrame(
                {
                    "time": times,
                    "run": run,
                    "available_at": run + pd.Timedelta(hours=4),
                    "historical_eligibility": "verified",
                    "wind_speed_10m": wind * 0.7,
                    "wind_speed_100m": wind,
                    "temperature_2m": temperature,
                    "wind_direction_100m": 120,
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    result.attrs["provenance"] = PROVENANCE
    return result


def test_weather_only_inference_ignores_even_poisoned_future_scada():
    weather = weather_data(days=5)
    issue = "2025-01-02T12:00:00Z"
    original = build_nwp_requests(None, weather, [issue], require_targets=False)
    poisoned = pd.DataFrame({"never_read_this_as_scada": [object()]})
    altered = build_nwp_requests(poisoned, weather, [issue], require_targets=False)
    assert_frame_equal(original.examples, altered.examples)
    assert len(original.examples) == 48
    assert original.examples.target_power.isna().all()
    assert original.examples.weather_available_at.le(original.examples.issue_time).all()
    assert not any("power" in column or "last_" in column for column in FEATURE_COLUMNS)


def test_new_forecasts_published_after_issue_cannot_change_prediction_features():
    weather = weather_data(days=5)
    issue = pd.Timestamp("2025-01-02T12:00:00Z")
    changed = weather.copy()
    changed.loc[changed.available_at.gt(issue), ["wind_speed_10m", "wind_speed_100m"]] = 45
    original = build_nwp_requests(None, weather, [issue], require_targets=False).examples
    updated = build_nwp_requests(None, changed, [issue], require_targets=False).examples
    assert_frame_equal(original, updated)
    assert original.weather_run.eq(pd.Timestamp("2025-01-02T00:00:00Z")).all()


def test_whole_horizon_uses_latest_eligible_run_not_latest_republication():
    issue = pd.Timestamp("2025-01-03T12:00:00Z")
    times = pd.date_range(issue + pd.Timedelta(hours=1), periods=48, freq="h")
    frames = []
    for age, publication_age, wind in ((36, 1, 8), (12, 8, 14)):
        frames.append(
            pd.DataFrame(
                {
                    "time": times,
                    "run": issue - pd.Timedelta(hours=age),
                    "available_at": issue - pd.Timedelta(hours=publication_age),
                    "historical_eligibility": "verified",
                    "wind_speed_10m": wind,
                    "wind_speed_100m": wind,
                    "temperature_2m": 5,
                    "wind_direction_100m": 120,
                }
            )
        )
    weather = pd.concat(frames, ignore_index=True)
    rows = build_nwp_requests(None, weather, [issue], require_targets=False).examples
    assert rows.weather_run.eq(issue - pd.Timedelta(hours=12)).all()
    # One missing latest-run lead forces the entire request onto the older complete run.
    weather = weather.iloc[:-1]
    rows = build_nwp_requests(None, weather, [issue], require_targets=False).examples
    assert len(rows) == 48
    assert rows.weather_run.eq(issue - pd.Timedelta(hours=36)).all()
    assert rows.wind_speed_100m.eq(8).all()


def test_unverified_and_impossible_weather_are_rejected():
    weather = weather_data(days=3)
    weather["historical_eligibility"] = "unverified"
    built = build_nwp_requests(None, weather, ["2025-01-01T12:00:00Z"], require_targets=False)
    assert built.examples.empty
    assert built.audit.reason.eq("no_as_issued_weather").all()
    weather["historical_eligibility"] = "verified"
    weather["available_at"] = weather.run - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="available_at"):
        build_nwp_requests(None, weather, ["2025-01-01T12:00:00Z"], require_targets=False)


def test_training_cannot_choose_runs_outside_runtime_archive_policy():
    weather = weather_data(days=3)
    issue = pd.Timestamp("2025-01-03T12:00:00Z")
    weather = weather.loc[weather.run.eq(pd.Timestamp("2025-01-01T00:00:00Z"))].copy()
    built = build_nwp_requests(None, weather, [issue], horizons=[1], require_targets=False)
    assert built.examples.empty  # Two days old: runtime never tries this cycle.
    weather["run"] = pd.Timestamp("2025-01-03T06:00:00Z")
    weather["available_at"] = pd.Timestamp("2025-01-03T10:00:00Z")
    weather = weather.loc[weather.time.gt(weather.run)]
    built = build_nwp_requests(None, weather, [issue], horizons=[1], require_targets=False)
    assert built.examples.empty  # The archive contract uses 00 UTC cycles only.


def test_complete_february_daily_forecasts_need_no_february_scada():
    weather = weather_data("2026-01-31T00:00:00Z", 30)

    class WeatherModel:
        def predict(self, features):
            return np.clip(features.wind_speed_100m / 20, 0, 1)

    outputs = []
    for issue in pd.date_range("2026-01-31T12:00:00Z", "2026-02-28T12:00:00Z", freq="24h"):
        predictions = predict_nwp_model(WeatherModel(), weather, issue, 48)
        assert len(predictions) == 48
        assert predictions.power.between(0, 1).all()
        assert predictions.weather_available_at.le(issue).all()
        outputs.extend(predictions.time.tolist())
    expected = set(pd.date_range("2026-02-01T00:00:00Z", periods=672, freq="h"))
    assert expected.issubset(set(outputs))


def test_missing_weather_is_not_silently_filled():
    weather = weather_data(days=3)
    missing = pd.Timestamp("2025-01-02T00:00:00Z")
    weather = weather.loc[~weather.time.eq(missing)]
    with pytest.raises(ValueError, match="весь горизонт"):
        predict_nwp_model(None, weather, "2025-01-01T12:00:00Z", 48)


@pytest.fixture
def trained(tmp_path):
    raw, weather = raw_data(), weather_data()
    config = {
        "turbine_id": 1,
        "source_sha256": "a" * 64,
        "timezone": "UTC",
        "semantics": "interval_start",
        "weather_provenance": PROVENANCE,
        "train_end": "2025-02-15T00:00:00Z",
        "validation_end": "2025-03-01T00:00:00Z",
        "control_end": "2025-03-15T12:00:00Z",
        "final_issue_at": "2025-03-15T12:00:00Z",
    }
    directory = tmp_path / "nwp"
    manifest = train_nwp(raw, weather, directory=directory, **config)
    return raw, weather, config, directory, manifest


def test_evaluation_and_final_fit_are_separate_and_future_targets_do_not_leak(trained, tmp_path):
    raw, weather, config, directory, manifest = trained
    report = json.loads((directory / "report.json").read_text())
    assert set(report["target_overlap_counts"].values()) == {0}
    assert report["final_fit_is_separate_from_evaluation"] is True
    assert report["validation_control_telemetry_features_used"] is False
    assert report["february_metrics"] is None
    assert report["promotion"]["frozen_before_control"] is True
    assert manifest["promoted"] is True
    assert manifest["model_sha256"] != manifest["evaluation_model_sha256"]
    fit = pd.read_parquet(directory / "fit_rows.parquet")
    assert fit.target_available_at.le(pd.Timestamp(config["final_issue_at"])).all()
    assert fit.weather_available_at.le(fit.issue_time).all()
    assert fit.weather_run.le(fit.weather_available_at).all()
    changed = raw.copy()
    changed.loc[changed.time.ge("2025-03-15 12:00:00"), "power"] = 0.99
    future = train_nwp(changed, weather, directory=tmp_path / "future", **config)
    assert manifest["model_sha256"] == future["model_sha256"]
    assert manifest["evaluation_model_sha256"] == future["evaluation_model_sha256"]
    changed.loc[changed.time.ge("2025-03-01"), "power"] = 0.99
    control = train_nwp(changed, weather, directory=tmp_path / "control", **config)
    assert manifest["evaluation_model_sha256"] == control["evaluation_model_sha256"]
    assert manifest["validation_mae"] == control["validation_mae"]
    assert manifest["promoted"] == control["promoted"]
    # Final refit intentionally includes the known January/control targets after selection.
    assert manifest["model_sha256"] != control["model_sha256"]


def test_registry_requires_known_weights_source_config_and_weather_contract(trained, monkeypatch):
    _, weather, _, directory, manifest = trained
    monkeypatch.setattr(runtime, "artifact_directory", lambda *_: directory)
    dataset = {"id": 1, "sha256": "a" * 64}
    kwargs = {
        "timezone": "UTC",
        "semantics": "interval_start",
        "issue_at": "2025-03-16T12:00:00Z",
        "horizon": 48,
    }
    assert runtime.model_readiness(dataset, **kwargs)["ok"] is True
    points, metadata = runtime.predict_with_nwp(dataset, weather, **kwargs)
    assert len(points) == 48
    assert metadata["weather_used"] is True
    assert metadata["requires_current_scada"] is False
    assert (
        runtime.model_readiness(
            dataset,
            **{**kwargs, "issue_at": "2025-03-14T12:00:00Z"},
        )["ok"]
        is False
    )
    assert runtime.model_readiness({**dataset, "sha256": "b" * 64}, **kwargs)["ok"] is False
    altered_weather = weather.copy()
    altered_weather.attrs["provenance"] = {**PROVENANCE, "model": "different_source"}
    points, metadata = runtime.predict_with_nwp(dataset, altered_weather, **kwargs)
    assert points is None
    assert "изменились" in metadata["reason"]
    assert manifest["weather_provenance"] == PROVENANCE


def test_pinned_model_and_toctou_weights_change_fail_closed(trained, monkeypatch):
    _, weather, _, directory, _ = trained
    monkeypatch.setattr(runtime, "artifact_directory", lambda *_: directory)
    dataset = {"id": 1, "sha256": "a" * 64}
    kwargs = {
        "timezone": "UTC",
        "semantics": "interval_start",
        "issue_at": "2025-03-16T12:00:00Z",
        "horizon": 48,
    }
    points, metadata = runtime.predict_with_nwp(
        dataset,
        weather,
        **kwargs,
        expected_model_sha256="0" * 64,
    )
    assert points is None
    assert "изменилась" in metadata["reason"]
    readiness = runtime.model_readiness

    def replace_after_check(*args, **options):
        result = readiness(*args, **options)
        assert result["ok"] is True
        (directory / "model.joblib").write_bytes(b"changed after checksum verification")
        return result

    monkeypatch.setattr(runtime, "model_readiness", replace_after_check)
    points, metadata = runtime.predict_with_nwp(dataset, weather, **kwargs)
    assert points is None
    assert "контрольная сумма" in metadata["reason"]


def test_no_implicit_provenance_training(tmp_path):
    with pytest.raises(ValueError, match="доказательство"):
        train_nwp(
            raw_data(),
            weather_data(),
            turbine_id=1,
            source_sha256="a" * 64,
            timezone="UTC",
            semantics="interval_start",
            directory=tmp_path,
            weather_provenance={},
        )


def test_training_targets_require_complete_hours():
    hourly = hourly_telemetry(raw_data(days=4), "UTC", "interval_start")
    target = pd.Timestamp("2025-01-02T02:00:00Z")
    hourly.loc[hourly.time.eq(target), "quality"] = "incomplete"
    built = build_nwp_requests(hourly, weather_data(days=4), ["2025-01-01T12:00:00Z"])
    assert built.audit.loc[built.audit.target_time.eq(target), "reason"].item() == "missing_target"
