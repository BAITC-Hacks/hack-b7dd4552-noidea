from argparse import Namespace
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from wind_ml import cli, evaluation
from wind_ml.artifacts import validate_artifact_request
from wind_ml.evaluation import ValidationFold, _training_measurements_before, validate_walk_forward
from wind_ml.features import FEATURE_COLUMNS, build_persistence_requests
from wind_ml.io import RAW_COLUMNS, hourly_from_scada_csv, read_hourly_csv, write_hourly_csv
from wind_ml.models import fit_boosting
from wind_ml.schemas import MeasurementTimeConfig, SchemaError, measurement_time_to_utc


def end_stamped_raw(future_power):
    times = pd.date_range("2025-01-01", periods=19, freq="10min")
    raw = pd.DataFrame(
        {
            RAW_COLUMNS[0]: times,
            RAW_COLUMNS[1]: 5,
            RAW_COLUMNS[2]: np.where(times <= pd.Timestamp("2025-01-01T01:00"), 0.1, future_power),
            RAW_COLUMNS[3]: 2,
        }
    )
    return StringIO(raw.to_csv(index=False))


def test_interval_end_aggregation_never_reads_future_ten_minute_samples(tmp_path):
    config = MeasurementTimeConfig("UTC", "interval_end")
    issue = pd.Timestamp("2025-01-01T01:00:00Z")
    first = hourly_from_scada_csv(end_stamped_raw(0.9), config=config)
    changed = hourly_from_scada_csv(end_stamped_raw(0.4), config=config)
    predictions = []
    for frame in (first, changed):
        result = build_persistence_requests(frame, config, [issue], [1], require_targets=False)
        predictions.append(result.audit.iloc[0]["prediction"])
        assert result.audit.iloc[0]["measurement_available_at"] == issue
        assert result.audit.iloc[0]["lag_age_hours"] == 0
    assert predictions == pytest.approx([0.1, 0.1])
    assert not first.power.equals(changed.power)
    saved = tmp_path / "hourly.csv"
    write_hourly_csv(first, saved)
    converted = measurement_time_to_utc(read_hourly_csv(saved), config)
    pd.testing.assert_series_equal(converted["available_at"], first["available_at"])


def test_legacy_interval_end_requires_raw_reaggregation(hourly_measurements):
    with pytest.raises(SchemaError, match="raw 10-min"):
        measurement_time_to_utc(hourly_measurements(), MeasurementTimeConfig("UTC", "interval_end"))


def test_canonical_hourly_rejects_time_config_change():
    config = MeasurementTimeConfig("UTC", "interval_end")
    frame = hourly_from_scada_csv(end_stamped_raw(0.5), config=config)
    with pytest.raises(SchemaError, match="не совпадает"):
        measurement_time_to_utc(frame, MeasurementTimeConfig("UTC", "interval_start"))


def test_cutoff_filters_by_label_availability_even_between_hours(hourly_measurements, utc_config):
    measurements = hourly_measurements(start="2025-01-01T09:00", periods=3)
    cutoff = pd.Timestamp("2025-01-01T10:30:00Z")
    for select in (
        lambda frame: cli._training_subset(frame, utc_config, cutoff)[0],
        lambda frame: _training_measurements_before(frame, utc_config, cutoff),
    ):
        before = select(measurements)
        assert before.time.tolist() == [pd.Timestamp("2025-01-01T09:00")]
        changed = measurements.copy()
        changed.loc[changed.time >= pd.Timestamp("2025-01-01T10:00"), "power"] = 0.999
        pd.testing.assert_frame_equal(before, select(changed))
    with pytest.raises(ValueError, match="train-end"):
        cli._training_subset(measurements, utc_config, None)


def test_train_cli_requires_explicit_cutoff():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "train",
                "--model",
                "curve",
                "--measurements",
                "input.csv",
                "--turbine-id",
                "1",
                "--artifacts-dir",
                "artifact",
                "--measurement-timezone",
                "UTC",
                "--timestamp-semantics",
                "interval_start",
            ]
        )


def test_future_targets_do_not_change_frozen_fit(
    hourly_measurements, weather_archive, utc_config, monkeypatch
):
    measurements = hourly_measurements(periods=200)
    weather = weather_archive(periods=160)
    start = pd.Timestamp("2025-01-05T12:00Z")
    captured = []

    class Frozen:
        def predict(self, examples):
            return examples.last_power.to_numpy()

    def fit(rows):
        captured.append(rows.copy())
        return Frozen()

    monkeypatch.setattr(evaluation, "fit_boosting", fit)
    for changed in (False, True):
        current = measurements.copy()
        if changed:
            current.loc[current.time >= start.tz_localize(None), "power"] = 0.999
        validate_walk_forward(
            model_type="boosting",
            measurements=current,
            time_config=utc_config,
            folds=[ValidationFold.create(str(start), "2025-01-06T12:00Z")],
            turbine_id="1",
            weather=weather,
        )
    pd.testing.assert_frame_equal(captured[0], captured[1])
    assert captured[0].target_available_at.le(start).all()
    assert captured[0].target_time.lt(start).all()


@pytest.mark.parametrize("second_start", ["2025-01-02T00:00Z", "2025-01-02T12:00Z"])
def test_overlapping_or_duplicate_folds_rejected(hourly_measurements, utc_config, second_start):
    with pytest.raises(ValueError, match="пересекаться"):
        validate_walk_forward(
            model_type="persistence",
            measurements=hourly_measurements(),
            time_config=utc_config,
            turbine_id="1",
            folds=[
                ValidationFold.create("2025-01-02T00:00Z", "2025-01-03T00:00Z"),
                ValidationFold.create(second_start, "2025-01-03T00:00Z"),
            ],
        )


def test_adjacent_folds_do_not_score_the_same_target(hourly_measurements, utc_config):
    _, audit = validate_walk_forward(
        model_type="persistence",
        measurements=hourly_measurements(),
        time_config=utc_config,
        turbine_id="1",
        folds=[
            ValidationFold.create("2025-01-02T00:00Z", "2025-01-03T00:00Z"),
            ValidationFold.create("2025-01-03T00:00Z", "2025-01-04T00:00Z"),
        ],
    )
    scored = audit.loc[audit.reason.isna()]
    first = set(scored.loc[scored.fold.eq(1), "target_time"])
    second = set(scored.loc[scored.fold.eq(2), "target_time"])
    assert first and second and not first.intersection(second)
    assert audit.loc[audit.reason.eq("target_outside_fold"), "prediction"].isna().all()


def artifact_manifest():
    return {
        "turbine_id": "1",
        "measurement_time": MeasurementTimeConfig("UTC", "interval_start").to_dict(),
        "training_bounds": {"target_time_max_exclusive_utc": "2025-02-01T00:00Z"},
        "trained_horizon_hours": 24,
    }


@pytest.mark.parametrize(
    "change,error",
    [
        ({"issue_at": "2025-01-31T23:00Z"}, "раньше границы"),
        ({"horizon": 48}, "обученный диапазон"),
        ({"turbine_id": "2"}, "другой турбины"),
        ({"time_config": MeasurementTimeConfig("Etc/GMT-6", "interval_start")}, "отличается"),
    ],
)
def test_inference_rejects_incompatible_artifact(change, error):
    request = {
        "turbine_id": "1",
        "time_config": MeasurementTimeConfig("UTC", "interval_start"),
        "issue_at": "2025-02-01T00:00Z",
        "horizon": 24,
    }
    validate_artifact_request(artifact_manifest(), **request)
    with pytest.raises(ValueError, match=error):
        validate_artifact_request(artifact_manifest(), **(request | change))


def test_cli_infer_applies_artifact_cutoff_before_prediction(monkeypatch, hourly_measurements):
    monkeypatch.setattr(cli, "load_artifact", lambda _: (artifact_manifest(), object()))
    monkeypatch.setattr(cli, "read_hourly_csv", lambda _: hourly_measurements())
    with pytest.raises(ValueError, match="раньше границы"):
        cli.command_infer(
            Namespace(
                artifacts_dir="fixture",
                measurements="fixture",
                measurement_timezone=None,
                timestamp_semantics=None,
                issue_at="2025-01-31T23:00Z",
                horizon=24,
            )
        )


@pytest.mark.parametrize("missing", ["training_bounds", "trained_horizon_hours"])
def test_legacy_artifact_without_provable_limits_requires_retraining(missing):
    manifest = artifact_manifest()
    manifest.pop(missing)
    with pytest.raises(ValueError, match="переобучите"):
        validate_artifact_request(
            manifest,
            turbine_id="1",
            time_config=MeasurementTimeConfig("UTC", "interval_start"),
            issue_at="2025-02-01T00:00Z",
            horizon=24,
        )


def test_stale_measurements_are_excluded(hourly_measurements, utc_config):
    frame = hourly_measurements(periods=3)
    result = build_persistence_requests(
        frame, utc_config, [pd.Timestamp("2025-01-01T08:00Z")], [1], require_targets=False
    )
    assert result.examples.empty
    assert result.audit.iloc[0].reason == "stale_telemetry"
    assert result.audit.iloc[0].lag_age_hours == 5


def test_boosting_has_no_random_inner_validation_split():
    generator = np.random.default_rng(42)
    examples = pd.DataFrame(
        generator.normal(size=(10_001, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    examples["target_power"] = np.clip(examples["last_power"], 0, 1)
    fitted = fit_boosting(examples)
    assert fitted.estimator.do_early_stopping_ is False
    assert fitted.estimator.validation_score_.size == 0
