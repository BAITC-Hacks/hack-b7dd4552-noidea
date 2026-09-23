from __future__ import annotations

import pandas as pd
import pytest

from wind_ml import evaluation
from wind_ml.evaluation import ValidationFold, metrics_from_audit, validate_walk_forward


def test_metrics_are_separate_for_horizon_bands() -> None:
    audit = pd.DataFrame(
        {
            "horizon_hours": [1, 24, 25, 48],
            "target_power": [0.1, 0.3, 0.6, 0.8],
            "prediction": [0.2, 0.4, 0.5, None],
            "reason": [None, None, None, "missing_target"],
        }
    )
    report = metrics_from_audit(audit, turbine_id="1", model_type="persistence")

    short, long = report["by_horizon"]
    assert short["evaluated_hours"] == 2
    assert short["mae"] == pytest.approx(0.1)
    assert long["evaluated_hours"] == 1
    assert long["excluded_share"] == 0.5


def test_boosting_fit_is_frozen_before_validation_targets(
    hourly_measurements, weather_archive, utc_config, monkeypatch
) -> None:
    measurements = hourly_measurements(periods=180)
    weather = weather_archive(periods=150)
    captured = []

    class DummyModel:
        def predict(self, examples):
            return examples["last_power"].to_numpy()

    def fake_fit(examples, **_):
        captured.append(examples["target_time"].max())
        return DummyModel()

    monkeypatch.setattr(evaluation, "fit_boosting", fake_fit)
    fold = ValidationFold.create("2025-01-05T12:00:00Z", "2025-01-05T16:00:00Z")

    _, audit = validate_walk_forward(
        model_type="boosting",
        measurements=measurements,
        time_config=utc_config,
        folds=[fold],
        turbine_id="1",
        weather=weather,
    )

    assert captured and captured[0] < fold.issue_start
    assert (audit["fold_issue_start"] == fold.issue_start).all()
