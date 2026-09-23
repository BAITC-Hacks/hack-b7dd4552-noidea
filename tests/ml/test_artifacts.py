from __future__ import annotations

from wind_ml.artifacts import load_artifact, save_artifact
from wind_ml.models import PowerCurve


def test_artifact_keeps_time_config_bounds_and_curve(tmp_path, utc_config) -> None:
    curve = PowerCurve(0.5, 2, [1.0, 2.0], [0.1, 0.3], [4, 5])
    destination = save_artifact(
        tmp_path / "artifact",
        model_type="curve",
        payload=curve,
        turbine_id="2",
        time_config=utc_config,
        training_bounds={
            "target_time_min_utc": "2025-01-01T00:00:00+00:00",
            "target_time_max_exclusive_utc": None,
        },
    )

    manifest, restored = load_artifact(destination)

    assert manifest["turbine_id"] == "2"
    assert manifest["measurement_time"]["timestamp_semantics"] == "interval_start"
    assert restored.predict([2.0]).tolist() == [0.3]
