import hashlib
import json

import joblib
import pandas as pd
import pytest
import sklearn
from sklearn.dummy import DummyRegressor

from wind.agent import AgentRequest, ToolSession, factual_summary
from wind.ingest import import_csv
from wind.ml import MODEL_TYPE, artifact_directory, model_catalog, predict_with_ml
from wind.simulate import make_measurements
from wind.storage import create_turbine, delete_turbine
from wind_ml.telemetry import FEATURE_COLUMNS


@pytest.fixture
def trained(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "storage"))
    create_turbine({"name": "ML test", "latitude": 43.6, "longitude": 78.5})
    source = tmp_path / "source.csv"
    make_measurements(source, "healthy")
    metadata = import_csv(source, 1)
    directory = artifact_directory(1, "UTC", "interval_start")
    directory.mkdir(parents=True)
    model = DummyRegressor(strategy="constant", constant=0.7)
    model.fit(pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS), [0.7])
    joblib.dump(model, directory / "model.joblib")
    manifest = {
        "artifact_version": 1,
        "model_type": MODEL_TYPE,
        "turbine_id": 1,
        "source_sha256": metadata["sha256"],
        "timezone": "UTC",
        "timestamp_semantics": "interval_start",
        "feature_columns": list(FEATURE_COLUMNS),
        "sklearn_version": sklearn.__version__,
        "promoted": True,
        "training_end": "2025-10-01T00:00:00Z",
        "usable_from": "2026-01-01T00:00:00Z",
        "validation_end": "2026-01-01T00:00:00Z",
        "test_end": "2026-02-01T00:00:00Z",
        "horizon_max": 48,
        "validation_mae": 0.1,
        "persistence_validation_mae": 0.2,
        "model_sha256": hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return metadata, directory, manifest


def tool_session():
    return ToolSession(
        AgentRequest(
            turbine_id=1,
            issue_at="2026-01-31T00:00:00Z",
            measurement_timezone="UTC",
            timestamp_semantics="interval_start",
            horizon=48,
        )
    )


def test_trained_model_reaches_agent_saved_forecast(trained):
    session = tool_session()
    assert session.inspect_data()["ok"]
    assert session.prepare_features()["ok"]
    result = session.predict_power()
    assert result["model"] == MODEL_TYPE
    assert result["weather_used"] is False
    assert result["power_min"] == result["power_max"] == pytest.approx(0.7)
    assert session.check_forecast()["ok"]
    assert session.save_forecast()["ok"]
    from wind.storage import data_dir

    saved = json.loads((data_dir() / "forecasts" / f"{session.id}.json").read_text())
    assert saved["model"] == MODEL_TYPE
    assert saved["model_metadata"]["training_end"] == trained[2]["training_end"]
    assert len(saved["points"]) == 48
    summary = factual_summary(
        {
            "status": "forecast_saved",
            "steps": [
                {"tool": "predict_power", "result": result},
            ],
        }
    )
    assert "ML-прогноз" in summary and "резервный" not in summary


@pytest.mark.parametrize(
    "field,value",
    [
        ("turbine_id", 9),
        ("source_sha256", "different-file"),
        ("timezone", "Etc/GMT-6"),
        ("timestamp_semantics", "interval_end"),
        ("promoted", False),
        ("usable_from", "2026-02-01T00:00:00Z"),
        ("training_end", "2026-02-01T00:00:00Z"),
        ("feature_columns", ["future_power"]),
        ("sklearn_version", "wrong-version"),
        ("model_sha256", "corrupt"),
        ("horizon_max", 24),
    ],
)
def test_incompatible_or_future_trained_model_is_never_loaded(trained, monkeypatch, field, value):
    _, directory, manifest = trained
    manifest[field] = value
    (directory / "manifest.json").write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe model must be rejected BEFORE deserialization or inference")

    monkeypatch.setattr("wind.ml._load_model", forbidden)
    session = tool_session()
    session.inspect_data()
    session.prepare_features()
    result = session.predict_power()
    assert result["model"] == "persistence_baseline"
    assert result["fallback_reason"]
    assert session.forecast[0]["power"] == pytest.approx(0.4)


def test_historical_issue_cannot_use_validation_selected_model(trained, monkeypatch):
    dataset, _, _ = trained

    def forbidden(*args, **kwargs):
        pytest.fail("A model selected later must not serve an earlier forecast")

    monkeypatch.setattr("wind.ml._load_model", forbidden)
    points, result = predict_with_ml(
        pd.DataFrame(),
        dataset,
        timezone="UTC",
        semantics="interval_start",
        issue_at="2025-12-31T23:00:00Z",
        horizon=48,
    )
    assert points is None and "раньше" in result["fallback_reason"]


def test_catalog_has_provenance_and_excludes_deleted_turbine(trained):
    items = model_catalog()
    assert len(items) == 1 and items[0]["source_matches"] is True
    assert items[0]["test_is_untouched"] is False
    delete_turbine(1)
    assert not model_catalog()
