from datetime import UTC, datetime, timedelta

import pytest

from wind.agent import AgentRequest, ToolSession, run_tools
from wind.ingest import import_csv
from wind.simulate import make_measurements
from wind.storage import create_turbine


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})
    source = tmp_path / "source.csv"
    make_measurements(source, "healthy")
    import_csv(source, 1)
    monkeypatch.setattr(
        "wind.nwp.model_readiness", lambda *a, **k: {"ok": True, "model_sha256": "pinned-model"}
    )
    return AgentRequest(
        turbine_id=1,
        issue_at=datetime(2026, 2, 28, 12, tzinfo=UTC),
        measurement_timezone="UTC",
        timestamp_semantics="interval_start",
        forecast_mode="weather",
    )


def weather(turbine_id, issue_at, horizon=48, previous_run=False):
    return {
        "points": [
            {"time": (issue_at + timedelta(hours=h)).isoformat(), "wind_speed_100m": 10}
            for h in range(1, horizon + 1)
        ],
        "provenance": {"provider": "test-operational-archive"},
        "historical_eligibility": "verified",
    }


def predictor(dataset, frame, **options):
    assert options["expected_model_sha256"] == "pinned-model"
    assert frame.attrs["provenance"]["provider"] == "test-operational-archive"
    return [{"time": row.time, "power": row.wind_speed_100m / 20} for row in frame.itertuples()], {
        "model": "nwp_hist_gradient_boosting",
        "weather_used": True,
        "model_sha256": "pinned-model",
    }


def test_weather_route_works_after_end_of_scada(prepared, monkeypatch):
    monkeypatch.setattr("wind.nwp.predict_with_nwp", predictor)
    session = ToolSession(prepared, issue_weather_provider=weather)
    session.inspect_data()
    ready = session.prepare_features()
    assert ready["ok"] and ready["telemetry_required"] is False
    assert session.predict_power()["code"] == "eligible_weather_required"
    assert session.fetch_weather(False)["ok"]
    assert session.predict_power()["weather_used"] is True
    assert session.analyze_forecast()["power_mean"] == 0.5
    assert session.save_forecast()["ok"]
    session.weather["points"][0]["wind_speed_100m"] = 2
    session.predict_power()
    assert session.forecast[0]["power"] == 0.1


def test_tools_replay_is_labelled_and_contains_analysis(prepared, monkeypatch):
    monkeypatch.setattr("wind.nwp.predict_with_nwp", predictor)
    report = run_tools(prepared, issue_weather_provider=weather)
    assert report["execution_mode"] == "tools"
    assert report["input_tokens"] == report["output_tokens"] == 0
    assert report["status"] == "forecast_saved"
    assert [step["tool"] for step in report["steps"]] == [
        "inspect_data",
        "prepare_features",
        "fetch_weather",
        "predict_power",
        "check_forecast",
        "analyze_forecast",
        "save_forecast",
    ]
    assert "погодный" in report["summary"]


def test_missing_weather_blocks_instead_of_silent_persistence(prepared, monkeypatch):
    calls = []

    def outage(*args, **kwargs):
        calls.append(kwargs["previous_run"])
        raise ValueError("No eligible forecast")

    monkeypatch.setattr("wind.nwp.predict_with_nwp", lambda *a, **k: pytest.fail("No weather"))
    report = run_tools(prepared, issue_weather_provider=outage)
    assert report["status"] == "stopped_without_forecast"
    assert calls == [False, True]
    assert not any(step["tool"] == "save_forecast" for step in report["steps"])
