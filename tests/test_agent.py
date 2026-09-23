import json
from datetime import UTC, datetime

import pytest

from wind.agent import AgentRequest, ToolSession, factual_summary
from wind.ingest import import_csv
from wind.simulate import make_measurements, outage
from wind.storage import create_turbine


def session(tmp_path, monkeypatch, scenario="healthy", timezone="UTC", semantics="interval_start"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "storage"))
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})
    source = tmp_path / "source.csv"
    make_measurements(source, scenario)
    import_csv(source, 1)
    return ToolSession(
        AgentRequest(
            turbine_id=1,
            issue_at=datetime(2026, 1, 31, tzinfo=UTC),
            measurement_timezone=timezone,
            timestamp_semantics=semantics,
        ),
        outage,
    )


def test_future_power_cannot_leak_and_forecast_requires_check(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    assert not s.predict_power()["ok"]
    s.inspect_data()
    assert s.prepare_features()["last_power"] == pytest.approx(0.4)
    assert s.predict_power()["hours"] == 48
    assert not s.save_forecast()["ok"]
    assert s.check_forecast()["ok"]
    assert s.save_forecast()["ok"]
    path = tmp_path / "storage/forecasts" / f"{s.id}.json"
    assert len(json.loads(path.read_text())["points"]) == 48
    s.forecast[0]["power"] = 2
    assert not s.save_forecast()["ok"]


@pytest.mark.parametrize(
    "scenario,timezone,code",
    [
        ("stale_telemetry", "UTC", "stale_telemetry"),
        ("healthy", None, "time_unconfirmed"),
    ],
)
def test_data_blocks_baseline(tmp_path, monkeypatch, scenario, timezone, code):
    s = session(tmp_path, monkeypatch, scenario, timezone)
    s.inspect_data()
    assert s.prepare_features()["code"] == code
    assert not s.predict_power()["ok"]


def test_outage_bounded_and_tool_arguments_restricted(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    assert s.execute("fetch_weather", {"previous_run": "false"})["code"] == "invalid_arguments"
    assert s.execute("shell", {})["code"] == "unknown_tool"
    assert s.fetch_weather(False)["code"] == "weather_unavailable"
    assert s.fetch_weather(True)["code"] == "weather_unavailable"
    assert s.fetch_weather(True)["code"] == "weather_retry_limit"


def test_interval_end_excludes_incomplete_future_hour(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch, semantics="interval_end")
    s.inspect_data()
    # 00:00 is an interval-end measurement already available at issue time;
    # later samples must still not enter the hourly feature.
    value = s.prepare_features()["last_power"]
    assert value == pytest.approx((0.4 * 5 + 0.99) / 6)


def test_summary_does_not_turn_unverified_weather_into_missing():
    report = {
        "status": "forecast_saved",
        "steps": [
            {
                "tool": "fetch_weather",
                "result": {
                    "ok": True,
                    "missing_values": {},
                    "historical_eligibility": "unverified",
                },
            },
        ],
    }
    summary = factual_summary(report)
    assert "получен" in summary
    assert "не подтверждена" in summary
    assert "недоступен" not in summary
