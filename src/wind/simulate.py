"""Synthetic events, real OpenAI decisions; never writes into the user's catalog."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
import pandas as pd

from wind.agent import AgentRequest, run_agent
from wind.ingest import import_csv
from wind.storage import create_turbine
from wind.weather import fetch_weather


def make_measurements(path: Path, scenario: str):
    times = pd.date_range("2026-01-30", periods=36 * 6, freq="10min")
    frame = pd.DataFrame(
        {
            "Статистическое время": times,
            "Средняя скорость ветра(m/s)": 6.0,
            "Нормализованная активная мощность": [
                (0.65 if scenario == "data_updated" else 0.4)
                if t < pd.Timestamp("2026-01-31")
                else 0.99
                for t in times
            ],
            "Средняя температура окружающей среды(°C)": -2.0,
        }
    )
    if scenario == "stale_telemetry":
        frame = frame[
            (frame.iloc[:, 0] < pd.Timestamp("2026-01-30 18:00"))
            | (frame.iloc[:, 0] >= pd.Timestamp("2026-01-31"))
        ]
    if scenario == "missing_hours":
        frame = frame.drop(index=range(60, 78))
    frame.to_csv(path, index=False)


def fixture_weather(request):
    return {
        "run": request.run.isoformat(),
        "hours": 72,
        "missing_values": {"wind_speed_100m": 0},
        "historical_eligibility": "unverified",
        "available_at": None,
    }


def outage(_):
    raise httpx.ConnectError("Simulated provider outage")


def main():
    target = Path("data/agent-validation") / uuid4().hex
    target.mkdir(parents=True)
    original = os.environ.get("DATA_DIR")
    reports = []
    try:
        for scenario in [
            "healthy",
            "missing_hours",
            "weather_outage",
            "stale_telemetry",
            "time_unconfirmed",
            "data_updated",
        ]:
            with TemporaryDirectory(prefix="wind-agent-validation-") as folder:
                root = Path(folder)
                os.environ["DATA_DIR"] = str(root / "storage")
                create_turbine(
                    {"name": "Синтетическая проверка", "latitude": 43.64515, "longitude": 78.535604}
                )
                make_measurements(root / "input.csv", scenario)
                import_csv(root / "input.csv", 1, "simulation")
                request = AgentRequest(
                    turbine_id=1,
                    issue_at=datetime(2026, 1, 31, tzinfo=UTC),
                    measurement_timezone=None if scenario == "time_unconfirmed" else "UTC",
                    timestamp_semantics=None
                    if scenario == "time_unconfirmed"
                    else "interval_start",
                    event="data_updated" if scenario == "data_updated" else "manual",
                )
                provider = (
                    fetch_weather
                    if scenario == "healthy"
                    else (outage if scenario == "weather_outage" else fixture_weather)
                )
                report = run_agent(request, weather_provider=provider, simulated=True)
                report["scenario"] = scenario
                report["weather_source"] = "live_archive" if scenario == "healthy" else "simulated"
                steps = report["steps"]
                saved = report["status"] == "forecast_saved"
                if scenario in ("time_unconfirmed", "stale_telemetry"):
                    passed = not saved and report["status"] == "stopped_without_forecast"
                    if scenario == "stale_telemetry":
                        passed = passed and any(
                            s["result"].get("code") == "stale_telemetry" for s in steps
                        )
                    else:
                        passed = passed and any(
                            s["tool"] == "inspect_data" and not s["result"].get("time_configured")
                            for s in steps
                        )
                else:
                    expected = 0.65 if scenario == "data_updated" else 0.4
                    forecast_path = root / "storage/forecasts" / f"{report['id']}.json"
                    forecast = (
                        json.loads(forecast_path.read_text()) if forecast_path.exists() else {}
                    )
                    passed = (
                        saved
                        and len(forecast.get("points", [])) == 48
                        and all(
                            abs(p["power"] - expected) < 1e-9 for p in forecast.get("points", [])
                        )
                    )
                    report["forecast"] = forecast
                    if scenario == "weather_outage":
                        passed = passed and sum(s["tool"] == "fetch_weather" for s in steps) == 2
                report["passed"] = bool(passed)
                reports.append(report)
                (target / f"{scenario}.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2)
                )
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "passed": report["passed"],
                            "status": report["status"],
                            "tools": [s["tool"] for s in steps],
                            "input_tokens": report["input_tokens"],
                            "output_tokens": report["output_tokens"],
                        }
                    ),
                    flush=True,
                )
    finally:
        if original is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = original
    summary = {
        "passed": sum(r["passed"] for r in reports),
        "total": len(reports),
        "input_tokens": sum(r["input_tokens"] for r in reports),
        "output_tokens": sum(r["output_tokens"] for r in reports),
        "directory": str(target),
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
