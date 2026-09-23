from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wind.agent import ToolSession
from wind.app import app, mount_frontend
from wind.ingest import import_csv
from wind.simulate import make_measurements
from wind.storage import create_turbine


@pytest.fixture
def local_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "storage"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not call an API, load a key, or create a forecast")

    monkeypatch.setattr("wind.app.run_agent", forbidden)
    monkeypatch.setattr("wind.app.load_dotenv", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    for name in ("fetch_weather", "predict_power", "check_forecast", "save_forecast"):
        monkeypatch.setattr(ToolSession, name, forbidden)
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})
    with TestClient(app) as client:
        yield client
    assert not (tmp_path / "storage/forecasts").exists()
    assert not (tmp_path / "storage/agent-runs").exists()


def payload(**overrides):
    return {
        "turbine_id": 1,
        "issue_at": "2026-01-31T00:00:00Z",
        "measurement_timezone": "UTC",
        "timestamp_semantics": "interval_start",
        "horizon": 48,
        **overrides,
    }


def measurements(tmp_path, scenario="healthy"):
    source = tmp_path / "source.csv"
    make_measurements(source, scenario)
    return import_csv(source, 1)


@pytest.mark.parametrize("horizon", [24, 48])
def test_ready_without_key_and_future_measurements_excluded(local_client, tmp_path, horizon):
    measurements(tmp_path)
    response = local_client.post("/api/agent/preflight", json=payload(horizon=horizon))
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert [check["tool"] for check in body["checks"]] == ["inspect_data", "prepare_features"]
    result = body["checks"][-1]["result"]
    assert result["last_power"] == pytest.approx(0.4)
    assert result["available_at"] == "2026-01-31T00:00:00+00:00"
    assert result["age_hours"] == 0


def test_no_measurements_blocks_before_preparation(local_client):
    response = local_client.post("/api/agent/preflight", json=payload())
    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "checks": [
            {"tool": "inspect_data", "result": {"ok": False, "code": "no_measurements"}},
        ],
    }


@pytest.mark.parametrize(
    "scenario,overrides,code",
    [
        ("stale_telemetry", {}, "stale_telemetry"),
        ("healthy", {"measurement_timezone": None}, "time_unconfirmed"),
        ("healthy", {"timestamp_semantics": None}, "time_unconfirmed"),
        ("healthy", {"issue_at": "2026-01-01T00:00:00Z"}, "no_complete_history"),
    ],
)
def test_unready_inputs_report_blocker(local_client, tmp_path, scenario, overrides, code):
    measurements(tmp_path, scenario)
    response = local_client.post("/api/agent/preflight", json=payload(**overrides))
    assert response.status_code == 200
    assert response.json()["ok"] is False
    result = response.json()["checks"][-1]["result"]
    assert result["ok"] is False
    assert result["code"] == code
    if code == "stale_telemetry":
        assert result["age_hours"] == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"horizon": 23},
        {"horizon": 49},
        {"issue_at": "2026-01-31T00:00:00"},
        {"issue_at": "2026-01-31T00:30:00Z"},
        {"measurement_timezone": "unknown/timezone"},
        {"timestamp_semantics": "unknown"},
    ],
)
def test_request_validation_matches_agent_contract(local_client, overrides):
    assert local_client.post("/api/agent/preflight", json=payload(**overrides)).status_code == 422


def test_future_issue_rejected(local_client):
    future = (datetime.now(UTC) + timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    response = local_client.post("/api/agent/preflight", json=payload(issue_at=future.isoformat()))
    assert response.status_code == 422


def test_unknown_turbine_returns_not_found(local_client):
    response = local_client.post("/api/agent/preflight", json=payload(turbine_id=99))
    assert response.status_code == 404


def test_unreadable_import_returns_validation_error(local_client, tmp_path):
    metadata = measurements(tmp_path)
    raw = metadata["hourly_path"].replace("-hourly.parquet", "-10min.parquet")
    (tmp_path / "storage" / raw).unlink()
    response = local_client.post("/api/agent/preflight", json=payload())
    assert response.status_code == 422
    assert "измерения" in response.json()["detail"]


@pytest.mark.parametrize("page", ["turbines", "forecast", "weather", "sources"])
def test_frontend_pages_and_canonical_redirects(tmp_path, page):
    markup = "<!doctype html><title>Wind</title><div id='root'></div>"
    (tmp_path / "index.html").write_text(markup)
    frontend = FastAPI()
    mount_frontend(frontend, tmp_path)
    query = "?turbine=2&filter=wind%2Fpower&label=a+b"
    with TestClient(frontend) as client:
        direct = client.get(f"/{page}{query}")
        assert direct.status_code == 200
        assert direct.headers["content-type"].startswith("text/html")
        assert direct.text == markup
        redirect = client.get(f"/{page}/{query}", follow_redirects=False)
        assert redirect.status_code == 308
        assert redirect.headers["location"] == f"/{page}{query}"
        assert client.get(redirect.headers["location"]).text == markup
        assert client.get(f"/{page}/", follow_redirects=False).headers["location"] == f"/{page}"
