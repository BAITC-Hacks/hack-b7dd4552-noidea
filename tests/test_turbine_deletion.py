import json
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wind.app import app, mount_frontend
from wind.discovery import write_plan
from wind.ingest import import_csv
from wind.simulate import make_measurements
from wind.storage import (
    TurbineNotFoundError,
    all_metadata,
    create_turbine,
    get_turbine,
    save_metadata,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "storage"))

    def forbidden(*args, **kwargs):
        pytest.fail("Deleting a turbine or rejecting its operations must not call external APIs")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr("wind.app.run_agent", forbidden)
    monkeypatch.setattr("wind.app.fetch_weather", forbidden)
    monkeypatch.setattr("wind.discovery.read_page", forbidden)
    with TestClient(app) as value:
        yield value


def turbine(name="Test"):
    return create_turbine({"name": name, "latitude": 43.6, "longitude": 78.5})


def measured_turbine(tmp_path, name="Test"):
    registered = turbine(name)
    source = tmp_path / "source.csv"
    make_measurements(source, "healthy")
    return import_csv(source, registered["id"])


def test_delete_and_restore_preserve_data_artifacts_and_neighbors(client, tmp_path):
    first = measured_turbine(tmp_path, "First")
    second = measured_turbine(tmp_path, "Second")
    original_catalog = client.get("/api/turbines").json()["items"]
    window = "?start=2026-01-30&end=2026-01-31"
    export = client.get(f"/api/turbines/{first['id']}/export{window}").content
    store = tmp_path / "storage"
    weather_id, forecast_id = "a" * 24, "b" * 32
    (store / "weather").mkdir()
    (store / "weather/saved.json").write_text('{"hourly": []}')
    save_metadata(
        "weather",
        weather_id,
        {
            "id": weather_id,
            "turbine_id": first["id"],
            "run": "2026-01-30T12:00:00Z",
            "raw_path": "weather/saved.json",
        },
    )
    (store / "forecasts").mkdir()
    (store / "forecasts" / f"{forecast_id}.json").write_text('{"points": []}')
    original_files = {
        path.relative_to(store): path.read_bytes()
        for path in store.rglob("*")
        if path.is_file() and path.name != "catalog.sqlite"
    }
    original_datasets, original_weather = all_metadata("datasets"), all_metadata("weather")

    response = client.delete(f"/api/turbines/{first['id']}")
    assert response.status_code == 200
    assert response.json() == {"id": first["id"], "deleted": True}
    assert client.get("/api/turbines").json()["items"] == [original_catalog[1]]
    archived = client.get("/api/turbines/deleted").json()["items"]
    assert len(archived) == 1
    assert archived[0]["id"] == first["id"]
    assert archived[0]["has_data"] is True
    assert archived[0]["sha256"] == first["sha256"]
    assert datetime.fromisoformat(archived[0]["deleted_at"]).utcoffset().total_seconds() == 0
    assert [item["id"] for item in all_metadata("turbines")] == [second["id"]]
    with pytest.raises(TurbineNotFoundError, match="корзины"):
        get_turbine(first["id"])
    assert client.get(f"/api/turbines/{second['id']}/series{window}").status_code == 200
    assert all_metadata("datasets") == original_datasets
    assert all_metadata("weather") == original_weather
    assert client.get(f"/api/weather/{weather_id}/raw").status_code == 200
    assert client.get(f"/api/forecasts/{forecast_id}").status_code == 200
    for path, content in original_files.items():
        assert (store / path).read_bytes() == content

    third = client.post(
        "/api/turbines", json={"name": "Third", "latitude": 43.6, "longitude": 78.5}
    )
    assert third.status_code == 201
    assert third.json()["id"] > second["id"]
    restored = client.post(f"/api/turbines/{first['id']}/restore")
    assert restored.status_code == 200
    assert restored.json() == original_catalog[0]
    assert client.get("/api/turbines/deleted").json() == {"items": []}
    assert len(client.get("/api/turbines").json()["items"]) == 3
    assert client.get(f"/api/turbines/{first['id']}/export{window}").content == export
    assert get_turbine(first["id"])["name"] == "First"


def test_empty_turbine_archive_and_restore_are_idempotent(client):
    registered = turbine()
    url = f"/api/turbines/{registered['id']}"
    assert client.delete(url).status_code == 200
    archived = client.get("/api/turbines/deleted").json()
    assert archived["items"][0]["has_data"] is False
    assert client.delete(url).status_code == 200
    assert client.get("/api/turbines/deleted").json() == archived
    expected = {**registered, "has_data": False}
    assert client.post(url + "/restore").json() == expected
    assert client.post(url + "/restore").json() == expected


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_unknown_turbine_returns_404(client, operation):
    response = (
        client.delete("/api/turbines/999")
        if operation == "delete"
        else client.post("/api/turbines/999/restore")
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Турбина не найдена"


@pytest.mark.parametrize(
    "operation", ["series", "export", "import", "preflight", "agent", "weather", "discovery"]
)
def test_deleted_turbine_rejects_new_operations(client, tmp_path, operation):
    measured_turbine(tmp_path)
    assert client.delete("/api/turbines/1").status_code == 200
    if operation in {"series", "export"}:
        response = client.get(f"/api/turbines/1/{operation}?start=2026-01-30&end=2026-01-31")
    elif operation == "import":
        response = client.post(
            "/api/turbines/1/import",
            files={"file": ("source.csv", (tmp_path / "source.csv").read_bytes(), "text/csv")},
        )
    elif operation in {"preflight", "agent"}:
        response = client.post(
            "/api/agent/" + ("preflight" if operation == "preflight" else "runs"),
            json={
                "turbine_id": 1,
                "issue_at": "2026-01-31T00:00:00Z",
                "measurement_timezone": "UTC",
                "timestamp_semantics": "interval_start",
            },
        )
    elif operation == "weather":
        response = client.post(
            "/api/weather", json={"turbine_id": 1, "run": "2026-01-30T12:00:00Z"}
        )
    else:
        response = client.post(
            "/api/discovery/plan", json={"site": "https://example.org", "turbine_id": 1}
        )
    assert response.status_code == 404
    assert "корзины" in response.json()["detail"]


def test_deleted_turbine_cannot_resume_discovery_plan(client, tmp_path):
    measured_turbine(tmp_path)
    report = {
        "id": "c" * 32,
        "status": "ready",
        "request": {"site": "https://example.org", "turbine_id": 1},
        "context": {},
    }
    write_plan(report)
    client.delete("/api/turbines/1")
    response = client.post(f"/api/discovery/plans/{report['id']}/next", json={})
    assert response.status_code == 404
    assert client.get(f"/api/discovery/plans/{report['id']}").json() == report
    saved = tmp_path / "storage/discovery" / f"{report['id']}.json"
    assert json.loads(saved.read_text()) == report


def test_new_turbine_page_and_canonical_redirect(tmp_path):
    (tmp_path / "index.html").write_text("<div id='root'></div>")
    frontend = FastAPI()
    mount_frontend(frontend, tmp_path)
    with TestClient(frontend) as client:
        assert client.get("/turbines/new").status_code == 200
        response = client.get("/turbines/new/?from=forecast", follow_redirects=False)
        assert response.status_code == 308
        assert response.headers["location"] == "/turbines/new?from=forecast"
