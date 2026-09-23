from pathlib import Path

from fastapi.testclient import TestClient

from wind.app import app


def test_empty_start_create_import_and_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    sample = Path(__file__).resolve().parents[1] / "samples/demo turbine 1.csv"
    with TestClient(app) as client:
        assert client.get("/api/turbines").json() == {"items": []}
        for i in range(3):
            response = client.post(
                "/api/turbines",
                json={
                    "name": f"Турбина {i}",
                    "latitude": 43.645,
                    "longitude": 78.535,
                },
            )
            assert response.status_code == 201
        turbine_id = response.json()["id"]
        assert turbine_id == 3
        assert not client.get("/api/turbines").json()["items"][2]["has_data"]
        response = client.post(
            f"/api/turbines/{turbine_id}/import",
            files={
                "file": ("measurements.csv", sample.read_bytes(), "text/csv"),
            },
        )
        assert response.status_code == 200
        assert response.json()["rows"] == 1008
        checksum = response.json()["sha256"]
        bad = client.post(
            f"/api/turbines/{turbine_id}/import",
            files={
                "file": ("broken.csv", b"not,a,valid,dataset", "text/csv"),
            },
        )
        assert bad.status_code == 422
        catalog = client.get("/api/turbines").json()["items"]
        assert catalog[2]["sha256"] == checksum
        assert catalog[2]["has_data"] is True
        assert client.get("/api/weather?turbine_id=3").json() == {"items": []}
    with TestClient(app) as client:
        assert len(client.get("/api/turbines").json()["items"]) == 3


def test_invalid_turbine_and_import(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        for name, latitude in [(" ", 43), ("Test", 91)]:
            assert (
                client.post(
                    "/api/turbines",
                    json={
                        "name": name,
                        "latitude": latitude,
                        "longitude": 78,
                    },
                ).status_code
                == 422
            )
        assert (
            client.post(
                "/api/turbines/99/import",
                files={
                    "file": ("data.csv", b"test", "text/csv"),
                },
            ).status_code
            == 404
        )
