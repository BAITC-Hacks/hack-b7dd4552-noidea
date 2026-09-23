import io
import json

import pytest
from fastapi.testclient import TestClient

from wind.app import app
from wind.automation import (
    AutomationSettings,
    enqueue_event,
    get_settings,
    list_events,
    recover_interrupted,
    run_event,
    set_settings,
)
from wind.ingest import import_csv
from wind.replay import _worker_lock, write_json
from wind.simulate import make_measurements
from wind.storage import all_metadata, create_turbine, data_dir


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})
    source = tmp_path / "measurements.csv"
    make_measurements(source, "healthy")
    import_csv(source, 1)
    return source


def enabled(**changes):
    return AutomationSettings(
        **{
            "enabled": True,
            "turbine_ids": [1],
            "measurement_timezone": "UTC",
            "timestamp_semantics": "interval_start",
            **changes,
        }
    )


def test_disabled_never_schedules_and_explicit_events_deduplicate(workspace):
    assert get_settings().enabled is False
    assert enqueue_event("data_updated", 1, "new-hash") is None
    set_settings(enabled())
    revision = all_metadata("datasets")[0]["sha256"]
    event_id = enqueue_event("data_updated", 1, revision)
    assert event_id
    assert enqueue_event("data_updated", 1, revision) is None
    calls = []

    def runner(request):
        calls.append(request)
        return {"status": "forecast_saved", "id": "test"}

    assert run_event(event_id, runner=runner)["status"] == "completed"
    assert run_event(event_id, runner=runner)["status"] == "completed"
    assert len(calls) == 1
    assert calls[0].event == "data_updated"
    assert list_events()[0]["settings"]["execution_mode"] == "tools"


def test_update_keeps_historical_issue_and_disabling_cancels_queued(workspace):
    stamp = "2026-02-05T12:00:00+00:00"
    write_json(
        data_dir() / "forecasts" / "known.json", {"request": {"turbine_id": 1, "issue_at": stamp}}
    )
    set_settings(enabled())
    event_id = enqueue_event("weather_updated", 1, "weather-revision")
    assert list_events()[0]["issue_at"] == stamp
    set_settings(AutomationSettings())

    def forbidden(request):
        pytest.fail("Disabled automation must not invoke a model")

    assert run_event(event_id, runner=forbidden)["status"] == "cancelled"


def test_import_api_triggers_only_opted_in_background_job(workspace, monkeypatch):
    called = []
    monkeypatch.setattr("wind.app.run_event", lambda key: called.append(key))
    with TestClient(app) as client:
        assert client.get("/api/automation").json()["enabled"] is False
        config = enabled().model_dump(mode="json")
        assert client.put("/api/automation", json=config).status_code == 200
        with workspace.open("rb") as file:
            response = client.post(
                "/api/turbines/1/import", files={"file": ("new.csv", file, "text/csv")}
            )
        assert response.status_code == 200
        assert called == [response.json()["automation_event_id"]]
        response = client.post(
            "/api/turbines/1/import",
            files={"file": ("new.csv", io.BytesIO(workspace.read_bytes()), "text/csv")},
        )
        assert response.status_code == 200
        assert len(called) == 1


def test_weather_api_triggers_revision_event(workspace, monkeypatch):
    called = []
    monkeypatch.setattr("wind.app.run_event", lambda key: called.append(key))
    monkeypatch.setattr(
        "wind.archive.fetch_issue_weather", lambda *a, **k: {"sha256": "forecast-revision"}
    )
    with TestClient(app) as client:
        client.put("/api/automation", json=enabled().model_dump(mode="json"))
        response = client.post(
            "/api/weather/gfs", json={"turbine_id": 1, "issue_at": "2026-01-31T12:00:00Z"}
        )
        assert response.status_code == 200
        assert called == [response.json()["automation_event_id"]]
        assert list_events()[0]["event"] == "weather_updated"


def test_enable_requires_explicit_time_and_turbines():
    with pytest.raises(ValueError):
        AutomationSettings(enabled=True)
    with pytest.raises(ValueError):
        enabled(measurement_timezone="not/a-zone")
    assert AutomationSettings().measurement_timezone is None


def test_superseded_measurements_do_not_spend_again(workspace):
    set_settings(enabled())
    event_id = enqueue_event("data_updated", 1, "older-revision")
    result = run_event(event_id, runner=lambda r: pytest.fail("Must skip superseded inputs"))
    assert result["status"] == "superseded"


def test_disabling_while_waiting_for_worker_cancels_request(workspace, monkeypatch):
    set_settings(enabled())
    event_id = enqueue_event("weather_updated", 1, "weather-revision")

    class DisableOnAcquire:
        def __enter__(self):
            set_settings(AutomationSettings())

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("wind.automation.RUN_LOCK", DisableOnAcquire())
    result = run_event(event_id, runner=lambda r: pytest.fail("Disabled while queued"))
    assert result["status"] == "cancelled"


@pytest.mark.parametrize("succeeded,status", [(0, "failed"), (1, "partial")])
def test_api_startup_keeps_live_cli_replay_and_recovers_after_lock_release(
    workspace, succeeded, status
):
    path = data_dir() / "replays" / ("a" * 32) / "job.json"
    job = {"status": "running", "progress": {"succeeded": succeeded}, "errors": []}
    write_json(path, job)
    with _worker_lock():
        # Startup recovery must return without waiting for or changing the live worker.
        recover_interrupted()
        assert json.loads(path.read_text()) == job
    recover_interrupted()
    recovered = json.loads(path.read_text())
    assert recovered["status"] == status
    assert recovered["interrupted"] is True
    assert len(recovered["errors"]) == 1
    recover_interrupted()
    assert json.loads(path.read_text()) == recovered
