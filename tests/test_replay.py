import csv
import io
import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from wind.app import app
from wind.replay import (
    ReplayRequest,
    coverage_rows,
    create_job,
    export_csv,
    folder,
    get_job,
    run_job,
    write_json,
)
from wind.storage import all_metadata, create_turbine, data_dir, save_metadata


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for number in (1, 2):
        create_turbine({"name": f"Test {number}", "latitude": 43.6, "longitude": 78.5})
        save_metadata("datasets", number, {"id": number, "sha256": f"source-{number}"})
    return tmp_path


def request(**changes):
    return ReplayRequest(
        **{
            "turbine_ids": [1, 2],
            "measurement_timezone": "Etc/GMT-6",
            "timestamp_semantics": "interval_start",
            **changes,
        }
    )


def fake_runner(item, *, run_id=None):
    key = run_id or uuid4().hex
    points = [
        {"time": (item.issue_at + timedelta(hours=hour)).isoformat(), "power": 0.4}
        for hour in range(1, item.horizon + 1)
    ]
    write_json(
        data_dir() / "forecasts" / f"{key}.json",
        {
            "request": item.model_dump(mode="json"),
            "points": points,
            "model": "nwp_hist_gradient_boosting",
            "model_metadata": {
                "weather_used": True,
                "source_sha256": f"source-{item.turbine_id}",
                "model_sha256": "a" * 64,
                "training_end": "2026-01-31T12:00:00+00:00",
                "usable_from": "2026-01-31T12:00:00+00:00",
                "weather_runs": [
                    {
                        "run": (item.issue_at - timedelta(hours=12)).isoformat(),
                        "available_at": (item.issue_at - timedelta(hours=8)).isoformat(),
                    }
                ],
            },
        },
    )
    return {"id": key, "status": "forecast_saved", "input_tokens": 0, "output_tokens": 0}


def test_entire_february_and_no_duplicate_calls_on_resume(workspace):
    calls = []

    def runner(item, **kwargs):
        assert item.forecast_mode == "weather"
        calls.append(item)
        return fake_runner(item, **kwargs)

    job = run_job(create_job(request(), validate_models=False)["id"], runner=runner)
    assert len(calls) == 58
    assert job["progress"] == {"completed": 58, "total": 58, "succeeded": 58, "failed": 0}
    assert job["summary"]["coverage_hours"] == 2 * 672
    assert job["summary"]["missing_hours"] == 0
    assert job["summary"]["accuracy_evaluated"] is False
    rows = coverage_rows(job)
    assert all(
        datetime.fromisoformat(r["issue_at"]) < datetime.fromisoformat(r["target_time"])
        for r in rows
    )
    noon = next(
        r for r in rows if r["turbine_id"] == 1 and r["target_time"] == "2026-02-01T12:00:00+00:00"
    )
    assert noon["issue_at"] == "2026-01-31T12:00:00+00:00"
    after = next(
        r for r in rows if r["turbine_id"] == 1 and r["target_time"] == "2026-02-01T13:00:00+00:00"
    )
    assert after["issue_at"] == "2026-02-01T12:00:00+00:00"
    assert len(list(csv.DictReader(io.StringIO(export_csv(job, "forecasts"))))) == 58 * 48
    run_job(job["id"], runner=runner)
    assert len(calls) == 58


def test_dataset_revision_blocks_mixed_result(workspace):
    calls = []

    def runner(item, **kwargs):
        calls.append(item)
        report = fake_runner(item, **kwargs)
        save_metadata("datasets", 1, {"id": 1, "sha256": "changed"})
        return report

    job = create_job(request(turbine_ids=[1], end_date="2026-02-01"), validate_models=False)
    job = run_job(job["id"], runner=runner)
    assert len(calls) == 1
    assert job["status"] == "failed"
    assert job["progress"]["failed"] == 2
    assert "изменились" in job["errors"][0]["detail"]


def test_weather_model_cannot_be_replaced_by_telemetry(workspace):
    def wrong_runner(item, **kwargs):
        report = fake_runner(item, **kwargs)
        path = data_dir() / "forecasts" / f"{report['id']}.json"
        payload = json.loads(path.read_text())
        payload["model_metadata"]["weather_used"] = False
        write_json(path, payload)
        return report

    job = create_job(request(turbine_ids=[1], end_date="2026-01-31"), validate_models=False)
    result = run_job(job["id"], runner=wrong_runner)
    assert result["status"] == "failed"
    assert "baseline" in result["errors"][0]["detail"]


@pytest.mark.parametrize(
    "changes",
    [
        {"turbine_ids": [1, 1]},
        {"measurement_timezone": ""},
        {"start_date": "2026-02-28", "end_date": "2026-01-31"},
        {"horizon": 49},
        {"execution_mode": "llm", "start_date": "2025-01-01"},
    ],
)
def test_invalid_replay_parameters(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_api_requires_model_and_bounds_export_paths(workspace):
    with TestClient(app) as client:
        response = client.post("/api/replays", json=request().model_dump(mode="json"))
        assert response.status_code == 422
        assert "модель" in response.json()["detail"]
        assert client.get("/api/replays/not-a-key").status_code == 404
        assert client.get("/api/replays/abc/export?kind=invalid").status_code == 422
        assert client.get("/api/replays").json() == {"items": []}


def test_physical_power_only_uses_confirmed_capacity_snapshot(workspace):
    no_capacity = create_job(request(), validate_models=False)
    with pytest.raises(ValueError, match="номинальную"):
        export_csv(no_capacity, "plant")
    for turbine in all_metadata("turbines"):
        turbine.update(
            {
                "rated_power_kw": 2500 if turbine["id"] == 1 else 1000,
                "capacity_source_url": "https://www.openstreetmap.org/node/1",
                "capacity_status": "user_confirmed_osm",
            }
        )
        save_metadata("turbines", turbine["id"], turbine)
    job = create_job(request(end_date="2026-02-01"), validate_models=False)
    job = run_job(job["id"], runner=fake_runner)
    assert job["status"] == "completed"
    rows = list(csv.DictReader(io.StringIO(export_csv(job, "plant"))))
    assert len(rows) == 24
    assert float(rows[0]["power_kw"]) == pytest.approx(1400)
    assert float(rows[0]["energy_kwh"]) == pytest.approx(1400)
    # Old exports retain their declared provenance; current card edits don't rewrite history.
    changed = all_metadata("turbines")[0]
    changed["rated_power_kw"] = 9000
    save_metadata("turbines", changed["id"], changed)
    assert float(list(csv.DictReader(io.StringIO(export_csv(job, "plant"))))[0]["power_kw"]) == 1400


def test_single_issue_coverage_is_next_calendar_day(workspace):
    job = create_job(request(turbine_ids=[1], end_date="2026-01-31"), validate_models=False)
    job = run_job(job["id"], runner=fake_runner)
    assert job["summary"]["expected_hours"] == 24
    assert job["status"] == "completed"


def test_crash_after_saved_forecast_recovers_without_repeating_paid_call(workspace):
    calls = []

    def crash_after_save(item, **kwargs):
        calls.append(kwargs["run_id"])
        fake_runner(item, **kwargs)
        raise KeyboardInterrupt("process crash")

    job = create_job(
        request(execution_mode="llm", turbine_ids=[1], end_date="2026-02-01"), validate_models=False
    )
    with pytest.raises(KeyboardInterrupt):
        run_job(job["id"], runner=crash_after_save)
    persisted = get_job(job["id"])
    assert not persisted["results"]
    assert len(persisted["attempts"]) == 1

    def resumed(item, **kwargs):
        calls.append(kwargs["run_id"])
        return fake_runner(item, **kwargs)

    result = run_job(job["id"], runner=resumed)
    assert result["status"] == "completed"
    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert result["results"][0]["recovered_after_interruption"] is True
    assert result["results"][0]["token_usage_unknown"] is True


def test_unknown_paid_attempt_never_retries_automatically(workspace):
    calls = []

    def crash(item, **kwargs):
        calls.append(kwargs["run_id"])
        raise KeyboardInterrupt("paid response was lost")

    job = create_job(
        request(execution_mode="llm", turbine_ids=[1], end_date="2026-02-01"), validate_models=False
    )
    with pytest.raises(KeyboardInterrupt):
        run_job(job["id"], runner=crash)

    def resumed(item, **kwargs):
        calls.append(kwargs["run_id"])
        return fake_runner(item, **kwargs)

    result = run_job(job["id"], runner=resumed)
    assert len(calls) == 2  # failed January31 is not paid for twice; February1 runs once.
    assert len(set(calls)) == 2
    assert result["status"] == "partial"
    assert result["summary"]["missing_hours"] > 0
    assert "автоматический повтор запрещён" in result["errors"][0]["detail"]
    run_job(job["id"], runner=resumed)
    assert len(calls) == 2


def test_tools_crash_can_resume_same_durable_id(workspace):
    ids = []

    def crash(item, **kwargs):
        ids.append(kwargs["run_id"])
        raise KeyboardInterrupt("before deterministic computation")

    job = create_job(request(turbine_ids=[1], end_date="2026-02-01"), validate_models=False)
    with pytest.raises(KeyboardInterrupt):
        run_job(job["id"], runner=crash)

    def resume(item, **kwargs):
        ids.append(kwargs["run_id"])
        return fake_runner(item, **kwargs)

    result = run_job(job["id"], runner=resume)
    assert result["status"] == "completed"
    assert ids[0] == ids[1]


@pytest.mark.parametrize("corruption", ["missing_hour", "duplicate_hour", "nan", "wrong_issue"])
def test_bad_saved_payload_never_counts_as_completed(workspace, corruption):
    def corrupt(item, **kwargs):
        report = fake_runner(item, **kwargs)
        path = data_dir() / "forecasts" / f"{report['id']}.json"
        payload = json.loads(path.read_text())
        if corruption == "missing_hour":
            payload["points"].pop()
        elif corruption == "duplicate_hour":
            payload["points"][-1] = payload["points"][0]
        elif corruption == "nan":
            payload["points"][0]["power"] = float("nan")
        else:
            payload["request"]["issue_at"] = "2026-01-30T12:00:00Z"
        path.write_text(json.dumps(payload))
        return report

    job = create_job(request(turbine_ids=[1], end_date="2026-02-01"), validate_models=False)
    result = run_job(job["id"], runner=corrupt)
    assert result["status"] == "failed"
    assert result["progress"]["succeeded"] == 0
    assert result["summary"]["complete"] is False


def test_saved_forecast_corruption_invalidates_completed_job_on_resume(workspace):
    job = create_job(request(turbine_ids=[1], end_date="2026-02-01"), validate_models=False)
    result = run_job(job["id"], runner=fake_runner)
    assert result["status"] == "completed"
    key = result["results"][0]["forecast_id"]
    (data_dir() / "forecasts" / f"{key}.json").unlink()
    result = run_job(job["id"], runner=lambda *_args, **_kwargs: pytest.fail("must not repeat"))
    assert result["status"] == "partial"
    assert result["summary"]["complete"] is False


def test_weights_changed_after_job_creation_are_blocked_before_runner(workspace, monkeypatch):
    from wind import replay

    snapshot = {
        "model_sha256": "a" * 64,
        "training_end": "2026-01-31T12:00:00+00:00",
        "usable_from": "2026-01-31T12:00:00+00:00",
        "weather_contract": {"model": "gfs"},
    }
    monkeypatch.setattr(replay, "model_snapshot", lambda *_args: snapshot.copy())
    job = create_job(request(turbine_ids=[1], end_date="2026-02-01"))
    snapshot["model_sha256"] = "b" * 64
    result = run_job(job["id"], runner=lambda *_args, **_kwargs: pytest.fail("changed weights"))
    assert result["status"] == "failed"
    assert "Модель" in result["errors"][0]["detail"]


def test_changed_weights_during_runner_invalidate_the_current_issue(workspace, monkeypatch):
    from wind import replay

    snapshot = {
        "model_sha256": "a" * 64,
        "training_end": "2026-01-31T12:00:00+00:00",
        "usable_from": "2026-01-31T12:00:00+00:00",
        "weather_contract": {"model": "gfs"},
    }
    monkeypatch.setattr(replay, "model_snapshot", lambda *_args: snapshot.copy())
    job = create_job(request(turbine_ids=[1], end_date="2026-02-01"))

    def runner(item, **kwargs):
        report = fake_runner(item, **kwargs)
        snapshot["model_sha256"] = "b" * 64
        return report

    result = run_job(job["id"], runner=runner)
    assert result["status"] == "failed"
    assert result["progress"]["succeeded"] == 0


def test_existing_recovery_note_cannot_force_partial_after_success(workspace):
    job = create_job(request(turbine_ids=[1], end_date="2026-02-01"), validate_models=False)
    job["errors"] = [{"detail": "previous process interrupted"}]
    write_json(folder(job["id"]) / "job.json", job)
    result = run_job(job["id"], runner=fake_runner)
    assert result["status"] == "completed"
    assert result["errors"] == []
