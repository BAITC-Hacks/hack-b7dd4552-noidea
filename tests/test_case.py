import hashlib
import json
from functools import partial
from pathlib import Path

import pandas as pd
import pytest

from wind import case
from wind.storage import create_turbine, save_metadata


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    turbine = create_turbine({"name": "Case turbine", "latitude": 43.6, "longitude": 78.5})
    raw = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-25", "2026-01-31 23:50", freq="10min"),
            "power": 0.4,
            "wind_speed": 5.0,
            "temperature": 2.0,
        }
    )
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    raw.to_parquet(tmp_path / "1-source-10min.parquet", index=False)
    dataset = {
        "id": turbine["id"],
        "sha256": digest,
        "raw_path": "source.csv",
        "hourly_path": "1-source-hourly.parquet",
        "dataset_kind": "full",
    }
    save_metadata("datasets", turbine["id"], dataset)
    return tmp_path, {
        "turbine_id": turbine["id"],
        "name": turbine["name"],
        "latitude": 43.6,
        "longitude": 78.5,
        "source_sha256": digest,
        "dataset": dataset,
        "raw": raw,
    }


def arguments(**changes):
    return {
        "all_turbines": True,
        "measurement_timezone": "Etc/GMT-6",
        "timestamp_semantics": "interval_start",
        **changes,
    }


def no_network(*args, **kwargs):
    pytest.fail("A network or paid phase was reached unexpectedly")


def test_missing_full_history_fails_before_download(workspace, monkeypatch):
    root, item = workspace
    monkeypatch.setattr(case, "download_archive", no_network)
    monkeypatch.setattr(case, "run_agent", no_network)
    with pytest.raises(ValueError, match="недостаточно полной истории"):
        case.run_case(**arguments(execution_mode="llm"))
    assert hashlib.sha256((root / "source.csv").read_bytes()).hexdigest() == item["source_sha256"]
    report = json.loads(next((root / "case-runs").glob("*/report.json")).read_text())
    assert report["status"] == "failed"
    assert not any(p["phase"] == "download_archive" for p in report["phases"])


def test_original_csv_tampering_fails_before_download(workspace, monkeypatch):
    root, _ = workspace
    (root / "source.csv").write_text("changed")
    monkeypatch.setattr(case, "download_archive", no_network)
    with pytest.raises(ValueError, match="CSV изменился"):
        case.run_case(**arguments())


def test_skip_train_incompatible_model_fails_before_network(workspace, monkeypatch):
    _, item = workspace
    monkeypatch.setattr(case, "validate_measurements", lambda *args: [item])
    monkeypatch.setattr(
        case, "model_readiness", lambda *args, **kwargs: {"ok": False, "reason": "wrong CSV"}
    )
    monkeypatch.setattr(case, "download_archive", no_network)
    with pytest.raises(ValueError, match="skip-train: wrong CSV"):
        case.run_case(**arguments(skip_train=True))


@pytest.fixture
def fake_phases(workspace, monkeypatch):
    root, item = workspace
    calls = []
    monkeypatch.setattr(case, "validate_measurements", lambda *args: [item])
    monkeypatch.setattr(case, "model_readiness", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(case, "_contract", lambda value: value)
    weather = pd.DataFrame({"time": ["2025-01-01T13:00:00Z"]})
    weather.attrs["provenance"] = {"test_fixture": True}

    def download(ids, start, end, **kwargs):
        calls.append(("download", ids, start, end, kwargs["workers"]))
        return {"completed": 424, "failed": [], "tasks": 424}

    def load(turbine_id):
        calls.append(("load", turbine_id))
        return weather

    def train(raw, forecast, **kwargs):
        calls.append(("train", kwargs))
        directory = Path(kwargs["directory"])
        directory.mkdir(parents=True)
        (directory / "model.joblib").write_bytes(b"fake locally trained model")
        manifest = {"promoted": True, "turbine_id": kwargs["turbine_id"]}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def create(request):
        calls.append(("create", request))
        return {"id": "a" * 32, "request": request.model_dump(mode="json")}

    def run(job_id, *, runner):
        calls.append(("run", job_id, runner))
        return {
            "id": job_id,
            "status": "completed",
            "progress": {"completed": 29, "total": 29, "succeeded": 29, "failed": 0},
            "summary": {"coverage_hours": 672, "expected_hours": 672, "accuracy_evaluated": False},
            "errors": [],
        }

    monkeypatch.setattr(case, "download_archive", download)
    monkeypatch.setattr(case, "load_archive_frame", load)
    monkeypatch.setattr(case, "train_nwp", train)
    monkeypatch.setattr(case, "create_job", create)
    monkeypatch.setattr(case, "run_job", run)
    monkeypatch.setattr(case, "export_csv", lambda job, kind: f"kind\n{kind}\n")
    return root, item, calls


@pytest.mark.parametrize("mode", ["tools", "llm"])
def test_pipeline_order_modes_exports_and_existing_artifact_preservation(fake_phases, mode):
    root, item, calls = fake_phases
    previous = case.artifact_directory(item["turbine_id"], "Etc/GMT-6", "interval_start")
    previous.mkdir(parents=True)
    (previous / "old-model.joblib").write_bytes(b"previous model")
    events = []
    report = case.run_case(**arguments(execution_mode=mode, progress=events.append, workers=2))
    assert [c[0] for c in calls] == ["download", "load", "train", "create", "run"]
    assert calls[0][2:] == (case.ARCHIVE_START, case.ARCHIVE_END, 2)
    training = calls[2][1]
    assert training["train_end"] == case.TRAIN_END
    assert training["validation_end"] == case.VALIDATION_END
    assert training["control_end"] == training["final_issue_at"] == case.FIRST_ISSUE
    assert calls[3][1].execution_mode == mode
    assert calls[4][2] is (case.run_agent if mode == "llm" else case.run_tools)
    assert report["status"] == "completed"
    assert Path(report["exports"]["coverage"]).read_text() == "kind\ncoverage\n"
    assert "plant" not in report["exports"]
    assert (
        Path(report["model_paths"][0]["previous"], "old-model.joblib").read_bytes()
        == b"previous model"
    )
    assert (previous / "model.joblib").read_bytes() == b"fake locally trained model"
    assert hashlib.sha256((root / "source.csv").read_bytes()).hexdigest() == item["source_sha256"]
    assert [e["phase"] for e in events if e["status"] == "completed"] == [
        "validate_inputs",
        "download_archive",
        "load_weather",
        "train_models",
        "replay",
        "export",
    ]


def test_offline_mode_never_downloads_even_during_replay(fake_phases, monkeypatch):
    _, _, calls = fake_phases
    monkeypatch.setattr(case, "download_archive", no_network)
    monkeypatch.setattr(case, "train_nwp", no_network)
    report = case.run_case(**arguments(offline=True, skip_train=True))
    assert report["status"] == "completed"
    assert [c[0] for c in calls] == ["load", "create", "run"]
    runner = calls[-1][2]
    assert isinstance(runner, partial) and runner.func is case.run_tools
    provider = runner.keywords["issue_weather_provider"]
    assert isinstance(provider, partial) and provider.func is case.fetch_issue_weather
    assert provider.keywords == {"offline": True}
    assert [(p["phase"], p["status"]) for p in report["phases"] if p["status"] == "skipped"] == [
        ("download_archive", "skipped"),
        ("train_models", "skipped"),
    ]


def test_unpromoted_model_never_replaces_existing_or_starts_replay(fake_phases, monkeypatch):
    _, item, calls = fake_phases
    previous = case.artifact_directory(item["turbine_id"], "Etc/GMT-6", "interval_start")
    previous.mkdir(parents=True)
    (previous / "model.joblib").write_bytes(b"previous model")
    monkeypatch.setattr(case, "train_nwp", lambda *args, **kwargs: {"promoted": False})
    monkeypatch.setattr(case, "create_job", no_network)
    with pytest.raises(ValueError, match="не прошла валидацию"):
        case.run_case(**arguments())
    assert (previous / "model.joblib").read_bytes() == b"previous model"
    assert [c[0] for c in calls] == ["download", "load"]


def test_offline_llm_is_rejected_before_any_phase(workspace, monkeypatch):
    monkeypatch.setattr(case, "validate_measurements", no_network)
    with pytest.raises(ValueError, match="LLM требует сетевого"):
        case.run_case(**arguments(offline=True, execution_mode="llm"))


def test_confirmed_snapshot_adds_physical_export(fake_phases, monkeypatch):
    _, item, _ = fake_phases
    run_job = case.run_job

    def with_confirmed_capacity(job_id, **kwargs):
        job = run_job(job_id, **kwargs)
        job["input_snapshots"] = {str(item["turbine_id"]): {"rated_power_kw": 2500}}
        return job

    monkeypatch.setattr(case, "run_job", with_confirmed_capacity)
    report = case.run_case(**arguments(offline=True, skip_train=True))
    assert Path(report["exports"]["plant"]).read_text() == "kind\nplant\n"
