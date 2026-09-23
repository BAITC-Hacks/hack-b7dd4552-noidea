"""Durable historical issue replay and explicit, labelled result exports."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
from contextlib import contextmanager
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wind.agent import AgentRequest, run_agent, run_tools
from wind.storage import all_metadata, data_dir, get_turbine

RUN_LOCK = Lock()
JOB_LOCK = Lock()
TERMINAL = {"completed", "partial", "failed"}


def now():
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value):
    """Readers see either complete previous JSON or complete new JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turbine_ids: list[int] = Field(min_length=1, max_length=32)
    measurement_timezone: str
    timestamp_semantics: Literal["interval_start", "interval_end"]
    start_date: date = date(2026, 1, 31)
    end_date: date = date(2026, 2, 28)
    issue_hour_utc: int = Field(default=12, ge=0, le=23)
    horizon: int = Field(default=48, ge=24, le=48)
    execution_mode: Literal["tools", "llm"] = "tools"

    @field_validator("turbine_ids")
    @classmethod
    def ids(cls, value):
        if any(item < 1 for item in value) or len(set(value)) != len(value):
            raise ValueError("Укажите уникальные положительные ID турбин")
        return value

    @field_validator("measurement_timezone")
    @classmethod
    def timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ValueError, KeyError) as exc:
            raise ValueError("Укажите часовой пояс IANA") from exc
        return value

    @model_validator(mode="after")
    def dates(self):
        days = (self.end_date - self.start_date).days + 1
        if not 1 <= days <= 366:
            raise ValueError("Диапазон выпусков должен содержать от 1 до 366 дней")
        if datetime.combine(self.end_date, time(self.issue_hour_utc), UTC) > datetime.now(UTC):
            raise ValueError("Момент исторического выпуска не может быть в будущем")
        if self.execution_mode == "llm" and days * len(self.turbine_ids) > 128:
            raise ValueError("Один платный прогон ограничен 128 выпусками; разбейте диапазон")
        return self

    def issues(self):
        return [
            datetime.combine(self.start_date + timedelta(days=day), time(self.issue_hour_utc), UTC)
            for day in range((self.end_date - self.start_date).days + 1)
        ]


def folder(job_id):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Некорректный идентификатор прогона")
    return data_dir() / "replays" / job_id


def get_job(job_id):
    path = folder(job_id) / "job.json"
    if not path.exists():
        raise ValueError("Прогон не найден")
    return json.loads(path.read_text())


def list_jobs():
    paths = sorted(
        (data_dir() / "replays").glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return [json.loads(path.read_text()) for path in paths[:50]]


def dataset_snapshot(turbine_id):
    turbine = get_turbine(turbine_id)
    dataset = next((d for d in all_metadata("datasets") if d["id"] == turbine_id), None)
    if dataset is None:
        raise ValueError(f"Турбина {turbine_id}: нет импортированных измерений")
    return {
        "sha256": dataset["sha256"],
        "latitude": turbine["latitude"],
        "longitude": turbine["longitude"],
        "rated_power_kw": turbine.get("rated_power_kw")
        if turbine.get("capacity_status") == "user_confirmed_osm"
        else None,
        "capacity_source_url": turbine.get("capacity_source_url")
        if turbine.get("capacity_status") == "user_confirmed_osm"
        else None,
    }


def model_snapshot(turbine_id, request, issue):
    from wind.nwp import model_readiness

    dataset = next(d for d in all_metadata("datasets") if d["id"] == turbine_id)
    ready = model_readiness(
        dataset,
        timezone=request.measurement_timezone,
        semantics=request.timestamp_semantics,
        issue_at=issue,
        horizon=request.horizon,
    )
    if not ready["ok"]:
        raise ValueError(
            f"Турбина {turbine_id}: {ready.get('reason', 'погодная модель не готова')}"
        )
    return {
        key: ready[key]
        for key in (
            "model_sha256",
            "training_end",
            "usable_from",
            "weather_contract",
        )
    }


@contextmanager
def _worker_lock():
    """Serialize API and CLI replays even when they run in separate processes."""
    path = data_dir() / "replays" / ".worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def create_job(request: ReplayRequest, *, validate_models=True):
    snapshots = {str(item): dataset_snapshot(item) for item in request.turbine_ids}
    models = (
        {
            str(item): model_snapshot(item, request, request.issues()[0])
            for item in request.turbine_ids
        }
        if validate_models
        else {}
    )
    job = {
        "id": uuid4().hex,
        "status": "queued",
        "request": request.model_dump(mode="json"),
        "created_at": now(),
        "updated_at": now(),
        "input_snapshots": snapshots,
        "model_snapshots": models,
        "attempts": {},
        "progress": {
            "completed": 0,
            "total": len(request.issues()) * len(request.turbine_ids),
            "succeeded": 0,
            "failed": 0,
        },
        "results": [],
        "errors": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "target_semantics": "hour_interval_start",
        "target_timezone": "UTC",
        "units": "fraction_of_turbine_rated_power",
        "issue_policy": "daily at configured UTC hour; targets issue+1h through issue+horizon",
    }
    write_json(folder(job["id"]) / "job.json", job)
    return job


def _agent_request(request, turbine_id, issue):
    return AgentRequest(
        turbine_id=turbine_id,
        issue_at=issue,
        measurement_timezone=request.measurement_timezone,
        timestamp_semantics=request.timestamp_semantics,
        horizon=request.horizon,
        forecast_mode="weather",
    )


def _check_inputs(job, request, turbine_id, issue):
    if dataset_snapshot(turbine_id) != job["input_snapshots"][str(turbine_id)]:
        raise ValueError("Измерения или координаты изменились; создайте новый прогон")
    expected = job.get("model_snapshots", {}).get(str(turbine_id))
    if expected and model_snapshot(turbine_id, request, issue) != expected:
        raise ValueError("Модель или погодный контракт изменились; создайте новый прогон")


def _forecast(job, entry):
    """Check persisted output independently of an agent's status or self-report."""
    key = entry.get("forecast_id") or entry.get("run_id")
    if not isinstance(key, str) or not re.fullmatch("[a-f0-9]{32}", key):
        raise ValueError("Некорректный идентификатор прогноза")
    content = (data_dir() / "forecasts" / f"{key}.json").read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if entry.get("forecast_sha256") and entry["forecast_sha256"] != digest:
        raise ValueError("Сохранённый прогноз изменился после проверки")
    payload = json.loads(content)
    request = ReplayRequest(**job["request"])
    expected = _agent_request(request, entry["turbine_id"], entry["issue_at"])
    if AgentRequest(**payload["request"]) != expected:
        raise ValueError("Сохранённый прогноз относится к другому запросу")
    metadata = payload["model_metadata"]
    if (
        metadata.get("weather_used") is not True
        or payload.get("model") != "nwp_hist_gradient_boosting"
    ):
        raise ValueError("Февральский прогон требует погодную модель; подмена baseline запрещена")
    if metadata.get("source_sha256") != job["input_snapshots"][str(entry["turbine_id"])]["sha256"]:
        raise ValueError("Прогноз использует другую версию измерений")
    if not re.fullmatch("[a-f0-9]{64}", metadata.get("model_sha256", "")):
        raise ValueError("Нет контрольной суммы модели в прогнозе")
    selected = job.get("model_snapshots", {}).get(str(entry["turbine_id"]))
    if selected and any(metadata.get(key) != value for key, value in selected.items()):
        raise ValueError("Прогноз использует другую модель или погодный контракт")
    issue = expected.issue_at
    trained = datetime.fromisoformat(metadata["training_end"])
    usable = datetime.fromisoformat(metadata["usable_from"])
    if any(value.tzinfo is None for value in (trained, usable)) or not trained <= usable <= issue:
        raise ValueError("Модель использует сведения позже момента прогноза")
    runs = metadata.get("weather_runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("В прогнозе нет происхождения погодного выпуска")
    for run in runs:
        initialized, available = map(datetime.fromisoformat, (run["run"], run["available_at"]))
        if any(value.tzinfo is None for value in (initialized, available)) or not (
            initialized <= available <= issue
        ):
            raise ValueError("Погодный выпуск был недоступен на момент расчёта")
    points = payload["points"]
    if len(points) != request.horizon:
        raise ValueError("Прогноз не покрывает весь запрошенный горизонт")
    times = [datetime.fromisoformat(point["time"]) for point in points]
    if times != [issue + timedelta(hours=h) for h in range(1, request.horizon + 1)]:
        raise ValueError("Прогноз содержит пропущенные, повторные или неверные часы")
    if any(
        isinstance(point["power"], bool)
        or not isinstance(point["power"], (int, float))
        or not math.isfinite(point["power"])
        or not 0 <= point["power"] <= 1
        for point in points
    ):
        raise ValueError("Некорректная нормированная мощность в прогнозе")
    return payload, digest


def _progress(job):
    succeeded = sum(row["status"] == "completed" for row in job["results"])
    failed = sum(row["status"] == "failed" for row in job["results"])
    job["progress"].update(completed=succeeded + failed, succeeded=succeeded, failed=failed)
    job["errors"] = [row.copy() for row in job["results"] if row["status"] == "failed"]
    for key in ("input_tokens", "output_tokens"):
        job[key] = sum(row.get(key, 0) for row in job["results"])


def _saved_report(run_id):
    path = data_dir() / "agent-runs" / f"{run_id}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_job(job_id, *, runner=None):
    """At most one paid attempt per issue, including crash/restart uncertainty."""
    with JOB_LOCK, _worker_lock():
        job = get_job(job_id)
        request = ReplayRequest(**job["request"])
        runner = runner or (run_agent if request.execution_mode == "llm" else run_tools)
        attempts = job.setdefault("attempts", {})
        # Revalidate completed outputs on resume; a missing/corrupted file is not coverage.
        for row in job["results"]:
            if row["status"] == "completed":
                try:
                    _forecast(job, row)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    row.update(status="failed", detail=str(exc)[:500])
        finished = {(row["turbine_id"], row["issue_at"]) for row in job["results"]}
        job.update(status="running", updated_at=now())
        write_json(folder(job_id) / "job.json", job)
        for issue in request.issues():
            for turbine_id in request.turbine_ids:
                stamp = issue.isoformat()
                if (turbine_id, stamp) in finished:
                    continue
                attempt_key = f"{turbine_id}:{stamp}"
                previous = attempts.get(attempt_key)
                entry = {"turbine_id": turbine_id, "issue_at": stamp}
                run_id = previous["run_id"] if previous else uuid4().hex
                entry["run_id"] = run_id
                report = None
                try:
                    with RUN_LOCK:
                        _check_inputs(job, request, turbine_id, issue)
                        recovered = False
                        if previous:
                            try:
                                _, digest = _forecast(job, entry)
                                entry.update(forecast_id=run_id, forecast_sha256=digest)
                                report = _saved_report(run_id)
                                recovered = True
                                entry["recovered_after_interruption"] = True
                            except (OSError, ValueError, TypeError, KeyError):
                                if request.execution_mode == "llm":
                                    raise ValueError(
                                        "Исход платного вызова после прерывания неизвестен; "
                                        "автоматический повтор запрещён. Проверьте журнал запуска "
                                        f"{run_id} перед новым прогоном."
                                    )
                        if not recovered:
                            attempts[attempt_key] = {
                                "run_id": run_id,
                                "status": "in_flight",
                                "started_at": now(),
                            }
                            # This journal must reach disk BEFORE any paid/network call.
                            write_json(folder(job_id) / "job.json", job)
                            report = runner(
                                _agent_request(request, turbine_id, issue), run_id=run_id
                            )
                            if report.get("id") != run_id:
                                raise ValueError("Агент вернул другой идентификатор попытки")
                            _check_inputs(job, request, turbine_id, issue)
                            if report.get("status") != "forecast_saved":
                                failures = [
                                    s["result"]
                                    for s in report.get("steps", [])
                                    if not s["result"].get("ok")
                                ]
                                reason = (
                                    failures[-1]
                                    if failures
                                    else report.get(
                                        "error", report.get("summary", "Прогноз не сохранён")
                                    )
                                )
                                raise ValueError(str(reason)[:500])
                            _, digest = _forecast(job, entry)
                            entry.update(forecast_id=run_id, forecast_sha256=digest)
                        entry["status"] = "completed"
                        attempts[attempt_key]["status"] = "completed"
                except Exception as exc:
                    entry.update(status="failed", detail=str(exc)[:500])
                    if attempt_key in attempts:
                        attempts[attempt_key]["status"] = "failed"
                if report is not None:
                    for key in ("input_tokens", "output_tokens"):
                        entry[key] = report.get(key, 0)
                elif previous and request.execution_mode == "llm":
                    entry["token_usage_unknown"] = True
                job["results"].append(entry)
                finished.add((turbine_id, stamp))
                _progress(job)
                job["updated_at"] = now()
                write_json(folder(job_id) / "job.json", job)
        _progress(job)
        job["summary"] = coverage_summary(job)
        complete = (
            job["progress"]["succeeded"] == job["progress"]["total"]
            and job["summary"]["complete"]
            and not job["errors"]
        )
        job["status"] = (
            "completed" if complete else ("partial" if job["progress"]["succeeded"] else "failed")
        )
        job["updated_at"] = now()
        write_json(folder(job_id) / "job.json", job)
        return job


def forecast_rows(job):
    rows = []
    for result in job["results"]:
        if result["status"] != "completed":
            continue
        payload, _ = _forecast(job, result)
        meta = payload["model_metadata"]
        weather = meta.get("weather_provenance", {})
        snapshot = job["input_snapshots"][str(result["turbine_id"])]
        rated = snapshot.get("rated_power_kw")
        for point in payload["points"]:
            rows.append(
                {
                    "turbine_id": result["turbine_id"],
                    "issue_at": result["issue_at"],
                    "target_time": point["time"],
                    "power": point["power"],
                    "rated_power_kw": rated,
                    "power_kw": point["power"] * rated if rated else None,
                    "energy_kwh": point["power"] * rated if rated else None,
                    "capacity_source_url": snapshot.get("capacity_source_url"),
                    "normalization_assumption": "power = actual_power_kw / rated_power_kw",
                    "lead_hours": int(
                        (
                            datetime.fromisoformat(point["time"])
                            - datetime.fromisoformat(result["issue_at"])
                        ).total_seconds()
                        / 3600
                    ),
                    "forecast_id": result["forecast_id"],
                    "model": payload["model"],
                    "model_sha256": meta.get("model_sha256", ""),
                    "weather_run": weather.get("run", ""),
                    "weather_available_at": weather.get("available_at", ""),
                    "weather_source": weather.get("provider", weather.get("source", "")),
                    "weather_sha256": weather.get("sha256", ""),
                    "units": "fraction_of_turbine_rated_power",
                }
            )
    return sorted(rows, key=lambda row: (row["turbine_id"], row["issue_at"], row["target_time"]))


def coverage_rows(job):
    """Latest forecast strictly before each target, never a future-issued revision."""
    request = ReplayRequest(**job["request"])
    # A replay beginning Jan31 is evaluated Feb1..Feb28, in explicit UTC.
    start = datetime.combine(request.start_date + timedelta(days=1), time(), UTC)
    last_target_day = max(request.end_date, request.start_date + timedelta(days=1))
    end = datetime.combine(last_target_day + timedelta(days=1), time(), UTC)
    candidates = {}
    for row in forecast_rows(job):
        target, issue = map(datetime.fromisoformat, (row["target_time"], row["issue_at"]))
        if start <= target < end and issue < target:
            key = row["turbine_id"], target
            if key not in candidates or issue > datetime.fromisoformat(candidates[key]["issue_at"]):
                candidates[key] = row
    rows = []
    for turbine_id in request.turbine_ids:
        target = start
        while target < end:
            selected = candidates.get((turbine_id, target))
            rows.append(
                selected
                or {
                    "turbine_id": turbine_id,
                    "target_time": target.isoformat(),
                    "power": None,
                    "issue_at": "",
                    "units": "fraction_of_turbine_rated_power",
                    "normalization_assumption": "power = actual_power_kw / rated_power_kw",
                }
            )
            target += timedelta(hours=1)
    return rows


def coverage_summary(job):
    rows = coverage_rows(job)
    covered = sum(row["power"] is not None for row in rows)
    return {
        "coverage_hours": covered,
        "expected_hours": len(rows),
        "missing_hours": len(rows) - covered,
        "turbines": len(job["request"]["turbine_ids"]),
        "coverage_policy": "latest issue before target; UTC days after start_date through end_date",
        "complete": bool(rows) and covered == len(rows),
        "accuracy_evaluated": False,
        "aggregation": "per_turbine_normalized_power; plant energy needs rated capacities",
    }


def export_csv(job, kind):
    if kind == "plant":
        return export_plant(job)
    if kind not in {"forecasts", "coverage"}:
        raise ValueError("Неизвестный формат выгрузки")
    rows = forecast_rows(job) if kind == "forecasts" else coverage_rows(job)
    fields = [
        "turbine_id",
        "issue_at",
        "target_time",
        "power",
        "rated_power_kw",
        "power_kw",
        "energy_kwh",
        "capacity_source_url",
        "normalization_assumption",
        "lead_hours",
        "forecast_id",
        "model",
        "model_sha256",
        "weather_run",
        "weather_available_at",
        "weather_source",
        "weather_sha256",
        "units",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def export_plant(job):
    """Sum only declared capacities for every turbine, never assume equal ratings."""
    ids = job["request"]["turbine_ids"]
    if any(not job["input_snapshots"][str(item)].get("rated_power_kw") for item in ids):
        raise ValueError(
            "Для суммарной выработки подтвердите номинальную мощность всех турбин "
            "и создайте новый прогон; текущая выгрузка содержит нормированную мощность"
        )
    by_time = {}
    for row in coverage_rows(job):
        by_time.setdefault(row["target_time"], []).append(row)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "target_time",
            "power_kw",
            "energy_kwh",
            "turbines",
            "complete",
            "normalization_assumption",
        ],
    )
    writer.writeheader()
    for target, rows in sorted(by_time.items()):
        complete = len(rows) == len(ids) and all(row.get("power_kw") is not None for row in rows)
        power = sum(row["power_kw"] for row in rows) if complete else None
        writer.writerow(
            {
                "target_time": target,
                "power_kw": power,
                "energy_kwh": power,
                "turbines": len(ids),
                "complete": complete,
                "normalization_assumption": "power = actual_power_kw / rated_power_kw",
            }
        )
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Ежедневный исторический прогноз без февральской SCADA"
    )
    parser.add_argument("--turbine-ids", type=int, nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--measurement-timezone")
    parser.add_argument("--timestamp-semantics", choices=["interval_start", "interval_end"])
    parser.add_argument("--start-date", default="2026-01-31")
    parser.add_argument("--end-date", default="2026-02-28")
    parser.add_argument("--issue-hour-utc", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--execution-mode", choices=["tools", "llm"], default="tools")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.resume:
        job = get_job(args.resume)
    else:
        ids = [t["id"] for t in all_metadata("turbines")] if args.all else args.turbine_ids
        job = create_job(
            ReplayRequest(
                **{
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"all", "resume", "turbine_ids"}
                },
                turbine_ids=ids,
            )
        )
    print(json.dumps({"id": job["id"], "status": "starting"}), flush=True)
    job = run_job(job["id"])
    for kind in ("forecasts", "coverage"):
        (folder(job["id"]) / f"{kind}.csv").write_text(export_csv(job, kind))
    print(json.dumps(job, ensure_ascii=False), flush=True)
    if job["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
