"""Opt-in, durable update events. Importing data never silently enables paid LLMs."""

import fcntl
import hashlib
import json
from datetime import UTC, datetime
from threading import Lock
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wind.agent import AgentRequest, run_agent, run_tools
from wind.replay import RUN_LOCK, now, write_json
from wind.storage import all_metadata, data_dir, get_turbine
from wind_ml.telemetry import hourly_telemetry

EVENT_LOCK = Lock()


class AutomationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    turbine_ids: list[int] = Field(default_factory=list, max_length=32)
    measurement_timezone: str | None = None
    timestamp_semantics: Literal["interval_start", "interval_end"] | None = None
    horizon: int = Field(default=48, ge=24, le=48)
    execution_mode: Literal["tools", "llm"] = "tools"

    @model_validator(mode="after")
    def validate_enabled(self):
        if any(value < 1 for value in self.turbine_ids) or len(set(self.turbine_ids)) != len(
            self.turbine_ids
        ):
            raise ValueError("ID турбин должны быть положительными и уникальными")
        if self.enabled:
            if (
                not self.turbine_ids
                or not self.measurement_timezone
                or not self.timestamp_semantics
            ):
                raise ValueError("Для автоматизации нужны турбины и явные настройки времени")
            AgentRequest(
                turbine_id=self.turbine_ids[0],
                issue_at=datetime(2026, 1, 1, tzinfo=UTC),
                measurement_timezone=self.measurement_timezone,
                timestamp_semantics=self.timestamp_semantics,
            )
        return self


def settings_path():
    return data_dir() / "automation" / "settings.json"


def get_settings():
    if not settings_path().exists():
        return AutomationSettings()
    return AutomationSettings(**json.loads(settings_path().read_text()))


def set_settings(settings: AutomationSettings):
    for turbine_id in settings.turbine_ids:
        get_turbine(turbine_id)
    write_json(settings_path(), settings.model_dump(mode="json"))
    return settings


def list_events():
    paths = sorted(
        (data_dir() / "automation" / "events").glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [json.loads(path.read_text()) for path in paths[:100]]


def issue_for_update(turbine_id, settings):
    """Recompute the latest existing issue, retaining the historical clock."""
    latest = []
    for path in (data_dir() / "forecasts").glob("*.json"):
        value = json.loads(path.read_text())
        request = value.get("request", {})
        if request.get("turbine_id") == turbine_id:
            latest.append(pd.Timestamp(request["issue_at"]))
    if latest:
        return max(latest).to_pydatetime()
    dataset = next((d for d in all_metadata("datasets") if d["id"] == turbine_id), None)
    if not dataset:
        raise ValueError("Нет измерений для выбора даты первого прогноза")
    path = dataset["hourly_path"].replace("-hourly.parquet", "-10min.parquet")
    hourly = hourly_telemetry(
        pd.read_parquet(data_dir() / path),
        settings.measurement_timezone,
        settings.timestamp_semantics,
    )
    available = hourly.loc[hourly.quality.eq("complete"), "available_at"].dropna()
    current = pd.Timestamp.now(tz="UTC").floor("h")
    available = available[available.le(current)]
    if available.empty:
        raise ValueError("Нет полного уже доступного часа измерений")
    return available.max().to_pydatetime()


def enqueue_event(
    event: Literal["data_updated", "weather_updated"],
    turbine_id: int,
    revision: str,
    *,
    issue_at: datetime | None = None,
):
    settings = get_settings()
    if not settings.enabled or turbine_id not in settings.turbine_ids:
        return None
    # The settings revision is part of identity; resaving the same values isn't a new event.
    identity = {
        "event": event,
        "turbine_id": turbine_id,
        "revision": revision,
        "settings": settings.model_dump(mode="json"),
        "requested_issue_at": issue_at.isoformat() if issue_at else None,
    }
    event_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = data_dir() / "automation" / "events" / f"{event_id}.json"
    with EVENT_LOCK:
        if path.exists():
            return None
        item = {
            "id": event_id,
            **identity,
            "status": "queued",
            "created_at": now(),
            "updated_at": now(),
        }
        try:
            item["issue_at"] = (issue_at or issue_for_update(turbine_id, settings)).isoformat()
        except (ValueError, OSError) as exc:
            item.update({"status": "failed", "error": str(exc)[:300]})
        write_json(path, item)
    return event_id if item["status"] == "queued" else None


def run_event(event_id, *, runner=None):
    if len(event_id) != 64 or any(char not in "0123456789abcdef" for char in event_id):
        raise ValueError("Некорректное событие")
    path = data_dir() / "automation" / "events" / f"{event_id}.json"
    with EVENT_LOCK:
        item = json.loads(path.read_text())
        if item["status"] != "queued":
            return item
        settings = AutomationSettings(**item["settings"])
        current = get_settings()
        if not current.enabled or current.model_dump() != settings.model_dump():
            item.update({"status": "cancelled", "error": "Настройки автоматизации изменены"})
            write_json(path, item)
            return item
        item.update({"status": "running", "updated_at": now()})
        write_json(path, item)
    try:
        request = AgentRequest(
            turbine_id=item["turbine_id"],
            issue_at=item["issue_at"],
            measurement_timezone=settings.measurement_timezone,
            timestamp_semantics=settings.timestamp_semantics,
            horizon=settings.horizon,
            event=item["event"],
            forecast_mode="auto",
        )
        runner = runner or (run_agent if settings.execution_mode == "llm" else run_tools)
        with RUN_LOCK:
            get_turbine(item["turbine_id"])
            current = get_settings()
            if not current.enabled or current.model_dump() != settings.model_dump():
                item.update(
                    {"status": "cancelled", "error": "Автоматизация отключена или изменена"}
                )
                item["updated_at"] = now()
                write_json(path, item)
                return item
            if item["event"] == "data_updated":
                dataset = next(
                    (d for d in all_metadata("datasets") if d["id"] == item["turbine_id"]), {}
                )
                if dataset.get("sha256") != item["revision"]:
                    item.update(
                        {"status": "superseded", "error": "Есть более новая версия измерений"}
                    )
                    item["updated_at"] = now()
                    write_json(path, item)
                    return item
            report = runner(request)
        item.update(
            {
                "status": "completed" if report["status"] == "forecast_saved" else "failed",
                "report": report,
            }
        )
    except Exception as exc:
        item.update({"status": "failed", "error": str(exc)[:300]})
    item["updated_at"] = now()
    write_json(path, item)
    return item


def recover_interrupted():
    """Do not automatically repeat a possibly-paid request after a process crash."""
    for path in (data_dir() / "automation" / "events").glob("*.json"):
        item = json.loads(path.read_text())
        if item["status"] in {"running", "queued"}:
            item.update(
                {
                    "status": "failed",
                    "updated_at": now(),
                    "error": "Сервер перезапущен; повтор платного запроса не выполняется",
                }
            )
            write_json(path, item)
    replay_dir = data_dir() / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    with (replay_dir / ".worker.lock").open("a") as handle:
        try:
            # A separate CLI process may still own the replay worker at API startup.
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            for path in replay_dir.glob("*/job.json"):
                item = json.loads(path.read_text())
                if item["status"] in {"running", "queued"}:
                    item.update(
                        {
                            "status": "partial" if item["progress"]["succeeded"] else "failed",
                            "updated_at": now(),
                            "interrupted": True,
                        }
                    )
                    item["errors"].append(
                        {
                            "detail": "Прогон прерван перезапуском. Результаты доступны; "
                            "продолжение через CLI --resume."
                        }
                    )
                    write_json(path, item)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
