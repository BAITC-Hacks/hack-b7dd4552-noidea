import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from wind.agent import PROJECT as AGENT_PROJECT
from wind.agent import AgentRequest, ToolSession, run_agent
from wind.discovery import router as discovery_router
from wind.ingest import import_csv
from wind.ml import MODEL_TYPE, model_catalog
from wind.sources import router as sources_router
from wind.storage import (
    all_metadata,
    create_turbine,
    data_dir,
    delete_turbine,
    deleted_turbines,
    get_turbine,
    restore_turbine,
)
from wind.weather import WeatherRequest, fetch_weather, weather_detail

PROJECT = Path(__file__).resolve().parents[2]


app = FastAPI(title="ВЭС · Data Explorer", version="0.2.0")


app.include_router(sources_router)
app.include_router(discovery_router)


class TurbineInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if not value.strip():
            raise ValueError("Укажите название")
        return value.strip()


@app.post("/api/turbines", status_code=201)
def add_turbine(body: TurbineInput):
    return create_turbine(body.model_dump())


@app.delete("/api/turbines/{turbine_id}")
def archive_turbine(turbine_id: int):
    try:
        delete_turbine(turbine_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"id": turbine_id, "deleted": True}


@app.get("/api/turbines/deleted")
def turbine_trash():
    return {"items": turbine_metadata(deleted_turbines())}


@app.post("/api/turbines/{turbine_id}/restore")
def unarchive_turbine(turbine_id: int):
    try:
        turbine = restore_turbine(turbine_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return turbine_metadata([turbine])[0]


@app.post("/api/turbines/{turbine_id}/import")
def upload_csv(turbine_id: int, file: UploadFile):
    try:
        get_turbine(turbine_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(422, "Выберите файл CSV в UTF-8")
    try:
        with TemporaryDirectory(prefix="wind-import-") as directory:
            path = Path(directory) / Path(file.filename.replace("\\", "/")).name
            size = 0
            with path.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 25 * 1024 * 1024:
                        raise HTTPException(413, "Максимальный размер файла — 25 МБ")
                    target.write(chunk)
            return import_csv(path, turbine_id, "user")
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(422, f"CSV не импортирован: {exc}") from exc
    finally:
        file.file.close()


def dataset(turbine_id: int):
    try:
        get_turbine(turbine_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    result = next((d for d in all_metadata("datasets") if d["id"] == turbine_id), None)
    if result is None:
        raise HTTPException(404, "Данные турбины ещё не импортированы")
    return result


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/turbines")
def turbines():
    return {"items": turbine_metadata(all_metadata("turbines"))}


def turbine_metadata(items: list[dict]) -> list[dict]:
    datasets = {d["id"]: d for d in all_metadata("datasets")}
    return [{**datasets.get(t["id"], {}), **t, "has_data": t["id"] in datasets} for t in items]


def select_hours(turbine_id: int, start: date, end: date) -> pd.DataFrame:
    if end < start or (end - start).days > 92:
        raise HTTPException(422, "Выберите период от 1 до 93 дней")
    meta = dataset(turbine_id)
    return pd.read_parquet(
        data_dir() / meta["hourly_path"],
        filters=[
            ("time", ">=", pd.Timestamp(start)),
            ("time", "<", pd.Timestamp(end + timedelta(days=1))),
        ],
    )


@app.get("/api/turbines/{turbine_id}/series")
def series(turbine_id: int, start: date, end: date):
    frame = select_hours(turbine_id, start, end)
    return {
        "timezone": "unconfirmed",
        "resolution": "hourly",
        "points": json.loads(frame.to_json(orient="records", date_format="iso")),
        "selected_hours": len(frame),
        "complete_hours": int(frame.quality.eq("complete").sum()),
    }


@app.get("/api/turbines/{turbine_id}/export")
def export(turbine_id: int, start: date, end: date):
    frame = select_hours(turbine_id, start, end)
    return Response(
        frame.to_csv(index=False),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="turbine-{turbine_id}-hourly.csv"'},
    )


@app.get("/api/weather")
def weather_list(turbine_id: int = Query(ge=1)):
    return {
        "items": sorted(
            [m for m in all_metadata("weather") if m["turbine_id"] == turbine_id],
            key=lambda x: x["run"],
            reverse=True,
        )
    }


def weather_metadata(key: str):
    if not re.fullmatch(r"[a-f0-9]{24}", key):
        raise HTTPException(404, "Выпуск не найден")
    meta = next((m for m in all_metadata("weather") if m["id"] == key), None)
    if meta is None:
        raise HTTPException(404, "Выпуск не найден")
    return meta


@app.get("/api/weather/{key}")
def weather_get(key: str):
    return weather_detail(weather_metadata(key))


@app.get("/api/weather/{key}/raw")
def weather_raw(key: str):
    meta = weather_metadata(key)
    return FileResponse(data_dir() / meta["raw_path"], filename=f"weather-{key}.json")


@app.post("/api/weather")
def weather_fetch(body: WeatherRequest):
    try:
        get_turbine(body.turbine_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        return fetch_weather(body)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            502,
            f"Источник погоды вернул HTTP {exc.response.status_code}. "
            "Попробуйте другой выпуск или повторите позже.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, "Источник погоды недоступен. Проверьте сеть и повторите.") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(502, f"Ответ погоды не прошёл проверку: {exc}") from exc


_agent_lock = Lock()


@app.get("/api/ml/models")
def ml_models():
    return {"items": model_catalog()}


@app.get("/api/agent/status")
def agent_status():
    load_dotenv(AGENT_PROJECT / ".env", override=False)
    models = model_catalog()
    return {
        "configured": bool(os.environ.get("OPENAI_API_KEY")),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        "forecast_model": MODEL_TYPE
        if any(m.get("promoted") and m.get("source_matches") for m in models)
        else "persistence_baseline",
        "forecast_policy": "validated_ml_or_persistence",
    }


@app.post("/api/agent/runs")
def agent_run(body: AgentRequest):
    try:
        get_turbine(body.turbine_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not _agent_lock.acquire(blocking=False):
        raise HTTPException(409, "Агент уже выполняет расчёт. Дождитесь завершения")
    try:
        return run_agent(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        _agent_lock.release()


@app.post("/api/agent/preflight")
def agent_preflight(body: AgentRequest):
    """Check local forecast inputs without model calls or forecast generation."""
    try:
        get_turbine(body.turbine_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        session = ToolSession(body)
        checks = []
        for name in ("inspect_data", "prepare_features"):
            result = session.execute(name, {})
            checks.append({"tool": name, "result": result})
            if not result.get("ok"):
                return {"ok": False, "checks": checks}
        return {"ok": True, "checks": checks}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(422, "Не удалось прочитать импортированные измерения") from exc


@app.get("/api/agent/runs")
def agent_history(turbine_id: int = Query(ge=1)):
    folder = data_dir() / "agent-runs"
    items = []
    if folder.exists():
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            report = json.loads(path.read_text())
            if report["request"]["turbine_id"] == turbine_id:
                items.append(report)
            if len(items) == 20:
                break
    return {"items": items}


@app.get("/api/forecasts/{forecast_id}")
def get_forecast(forecast_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", forecast_id):
        raise HTTPException(404, "Прогноз не найден")
    path = data_dir() / "forecasts" / f"{forecast_id}.json"
    if not path.exists():
        raise HTTPException(404, "Прогноз не найден")
    return FileResponse(path, filename=f"forecast-{forecast_id}.json")


def mount_frontend(application: FastAPI, directory: Path):
    @application.get("/sources", include_in_schema=False)
    @application.get("/turbines", include_in_schema=False)
    @application.get("/turbines/new", include_in_schema=False)
    @application.get("/forecast", include_in_schema=False)
    @application.get("/weather", include_in_schema=False)
    def frontend_page():
        return FileResponse(directory / "index.html")

    @application.get("/sources/", include_in_schema=False)
    @application.get("/turbines/", include_in_schema=False)
    @application.get("/turbines/new/", include_in_schema=False)
    @application.get("/forecast/", include_in_schema=False)
    @application.get("/weather/", include_in_schema=False)
    def canonical_frontend_page(request: Request):
        target = request.url.path.rstrip("/")
        if request.url.query:
            target += "?" + request.url.query
        return RedirectResponse(target, status_code=308)

    application.mount("/", StaticFiles(directory=directory, html=True), name="frontend")


dist = Path(os.environ.get("FRONTEND_DIST", str(PROJECT / "frontend/dist")))
if dist.exists():
    mount_frontend(app, dist)
