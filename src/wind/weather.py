import hashlib
import json
import math
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import BaseModel, Field, field_validator

from wind.storage import all_metadata, data_dir, get_turbine, save_metadata

ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["temperature_2m", "wind_speed_10m", "wind_speed_100m", "wind_direction_100m"]
UNITS = ["°C", "m/s", "m/s", "°"]


class WeatherRequest(BaseModel):
    turbine_id: int = Field(ge=1)
    run: datetime

    @field_validator("run")
    @classmethod
    def validate_run(cls, value: datetime):
        if value.tzinfo is None:
            raise ValueError("Укажите часовой пояс выпуска, например +00:00")
        value = value.astimezone(UTC)
        if value.hour not in (0, 6, 12, 18) or value.minute or value.second or value.microsecond:
            raise ValueError("Выпуски: 00, 06, 12, 18 UTC, без минут и секунд")
        if not datetime(2024, 3, 1, tzinfo=UTC) <= value <= datetime.now(UTC):
            raise ValueError("Выпуск должен быть с марта 2024 года и не в будущем")
        return value


def validate_payload(payload: dict, run: datetime) -> list[dict]:
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Источник вернул время не в UTC")
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    expected = [(run + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(72)]
    if times != expected:
        raise ValueError("Источник не вернул ожидаемые 72 последовательных часа от выпуска")
    for key, unit in zip(VARIABLES, UNITS, strict=True):
        values = hourly.get(key, [])
        if len(values) != 72 or payload.get("hourly_units", {}).get(key) != unit:
            raise ValueError(f"Неверные единицы или длина ряда: {key}")
        if any(
            v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v))
            for v in values
        ):
            raise ValueError(f"Некорректные значения: {key}")
        if all(v is None for v in values):
            raise ValueError(f"Полностью отсутствует переменная {key}")
        if key.startswith("wind_speed") and any(v is not None and v < 0 for v in values):
            raise ValueError("Источник вернул отрицательную скорость ветра")
        if key.startswith("wind_direction") and any(
            v is not None and not 0 <= v <= 360 for v in values
        ):
            raise ValueError("Направление ветра вне диапазона 0–360°")
    return [
        {"time": time + ":00+00:00", **{key: hourly[key][i] for key in VARIABLES}}
        for i, time in enumerate(times)
    ]


def weather_detail(metadata: dict) -> dict:
    payload = json.loads((data_dir() / metadata["raw_path"]).read_bytes())
    return {
        **metadata,
        "points": validate_payload(payload, datetime.fromisoformat(metadata["run"])),
    }


def fetch_weather(request: WeatherRequest) -> dict:
    turbine = get_turbine(request.turbine_id)
    params = {
        "latitude": turbine["latitude"],
        "longitude": turbine["longitude"],
        "run": request.run.strftime("%Y-%m-%dT%H:%M"),
        "models": "ecmwf_ifs",
        "hourly": ",".join(VARIABLES),
        "wind_speed_unit": "ms",
        "forecast_hours": 72,
        "timezone": "GMT",
    }
    key = hashlib.sha256(
        json.dumps({**params, "turbine_id": request.turbine_id}, sort_keys=True).encode()
    ).hexdigest()[:24]
    cached = next((m for m in all_metadata("weather") if m["id"] == key), None)
    if cached:
        return {**weather_detail(cached), "cached": True}
    with httpx.Client(timeout=45, follow_redirects=False) as client:
        response = client.get(ENDPOINT, params=params)
        response.raise_for_status()
    payload = response.json()
    points = validate_payload(payload, request.run)
    root = data_dir() / "weather"
    root.mkdir(exist_ok=True)
    digest = hashlib.sha256(response.content).hexdigest()
    path = root / f"{digest}.json"
    path.write_bytes(response.content)
    metadata = {
        "id": key,
        "turbine_id": request.turbine_id,
        "run": request.run.isoformat(),
        "model": "ecmwf_ifs",
        "source": "Open-Meteo / ECMWF",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "request_url": str(response.url),
        "request_params": params,
        "sha256": digest,
        "raw_path": f"weather/{digest}.json",
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "units": payload["hourly_units"],
        "hours": len(points),
        "missing_values": {key: sum(p[key] is None for p in points) for key in VARIABLES},
        # Run initialisation time is NOT publication/availability time.
        "available_at": None,
        "historical_eligibility": "unverified",
    }
    save_metadata("weather", key, metadata)
    return {**metadata, "points": points, "cached": False}
