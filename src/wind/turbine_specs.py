"""Public OSM equipment candidates; capacities are applied only by user confirmation."""

import json
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from wind.storage import TurbineNotFoundError, _read_turbine, connect, get_turbine

router = APIRouter(tags=["turbine specifications"])
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_API = "https://api.openstreetmap.org/api/0.6/node/"
RADIUS_M = 100
_CACHE = {}
# Public candidates for the two coordinates in the assignment; never auto-applied.
KNOWN_NODES = (
    (9690012914, 43.6452141, 78.5355410),
    (9690012913, 43.6432764, 78.5387743),
)
CURRENT_FIELDS = (
    "rated_power_kw",
    "manufacturer",
    "turbine_model",
    "capacity_source_url",
    "capacity_status",
    "capacity_confirmed_at",
    "osm_node_id",
    "identity_distance_m",
)


class ConfirmSpecifications(BaseModel):
    model_config = ConfigDict(extra="forbid")
    osm_node_id: int = Field(gt=0, le=10**15, strict=True)


def distance_m(latitude, longitude, other_latitude, other_longitude):
    lat1, lat2 = map(math.radians, (latitude, other_latitude))
    dlat, dlon = lat2 - lat1, math.radians(other_longitude - longitude)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(min(1, math.sqrt(max(0, value))))


def parse_capacity(value):
    """Only explicit electrical kW/MW tags, never a site-wide or guessed default."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(kW|MW)\s*", value)
    if not match:
        return None
    capacity = float(match[1].replace(",", ".")) * (1000 if match[2] == "MW" else 1)
    return capacity if 0 < capacity <= 100_000 else None


def _candidate(node, turbine):
    if not isinstance(node, dict) or not isinstance(node.get("tags"), dict):
        return None
    tags = node.get("tags", {})
    if (
        node.get("type") != "node"
        or tags.get("power") != "generator"
        or not (
            tags.get("generator:source") == "wind" or tags.get("generator:method") == "wind_turbine"
        )
    ):
        return None
    try:
        node_id = int(node["id"])
        latitude, longitude = float(node["lat"]), float(node["lon"])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180 or node_id <= 0:
            return None
        distance = distance_m(turbine["latitude"], turbine["longitude"], latitude, longitude)
    except (ValueError, TypeError, KeyError):
        return None
    if distance > RADIUS_M:
        return None
    return {
        "osm_node_id": node_id,
        "latitude": latitude,
        "longitude": longitude,
        "distance_m": round(distance, 2),
        "rated_power_kw": parse_capacity(tags.get("generator:output:electricity")),
        "manufacturer": str(tags["manufacturer"])[:150] if tags.get("manufacturer") else None,
        "model": str(tags["model"])[:150] if tags.get("model") else None,
        "source_url": f"https://www.openstreetmap.org/node/{node_id}",
        "source": "OpenStreetMap",
        "identity_confirmed": False,
    }


def _response(url, *, params=None):
    with httpx.Client(timeout=15, follow_redirects=False) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError("Слишком большой ответ источника")
        return response


def _node(node_id):
    # Node IDs are integers; user-supplied hosts and redirects are never fetched.
    response = _response(OSM_API + str(node_id))
    root = ET.fromstring(response.content)
    matches = [item for item in root.findall("node") if item.get("id") == str(node_id)]
    if len(matches) != 1:
        raise ValueError("OSM не вернул выбранный объект")
    item = matches[0]
    return {
        "type": "node",
        "id": node_id,
        "lat": item.get("lat"),
        "lon": item.get("lon"),
        "tags": {tag.get("k"): tag.get("v") for tag in item.findall("tag")},
    }


def _nearby(turbine):
    key = turbine["latitude"], turbine["longitude"]
    cached = _CACHE.get(key)
    if cached and time.monotonic() - cached[0] < 300:
        return cached[1], cached[2]
    query = (
        '[out:json][timeout:10];node["power"="generator"]["generator:source"="wind"]'
        f"(around:{RADIUS_M},{float(key[0])},{float(key[1])});out body 100;"
    )
    warning = None
    try:
        payload = _response(OVERPASS_URL, params={"data": query}).json()
        if (
            not isinstance(payload, dict)
            or payload.get("remark")
            or not isinstance(payload.get("elements"), list)
        ):
            raise ValueError("Неполный ответ поиска OpenStreetMap")
        nodes = payload["elements"]
    except (httpx.HTTPError, ValueError, TypeError):
        ids = [item[0] for item in KNOWN_NODES if distance_m(*key, item[1], item[2]) <= RADIUS_M]
        if not ids:
            raise ValueError("Поиск OpenStreetMap временно недоступен") from None
        nodes = [_node(node_id) for node_id in ids]
        warning = "Поиск недоступен; показаны повторно проверенные объекты из координат задания."
    candidates = {}
    for node in nodes[:100]:
        candidate = _candidate(node, turbine)
        if candidate:
            candidates[candidate["osm_node_id"]] = candidate
    result = sorted(candidates.values(), key=lambda item: item["distance_m"])
    if len(_CACHE) >= 128:
        _CACHE.clear()
    _CACHE[key] = time.monotonic(), result, warning
    return result, warning


def _current(turbine):
    return {key: turbine.get(key) for key in CURRENT_FIELDS}


@router.get("/api/turbines/{turbine_id}/specifications")
def get_specifications(turbine_id: int):
    try:
        turbine = get_turbine(turbine_id)
        candidates, warning = _nearby(turbine)
        # A deleted turbine must not reappear after a slow lookup.
        current = get_turbine(turbine_id)
        if (current["latitude"], current["longitude"]) != (
            turbine["latitude"],
            turbine["longitude"],
        ):
            raise ValueError("Координаты изменились; повторите поиск")
    except TurbineNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except (httpx.HTTPError, ValueError, ET.ParseError) as error:
        raise HTTPException(502, str(error)) from error
    return {
        "current": _current(current),
        "candidates": candidates,
        "warning": warning,
        "confirmation_required": True,
        "note": "OSM — открытая карта, не паспорт оборудования. Подтвердите совпадение турбины.",
    }


@router.post("/api/turbines/{turbine_id}/specifications")
def confirm_specifications(turbine_id: int, body: ConfirmSpecifications):
    try:
        get_turbine(turbine_id)
        node = _node(body.osm_node_id)  # Bypass search cache on every explicit confirmation.
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            turbine = _read_turbine(db, turbine_id)
            if turbine.get("deleted_at"):
                raise TurbineNotFoundError("Турбина удалена; характеристики не сохранены")
            candidate = _candidate(node, turbine)
            if not candidate or candidate["rated_power_kw"] is None:
                raise ValueError("Нужен ветрогенератор в пределах 100 м с явной мощностью кВт/МВт")
            turbine.update(
                rated_power_kw=candidate["rated_power_kw"],
                manufacturer=candidate["manufacturer"],
                turbine_model=candidate["model"],
                capacity_source_url=candidate["source_url"],
                capacity_status="user_confirmed_osm",
                capacity_confirmed_at=datetime.now(UTC).isoformat(),
                osm_node_id=body.osm_node_id,
                identity_distance_m=candidate["distance_m"],
            )
            db.execute(
                "UPDATE turbines SET metadata=? WHERE id=?", (json.dumps(turbine), turbine_id)
            )
    except TurbineNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except (ValueError, ET.ParseError) as error:
        raise HTTPException(400, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(502, "OpenStreetMap временно недоступен") from error
    return {"current": _current(turbine), "turbine": turbine}
