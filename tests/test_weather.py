from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from wind.app import app
from wind.storage import all_metadata, create_turbine, data_dir
from wind.weather import UNITS, VARIABLES, WeatherRequest, fetch_weather, validate_payload

RUN = datetime(2026, 1, 31, tzinfo=UTC)


def payload():
    return {
        "utc_offset_seconds": 0,
        "latitude": 43.62,
        "longitude": 78.48,
        "hourly_units": dict(zip(VARIABLES, UNITS)),
        "hourly": {
            "time": [(RUN + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(72)],
            **{key: [5.0] * 72 for key in VARIABLES},
        },
    }


@pytest.mark.parametrize(
    "run",
    ["2026-01-31T00:00:00", "2026-01-31T01:00:00Z", "2023-01-31T00:00:00Z", "2099-01-31T00:00:00Z"],
)
def test_reject_ambiguous_or_unsupported_run(run):
    with pytest.raises(ValidationError):
        WeatherRequest(turbine_id=1, run=run)


def test_weather_time_and_units_are_validated():
    data = payload()
    data["hourly"]["time"][0] = "2026-01-30T00:00"
    with pytest.raises(ValueError, match="72 последовательных"):
        validate_payload(data, RUN)
    data = payload()
    data["hourly_units"]["wind_speed_100m"] = "km/h"
    with pytest.raises(ValueError, match="единицы"):
        validate_payload(data, RUN)


def test_nulls_preserved_and_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})
    data = payload()
    data["hourly"]["wind_speed_100m"][4] = None
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "https://example.test"))
    fake = Mock()
    fake.__enter__ = Mock(return_value=fake)
    fake.__exit__ = Mock(return_value=False)
    fake.get.return_value = response
    monkeypatch.setattr("wind.weather.httpx.Client", lambda **kwargs: fake)
    request = WeatherRequest(turbine_id=1, run=RUN)
    result = fetch_weather(request)
    assert result["available_at"] is None
    assert result["historical_eligibility"] == "unverified"
    assert result["missing_values"]["wind_speed_100m"] == 1
    assert result["points"][4]["wind_speed_100m"] is None
    assert (data_dir() / result["raw_path"]).exists()
    assert fetch_weather(request)["cached"] is True
    assert fake.get.call_count == 1
    assert len(all_metadata("weather")) == 1


def test_api_provider_error_does_not_persist_bad_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})

    def fail(_):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("wind.app.fetch_weather", fail)
    with TestClient(app) as client:
        response = client.post("/api/weather", json={"turbine_id": 1, "run": RUN.isoformat()})
        assert response.status_code == 502
        assert client.get("/api/weather?turbine_id=1").json()["items"] == []
