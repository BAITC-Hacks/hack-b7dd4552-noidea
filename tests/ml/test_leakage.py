from __future__ import annotations

import pandas as pd

from wind_ml.features import build_forecast_requests, build_persistence_requests
from wind_ml.schemas import MeasurementTimeConfig, measurement_time_to_utc


def test_persistence_never_reads_measurement_from_future(hourly_measurements, utc_config) -> None:
    measurements = hourly_measurements(periods=5)
    measurements["power"] = [0.1, 0.2, 0.9, 0.4, 0.5]
    issue = pd.Timestamp("2025-01-01T02:00:00Z")

    result = build_persistence_requests(measurements, utc_config, [issue], [1])

    # interval_start: значение за 01:00 стало известно ровно в 02:00;
    # значение с меткой 02:00 (0.9) доступно только в 03:00.
    assert result.audit.iloc[0]["last_power"] == 0.2
    assert result.audit.iloc[0]["prediction"] == 0.2
    assert result.audit.iloc[0]["target_power"] == 0.4


def test_weather_must_be_verified_and_available_at_issue(hourly_measurements, utc_config) -> None:
    measurements = hourly_measurements(periods=6)
    issue = pd.Timestamp("2025-01-01T02:00:00Z")
    weather = pd.DataFrame(
        [
            {
                "time": "2025-01-01T03:00:00Z",
                "run": "2025-01-01T00:00:00Z",
                "available_at": "2025-01-01T01:00:00Z",
                "historical_eligibility": "verified",
                "temperature_2m": 1,
                "wind_speed_10m": 2,
                "wind_speed_100m": 3,
                "wind_direction_100m": 45,
            },
            {
                "time": "2025-01-01T03:00:00Z",
                "run": "2025-01-01T02:00:00Z",
                "available_at": "2025-01-01T03:00:00Z",
                "historical_eligibility": "verified",
                "temperature_2m": 99,
                "wind_speed_10m": 99,
                "wind_speed_100m": 99,
                "wind_direction_100m": 99,
            },
        ]
    )

    result = build_forecast_requests(measurements, weather, utc_config, [issue], [1])

    assert result.audit.iloc[0]["wind_speed_100m"] == 3
    assert result.audit.iloc[0]["weather_available_at"] == pd.Timestamp("2025-01-01T01:00:00Z")


def test_unverified_weather_is_reported_not_silently_used(hourly_measurements, utc_config) -> None:
    issue = pd.Timestamp("2025-01-01T02:00:00Z")
    weather = pd.DataFrame(
        {
            "time": ["2025-01-01T03:00:00Z"],
            "run": ["2025-01-01T00:00:00Z"],
            "available_at": [None],
            "historical_eligibility": ["unverified"],
            "temperature_2m": [1],
            "wind_speed_10m": [2],
            "wind_speed_100m": [3],
            "wind_direction_100m": [45],
        }
    )

    result = build_forecast_requests(
        hourly_measurements(periods=5), weather, utc_config, [issue], [1]
    )

    assert result.audit.iloc[0]["reason"] == "weather_unverified"


def test_unresolved_local_clock_hour_is_not_used_as_lag(hourly_measurements) -> None:
    measurements = hourly_measurements(start="2024-02-29 21:00", periods=5)
    config = MeasurementTimeConfig("Asia/Almaty", "interval_start")
    converted = measurement_time_to_utc(measurements, config)
    assert "unresolved_local_time" in set(converted["time_status"])

    result = build_persistence_requests(
        measurements, config, [pd.Timestamp("2024-02-29T18:00:00Z")], [1]
    )

    assert result.audit.iloc[0]["last_power"] != measurements.iloc[-1]["power"]
