from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_ml.schemas import MeasurementTimeConfig


@pytest.fixture
def utc_config() -> MeasurementTimeConfig:
    return MeasurementTimeConfig("UTC", "interval_start")


@pytest.fixture
def hourly_measurements() -> callable:
    def build(start: str = "2025-01-01", periods: int = 200) -> pd.DataFrame:
        times = pd.date_range(start, periods=periods, freq="h")
        wind = 6 + 3 * np.sin(np.arange(periods) / 9)
        power = np.clip((wind - 2) / 10, 0, 1)
        return pd.DataFrame(
            {
                "time": times,
                "wind_speed": wind,
                "power": power,
                "temperature": 5 + np.cos(np.arange(periods) / 6),
                "sample_count": 6,
                "valid_count": 6,
                "completeness": 1.0,
                "quality": "complete",
            }
        )

    return build


@pytest.fixture
def weather_archive() -> callable:
    def build(start: str = "2025-01-01", periods: int = 150, horizon: int = 48) -> pd.DataFrame:
        """По одному проверенному выпуску каждый час, достаточному для fixtures."""

        issues = pd.date_range(start, periods=periods, freq="h", tz="UTC")
        rows = []
        for issue in issues:
            for lead in range(1, horizon + 1):
                rows.append(
                    {
                        "time": issue + pd.Timedelta(lead, unit="h"),
                        "run": issue,
                        "available_at": issue,
                        "historical_eligibility": "verified",
                        "temperature_2m": 5.0,
                        "wind_speed_10m": 4.0,
                        "wind_speed_100m": 7.0,
                        "wind_direction_100m": 90.0,
                    }
                )
        return pd.DataFrame(rows)

    return build
