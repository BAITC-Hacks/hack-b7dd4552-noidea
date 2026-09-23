from __future__ import annotations

import json

from wind_ml.io import weather_from_open_meteo_json


def test_open_meteo_normalizer_keeps_unknown_publication_unverified(tmp_path) -> None:
    source = tmp_path / "weather.json"
    source.write_text(
        json.dumps(
            {
                "timezone": "GMT",
                "hourly": {
                    "time": ["2026-01-31T00:00", "2026-01-31T01:00"],
                    "temperature_2m": [1, 2],
                    "wind_speed_10m": [3, 4],
                    "wind_speed_100m": [5, 6],
                    "wind_direction_100m": [7, 8],
                },
            }
        ),
        encoding="utf-8",
    )

    result = weather_from_open_meteo_json(
        source,
        run="2026-01-31T00:00:00Z",
        available_at=None,
        historical_eligibility="unverified",
    )

    assert len(result) == 2
    assert result["available_at"].isna().all()
    assert result["historical_eligibility"].eq("unverified").all()
