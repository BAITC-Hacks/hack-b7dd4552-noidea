from __future__ import annotations

from pathlib import Path

from wind_ml.io import hourly_from_scada_csv


def test_missing_hours_stay_null_and_are_not_interpolated(tmp_path: Path) -> None:
    source = tmp_path / "scada.csv"
    source.write_text(
        "Статистическое время,Средняя скорость ветра(m/s),Нормализованная активная мощность,"
        "Средняя температура окружающей среды(°C)\n"
        "2025-01-01 00:00:00,5,0.3,2\n"
        "2025-01-01 00:10:00,5,0.3,2\n"
        "2025-01-01 00:20:00,5,0.3,2\n"
        "2025-01-01 00:30:00,5,0.3,2\n"
        "2025-01-01 00:40:00,5,0.3,2\n"
        "2025-01-01 00:50:00,5,0.3,2\n"
        "2025-01-01 02:00:00,8,0.7,3\n",
        encoding="utf-8",
    )

    result = hourly_from_scada_csv(source)

    assert result["quality"].tolist() == ["complete", "missing", "partial"]
    gap = result.iloc[1]
    assert gap["sample_count"] == 0
    assert gap[["wind_speed", "power", "temperature"]].isna().all()
    assert result.iloc[2][["wind_speed", "power", "temperature"]].isna().all()
