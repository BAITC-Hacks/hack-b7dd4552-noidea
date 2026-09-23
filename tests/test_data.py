from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from wind.app import app
from wind.ingest import COLUMNS, import_csv, normalize
from wind.storage import all_metadata, create_turbine, data_dir


def csv_file(tmp_path, times, powers=None):
    frame = pd.DataFrame(
        {
            "ID": range(len(times)),
            "Статистическое время": times,
            "Средняя скорость ветра(m/s)": [5.0] * len(times),
            "Нормализованная активная мощность": powers or [0.4] * len(times),
            "Средняя температура окружающей среды(°C)": [-2.0] * len(times),
        }
    )
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    return path


def test_missing_and_partial_hours_are_not_zero(tmp_path):
    times = pd.date_range("2026-01-01", periods=6, freq="10min").tolist()
    times.append(pd.Timestamp("2026-01-01 02:00"))
    _, hourly, summary = normalize(csv_file(tmp_path, times))
    assert hourly.quality.tolist() == ["complete", "missing", "partial"]
    assert hourly.power.iloc[0] == pytest.approx(0.4)
    assert hourly.power.iloc[1:].isna().all()
    assert hourly.completeness.tolist() == [1, 0, 1 / 6]
    assert summary["missing_slots"] == 6
    assert summary["largest_gaps"][0]["start"] == "2026-01-01T01:00:00"
    assert summary["timezone"] == "unconfirmed"


@pytest.mark.parametrize(
    "times,powers,message",
    [
        (["2026-01-01 00:00:00"] * 2, None, "Повторяющиеся"),
        (["2026-01-01 00:05:00"], None, "вне сетки"),
        (["not a date"], None, "Некорректное время"),
        (["2026-01-01 00:00:00"], [1.1], "вне"),
    ],
)
def test_invalid_inputs_rejected(tmp_path, times, powers, message):
    with pytest.raises(ValueError, match=message):
        normalize(csv_file(tmp_path, times, powers))


def test_invalid_measurement_does_not_make_complete_hour(tmp_path):
    times = pd.date_range("2026-01-01", periods=6, freq="10min")
    _, hourly, summary = normalize(csv_file(tmp_path, times, [0.0, 0.3, None, 0.4, 0.4, 0.4]))
    assert summary["invalid_values"]["power"] == 1
    assert hourly.sample_count.iloc[0] == 6
    assert hourly.valid_count.iloc[0] == 5
    assert pd.isna(hourly.power.iloc[0])


def test_import_keeps_raw_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})
    source = csv_file(tmp_path, pd.date_range("2026-01-01", periods=6, freq="10min"))
    first = import_csv(source, 1)
    second = import_csv(source, 1)
    assert first["sha256"] == second["sha256"]
    assert len(all_metadata("datasets")) == 1
    assert (data_dir() / first["raw_path"]).read_bytes() == source.read_bytes()


def test_api_period_and_csv_export(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    create_turbine({"name": "Test", "latitude": 43.6, "longitude": 78.5})
    source = csv_file(tmp_path, pd.date_range("2026-01-01", periods=12, freq="10min"))
    import_csv(source, 1)
    with TestClient(app) as client:
        data = client.get("/api/turbines/1/series?start=2026-01-01&end=2026-01-01").json()
        assert len(data["points"]) == 2
        assert data["points"][0]["power"] == pytest.approx(0.4)
        assert (
            client.get("/api/turbines/1/series?start=2026-02-01&end=2026-01-01").status_code == 422
        )
        assert (
            client.get("/api/turbines/1/series?start=2025-01-01&end=2026-01-01").status_code == 422
        )
        assert (
            client.get("/api/turbines/99/series?start=2026-01-01&end=2026-01-01").status_code == 404
        )
        export = client.get("/api/turbines/1/export?start=2026-01-01&end=2026-01-01")
        assert export.status_code == 200
        assert "completeness" in export.text
        empty = client.get("/api/turbines/1/series?start=2026-02-01&end=2026-02-02")
        assert empty.json()["points"] == []


def test_demo_samples_have_documented_period():
    root = Path(__file__).resolve().parents[1]
    for i in (1, 2):
        raw, hourly, summary = normalize(root / "samples" / f"demo turbine {i}.csv")
        assert len(raw) == 7 * 144
        assert len(hourly) == 7 * 24
        assert summary["start"] == "2026-01-25T00:00:00"
        assert set(COLUMNS.values()) <= set(raw.columns)
