"""Offline clock-offset comparison; never changes source or production timestamps.

Run with uv run python scripts/audit_timezone.py after caching Open-Meteo archive
responses in data/timezone-audit/{era5,ecmwf_ifs}.json. Weather is diagnostic only,
not an as-issued forecast suitable for backtesting. Compare exact clock-hour
SCADA records (10-minute measurement semantics remain uncertain).
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/timezone-audit"
PAIRS = {
    "wind100": ("wind_speed", "wind_speed_100m"),
    "wind10": ("wind_speed", "wind_speed_10m"),
    "power_wind100": ("power", "wind_speed_100m"),
    "temperature": ("temperature", "temperature_2m"),
}


def download():
    OUT.mkdir(parents=True, exist_ok=True)
    for model in ("era5", "ecmwf_ifs"):
        path = OUT / f"{model}.json"
        if path.exists():
            continue
        params = {
            "latitude": 43.645150, "longitude": 78.535604,
            "start_date": "2023-03-10", "end_date": "2026-02-01",
            "hourly": "wind_speed_10m,wind_speed_100m,temperature_2m",
            "models": model, "timezone": "GMT", "wind_speed_unit": "ms",
        }
        response = httpx.get("https://archive-api.open-meteo.com/v1/archive",
                             params=params, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
        (OUT / f"{model}-request.json").write_text(json.dumps({
            "url": str(response.url), "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(response.content).hexdigest(),
        }, indent=2))


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 100:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def main():
    rows, monthly, sensitivity = [], [], []
    provenance = []
    for model in ("era5", "ecmwf_ifs"):
        path = OUT / f"{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run with --download")
        payload = json.loads(path.read_text())
        assert payload["utc_offset_seconds"] == 0
        assert payload["hourly_units"]["wind_speed_100m"] == "m/s"
        weather = pd.DataFrame(payload["hourly"])
        weather.index = pd.to_datetime(weather.pop("time"))
        provenance.append({"model": model,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "grid": [payload["latitude"],
                                                   payload["longitude"]]})
        for turbine in (1, 2):
            source = next((ROOT / "data").glob(f"{turbine}-*-10min.parquet"))
            raw = pd.read_parquet(source).set_index("time").sort_index()
            hourly = raw.resample("h").mean().where(raw.resample("h").count() == 6)
            interpolated = weather.resample("10min").interpolate()
            # Six intervals stamped 00..50: mean physical midpoint is :30 for
            # interval-start labels, :20 for interval-end labels. Interpolation
            # is a sensitivity check, not additional weather observations.
            for semantics, minutes in (("interval_start", 30), ("interval_end", 20)):
                values = {}
                for shift in (5, 6, 7):
                    target = (hourly.index - pd.Timedelta(hours=shift)
                              + pd.Timedelta(minutes=minutes))
                    other = interpolated.reindex(target).set_axis(hourly.index)
                    values[str(shift)] = correlation(hourly.wind_speed.to_numpy(),
                                                    other.wind_speed_100m.to_numpy())
                sensitivity.append({"model": model, "turbine": turbine,
                                    "semantics": semantics, "r": values})
            measured = raw[raw.index.minute == 0].reindex(
                pd.date_range(raw.index.min().ceil("h"), raw.index.max().floor("h"), freq="h")
            )
            for offset in [*range(-12, 15), "Asia/Almaty"]:
                if isinstance(offset, int):
                    utc = measured.index - pd.Timedelta(hours=offset)
                else:
                    utc = measured.index.tz_localize(
                        "Asia/Almaty", ambiguous="NaT", nonexistent="NaT"
                    ).tz_convert("UTC").tz_localize(None)
                matched = weather.reindex(utc).set_axis(measured.index)
                for label, (local, external) in PAIRS.items():
                    x, y = measured[local], matched[external]
                    for period, select in {
                        "all": np.ones(len(x), dtype=bool),
                        "before_2024_03": x.index < "2024-03-01",
                        "after_2024_03": x.index >= "2024-03-01",
                    }.items():
                        rows.append({
                            "model": model, "turbine": turbine, "offset": str(offset),
                            "signal": label, "period": period,
                            "n": int((x[select].notna() & y[select].notna()).sum()),
                            "r": correlation(x[select].to_numpy(), y[select].to_numpy()),
                            "r_delta3h": correlation(x.diff(3)[select].to_numpy(),
                                                     y.diff(3)[select].to_numpy()),
                        })
                    for month in measured.index.to_period("M").unique():
                        select = measured.index.to_period("M") == month
                        monthly.append({
                            "model": model, "turbine": turbine, "offset": str(offset),
                            "signal": label, "month": str(month),
                            "r": correlation(x[select].to_numpy(), y[select].to_numpy()),
                        })
    report = {"provenance": provenance, "scores": rows, "monthly": monthly,
              "sensitivity": sensitivity}
    (OUT / "scores.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    scores = pd.DataFrame(rows)
    for keys, frame in scores.groupby(["model", "turbine", "signal", "period"]):
        if keys[2] not in ("wind100", "temperature"):
            continue
        best = frame.sort_values("r", ascending=False).head(3)
        print(keys, best[["offset", "r", "r_delta3h", "n"]].round(4).to_dict("records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing public archives")
    if parser.parse_args().download:
        download()
    main()
