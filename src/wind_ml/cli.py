"""CLI обучения, проверки, инференса и минимальной проверки погодного API."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .artifacts import load_artifact, save_artifact, validate_artifact_request
from .evaluation import ValidationFold, validate_walk_forward
from .features import (
    build_forecast_requests,
    build_persistence_requests,
    parse_utc,
    verified_issue_times,
)
from .io import (
    hourly_from_scada_csv,
    read_hourly_csv,
    read_weather_csv,
    weather_from_open_meteo_json,
    write_hourly_csv,
    write_weather_csv,
)
from .models import PowerCurve, fit_boosting, persistence_predict
from .schemas import MeasurementTimeConfig, measurement_time_to_utc


def _config(
    args: argparse.Namespace, manifest: dict[str, object] | None = None
) -> MeasurementTimeConfig:
    saved = (manifest or {}).get("measurement_time", {})
    timezone = args.measurement_timezone or (
        saved.get("source_timezone") if isinstance(saved, dict) else None
    )
    semantics = args.timestamp_semantics or (
        saved.get("timestamp_semantics") if isinstance(saved, dict) else None
    )
    if not timezone or not semantics:
        raise ValueError("Нужно явно указать --measurement-timezone и --timestamp-semantics")
    max_lag = saved.get("max_lag_hours", 2.0) if isinstance(saved, dict) else 2.0
    return MeasurementTimeConfig(str(timezone), str(semantics), max_lag_hours=float(max_lag))


def _bounds(
    measurements: pd.DataFrame, config: MeasurementTimeConfig, end: pd.Timestamp | None
) -> dict[str, str | None]:
    converted = measurement_time_to_utc(measurements, config)
    times = converted["time_utc"]
    return {
        "target_time_min_utc": None if times.empty else times.min().isoformat(),
        "target_time_max_exclusive_utc": end.isoformat()
        if end is not None
        else times.max().isoformat(),
        "target_available_at_max_utc": (
            None if converted.empty else converted["available_at"].max().isoformat()
        ),
    }


def _training_subset(
    measurements: pd.DataFrame, config: MeasurementTimeConfig, end: pd.Timestamp | None
) -> tuple[pd.DataFrame, pd.Timestamp]:
    converted = measurement_time_to_utc(measurements, config)
    if end is None:
        raise ValueError("Укажите явную границу обучения --train-end")
    known = converted.loc[
        (converted["time_utc"] < end) & (converted["available_at"] <= end), "time"
    ]
    return measurements.loc[measurements["time"].isin(known)].copy(), end


def _horizons(value: int) -> range:
    if not 1 <= value <= 48:
        raise ValueError("--horizon должен быть в диапазоне 1..48")
    return range(1, value + 1)


def command_prepare_hourly(args: argparse.Namespace) -> None:
    frame = hourly_from_scada_csv(args.input, config=_config(args))
    write_hourly_csv(frame, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "hours": len(frame),
                "quality": frame["quality"].value_counts().to_dict(),
            },
            ensure_ascii=False,
        )
    )


def command_normalize_weather(args: argparse.Namespace) -> None:
    frame = weather_from_open_meteo_json(
        args.input,
        run=args.run,
        available_at=args.available_at,
        historical_eligibility=args.historical_eligibility,
    )
    write_weather_csv(frame, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(frame),
                "eligibility": args.historical_eligibility,
            },
            ensure_ascii=False,
        )
    )


def command_train(args: argparse.Namespace) -> None:
    config = _config(args)
    _horizons(args.horizon)
    measurements = read_hourly_csv(args.measurements)
    end = parse_utc(args.train_end) if args.train_end else None
    train_measurements, cutoff = _training_subset(measurements, config, end)
    weather_contract: dict[str, object] = {}
    if args.model == "persistence":
        payload: object = {"type": "persistence"}
    elif args.model == "curve":
        payload = PowerCurve.fit(train_measurements, min_samples_per_bin=args.min_samples_per_bin)
    else:
        if not args.weather:
            raise ValueError("--weather обязателен для boosting")
        weather = read_weather_csv(args.weather)
        issues = verified_issue_times(weather, before=cutoff)
        rows = build_forecast_requests(
            measurements, weather, config, issues, _horizons(args.horizon), require_targets=True
        ).examples
        rows = rows.loc[
            (rows["target_time"] < cutoff) & (rows["target_available_at"] <= cutoff)
        ].copy()
        payload = fit_boosting(rows, random_state=args.random_state)
        weather_contract = {
            "time": "UTC",
            "required_historical_eligibility": "verified",
            "availability_rule": "available_at <= issue_time",
            "weather_features": [
                "temperature_2m",
                "wind_speed_10m",
                "wind_speed_100m",
                "wind_direction_100m",
            ],
            "training_rows": len(rows),
        }
    target = save_artifact(
        args.artifacts_dir,
        model_type=args.model,
        payload=payload,
        turbine_id=args.turbine_id,
        time_config=config,
        training_bounds=_bounds(train_measurements, config, cutoff),
        metrics={"note": "Финальное дообучение выполнено отдельно; это не оценка качества."},
        weather_contract=weather_contract,
        trained_horizon_hours=args.horizon,
    )
    print(
        json.dumps(
            {"artifact": str(target), "model": args.model, "train_end": cutoff.isoformat()},
            ensure_ascii=False,
        )
    )


def command_validate(args: argparse.Namespace) -> None:
    config = _config(args)
    measurements = read_hourly_csv(args.measurements)
    weather = read_weather_csv(args.weather) if args.weather else None
    folds = []
    for raw_fold in args.fold:
        try:
            start, end = raw_fold.split("/", maxsplit=1)
        except ValueError as exc:
            raise ValueError("--fold задаётся как START/END в UTC") from exc
        folds.append(ValidationFold.create(start, end))
    report, audit = validate_walk_forward(
        model_type=args.model,
        measurements=measurements,
        time_config=config,
        folds=folds,
        turbine_id=args.turbine_id,
        weather=weather,
        horizons=_horizons(args.horizon),
    )
    if args.audit_output:
        Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_output, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_infer(args: argparse.Namespace) -> None:
    manifest, payload = load_artifact(args.artifacts_dir)
    config = _config(args, manifest)
    measurements = read_hourly_csv(args.measurements)
    issue_at = parse_utc(args.issue_at)
    horizons = _horizons(args.horizon)
    validate_artifact_request(
        manifest,
        turbine_id=getattr(args, "turbine_id", None) or str(manifest.get("turbine_id")),
        time_config=config,
        issue_at=issue_at.isoformat(),
        horizon=args.horizon,
    )
    model_type = manifest["model_type"]
    if model_type == "persistence":
        result = build_persistence_requests(
            measurements, config, [issue_at], horizons, require_targets=False
        )
        audit = result.audit
        audit["prediction"] = np.where(
            audit["reason"].isna(), persistence_predict(audit["last_power"]), np.nan
        )
    else:
        if not args.weather:
            raise ValueError("--weather обязателен для curve и boosting")
        weather = read_weather_csv(args.weather)
        result = build_forecast_requests(
            measurements, weather, config, [issue_at], horizons, require_targets=False
        )
        audit = result.audit
        valid = audit["reason"].isna()
        audit["prediction"] = np.nan
        if model_type == "curve" and valid.any():
            audit.loc[valid, "prediction"] = payload.predict(audit.loc[valid, "wind_speed_100m"])
        elif model_type == "boosting" and valid.any():
            audit.loc[valid, "prediction"] = payload.predict(audit.loc[valid])
        elif model_type not in {"curve", "boosting"}:
            raise ValueError(f"Неизвестный model_type в артефакте: {model_type}")
    output = audit[["issue_time", "target_time", "horizon_hours", "prediction", "reason"]]
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
    print(output.to_json(orient="records", date_format="iso", force_ascii=False))


def command_weather_check(args: argparse.Namespace) -> None:
    """Проверить один небольшой запрос, не объявляя его исторически пригодным."""

    query = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "run": args.run,
        "models": "ecmwf_ifs",
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,wind_direction_100m",
        "wind_speed_unit": "ms",
        "forecast_hours": 72,
        "timezone": "GMT",
    }
    url = "https://single-runs-api.open-meteo.com/v1/forecast?" + urlencode(query)
    with urlopen(url, timeout=30) as response:  # nosec B310 - фиксированный публичный endpoint
        raw = response.read()
    payload = json.loads(raw)
    hourly = payload.get("hourly", {})
    required = [
        "time",
        "temperature_2m",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_100m",
    ]
    lengths = {key: len(hourly.get(key, [])) for key in required}
    if payload.get("timezone") != "GMT" or len(set(lengths.values())) != 1 or lengths["time"] != 72:
        raise ValueError(f"Неполный или несогласованный ответ Single Runs API: {lengths}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    metadata = {
        "run": args.run,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "model": "ecmwf_ifs",
        "request_params": query,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "available_at": None,
        "historical_eligibility": "unverified",
        "hourly_count": lengths["time"],
        "first_time": hourly["time"][0],
        "last_time": hourly["time"][-1],
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def _time_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--measurement-timezone", required=required)
    parser.add_argument(
        "--timestamp-semantics", choices=["interval_start", "interval_end"], required=required
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wind-ml")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-hourly", help="агрегировать исходный 10-минутный CSV")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    _time_arguments(prepare, required=True)
    prepare.set_defaults(func=command_prepare_hourly)

    weather = commands.add_parser("normalize-weather", help="нормализовать ответ Open-Meteo в CSV")
    weather.add_argument("--input", required=True)
    weather.add_argument("--output", required=True)
    weather.add_argument("--run", required=True)
    weather.add_argument("--available-at")
    weather.add_argument("--historical-eligibility", default="unverified")
    weather.set_defaults(func=command_normalize_weather)

    train = commands.add_parser("train", help="отдельно обучить финальный артефакт")
    train.add_argument("--model", choices=["persistence", "curve", "boosting"], required=True)
    train.add_argument("--measurements", required=True)
    train.add_argument("--weather")
    train.add_argument("--turbine-id", required=True)
    train.add_argument("--artifacts-dir", required=True)
    train.add_argument("--train-end", required=True, help="исключающая UTC-граница обучающих целей")
    train.add_argument("--horizon", type=int, default=48)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--min-samples-per-bin", type=int, default=30)
    _time_arguments(train, required=True)
    train.set_defaults(func=command_train)

    validate = commands.add_parser("validate", help="walk-forward с замороженным fit")
    validate.add_argument("--model", choices=["persistence", "curve", "boosting"], required=True)
    validate.add_argument("--measurements", required=True)
    validate.add_argument("--weather")
    validate.add_argument("--turbine-id", required=True)
    validate.add_argument(
        "--fold", action="append", required=True, help="START/END, оба значения в UTC"
    )
    validate.add_argument("--horizon", type=int, default=48)
    validate.add_argument("--audit-output")
    _time_arguments(validate, required=True)
    validate.set_defaults(func=command_validate)

    infer = commands.add_parser("infer", help="выдать 1..48 прогнозов из сохранённого артефакта")
    infer.add_argument("--artifacts-dir", required=True)
    infer.add_argument("--measurements", required=True)
    infer.add_argument("--weather")
    infer.add_argument("--issue-at", required=True, help="UTC-время выпуска")
    infer.add_argument("--horizon", type=int, default=48)
    infer.add_argument("--turbine-id", help="Проверить принадлежность артефакта турбине")
    infer.add_argument("--output")
    _time_arguments(infer, required=False)
    infer.set_defaults(func=command_infer)

    check = commands.add_parser("weather-check", help="малый проверочный запрос Single Runs API")
    check.add_argument("--latitude", type=float, required=True)
    check.add_argument("--longitude", type=float, required=True)
    check.add_argument("--run", required=True)
    check.add_argument("--output", required=True)
    check.set_defaults(func=command_weather_check)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
