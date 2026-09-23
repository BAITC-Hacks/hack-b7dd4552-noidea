"""Local weather model registry, training CLI and fail-closed inference."""

import argparse
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from wind.storage import all_metadata, data_dir, get_turbine
from wind_ml.nwp import FEATURE_COLUMNS, MODEL_TYPE, predict_nwp_model, train_nwp

CONTRACT_KEYS = (
    "provider",
    "model",
    "availability_policy",
    "policy_version",
    "latitude",
    "longitude",
    "grid_latitude",
    "grid_longitude",
    "dataset_snapshot",
)


def artifact_directory(turbine_id: int, timezone: str, semantics: str) -> Path:
    key = hashlib.sha256(f"{timezone}|{semantics}".encode()).hexdigest()[:16]
    return data_dir() / "nwp-models" / str(turbine_id) / key


def _manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def _contract(provenance):
    if not isinstance(provenance, dict) or not provenance.get("availability_evidence"):
        raise ValueError("Нет подтверждения исторической доступности погоды")
    if any(provenance.get(key) is None for key in CONTRACT_KEYS):
        raise ValueError("Неполный контракт происхождения погодных признаков")
    return {key: provenance[key] for key in CONTRACT_KEYS}


def _public(manifest, dataset):
    fields = (
        "turbine_id",
        "model_type",
        "timezone",
        "timestamp_semantics",
        "promoted",
        "usable_from",
        "training_end",
        "evaluation_training_end",
        "validation_end",
        "control_end",
        "validation_mae",
        "mean_validation_mae",
        "frozen_persistence_validation_mae",
        "model_sha256",
        "source_sha256",
        "weather_used",
        "requires_current_scada",
        "forecast_units",
    )
    return {
        **{key: manifest.get(key) for key in fields},
        "source_matches": manifest.get("source_sha256") == dataset.get("sha256"),
        "validation_start": manifest.get("evaluation_training_end"),
        "control_start": manifest.get("validation_end"),
        "test_is_untouched": False,
    }


def model_catalog():
    active = {t["id"] for t in all_metadata("turbines")}
    items = []
    for dataset in all_metadata("datasets"):
        if dataset["id"] not in active:
            continue
        paths = (data_dir() / "nwp-models" / str(dataset["id"])).glob("*/manifest.json")
        for path in sorted(paths):
            try:
                manifest = _manifest(path.parent)
                if manifest.get("model_type") == MODEL_TYPE:
                    items.append(_public(manifest, dataset))
            except (OSError, ValueError, TypeError):
                continue
    return items


def model_readiness(dataset, *, timezone, semantics, issue_at, horizon=48):
    """Read local metadata/checksum only; never requests weather or executes a model."""
    directory = artifact_directory(dataset["id"], timezone, semantics)
    try:
        manifest = _manifest(directory)
        expected = {
            "artifact_version": 1,
            "model_type": MODEL_TYPE,
            "turbine_id": dataset["id"],
            "source_sha256": dataset["sha256"],
            "timezone": timezone,
            "timestamp_semantics": semantics,
            "feature_columns": list(FEATURE_COLUMNS),
            "sklearn_version": sklearn.__version__,
            "weather_used": True,
            "requires_current_scada": False,
            "target_time_semantics": "interval_start",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Погодный артефакт не соответствует CSV, времени или версии кода")
        if manifest.get("promoted") is not True:
            raise ValueError("Погодная модель не прошла временную валидацию")
        issue = pd.Timestamp(issue_at)
        boundaries = [
            pd.Timestamp(manifest[key])
            for key in (
                "usable_from",
                "training_end",
                "fit_target_available_at_max",
            )
        ]
        if any(time.tzinfo is None for time in [issue, *boundaries]):
            raise ValueError("Не задан часовой пояс границ модели")
        usable, trained, latest_label = boundaries
        if issue < usable or usable < trained or trained < latest_label:
            raise ValueError("Обучение/выбор модели использует сведения позже момента прогноза")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not (1 <= horizon <= manifest.get("horizon_max", 0))
        ):
            raise ValueError("Неподдерживаемый горизонт прогноза")
        _contract(manifest.get("weather_provenance"))
        digest = hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
        if digest != manifest.get("model_sha256"):
            raise ValueError("Изменилась контрольная сумма погодной модели")
        return {
            "ok": True,
            "model": MODEL_TYPE,
            **_public(manifest, dataset),
            "weather_contract": _contract(manifest["weather_provenance"]),
        }
    except FileNotFoundError:
        return {"ok": False, "model": MODEL_TYPE, "reason": "Погодная ML-модель ещё не обучена"}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"ok": False, "model": MODEL_TYPE, "reason": str(exc)[:300]}


@lru_cache(maxsize=8)
def _load(content, digest):
    # Only models produced by the server CLI; no upload or user-selected file path.
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Изменилась контрольная сумма погодной модели")
    return joblib.load(io.BytesIO(content))


def predict_with_nwp(
    dataset,
    weather,
    *,
    timezone,
    semantics,
    issue_at,
    horizon=48,
    expected_model_sha256=None,
):
    readiness = model_readiness(
        dataset,
        timezone=timezone,
        semantics=semantics,
        issue_at=issue_at,
        horizon=horizon,
    )
    if not readiness["ok"]:
        return None, readiness
    directory = artifact_directory(dataset["id"], timezone, semantics)
    try:
        manifest = _manifest(directory)
        expected_digest = expected_model_sha256 or readiness["model_sha256"]
        if manifest.get("model_sha256") != expected_digest:
            raise ValueError("Погодная модель изменилась во время расчёта")
        if _public(manifest, dataset) != {
            key: readiness[key] for key in _public(manifest, dataset)
        }:
            raise ValueError("Метаданные погодной модели изменились во время расчёта")
        if _contract(weather.attrs.get("provenance")) != _contract(manifest["weather_provenance"]):
            raise ValueError("Источник, NWP-модель, координаты или политика публикации изменились")
        content = (directory / "model.joblib").read_bytes()
        model = _load(content, expected_digest)
        predictions = predict_nwp_model(model, weather, issue_at, horizon)
        points = [
            {"time": row.time.isoformat(), "power": float(row.power)}
            for row in predictions.itertuples()
        ]
        provenance = predictions[["weather_run", "weather_available_at"]].drop_duplicates()
        return points, {
            "ok": True,
            "model": MODEL_TYPE,
            "weather_used": True,
            "requires_current_scada": False,
            "model_sha256": manifest["model_sha256"],
            "source_sha256": dataset["sha256"],
            "training_end": manifest["training_end"],
            "usable_from": manifest["usable_from"],
            "evaluation_training_end": manifest["evaluation_training_end"],
            "validation_mae": manifest["validation_mae"],
            "weather_contract": _contract(manifest["weather_provenance"]),
            "weather_runs": [
                {
                    "run": row.weather_run.isoformat(),
                    "available_at": row.weather_available_at.isoformat(),
                }
                for row in provenance.itertuples()
            ],
            "forecast_units": "normalized_power",
            "target_time_semantics": "interval_start",
            "warning": "Погодная модель без новой SCADA. Финальные веса переобучены отдельно; "
            "метрики относятся к замороженной проверочной модели. Февральский факт не использован.",
        }
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        return None, {"ok": False, "model": MODEL_TYPE, "reason": str(exc)[:300]}


def main():
    parser = argparse.ArgumentParser(description="Обучить погодную модель без будущей SCADA")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--turbine-id", required=True, type=int)
    parser.add_argument(
        "--weather", type=Path, help="Verified archive parquet; default local cache"
    )
    parser.add_argument("--provenance", type=Path, help="JSON provenance; default parquet attrs")
    parser.add_argument("--measurement-timezone", required=True)
    parser.add_argument(
        "--timestamp-semantics", required=True, choices=["interval_start", "interval_end"]
    )
    parser.add_argument("--train-end", default="2025-10-01T00:00:00Z")
    parser.add_argument("--validation-end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--control-end", default="2026-01-31T12:00:00Z")
    parser.add_argument("--final-issue-at", default="2026-01-31T12:00:00Z")
    args = parser.parse_args()
    get_turbine(args.turbine_id)
    dataset = next(d for d in all_metadata("datasets") if d["id"] == args.turbine_id)
    if args.weather:
        weather = pd.read_parquet(args.weather)
    else:
        from wind.archive import load_archive_frame

        weather = load_archive_frame(args.turbine_id)
    provenance = (
        json.loads(args.provenance.read_text())
        if args.provenance
        else weather.attrs.get("provenance")
    )
    _contract(provenance)
    path = dataset["hourly_path"].replace("-hourly.parquet", "-10min.parquet")
    manifest = train_nwp(
        pd.read_parquet(data_dir() / path),
        weather,
        turbine_id=dataset["id"],
        source_sha256=dataset["sha256"],
        timezone=args.measurement_timezone,
        semantics=args.timestamp_semantics,
        directory=artifact_directory(
            dataset["id"], args.measurement_timezone, args.timestamp_semantics
        ),
        weather_provenance=provenance,
        train_end=args.train_end,
        validation_end=args.validation_end,
        control_end=args.control_end,
        final_issue_at=args.final_issue_at,
    )
    print(json.dumps(_public(manifest, dataset), ensure_ascii=False))


if __name__ == "__main__":
    main()
