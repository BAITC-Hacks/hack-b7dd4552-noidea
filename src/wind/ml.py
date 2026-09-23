"""Validated local ML artifacts; no remote calls or automatic training on requests."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from wind.storage import all_metadata, data_dir, get_turbine
from wind_ml.telemetry import FEATURE_COLUMNS, hourly_telemetry, predict_telemetry

MODEL_TYPE = "telemetry_hist_gradient_boosting"


def artifact_directory(turbine_id: int, timezone: str, semantics: str) -> Path:
    config = hashlib.sha256(f"{timezone}|{semantics}".encode()).hexdigest()[:16]
    return data_dir() / "models" / str(turbine_id) / config


def _read_manifest(directory: Path) -> dict:
    value = json.loads((directory / "manifest.json").read_text())
    if value.get("artifact_version") != 1 or value.get("model_type") != MODEL_TYPE:
        raise ValueError("Неподдерживаемый ML-артефакт")
    return value


def _public_manifest(manifest: dict, dataset: dict) -> dict:
    fields = (
        "turbine_id",
        "model_type",
        "timezone",
        "timestamp_semantics",
        "promoted",
        "usable_from",
        "training_end",
        "validation_end",
        "test_end",
        "validation_mae",
        "persistence_validation_mae",
        "model_sha256",
        "source_sha256",
    )
    return {
        **{key: manifest.get(key) for key in fields},
        "source_matches": manifest.get("source_sha256") == dataset.get("sha256"),
        "validation_start": manifest.get("training_end"),
        "control_start": manifest.get("validation_end"),
        "control_end": manifest.get("test_end"),
        "weather_used": False,
        "test_is_untouched": False,
    }


def model_catalog() -> list[dict]:
    active = {t["id"] for t in all_metadata("turbines")}
    datasets = {d["id"]: d for d in all_metadata("datasets") if d["id"] in active}
    items = []
    for turbine_id, dataset in datasets.items():
        for path in sorted((data_dir() / "models" / str(turbine_id)).glob("*/manifest.json")):
            try:
                manifest = _read_manifest(path.parent)
                if manifest.get("turbine_id") == turbine_id:
                    items.append(_public_manifest(manifest, dataset))
            except (OSError, ValueError, TypeError):
                continue
    return items


@lru_cache(maxsize=8)
def _load_model(path: str, sha256: str):
    # The checksum is validated before this function. Only server-produced files
    # under DATA_DIR/models are used; there is no model-upload API.
    return joblib.load(path)


def predict_with_ml(
    raw: pd.DataFrame,
    dataset: dict,
    *,
    timezone: str,
    semantics: str,
    issue_at,
    horizon: int,
) -> tuple[list[dict] | None, dict]:
    """Return a complete forecast or an explicit reason to use persistence."""
    fallback = {"model": "persistence_baseline", "weather_used": False}
    directory = artifact_directory(dataset["id"], timezone, semantics)
    if not (directory / "manifest.json").exists():
        return None, {
            **fallback,
            "fallback_reason": "Для этой турбины и настройки времени нет ML-модели",
        }
    try:
        manifest = _read_manifest(directory)
        expected = {
            "turbine_id": dataset["id"],
            "source_sha256": dataset["sha256"],
            "timezone": timezone,
            "timestamp_semantics": semantics,
            "feature_columns": list(FEATURE_COLUMNS),
            "sklearn_version": sklearn.__version__,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Изменились CSV, временные настройки или версия модели; нужно переобучение"
            )
        if manifest.get("promoted") is not True:
            raise ValueError("ML-модель не прошла сравнение с baseline на валидации")
        issue = pd.Timestamp(issue_at)
        trained = pd.Timestamp(manifest["training_end"])
        usable = pd.Timestamp(manifest["usable_from"])
        if any(time.tzinfo is None for time in (issue, trained, usable)) or usable < trained:
            raise ValueError("Некорректные временные границы ML-артефакта")
        if issue < usable:
            raise ValueError(
                "Дата расчёта раньше окончания проверки ML-модели; ретропрогноз заблокирован"
            )
        if not 1 <= horizon <= manifest.get("horizon_max", 0):
            raise ValueError("Горизонт не поддерживается ML-артефактом")
        model_path = directory / "model.joblib"
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != manifest.get("model_sha256"):
            raise ValueError("Контрольная сумма ML-модели не совпадает")
        model = _load_model(str(model_path), digest)
        hourly = hourly_telemetry(raw, timezone, semantics)
        forecast = predict_telemetry(model, hourly, issue, horizon)
        points = [
            {"time": pd.Timestamp(row.time).isoformat(), "power": float(row.power)}
            for row in forecast.itertuples()
        ]
        return points, {
            "model": MODEL_TYPE,
            "weather_used": False,
            "model_sha256": digest,
            "training_end": manifest["training_end"],
            "usable_from": manifest["usable_from"],
            "validation_mae": manifest.get("validation_mae"),
            "persistence_validation_mae": manifest.get("persistence_validation_mae"),
            "source_sha256": dataset["sha256"],
            "warning": "Модель прошлой телеметрии; будущая погода не используется. "
            "Часовой пояс — явно выбранная гипотеза. "
            "Январский контроль не является нетронутым тестом.",
        }
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        return None, {**fallback, "fallback_reason": str(exc)[:250]}


def main():
    parser = argparse.ArgumentParser(description="Обучение ML без примесей validation/test в fit")
    parser.add_argument("command", choices=["train"])
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--turbine-id", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--measurement-timezone", required=True)
    parser.add_argument(
        "--timestamp-semantics", choices=["interval_start", "interval_end"], required=True
    )
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-end", required=True)
    args = parser.parse_args()
    from wind_ml.training import train_telemetry

    active = {t["id"] for t in all_metadata("turbines")}
    if args.turbine_id is not None:
        get_turbine(args.turbine_id)
        active = {args.turbine_id}
    for dataset in all_metadata("datasets"):
        if dataset["id"] not in active:
            continue
        raw_path = dataset["hourly_path"].replace("-hourly.parquet", "-10min.parquet")
        manifest = train_telemetry(
            pd.read_parquet(data_dir() / raw_path),
            turbine_id=dataset["id"],
            source_sha256=dataset["sha256"],
            timezone=args.measurement_timezone,
            semantics=args.timestamp_semantics,
            directory=artifact_directory(
                dataset["id"], args.measurement_timezone, args.timestamp_semantics
            ),
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_end=args.test_end,
        )
        print(json.dumps(_public_manifest(manifest, dataset), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
