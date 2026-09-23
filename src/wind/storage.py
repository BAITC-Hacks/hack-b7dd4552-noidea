import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class TurbineNotFoundError(ValueError):
    """The requested turbine is missing or no longer active."""


def data_dir() -> Path:
    path = Path(os.environ.get("DATA_DIR", "data/workspace")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect():
    db = sqlite3.connect(data_dir() / "catalog.sqlite", timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS datasets (id INTEGER PRIMARY KEY, metadata TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS weather (id TEXT PRIMARY KEY, metadata TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS turbines (id INTEGER PRIMARY KEY AUTOINCREMENT, metadata TEXT)"
    )
    db.execute("CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, metadata TEXT)")
    try:
        with db:
            yield db
    finally:
        db.close()


def save_metadata(table: str, key: int | str, metadata: dict):
    assert table in {"datasets", "weather", "turbines", "sources"}
    with connect() as db:
        db.execute(
            f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
            (key, json.dumps(metadata, ensure_ascii=False)),
        )


def all_metadata(table: str, *, include_deleted: bool = False) -> list[dict]:
    assert table in {"datasets", "weather", "turbines", "sources"}
    with connect() as db:
        items = [
            json.loads(row[0]) for row in db.execute(f"SELECT metadata FROM {table} ORDER BY id")
        ]
    if table == "turbines" and not include_deleted:
        return [item for item in items if not item.get("deleted_at")]
    return items


def deleted_turbines() -> list[dict]:
    return [
        item for item in all_metadata("turbines", include_deleted=True) if item.get("deleted_at")
    ]


def create_turbine(values: dict) -> dict:
    with connect() as db:
        cursor = db.execute("INSERT INTO turbines (metadata) VALUES (?)", ("{}",))
        result = {**values, "id": cursor.lastrowid}
        db.execute("UPDATE turbines SET metadata=? WHERE id=?", (json.dumps(result), result["id"]))
    return result


def _read_turbine(db, turbine_id: int) -> dict:
    row = db.execute("SELECT metadata FROM turbines WHERE id=?", (turbine_id,)).fetchone()
    if row is None:
        raise TurbineNotFoundError("Турбина не найдена")
    return json.loads(row[0])


def get_turbine(turbine_id: int) -> dict:
    with connect() as db:
        result = _read_turbine(db, turbine_id)
    if result.get("deleted_at"):
        raise TurbineNotFoundError("Турбина удалена. Восстановите её из корзины")
    return result


def delete_turbine(turbine_id: int) -> dict:
    """Archive only the turbine registration, retaining every dataset and artifact."""
    return _set_turbine_deleted(turbine_id, True)


def restore_turbine(turbine_id: int) -> dict:
    return _set_turbine_deleted(turbine_id, False)


def _set_turbine_deleted(turbine_id: int, deleted: bool) -> dict:
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        result = _read_turbine(db, turbine_id)
        if deleted:
            result.setdefault("deleted_at", datetime.now(UTC).isoformat())
        else:
            result.pop("deleted_at", None)
        db.execute("UPDATE turbines SET metadata=? WHERE id=?", (json.dumps(result), turbine_id))
    return result
