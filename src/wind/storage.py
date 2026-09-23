import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


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


def all_metadata(table: str) -> list[dict]:
    assert table in {"datasets", "weather", "turbines", "sources"}
    with connect() as db:
        return [
            json.loads(row[0]) for row in db.execute(f"SELECT metadata FROM {table} ORDER BY id")
        ]


def create_turbine(values: dict) -> dict:
    with connect() as db:
        cursor = db.execute("INSERT INTO turbines (metadata) VALUES (?)", ("{}",))
        result = {**values, "id": cursor.lastrowid}
        db.execute("UPDATE turbines SET metadata=? WHERE id=?", (json.dumps(result), result["id"]))
    return result


def get_turbine(turbine_id: int) -> dict:
    result = next((t for t in all_metadata("turbines") if t["id"] == turbine_id), None)
    if result is None:
        raise ValueError("Сначала добавьте турбину")
    return result
