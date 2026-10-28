"""SQLite persistence for demo assessments and configurable bed availability."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

CAPACITIES = {"ICU": 15, "GENERAL": 60}
INITIAL_AVAILABLE = {"ICU": 12, "GENERAL": 45}
MAX_CAPACITY = {"ICU": 250, "GENERAL": 1000}


def validate_assessment(person_name, age, spo2, heart_rate, comorbidities):
    """Validate fictional demo inputs and return the trimmed display name."""
    name = person_name.strip()
    if not name or len(name) > 80 or any(ord(char) < 32 for char in name):
        raise ValueError("Enter a name of 1–80 characters without control characters.")
    for value, low, high, label in (
        (age, 1, 100, "Age"), (spo2, 70, 100, "SpO₂"),
        (heart_rate, 40, 160, "Heart rate"), (comorbidities, 0, 3, "Comorbidity count")
    ):
        if isinstance(value, bool) or not low <= int(value) <= high:
            raise ValueError(f"{label} must be between {low} and {high} for this demo.")
    return name


@contextmanager
def _connect(db_path):
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(db_path):
    """Create tables, apply the name-column migration, and seed resources once."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS resources (
            name TEXT PRIMARY KEY,
            capacity INTEGER NOT NULL CHECK (capacity >= 0),
            available INTEGER NOT NULL CHECK (available >= 0 AND available <= capacity)
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            person_name TEXT NOT NULL DEFAULT '',
            age INTEGER NOT NULL CHECK (age BETWEEN 1 AND 100),
            spo2 INTEGER NOT NULL CHECK (spo2 BETWEEN 70 AND 100),
            heart_rate INTEGER NOT NULL CHECK (heart_rate BETWEEN 40 AND 160),
            comorbidities INTEGER NOT NULL CHECK (comorbidities BETWEEN 0 AND 3),
            predicted_level INTEGER NOT NULL CHECK (predicted_level BETWEEN 0 AND 2),
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            suggested_resource TEXT,
            allocation_status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )""")
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(assessments)")}
        if "person_name" not in columns:
            connection.execute("ALTER TABLE assessments ADD COLUMN person_name TEXT NOT NULL DEFAULT ''")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_assessments_created ON assessments(created_at DESC)")
        for name, capacity in CAPACITIES.items():
            connection.execute(
                "INSERT OR IGNORE INTO resources(name, capacity, available) VALUES (?, ?, ?)",
                (name, capacity, INITIAL_AVAILABLE[name]),
            )


def get_resources(db_path):
    with _connect(db_path) as connection:
        rows = connection.execute("SELECT name, capacity, available FROM resources ORDER BY name").fetchall()
    return [dict(row) for row in rows]


def update_capacities(db_path, *, icu_capacity, general_capacity):
    """Set capacities while preserving current occupied counts; reject unsafe reductions."""
    requested = {"ICU": int(icu_capacity), "GENERAL": int(general_capacity)}
    for name, capacity in requested.items():
        if not 1 <= capacity <= MAX_CAPACITY[name]:
            raise ValueError(f"{name} capacity must be between 1 and {MAX_CAPACITY[name]}.")
    with _connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = {
            row["name"]: dict(row)
            for row in connection.execute("SELECT name, capacity, available FROM resources")
        }
        for name, capacity in requested.items():
            occupied = current[name]["capacity"] - current[name]["available"]
            if capacity < occupied:
                raise ValueError(f"{name} capacity cannot be lower than its {occupied} currently occupied demo slots.")
        for name, capacity in requested.items():
            occupied = current[name]["capacity"] - current[name]["available"]
            connection.execute(
                "UPDATE resources SET capacity = ?, available = ? WHERE name = ?",
                (capacity, capacity - occupied, name),
            )


def save_assessment(db_path, *, person_name, age, spo2, heart_rate, comorbidities,
                    predicted_level, predicted_label, confidence):
    """Persist an assessment and reserve a demo slot atomically when available."""
    person_name = validate_assessment(person_name, age, spo2, heart_rate, comorbidities)
    if predicted_level not in (0, 1, 2):
        raise ValueError("Predicted demo category must be 0, 1, or 2.")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("Model confidence must be between 0 and 1.")
    resource = "ICU" if predicted_level == 2 else "GENERAL"
    patient_id = f"DEMO-{uuid4().hex[:10].upper()}"
    with _connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        updated = connection.execute(
            "UPDATE resources SET available = available - 1 WHERE name = ? AND available > 0",
            (resource,),
        ).rowcount
        status = "allocated" if updated else "capacity_full"
        cursor = connection.execute("""INSERT INTO assessments(
            patient_id, person_name, age, spo2, heart_rate, comorbidities,
            predicted_level, predicted_label, confidence, suggested_resource, allocation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (patient_id, person_name, int(age), int(spo2), int(heart_rate), int(comorbidities),
             int(predicted_level), predicted_label, float(confidence), resource, status))
        record = connection.execute("SELECT * FROM assessments WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(record)


def release_allocation(db_path, assessment_id):
    """Release an allocated demo slot once; return False if already released/unallocated."""
    with _connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT suggested_resource FROM assessments WHERE id = ? AND allocation_status = 'allocated'",
            (int(assessment_id),),
        ).fetchone()
        if row is None:
            return False
        connection.execute(
            "UPDATE resources SET available = MIN(capacity, available + 1) WHERE name = ?",
            (row["suggested_resource"],),
        )
        connection.execute("UPDATE assessments SET allocation_status = 'released' WHERE id = ?", (int(assessment_id),))
    return True


def get_assessments(db_path, limit=500):
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM assessments ORDER BY created_at DESC, id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(row) for row in rows]




def delete_released_assessments(db_path, assessment_ids):
    """Permanently delete selected records only when they are already released."""
    ids = sorted({int(item) for item in assessment_ids})
    if not ids:
        return 0
    removed = 0
    with _connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for assessment_id in ids:
            row = connection.execute(
                "SELECT allocation_status FROM assessments WHERE id = ?",
                (assessment_id,),
            ).fetchone()
            if row is None or row["allocation_status"] != "released":
                continue
            connection.execute("DELETE FROM assessments WHERE id = ?", (assessment_id,))
            removed += 1
    return removed

