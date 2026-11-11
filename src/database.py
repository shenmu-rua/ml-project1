"""Merge the project CSVs into one SQLite database and load model splits from it.

Run from the repository root (after ``python -m src.data_pipeline all``)::

    python -m src.database

Tables
    collisions_raw    2024 + 2025 DfT collision CSVs, verbatim text
    weather_raw       weather_2024_2025.csv, verbatim text
    vehicles_raw      2024 + 2025 DfT vehicle CSVs, verbatim text
    casualties_raw    2024 + 2025 DfT casualty CSVs, verbatim text
    collisions_clean  A's cleaned collisions linked to weather and participants
    model_data        train/val/test rows with a ``split`` column
"""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    CATEGORICAL_INPUTS,
    INPUT_COLUMNS,
    NUMERIC_INPUTS,
    ROOT,
    TARGET_COLUMN,
    _check_inputs,
)
from src.participants import table_paths

DB_PATH = ROOT / "data/road_collisions.sqlite"
SPLITS = ("train", "val", "test")
CHUNK_ROWS = 50_000
TEXT_COLUMNS = {
    **{c: object for c in CATEGORICAL_INPUTS},
    "collision_index": str,
    "time_issue": str,
}


def _append_csv(connection, table, path, *, verbatim, extra=None):
    """Stream one CSV into a table; verbatim keeps every raw cell as text."""
    options = (
        {"dtype": str, "keep_default_na": False}
        if verbatim
        else {"dtype": TEXT_COLUMNS, "low_memory": False}
    )
    rows = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, **options):
        for name, value in (extra or {}).items():
            chunk[name] = value
        chunk.to_sql(table, connection, if_exists="append", index=False)
        rows += len(chunk)
    return rows


def build_database(data_dir=None, db_path=None):
    """Rebuild the database from scratch so reruns are deterministic."""
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
    db_path = Path(db_path) if db_path is not None else data_dir / DB_PATH.name
    sources = {
        "collisions_raw": [
            (data_dir / f"dft-road-casualty-statistics-collision-{year}.csv", None)
            for year in (2024, 2025)
        ],
        "weather_raw": [(data_dir / "weather_2024_2025.csv", None)],
        "vehicles_raw": [(p, None) for p in table_paths(data_dir, "vehicle")],
        "casualties_raw": [(p, None) for p in table_paths(data_dir, "casualty")],
        "collisions_clean": [(data_dir / "processed/collisions_clean.csv", None)],
        "model_data": [
            (data_dir / f"splits/{name}.csv", {"split": name}) for name in SPLITS
        ],
    }
    missing = [str(p) for files in sources.values() for p, _ in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python -m src.data_pipeline all` first."
        )
    temporary = db_path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    counts = {}
    with closing(sqlite3.connect(temporary)) as connection:
        for table, files in sources.items():
            counts[table] = sum(
                _append_csv(
                    connection,
                    table,
                    path,
                    verbatim=table.endswith("_raw"),
                    extra=extra,
                )
                for path, extra in files
            )
        for table, column in (
            ("collisions_raw", "collision_index"),
            ("collisions_clean", "collision_index"),
            ("model_data", "split"),
            ("weather_raw", "station_id"),
            ("vehicles_raw", "collision_index"),
            ("casualties_raw", "collision_index"),
        ):
            connection.execute(
                f"CREATE INDEX idx_{table}_{column} ON {table}({column})"
            )
        connection.commit()
    temporary.replace(db_path)
    print("DATABASE_COMPLETE", db_path, counts, flush=True)
    return counts


def load_splits(db_path=None):
    """Same contract as ``src.features.load_splits``, read from SQLite."""
    path = Path(db_path) if db_path is not None else DB_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python -m src.database` first.")
    columns = ", ".join(f'"{c}"' for c in [*INPUT_COLUMNS, TARGET_COLUMN])
    query = (
        f"SELECT {columns} FROM model_data WHERE split = ? "
        "ORDER BY collision_datetime_utc, collision_index"
    )
    output = {}
    with closing(sqlite3.connect(path)) as connection:
        for name in SPLITS:
            frame = pd.read_sql_query(query, connection, params=(name,))
            if frame.empty:
                raise ValueError(f"Split {name!r} has no rows in {path}.")
            frame["collision_datetime_utc"] = pd.to_datetime(
                frame.collision_datetime_utc, utc=True
            )
            # A column that is entirely NULL in a split comes back as object.
            frame[NUMERIC_INPUTS] = frame[NUMERIC_INPUTS].astype(float)
            for column in CATEGORICAL_INPUTS:
                values = frame[column].astype(object)
                frame[column] = values.where(pd.notna(values), np.nan)
            _check_inputs(frame)
            output[name] = (
                frame[INPUT_COLUMNS].copy(),
                frame[TARGET_COLUMN].astype("int64"),
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()
    build_database(args.data_dir, args.db_path)


if __name__ == "__main__":
    main()
