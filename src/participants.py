"""Collision-level features from the DfT vehicle and casualty tables.

Each collision has several vehicles and casualties. They become fixed-width model
inputs by counting rows per retained code and taking min/max/mean of numeric
fields. Injury outcome columns are listed in schema.json and never aggregated.

The API can build the same inputs from on-scene lists::

    aggregate(vehicles, "vehicle").join(aggregate(casualties, "casualty"))
"""

from pathlib import Path

import pandas as pd

STATISTICS = ("min", "max", "mean")
YEARS = (2024, 2025)


def feature_names(contract, table=None):
    """Column names in the fixed order shared by training and prediction."""
    names = []
    for key, spec in contract["participants"].items():
        if table not in (None, key):
            continue
        prefix = spec["prefix"]
        for column, codes in spec["counts"].items():
            names += [f"{prefix}_{column}_{code}" for code in codes]
        for column in spec["stats"]:
            names += [f"{prefix}_{column}_{stat}" for stat in STATISTICS]
    return names


def aggregate(frame, table, contract=None):
    """One row per collision_index; codes outside the contract are not counted."""
    if contract is None:
        from src.validation import load_schema

        contract = load_schema()
    spec = contract["participants"][table]
    leaked = sorted(
        set(spec["excluded_outcomes"]) & (set(spec["counts"]) | set(spec["stats"]))
    )
    if leaked:
        raise ValueError(f"Outcome columns cannot be model inputs: {leaked}")
    key = frame["collision_index"].astype(str)
    prefix = spec["prefix"]
    parts = []
    for column, codes in spec["counts"].items():
        counts = pd.crosstab(key, frame[column].astype(str).str.strip())
        counts = counts.reindex(columns=codes, fill_value=0)
        counts.columns = [f"{prefix}_{column}_{code}" for code in codes]
        parts.append(counts)
    for column in spec["stats"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        # Negative codes mean "not known"; keep them missing instead of numeric.
        summary = values.where(values >= 0).groupby(key).agg(list(STATISTICS))
        summary.columns = [f"{prefix}_{column}_{stat}" for stat in STATISTICS]
        parts.append(summary)
    result = pd.concat(parts, axis=1).astype(float)
    result.index.name = "collision_index"
    return result[feature_names(contract, table)]


def read_table(data_dir, table):
    """Raw rows of one participant table for both years, every cell as text."""
    return pd.concat(
        [
            pd.read_csv(path, dtype=str, keep_default_na=False)
            for path in table_paths(data_dir, table)
        ],
        ignore_index=True,
    )


def table_paths(data_dir, table):
    return [
        Path(data_dir) / f"dft-road-casualty-statistics-{table}-{year}.csv"
        for year in YEARS
    ]


def read_participants(data_dir, contract):
    """Vehicle and casualty features for every collision that has such rows."""
    tables = [
        aggregate(read_table(data_dir, table), table, contract)
        for table in contract["participants"]
    ]
    return tables[0].join(tables[1:], how="outer")
