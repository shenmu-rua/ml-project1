"""Pandera checks generated from the same schema used by the feature pipeline."""

import json
from pathlib import Path

import pandera.pandas as pa

from src.participants import feature_names


def load_schema(path=None):
    path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def dataframe_schema(*, modeling=False):
    contract = load_schema()
    fields = {
        "collision_index": pa.Column(str, unique=True, nullable=False),
        "is_severe": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=not modeling),
        "collision_datetime_utc": pa.Column(
            checks=pa.Check(lambda s: str(s.dt.tz) == "UTC"),
            nullable=not modeling,
        ),
    }
    for name, definition in contract["numeric_features"].items():
        checks = []
        if "min" in definition:
            checks.append(pa.Check.ge(definition["min"]))
        if "max" in definition:
            checks.append(pa.Check.le(definition["max"]))
        if "allowed" in definition:
            checks.append(pa.Check.isin(definition["allowed"]))
        required = (
            modeling
            and name in ["latitude", "longitude"] + contract["weather"]["core_fields"]
        )
        fields[name] = pa.Column(float, checks, nullable=not required)
    for name in feature_names(contract):
        fields[name] = pa.Column(float, pa.Check.ge(0), nullable=True)
    for name, definition in contract["categorical_features"].items():
        fields[name] = pa.Column(
            object, pa.Check.isin(list(definition["categories"])), nullable=True
        )
    return pa.DataFrameSchema(
        fields,
        strict=False,
        coerce=False,
        name="road_severity_model" if modeling else "road_collision_master",
    )


def validate_frame(frame, *, modeling=False):
    return dataframe_schema(modeling=modeling).validate(frame, lazy=True)
