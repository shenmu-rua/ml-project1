"""Shared Python feature contract for training, evaluation and API prediction.

Usage::

    splits = load_splits()
    X_train, y_train = splits["train"]
    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)  # never fit preprocessing on validation/test
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.astronomy import solar_angles, solar_phase
from src.participants import feature_names

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
TARGET_COLUMN = SCHEMA["target"]["name"]
PARTICIPANT_INPUTS = feature_names(SCHEMA)
NUMERIC_INPUTS = list(SCHEMA["numeric_features"]) + PARTICIPANT_INPUTS
CATEGORICAL_INPUTS = list(SCHEMA["categorical_features"])
INPUT_COLUMNS = [SCHEMA["datetime_column"]] + NUMERIC_INPUTS + CATEGORICAL_INPUTS
NUMERIC_FEATURES = NUMERIC_INPUTS + SCHEMA["engineered_numeric_features"]
CATEGORICAL_FEATURES = CATEGORICAL_INPUTS + SCHEMA["engineered_categorical_features"]
# Time and place drive the solar features and cannot be imputed.
REQUIRED_INPUTS = [SCHEMA["datetime_column"], "latitude", "longitude"]
# Inputs that one on-scene source supplies together and may be absent together.
INFO_GROUPS = {
    "weather": list(SCHEMA["weather"]["core_fields"]),
    "road": CATEGORICAL_INPUTS + ["speed_limit", "number_of_vehicles"],
    "vehicle": feature_names(SCHEMA, "vehicle"),
    "casualty": feature_names(SCHEMA, "casualty"),
}


def complete_inputs(frame):
    """Accept partial information: absent optional columns become missing values.

    ``model.predict_proba(complete_inputs(partial_frame))`` works with any subset of
    INFO_GROUPS; the model is trained on rows with these groups blanked.
    """
    absent = [
        name
        for name in REQUIRED_INPUTS
        if name not in frame.columns or frame[name].isna().any()
    ]
    if absent:
        raise ValueError(f"Prediction needs time and place: missing {absent}")
    return frame.reindex(columns=INPUT_COLUMNS)


def _check_inputs(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Pass a pandas DataFrame using the schema.json column names.")
    missing = sorted(set(INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")


class RoadFeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless calendar, solar and wind features; explicitly discard all labels/IDs."""

    def fit(self, X, y=None):
        _check_inputs(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self)
        _check_inputs(X)
        frame = X[NUMERIC_INPUTS + CATEGORICAL_INPUTS].copy()
        for name in NUMERIC_INPUTS:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
        for name in CATEGORICAL_INPUTS:
            frame[name] = frame[name].astype(object)
            frame[name] = frame[name].where(pd.notna(frame[name]), np.nan)
        utc = pd.to_datetime(X["collision_datetime_utc"], utc=True, errors="raise")
        local = utc.dt.tz_convert("Europe/London")
        hours = local.dt.hour + local.dt.minute / 60
        for name, values, period in (
            ("hour", hours, 24),
            ("weekday", local.dt.dayofweek, 7),
            ("month", local.dt.month - 1, 12),
        ):
            frame[name + "_sin"] = np.sin(2 * np.pi * values / period)
            frame[name + "_cos"] = np.cos(2 * np.pi * values / period)
        frame["is_weekend"] = local.dt.dayofweek.ge(5).astype(float)
        angles = solar_angles(X)
        frame["sun_elevation_deg"] = angles
        frame["solar_phase"] = solar_phase(angles)
        frame["rain_present"] = np.where(
            frame.prcp.isna(), np.nan, (frame.prcp > 0).astype(float)
        )
        radians = np.radians(frame.wdir)
        # Meteorological direction is where wind comes FROM.
        frame["wind_east_kmh"] = -frame.wspd * np.sin(radians)
        frame["wind_north_kmh"] = -frame.wspd * np.cos(radians)
        return frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)


def build_preprocessor():
    """Fit medians, scaling and categories on training rows only."""
    numeric = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant", fill_value="unknown", keep_empty_features=True
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def build_pipeline(estimator=None, *, memory=None):
    """Return the entire reusable pipeline; memory can cache transforms during tuning."""
    steps = [("features", RoadFeatureEngineer()), ("preprocess", build_preprocessor())]
    if estimator is not None:
        steps.append(("model", estimator))
    return Pipeline(steps, memory=memory)


def load_splits(data_dir=None):
    """B/C/D share this loader; the returned X contains no labels or audit metadata."""
    directory = Path(data_dir) if data_dir is not None else ROOT / "data/splits"
    output = {}
    for name in ("train", "val", "test"):
        path = directory / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `dvc repro` or `python -m src.data_pipeline all` first."
            )
        frame = pd.read_csv(
            path,
            dtype={**{c: object for c in CATEGORICAL_INPUTS}, "collision_index": str},
            low_memory=False,
        )
        frame["collision_datetime_utc"] = pd.to_datetime(
            frame.collision_datetime_utc, utc=True
        )
        _check_inputs(frame)
        output[name] = (
            frame[INPUT_COLUMNS].copy(),
            frame[TARGET_COLUMN].astype("int64"),
        )
    return output


def check_preprocessing(data_dir=None):
    """Exercise every saved split without fitting anything on validation/test."""
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
    splits = load_splits(data_dir / "splits")
    pipeline = build_pipeline()
    training, target = splits["train"]
    matrices = {"train": pipeline.fit_transform(training, target)}
    imputer = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
    )
    original_statistics = imputer.statistics_.copy()
    for name in ("val", "test"):
        matrices[name] = pipeline.transform(splits[name][0])
    if not np.array_equal(original_statistics, imputer.statistics_, equal_nan=True):
        raise AssertionError("Validation/test changed fitted training statistics.")
    for matrix in matrices.values():
        values = matrix.data if hasattr(matrix, "toarray") else np.asarray(matrix)
        if not np.isfinite(values).all():
            raise AssertionError("Non-finite values reached the model matrix.")
    names = pipeline.get_feature_names_out().tolist()
    if any(
        "severity" in name or "is_severe" in name or "casualties" in name
        for name in names
    ):
        raise AssertionError("An outcome or label leaked into model inputs.")
    report = {
        "fit_population": "train_only",
        "input_columns": INPUT_COLUMNS,
        "encoded_feature_count": len(names),
        "encoded_feature_names": names,
        "shapes": {name: list(matrix.shape) for name, matrix in matrices.items()},
        "training_input_medians": {
            column: float(imputer.statistics_[i])
            for i, column in enumerate(NUMERIC_INPUTS)
        },
        "validation_test_statistics_unchanged": True,
        "all_model_values_finite": True,
        "target_and_outcome_fields_excluded": True,
    }
    (data_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("PREPROCESSING_CHECK_COMPLETE", report["shapes"], flush=True)
    return report


if __name__ == "__main__":
    check_preprocessing()
