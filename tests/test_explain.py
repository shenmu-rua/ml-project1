"""SHAP tests on a tiny LightGBM pipeline built in memory.

The full 226-column input contract is filled with missing values, so the test
also proves that ``explain_single`` survives a realistically sparse record.
"""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.explain import collapse_to_source, explain_single
from src.features import INPUT_COLUMNS, build_pipeline
from src.triage import TriageClassifier


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    size = 120
    data = {column: np.nan for column in INPUT_COLUMNS}
    frame = pd.DataFrame(data, index=range(size))
    frame["collision_datetime_utc"] = pd.date_range(
        "2025-01-01", periods=size, freq="h", tz="UTC"
    )
    frame["latitude"] = 51.5 + rng.normal(0, 0.2, size)
    frame["longitude"] = -0.1 + rng.normal(0, 0.2, size)
    frame["speed_limit"] = rng.choice([20, 30, 60], size=size).astype(float)
    frame["number_of_vehicles"] = rng.integers(1, 4, size=size).astype(float)
    frame["urban_or_rural_area"] = rng.choice(["1", "2"], size=size)
    frame["light_conditions"] = rng.choice(["1", "4", "6"], size=size)
    return frame


@pytest.fixture
def model(frame):
    target = pd.Series((frame["speed_limit"].to_numpy() >= 50).astype(int))
    pipeline = build_pipeline(
        LGBMClassifier(n_estimators=8, num_leaves=4, verbose=-1, random_state=0)
    )
    pipeline.fit(frame, target)
    return TriageClassifier(pipeline, (0.2, 0.8))


def test_collapse_sums_one_hot_back_into_the_raw_field():
    names = ["cat__vehicle_type_1", "cat__vehicle_type_9", "num__speed_limit"]
    values = np.array([0.3, 0.2, -0.5])
    collapsed = collapse_to_source(names, values, ["vehicle_type", "speed_limit"])
    assert collapsed["vehicle_type"] == pytest.approx(0.5)
    assert collapsed["speed_limit"] == pytest.approx(-0.5)


def test_collapse_prefix_prefers_the_longest_field():
    names = ["cat__age_of_casualty_1"]
    collapsed = collapse_to_source(names, np.array([1.0]), ["age", "age_of_casualty"])
    assert collapsed == {"age_of_casualty": pytest.approx(1.0)}


def test_explanation_is_additive(frame, model):
    explanation = explain_single(frame.iloc[[0]], model=model, top_n=6)
    total = (
        explanation.base_value
        + sum(item.impact for item in explanation.contributions)
        + explanation.other_impact
    )
    preprocess = model.pipeline[:-1]
    matrix = preprocess.transform(frame.iloc[[0]])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    margin = float(model.pipeline[-1].predict(matrix, raw_score=True)[0])
    assert total == pytest.approx(margin, abs=1e-3)


def test_explanation_reports_tier_and_probability(frame, model):
    explanation = explain_single(frame.iloc[[3]], model=model)
    assert explanation.priority in (1, 2, 3)
    assert 0.0 <= explanation.probability <= 1.0
    assert explanation.contributions
    assert all(item.feature for item in explanation.contributions)


def test_explain_single_rejects_more_than_one_row(frame, model):
    with pytest.raises(ValueError):
        explain_single(frame, model=model)
