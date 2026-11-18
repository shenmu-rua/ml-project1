"""SHAP tests on a tiny LightGBM pipeline built in memory.

The full 226-column input contract is filled with missing values, so the test
also proves that ``explain_single`` survives a realistically sparse record.
"""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from scipy.special import expit
from xgboost import XGBClassifier

from src.explain import _psi_fallback, collapse_to_source, explain_single, global_shap
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


def test_xgboost_shap_explains_sparse_predictions(frame, tmp_path):
    rows = frame.iloc[[0] * 120].reset_index(drop=True)
    rows["urban_or_rural_area"] = ["1", "2"] * 60
    y = (rows["urban_or_rural_area"] == "1").astype(int)
    pipeline = build_pipeline(XGBClassifier(n_estimators=4, max_depth=1, n_jobs=1))
    pipeline.fit(rows, y)
    model = TriageClassifier(pipeline, (0.2, 0.8))
    sample = rows.iloc[:2]
    probabilities = model.predict_proba(sample)[:, 1]
    assert probabilities[0] > 0.5 > probabilities[1]
    for i, probability in enumerate(probabilities):
        result = explain_single(sample.iloc[[i]], model=model, top_n=1)
        margin = (
            result.base_value + result.contributions[0].impact + result.other_impact
        )
        assert expit(margin) == pytest.approx(probability, abs=1e-6)
    summary = global_shap(sample, model=model, out_dir=tmp_path)
    assert summary["groups"]["road"] == pytest.approx(1.0)


def test_psi_includes_out_of_range_observations(tmp_path):
    ref = pd.DataFrame({"v": np.tile(np.arange(100), 10)})
    cur = pd.DataFrame({"v": np.r_[np.tile(np.arange(100), 7), np.full(300, 200)]})
    report = _psi_fallback(ref, cur, tmp_path / "drift", RuntimeError())
    expected = 9 * (0.07 - 0.1) * np.log(0.7) + (0.37 - 0.1) * np.log(3.7)
    assert report["top"][0]["psi"] == pytest.approx(expected)
    assert report["drifted_share"] == 1.0


def test_psi_reports_insufficient_data(tmp_path):
    rows = pd.DataFrame({"v": range(60)})
    report = _psi_fallback(rows, rows.iloc[:24], tmp_path / "drift", RuntimeError())
    assert report["status"] == "insufficient_data"
    assert report["drifted_share"] is None
    assert pd.read_csv(tmp_path / "drift.csv").empty
