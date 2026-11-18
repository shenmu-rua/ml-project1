"""Mock-based tests for the evaluation module.

No MLflow, no SQLite, no model download, no network: everything here runs on
synthetic arrays and a tiny in-memory outcome frame.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    classification_metrics,
    expected_calibration_error,
    remap_cutoffs,
    select_calibrator,
    sweep_lower_cutoff,
)
from src.triage import acceptance, priorities, triage_metrics


@pytest.fixture
def outcomes():
    rng = np.random.default_rng(0)
    size = 400
    severity = rng.choice([1, 2, 3], size=size, p=[0.05, 0.15, 0.80])
    enhanced = np.where(
        severity == 1,
        1,
        np.where(severity == 2, 5, rng.choice([3, 7], size=size)),
    )
    return pd.DataFrame(
        {"collision_severity": severity, "enhanced_severity_collision": enhanced}
    )


def test_perfect_separation_scores_one():
    y = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y, probability, "test")
    assert metrics["test_roc_auc"] == pytest.approx(1.0)
    assert metrics["test_pr_auc"] == pytest.approx(1.0)
    assert metrics["test_f1"] == pytest.approx(1.0)


def test_metrics_are_prefixed_like_training():
    y = pd.Series([0, 1, 0, 1])
    metrics = classification_metrics(y, np.array([0.2, 0.7, 0.3, 0.8]), "val")
    assert all(key.startswith("val_") for key in metrics)
    assert {"val_brier", "val_log_loss", "val_recall"} <= set(metrics)


def test_ece_is_small_when_frequencies_match():
    probability = np.full(1000, 0.3)
    y = pd.Series(np.random.default_rng(0).binomial(1, 0.3, 1000))
    assert expected_calibration_error(y, probability) < 0.06


def test_ece_flags_overconfidence():
    probability = np.full(1000, 0.9)
    y = pd.Series(np.zeros(1000, dtype=int))
    assert expected_calibration_error(y, probability) > 0.8


def test_remapped_cutoffs_keep_the_tiers_stable():
    rng = np.random.default_rng(1)
    raw = rng.random(2000)
    calibrated = raw**2  # strictly monotone, so the ranking is unchanged
    cutoffs = (0.2, 0.7)
    remapped = remap_cutoffs(cutoffs, raw, calibrated)
    same = priorities(raw, cutoffs) == priorities(calibrated, remapped)
    assert same.mean() > 0.999


def test_remapped_cutoffs_move_to_the_calibrated_scale():
    raw = np.linspace(0.01, 0.99, 500)
    calibrated = raw**2
    low, high = remap_cutoffs((0.2, 0.7), raw, calibrated)
    assert low < 0.2 and high < 0.7


def test_sweep_returns_ordered_valid_points(outcomes):
    rng = np.random.default_rng(2)
    probability = rng.random(len(outcomes))
    points = sweep_lower_cutoff(outcomes, probability, high=0.8, shares=(0.2, 0.3))
    assert points
    assert all(0.0 <= point.p3_share <= 1.0 for point in points)
    assert points[0].cutoff < points[-1].cutoff


def test_acceptance_exposes_the_four_checks(outcomes):
    rng = np.random.default_rng(3)
    probability = rng.random(len(outcomes))
    metrics = triage_metrics(outcomes, probability, (0.2, 0.8), "test")
    checks = acceptance(metrics, prefix="test")
    assert set(checks) == {
        "undertriage_life",
        "undertriage_fatal",
        "routine_share",
        "immediate_lift",
    }
    assert all(isinstance(value, bool) for value in checks.values())


def test_calibration_selection_uses_later_unseen_timestamps(monkeypatch):
    times = pd.date_range("2025-01-01", periods=5, freq="D").repeat([2, 2, 2, 4, 2])
    x = pd.DataFrame({"collision_datetime_utc": times})
    y = pd.Series([0, 1] * 6)
    windows = []

    def fit(model, fit_x, fit_y, method):
        class Candidate:
            def predict_proba(self, holdout):
                assert fit_x.index.equals(fit_y.index)
                assert fit_x.iloc[-1, 0] < holdout.iloc[0, 0]
                windows.append((method, len(fit_x), len(holdout)))
                score = 0.5 if method == "sigmoid" else 0.95
                return np.tile([1 - score, score], (len(holdout), 1))

        return Candidate()

    monkeypatch.setattr("src.evaluate.fit_calibrator", fit)
    chosen, candidates, split_at = select_calibrator(None, x, y)
    assert split_at == 6
    assert windows == [("isotonic", 6, 6), ("sigmoid", 6, 6)]
    assert chosen == "sigmoid"
    assert candidates[chosen]["holdout_log_loss"] == pytest.approx(np.log(2))


def test_registry_uses_configured_tracking_and_loads_resolved_version(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import mlflow

    from src.registry import load_registered_model

    entry = SimpleNamespace(name="road-severity-model", version="5")
    client = Mock()
    client.get_model_version_by_alias.return_value = entry
    client.get_model_version.return_value = entry
    tracking = Mock()
    load = Mock()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: client)
    monkeypatch.setattr(mlflow, "set_tracking_uri", tracking)
    monkeypatch.setattr(mlflow.sklearn, "load_model", load)
    for version in (None, "5"):
        _, resolved = load_registered_model(version)
        assert resolved is entry
        tracking.assert_called_with("http://mlflow:5000")
        load.assert_called_with("models:/road-severity-model/5")
    client.get_model_version_by_alias.assert_called_once_with(
        "road-severity-model", "production"
    )
    client.get_model_version.assert_called_once_with("road-severity-model", "5")
