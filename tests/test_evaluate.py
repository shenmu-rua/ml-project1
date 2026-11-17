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
