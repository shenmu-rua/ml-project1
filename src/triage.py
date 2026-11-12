"""Dispatch priority tiers on top of the severity probability (member B).

The model cannot separate "serious" from "slight" sharply enough for a yes/no
alarm, but it ranks collisions well enough to order the emergency response:

    priority 1  immediate   the highest-scoring share of collisions
    priority 2  urgent      everything between the two cut-offs
    priority 3  routine     scores so low that few life-threatening cases fall here

Cut-offs come from the validation split only. Acceptance follows trauma triage
practice: life-threatening collisions sent to the routine tier (under-triage)
must stay within the limit in schema.json.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from src.validation import load_schema

TRIAGE = load_schema()["triage"]
GRADES = {int(code): float(grade) for code, grade in TRIAGE["grades"].items()}
LIFE_THREATENING = TRIAGE["life_threatening_min_grade"]
FATAL = max(GRADES.values())


def severity_grade(outcomes):
    """0 slight .. 4 fatal; NaN where a force recorded "serious" without a grade."""
    grade = outcomes["enhanced_severity_collision"].map(GRADES)
    official = outcomes["collision_severity"]
    grade = grade.where(grade.notna(), official.map({1: FATAL, 3: 0.0}))
    return grade.astype(float)


def graded_forces(outcomes):
    """Rows from forces whose records distinguish the grades of serious injury."""
    return outcomes["enhanced_severity_collision"].isin(list(GRADES)).to_numpy()


def training_weights(outcomes):
    """Graver outcomes weigh more, so the ranking is sharpest where it matters."""
    grade = severity_grade(outcomes)
    known_serious = grade[graded_forces(outcomes) & grade.between(1, FATAL - 1)]
    grade = grade.fillna(known_serious.mean())
    return (1 + grade**2 / 2).to_numpy()


def triage_cutoffs(outcomes, probability):
    """(routine/urgent, urgent/immediate) probability cut-offs from validation rows."""
    life = graded_forces(outcomes) & (severity_grade(outcomes) >= LIFE_THREATENING)
    low = np.quantile(probability[life.to_numpy()], TRIAGE["validation_undertriage"])
    high = np.quantile(probability, 1 - TRIAGE["immediate_share"])
    if high <= low:
        raise ValueError("The immediate tier would swallow the urgent tier.")
    return float(low), float(high)


def priorities(probability, cutoffs):
    low, high = cutoffs
    return 3 - (probability >= low).astype(int) - (probability >= high).astype(int)


class TriageClassifier(BaseEstimator):
    """A fitted probability pipeline plus the two cut-offs that turn it into tiers.

    ``predict`` returns the dispatch priority (1 immediate, 2 urgent, 3 routine),
    ``predict_proba`` is the pipeline's, and ``pipeline`` is the fitted sklearn
    Pipeline for explanation tools.
    """

    def __init__(self, pipeline, cutoffs):
        self.pipeline = pipeline
        self.cutoffs = cutoffs

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return priorities(self.predict_proba(X)[:, 1], self.cutoffs)


def triage_metrics(outcomes, probability, cutoffs, prefix):
    """How safely and how usefully the tiers split one evaluation population."""
    tier = priorities(probability, cutoffs)
    grade = severity_grade(outcomes).to_numpy()
    graded = graded_forces(outcomes)
    life = graded & (grade >= LIFE_THREATENING)
    fatal = outcomes["collision_severity"].eq(1).to_numpy()
    severe = outcomes["collision_severity"].isin([1, 2]).to_numpy()
    base = life[graded].mean()
    values = {
        "p1_share": np.mean(tier == 1),
        "p2_share": np.mean(tier == 2),
        "p3_share": np.mean(tier == 3),
        "undertriage_life": np.mean(tier[life] == 3),
        "undertriage_fatal": np.mean(tier[fatal] == 3),
        "life_in_p1": np.mean(tier[life] == 1),
        "life_base_rate": base,
        "p1_life_rate": life[graded & (tier == 1)].mean(),
        "p3_life_rate": life[graded & (tier == 3)].mean(),
        "p1_life_lift": life[graded & (tier == 1)].mean() / base,
        "p1_severe_rate": severe[tier == 1].mean(),
        "p3_severe_rate": severe[tier == 3].mean(),
        "severe_recall_p1_p2": np.mean(tier[severe] != 3),
        "roc_auc_life": roc_auc_score(life[graded], probability[graded]),
        "roc_auc_fatal": roc_auc_score(fatal, probability),
    }
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def acceptance(metrics, prefix="test"):
    """Minimum bar for supporting dispatch: each check maps to a schema.json limit."""
    limits = TRIAGE["acceptance"]
    return {
        "undertriage_life": metrics[f"{prefix}_undertriage_life"]
        <= limits["max_undertriage_life"],
        "undertriage_fatal": metrics[f"{prefix}_undertriage_fatal"]
        <= limits["max_undertriage_fatal"],
        "routine_share": metrics[f"{prefix}_p3_share"] >= limits["min_routine_share"],
        "immediate_lift": metrics[f"{prefix}_p1_life_lift"]
        >= limits["min_immediate_life_lift"],
    }
