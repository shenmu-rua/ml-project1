"""Evaluation, probability calibration, tier audit and error analysis (member C).

Run from the repository root after ``python -m src.train``::

    python -m src.evaluate                  # every evaluation block
    python -m src.evaluate --skip-cv        # skip time-series cross-validation
    python -m src.evaluate --skip-drift     # skip the Evidently drift report

Nothing here trusts a number reported by training: every metric is recomputed
from the frozen test split, and the recomputed values are reconciled against
what MLflow recorded for the production model. Artefacts land in
``artifacts/evaluation``.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.database import load_outcomes, load_splits
from src.features import ROOT, SCHEMA, build_pipeline
from src.triage import (
    TriageClassifier,
    acceptance,
    priorities,
    severity_grade,
    training_weights,
    triage_metrics,
)

MODEL_NAME = "road-severity-model"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
OUT = ROOT / "artifacts/evaluation"
TRIAGE = SCHEMA["triage"]
LIFE_MIN_GRADE = TRIAGE["life_threatening_min_grade"]
SEGMENT_MIN_ROWS = 300
# Hyper-parameters published in B's report; cross-validation reuses them so the
# stability check describes the model that was actually shipped.
CV_PARAMS = {
    "n_estimators": 900,
    "learning_rate": 0.0157,
    "num_leaves": 95,
    "min_child_samples": 128,
    "subsample": 0.986,
    "colsample_bytree": 0.865,
    "reg_alpha": 0.797,
    "reg_lambda": 0.271,
    "subsample_freq": 1,
    "n_jobs": -1,
    "random_state": SCHEMA["split"]["random_seed"],
    "verbose": -1,
}


# ---------------------------------------------------------------- metrics


def classification_metrics(
    y_true: pd.Series, probability: np.ndarray, prefix: str
) -> dict[str, float]:
    """Same definitions and 0.5 threshold as training, so the numbers reconcile."""
    label = (probability >= 0.5).astype(int)
    values = {
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "f1": f1_score(y_true, label, zero_division=0),
        "precision": precision_score(y_true, label, zero_division=0),
        "recall": recall_score(y_true, label, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, label),
        "brier": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability),
    }
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def expected_calibration_error(
    y_true: Iterable[int], probability: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-frequency ECE; equal-width bins go empty on a skewed probability scale."""
    y = np.asarray(y_true, dtype=float)
    order = np.argsort(probability, kind="stable")
    total = 0.0
    for index in np.array_split(order, n_bins):
        if index.size == 0:
            continue
        gap = abs(float(probability[index].mean()) - float(y[index].mean()))
        total += index.size / len(y) * gap
    return float(total)


def reliability_points(
    y_true: pd.Series, probability: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Mean predicted probability and observed rate for each equal-frequency bin."""
    y = np.asarray(y_true, dtype=float)
    order = np.argsort(probability, kind="stable")
    confidence, observed = [], []
    for index in np.array_split(order, n_bins):
        if index.size:
            confidence.append(float(probability[index].mean()))
            observed.append(float(y[index].mean()))
    return np.asarray(confidence), np.asarray(observed)


def fit_calibrator(
    model: TriageClassifier, x_val: pd.DataFrame, y_val: pd.Series, method: str
) -> CalibratedClassifierCV:
    """Fit on validation rows only; the test split never touches the calibrator."""
    calibrator = CalibratedClassifierCV(FrozenEstimator(model.pipeline), method=method)
    calibrator.fit(x_val, y_val)
    return calibrator


def remap_cutoffs(
    cutoffs: tuple[float, float],
    raw_val: np.ndarray,
    calibrated_val: np.ndarray,
) -> tuple[float, float]:
    """Move each cut-off to the same validation quantile after calibration.

    Calibration is monotone, so this keeps every row in the tier it had before
    (and therefore keeps the acceptance result) while making the displayed
    probability a real frequency.
    """
    shares = [float((raw_val < cutoff).mean()) for cutoff in cutoffs]
    return tuple(float(np.quantile(calibrated_val, share)) for share in shares)


# ---------------------------------------------------------------- tier audit


@dataclass(frozen=True)
class SweepPoint:
    cutoff: float
    p3_share: float
    undertriage_life: float
    undertriage_fatal: float
    p1_life_lift: float
    passed: bool


def sweep_lower_cutoff(
    outcomes: pd.DataFrame,
    probability: np.ndarray,
    high: float,
    shares: Iterable[float] = tuple(np.linspace(0.10, 0.45, 15)),
) -> list[SweepPoint]:
    """Hold the immediate cut-off, scan the routine/urgent cut-off."""
    points = []
    for share in shares:
        low = float(np.quantile(probability, share))
        if low >= high:
            continue
        metrics = triage_metrics(outcomes, probability, (low, high), "sweep")
        checks = acceptance(metrics, prefix="sweep")
        points.append(
            SweepPoint(
                cutoff=low,
                p3_share=float(metrics["sweep_p3_share"]),
                undertriage_life=float(metrics["sweep_undertriage_life"]),
                undertriage_fatal=float(metrics["sweep_undertriage_fatal"]),
                p1_life_lift=float(metrics["sweep_p1_life_lift"]),
                passed=all(checks.values()),
            )
        )
    return points


def sensitivity_analysis(
    outcomes: pd.DataFrame,
    probability: np.ndarray,
    cutoffs: tuple[float, float],
    deltas: Iterable[float] = (-0.02, -0.01, 0.0, 0.01, 0.02),
) -> list[dict[str, object]]:
    """Would a small move of the routine/urgent cut-off break the acceptance bar?"""
    rows = []
    for delta in deltas:
        moved = (cutoffs[0] + delta, cutoffs[1])
        metrics = triage_metrics(outcomes, probability, moved, "sens")
        checks = acceptance(metrics, prefix="sens")
        rows.append(
            {
                "delta": float(delta),
                "cutoff_low": float(moved[0]),
                "undertriage_life": float(metrics["sens_undertriage_life"]),
                "p3_share": float(metrics["sens_p3_share"]),
                "p1_life_lift": float(metrics["sens_p1_life_lift"]),
                "passed": all(checks.values()),
            }
        )
    return rows


# ---------------------------------------------------------------- subgroups


def _segment_columns(x: pd.DataFrame) -> dict[str, pd.Series]:
    """Readable slices from the raw input columns; a missing column drops the slice."""
    groups: dict[str, pd.Series] = {}
    urban = x.get("urban_or_rural_area")
    if urban is not None:
        groups["城乡"] = (
            urban.astype("string").map({"1": "城市", "2": "乡村"}).fillna("未知")
        )
    light = x.get("light_conditions")
    if light is not None:
        groups["光照"] = (
            light.astype("string").map({"1": "白天"}).fillna("黑暗/未知")
        )
    speed = x.get("speed_limit")
    if speed is not None:
        bands = pd.cut(
            pd.to_numeric(speed, errors="coerce"),
            bins=[0, 30, 50, 110],
            labels=["≤30", "31-50", ">50"],
        )
        groups["限速"] = bands.astype("string").fillna("未知")
    timestamp = x.get("collision_datetime_utc")
    if timestamp is not None:
        hour = (
            pd.to_datetime(timestamp, utc=True)
            .dt.tz_convert(SCHEMA["timezone"])
            .dt.hour
        )
        bands = pd.cut(
            hour,
            bins=[-1, 5, 9, 15, 19, 23],
            labels=["深夜", "早高峰", "日间", "晚高峰", "夜间"],
        )
        groups["时段"] = bands.astype("string").fillna("未知")
    return groups


def segment_report(
    x: pd.DataFrame,
    y: pd.Series,
    probability: np.ndarray,
    outcomes: pd.DataFrame,
    cutoffs: tuple[float, float],
    prefix: str = "test",
) -> pd.DataFrame:
    """Recompute every metric per subgroup to find where the model is weakest."""
    rows = []
    for dimension, series in _segment_columns(x).items():
        for value in series.dropna().unique():
            mask = (series == value).to_numpy()
            if mask.sum() < SEGMENT_MIN_ROWS:
                continue
            y_slice = y.to_numpy()[mask]
            if y_slice.min() == y_slice.max():
                continue
            probability_slice = probability[mask]
            row: dict[str, object] = {
                "维度": dimension,
                "分组": str(value),
                "样本数": int(mask.sum()),
                "正例率": float(y_slice.mean()),
                "PR-AUC": float(average_precision_score(y_slice, probability_slice)),
                "ROC-AUC": float(roc_auc_score(y_slice, probability_slice)),
            }
            try:
                metrics = triage_metrics(
                    outcomes.iloc[mask], probability_slice, cutoffs, "seg"
                )
                row.update(
                    {
                        "漏判率": float(metrics["seg_undertriage_life"]),
                        "三级占比": float(metrics["seg_p3_share"]),
                        "一级富集": float(metrics["seg_p1_life_lift"]),
                    }
                )
            except ValueError:
                # Too few graded or fatal rows in this slice for the tier metrics.
                row.update({"漏判率": np.nan, "三级占比": np.nan, "一级富集": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def error_cases(
    x: pd.DataFrame,
    probability: np.ndarray,
    outcomes: pd.DataFrame,
    cutoffs: tuple[float, float],
    top_n: int = 20,
) -> pd.DataFrame:
    """Worst under-triage and over-triage rows, ordered by model probability.

    ``load_splits`` returns no ``collision_index``, so only the row number and
    timestamp identify a case; join ``model_data`` on the same ordering key when
    the accident reference is needed.
    """
    grade = severity_grade(outcomes).to_numpy()
    life = grade >= LIFE_MIN_GRADE
    tier = priorities(probability, cutoffs)
    fatal = outcomes["collision_severity"].to_numpy() == 1
    slight = outcomes["collision_severity"].to_numpy() == 3
    frame = pd.DataFrame(
        {
            "行号": np.arange(len(probability)),
            "时间": pd.to_datetime(x["collision_datetime_utc"], utc=True).to_numpy(),
            "概率": probability,
            "分级": tier,
            "伤情等级": grade,
            "致死": fatal,
        }
    )
    missed = frame[life & (tier == 3)].sort_values("概率").head(top_n)
    over = frame[slight & (tier == 1)].sort_values("概率", ascending=False).head(top_n)
    missed = missed.assign(类型="漏判：危及生命却进入常规")
    over = over.assign(类型="过诊：轻伤却进入立即")
    return pd.concat([missed, over], ignore_index=True)


# ---------------------------------------------------------------- stability


def time_series_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    n_splits: int = 4,
    max_rows: int = 40000,
    estimator=None,
) -> dict[str, object]:
    """Ordered folds only: a random KFold would train on the future.

    ``estimator`` is cloned from the production pipeline so the stability check
    describes the algorithm that was actually shipped; without it the module
    falls back to the LightGBM hyper-parameters of B's report.
    """
    if estimator is None:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(**CV_PARAMS)

    if len(x) > max_rows:
        x = x.iloc[-max_rows:]
        y = y.iloc[-max_rows:]
        weights = weights[-max_rows:]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(x), start=1):
        pipeline = build_pipeline(clone(estimator))
        pipeline.fit(
            x.iloc[train_index],
            y.iloc[train_index],
            model__sample_weight=weights[train_index],
        )
        probability = pipeline.predict_proba(x.iloc[valid_index])[:, 1]
        folds.append(
            {
                "fold": fold,
                "train_rows": int(len(train_index)),
                "valid_rows": int(len(valid_index)),
                "pr_auc": float(average_precision_score(y.iloc[valid_index], probability)),
                "roc_auc": float(roc_auc_score(y.iloc[valid_index], probability)),
            }
        )
    pr = np.array([row["pr_auc"] for row in folds])
    roc = np.array([row["roc_auc"] for row in folds])
    return {
        "method": "TimeSeriesSplit",
        "n_splits": n_splits,
        "rows_used": int(len(x)),
        "folds": folds,
        "pr_auc_mean": float(pr.mean()),
        "pr_auc_std": float(pr.std()),
        "roc_auc_mean": float(roc.mean()),
        "roc_auc_std": float(roc.std()),
    }


# ---------------------------------------------------------------- reconcile


def load_production(uri: str = MODEL_URI) -> TriageClassifier:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(uri)


def reconcile(metrics: dict[str, float], tolerance: float = 1e-4) -> list[dict]:
    """Compare recomputed test metrics with what training logged for production."""
    import mlflow

    try:
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        logged = client.get_run(version.run_id).data.metrics
    except Exception as error:  # offline CI must not fail on a missing registry
        return [{"key": "mlflow", "note": f"skipped: {error}"}]
    rows = []
    for key, value in metrics.items():
        if not key.startswith("test_") or key not in logged:
            continue
        rows.append(
            {
                "key": key,
                "recomputed": float(value),
                "logged": float(logged[key]),
                "delta": float(value - logged[key]),
                "match": abs(value - logged[key]) <= tolerance,
            }
        )
    return rows


# ---------------------------------------------------------------- plots


def plot_reliability(
    y_true: pd.Series, raw: np.ndarray, calibrated: np.ndarray, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5.2, 5.2))
    axes.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1, label="Perfect")
    for probability, label, color in (
        (raw, "Raw", "#4B3FE3"),
        (calibrated, "Calibrated", "#27D2BF"),
    ):
        confidence, observed = reliability_points(y_true, probability)
        axes.plot(confidence, observed, "o-", color=color, linewidth=2, label=label)
    axes.set_xlabel("Predicted probability")
    axes.set_ylabel("Observed positive rate")
    axes.set_title("Reliability diagram (test)")
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_sweep(points: list[SweepPoint], current: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cutoffs = [point.cutoff for point in points]
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    axes.plot(
        cutoffs,
        [point.p3_share for point in points],
        color="#4B3FE3",
        label="Routine share",
    )
    axes.plot(
        cutoffs,
        [point.undertriage_life for point in points],
        color="#F87454",
        label="Life-threatening under-triage",
    )
    axes.axvline(current, color="0.45", linestyle="--", linewidth=1)
    axes.set_xlabel("Lower cut-off (routine / urgent)")
    axes.set_ylabel("Share")
    axes.set_title("Tier boundary trade-off (validation)")
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- orchestration


def evaluate(
    *, skip_cv: bool = False, skip_drift: bool = False, max_rows: int | None = None
) -> dict[str, object]:
    """Run every evaluation block and write ``artifacts/evaluation``.

    The function is the controller: each block is a pure function that can be
    tested without MLflow, SQLite or the real model.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    outcomes = load_outcomes()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]
    outcomes_test = outcomes["test"]
    if max_rows:
        x_test, y_test = x_test.iloc[:max_rows], y_test.iloc[:max_rows]
        outcomes_test = outcomes_test.iloc[:max_rows]

    model = load_production()
    cutoffs = tuple(float(value) for value in model.cutoffs)
    raw_val = model.predict_proba(x_val)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]

    report: dict[str, object] = {"cutoffs": list(cutoffs)}

    # (1) independent recomputation and reconciliation
    metrics = classification_metrics(y_test, raw_test, "test")
    metrics["test_ece"] = expected_calibration_error(y_test, raw_test)
    metrics.update(triage_metrics(outcomes_test, raw_test, cutoffs, "test"))
    report["metrics"] = metrics
    report["acceptance"] = acceptance(metrics, prefix="test")
    report["reconciliation"] = reconcile(metrics)

    # (2) calibration fitted on validation, chosen on validation
    candidates: dict[str, dict[str, object]] = {}
    fitted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in ("isotonic", "sigmoid"):
        calibrator = fit_calibrator(model, x_val, y_val, method)
        val_probability = calibrator.predict_proba(x_val)[:, 1]
        test_probability = calibrator.predict_proba(x_test)[:, 1]
        fitted[method] = (val_probability, test_probability)
        candidates[method] = {
            "val_brier": float(brier_score_loss(y_val, val_probability)),
            "val_log_loss": float(log_loss(y_val, val_probability)),
            "test_brier": float(brier_score_loss(y_test, test_probability)),
            "test_log_loss": float(log_loss(y_test, test_probability)),
            "test_ece": expected_calibration_error(y_test, test_probability),
            "test_pr_auc": float(average_precision_score(y_test, test_probability)),
            "test_roc_auc": float(roc_auc_score(y_test, test_probability)),
        }
    chosen = min(candidates, key=lambda name: float(candidates[name]["val_log_loss"]))
    calibrated_val, calibrated_test = fitted[chosen]
    remapped = remap_cutoffs(cutoffs, raw_val, calibrated_val)
    calibrated_metrics = triage_metrics(outcomes_test, calibrated_test, remapped, "test")
    report["calibration"] = {
        "baseline": {
            "val_brier": float(brier_score_loss(y_val, raw_val)),
            "val_log_loss": float(log_loss(y_val, raw_val)),
            "test_brier": metrics["test_brier"],
            "test_log_loss": metrics["test_log_loss"],
            "test_ece": metrics["test_ece"],
        },
        "candidates": candidates,
        "chosen": chosen,
        "remapped_cutoffs": list(remapped),
        "acceptance_after": acceptance(calibrated_metrics, "test"),
        "tier_share_shift": {
            key: float(calibrated_metrics[f"test_{key}"] - metrics[f"test_{key}"])
            for key in ("p1_share", "p2_share", "p3_share", "undertriage_life")
        },
    }
    plot_reliability(y_test, raw_test, calibrated_test, OUT / "reliability_test.png")

    # (3) tier audit on validation, stability check on test
    sweep = sweep_lower_cutoff(outcomes["val"], raw_val, cutoffs[1])
    report["sweep"] = [asdict(point) for point in sweep]
    report["sensitivity"] = sensitivity_analysis(outcomes_test, raw_test, cutoffs)
    if sweep:
        plot_sweep(sweep, cutoffs[0], OUT / "cutoff_tradeoff_val.png")

    # (4) subgroups and worst cases
    segments = segment_report(x_test, y_test, raw_test, outcomes_test, cutoffs)
    segments.to_csv(OUT / "segments_test.csv", index=False, encoding="utf-8-sig")
    report["segments"] = segments.to_dict(orient="records")
    errors = error_cases(x_test, raw_test, outcomes_test, cutoffs)
    errors.to_csv(OUT / "error_cases_test.csv", index=False, encoding="utf-8-sig")
    report["error_cases"] = errors.head(10).to_dict(orient="records")

    # (5) ordered cross-validation
    if not skip_cv:
        report["cv"] = time_series_cv(
            x_train,
            y_train,
            training_weights(outcomes["train"]),
            estimator=model.pipeline[-1],
        )

    # (6) drift, reference = training, current = test
    if not skip_drift:
        from src.explain import drift_report

        report["drift"] = drift_report(x_train, x_test, OUT / "drift_report")

    (OUT / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cv", action="store_true", help="skip time-series CV")
    parser.add_argument("--skip-drift", action="store_true", help="skip drift report")
    parser.add_argument("--max-rows", type=int, help="use only the first N test rows")
    args = parser.parse_args()
    report = evaluate(
        skip_cv=args.skip_cv,
        skip_drift=args.skip_drift,
        max_rows=args.max_rows,
    )
    print("EVALUATE_COMPLETE", json.dumps(report["acceptance"], ensure_ascii=False))
    print("ARTIFACTS", str(OUT), flush=True)


if __name__ == "__main__":
    main()
