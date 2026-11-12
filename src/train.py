"""Baselines, Optuna tuning, MLflow tracking and Model Registry (member B).

Run from the repository root after ``python -m src.database``::

    python -m src.train                          # 4 baselines + 20 trials each
    python -m src.train --trials 5 --models lr lgbm
    python -m src.train --from-run RUN_ID        # rebuild a final run, no search

Every model is selected on the validation split only. The registered artifact,
``models:/road-severity-model@production``, is a ``src.triage.TriageClassifier``:
``predict`` returns the dispatch priority (1 immediate, 2 urgent, 3 routine),
``predict_proba`` the severity probability, and ``model.pipeline`` the complete
fitted Pipeline (A's features + preprocessing + estimator).
"""

import argparse
import ast
import json
import os

import mlflow
import numpy as np
import optuna
import pandas as pd
import skops.io
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.database import load_outcomes, load_splits
from src.features import INFO_GROUPS, ROOT, SCHEMA, build_pipeline
from src.triage import (
    TriageClassifier,
    acceptance,
    training_weights,
    triage_cutoffs,
    triage_metrics,
)

SEED = SCHEMA["split"]["random_seed"]
EXPERIMENT = "road-severity"
MODEL_NAME = "road-severity-model"
MODEL_ALIAS = "production"
# Threshold-free and minority-sensitive; accuracy is misleading at ~25% positives.
SELECTION_METRIC = "val_pr_auc"
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
# Share of training rows repeated with information groups blanked, so the model
# degrades gracefully instead of reading "not reported" as a typical collision.
MASK_FRACTION = 0.5
MISSING_SCENARIOS = {
    **INFO_GROUPS,
    "vehicle_casualty": INFO_GROUPS["vehicle"] + INFO_GROUPS["casualty"],
}

# Fixed reference settings; tree settings follow the attached severity paper.
BASELINES = {
    "lr": {"C": 1.0},
    "rf": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"},
    "xgb": {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "lgbm": {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.05},
}


def build_estimator(name, params):
    """Imbalance is handled by the severity-grade sample weights passed to fit."""
    if name == "lr":
        return LogisticRegression(**params, max_iter=2000)
    if name == "rf":
        return RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)
    if name == "xgb":
        return XGBClassifier(**params, tree_method="hist", n_jobs=-1, random_state=SEED)
    if name == "lgbm":
        return LGBMClassifier(**params, n_jobs=-1, random_state=SEED, verbose=-1)
    raise ValueError(f"Unknown model family: {name}")


def suggest_params(name, trial):
    if name == "lr":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.05, 0.5),
        }
    boosting = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    if name == "xgb":
        return {
            **boosting,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 30, log=True
            ),
        }
    if name == "lgbm":
        return {
            **boosting,
            "subsample_freq": 1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 300, log=True
            ),
        }
    raise ValueError(f"Unknown model family: {name}")


def mask_information(frame, rng):
    """Copy rows with at least one information group blanked, as at a partial scene."""
    groups = list(INFO_GROUPS.values())
    hide = rng.random((len(frame), len(groups))) < 0.5
    untouched = ~hide.any(axis=1)
    hide[untouched, rng.integers(len(groups), size=untouched.sum())] = True
    masked = frame.copy()
    for rows, columns in zip(hide.T, groups):
        masked.loc[rows, columns] = np.nan
    return masked


def classification_metrics(y_true, probability, prefix):
    """Ranking, thresholded (0.5) and probability-quality metrics for one split."""
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


class Trainer:
    """Holds the train-only fitted preprocessing so every candidate reuses it."""

    def __init__(self, splits, outcomes):
        self.splits = splits
        self.outcomes = outcomes
        x_train, self.y_train = splits["train"]
        x_val, self.y_val = splits["val"]
        # Medians, scaling and categories come from the real rows only.
        self.preprocess = build_pipeline().fit(x_train, self.y_train)
        extra = x_train.sample(frac=MASK_FRACTION, random_state=SEED)
        masked = mask_information(extra, np.random.default_rng(SEED))
        weights = pd.Series(training_weights(outcomes["train"]), index=x_train.index)
        self.weights = pd.concat([weights, weights.loc[extra.index]]).to_numpy()
        self.y_train = pd.concat([self.y_train, self.y_train.loc[extra.index]])
        self.m_train = self.preprocess.transform(pd.concat([x_train, masked]))
        self.m_val = self.preprocess.transform(x_val)
        report = json.loads(
            (ROOT / "data/split_report.json").read_text(encoding="utf-8")
        )
        self.tags = {
            "schema_version": SCHEMA["schema_version"],
            "split_method": report["method"]["method"],
            "train_sha256": report["splits"]["train"]["sha256"],
            "imbalance_strategy": "severity_grade_sample_weight",
            "masked_training_fraction": str(MASK_FRACTION),
        }

    def fit(self, name, params):
        estimator = build_estimator(name, params)
        # sklearn's sparse tree splitter is ~10x slower on this mostly-dense matrix;
        # a forest grown on the dense copy still predicts on the pipeline's sparse output.
        matrix = self.m_train.toarray() if name == "rf" else self.m_train
        return estimator.fit(matrix, self.y_train, sample_weight=self.weights)

    def run(self, name, params, stage, *, nested=False):
        """Fit one candidate, log it to MLflow and return its validation record."""
        with mlflow.start_run(run_name=f"{stage}-{name}", nested=nested):
            estimator = self.fit(name, params)
            metrics = classification_metrics(
                self.y_val, estimator.predict_proba(self.m_val)[:, 1], "val"
            )
            mlflow.set_tags({**self.tags, "stage": stage, "model_family": name})
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        return {"model": name, "stage": stage, "params": params, **metrics}

    def tune(self, name, trials):
        """One parent run per family, one nested run per Optuna trial."""
        records = []

        def objective(trial):
            record = self.run(name, suggest_params(name, trial), "trial", nested=True)
            records.append(record)
            return record[SELECTION_METRIC]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            study_name=f"{EXPERIMENT}-{name}",
        )
        with mlflow.start_run(run_name=f"optuna-{name}"):
            mlflow.set_tags({**self.tags, "stage": "optuna", "model_family": name})
            study.optimize(objective, n_trials=trials)
            best = max(records, key=lambda record: record[SELECTION_METRIC])
            mlflow.log_params(best["params"])
            mlflow.log_metric(f"best_{SELECTION_METRIC}", study.best_value)
            mlflow.log_metric("n_trials", len(study.trials))
            mlflow.log_table(study.trials_dataframe(), "optuna_trials.json")
        return {**best, "stage": "tuned"}

    def register(self, best, leaderboard, *, register=True):
        """Refit the winner, set its priority tiers, log it and move the alias."""
        name, params = best["model"], best["params"]
        pipeline = Pipeline([*self.preprocess.steps, ("model", self.fit(name, params))])
        x_val, _ = self.splits["val"]
        x_test, y_test = self.splits["test"]
        val_probability = pipeline.predict_proba(x_val)[:, 1]
        # Tiers are cut on validation only; test shows how well they transfer.
        cutoffs = triage_cutoffs(self.outcomes["val"], val_probability)
        model = TriageClassifier(pipeline, cutoffs)
        metrics = {
            "cutoff_routine_urgent": cutoffs[0],
            "cutoff_urgent_immediate": cutoffs[1],
            **classification_metrics(self.y_val, val_probability, "val"),
            **triage_metrics(self.outcomes["val"], val_probability, cutoffs, "val"),
        }
        # Also with each information source absent, as at a partially reported scene.
        scenarios = {
            "test": [],
            **{f"test_no_{k}": v for k, v in MISSING_SCENARIOS.items()},
        }
        for prefix, columns in scenarios.items():
            frame = x_test.copy()
            frame[columns] = np.nan
            probability = pipeline.predict_proba(frame)[:, 1]
            metrics.update(classification_metrics(y_test, probability, prefix))
            metrics.update(
                triage_metrics(self.outcomes["test"], probability, cutoffs, prefix)
            )
        checks = acceptance(metrics)
        with mlflow.start_run(run_name=f"final-{name}") as run:
            mlflow.set_tags(
                {
                    **self.tags,
                    "stage": "final",
                    "model_family": name,
                    "selected_from": best["stage"],
                    "selection_metric": SELECTION_METRIC,
                    "fit_population": "train_only",
                    "acceptance_passed": str(all(checks.values())),
                    **{f"acceptance_{key}": str(ok) for key, ok in checks.items()},
                }
            )
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_table(leaderboard, "leaderboard.json")
            mlflow.log_artifact(str(ROOT / "data/split_report.json"))
            mlflow.log_artifact(str(ROOT / "schema.json"))
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                # No MLflow signature: its datetime type rejects the tz-aware UTC
                # column, and schema.json (logged above) is the input contract.
                # skops refuses unknown classes unless the saver vouches for them.
                skops_trusted_types=skops.io.get_untrusted_types(
                    data=skops.io.dumps(model)
                ),
                registered_model_name=MODEL_NAME if register else None,
            )
            if register:
                client, version = mlflow.MlflowClient(), info.registered_model_version
                client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
                for key, value in {
                    "triage_cutoffs": json.dumps(cutoffs),
                    "acceptance_passed": str(all(checks.values())),
                }.items():
                    client.set_model_version_tag(MODEL_NAME, version, key, value)
        return {
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "acceptance": checks,
            **metrics,
        }


def previous_selection(run_id):
    """Winner and leaderboard of an earlier final run, to rebuild it without a search."""
    run = mlflow.get_run(run_id)

    def parse(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    best = {
        "model": run.data.tags["model_family"],
        "stage": run.data.tags["selected_from"],
        "params": {key: parse(value) for key, value in run.data.params.items()},
    }
    return best, mlflow.load_table("leaderboard.json", run_ids=[run_id])


def search(trainer, models, trials):
    """Baselines and Optuna studies; returns the validation winner and all records."""
    records = []
    for name in models:
        records.append(trainer.run(name, BASELINES[name], "baseline"))
        if trials > 0:
            records.append(trainer.tune(name, trials))
    leaderboard = (
        pd.DataFrame(records)
        .drop(columns="params")
        .sort_values(SELECTION_METRIC, ascending=False, ignore_index=True)
    )
    return max(records, key=lambda record: record[SELECTION_METRIC]), leaderboard


def train(models, trials, *, register=True, from_run=None):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    trainer = Trainer(load_splits(), load_outcomes())
    if from_run:
        best, leaderboard = previous_selection(from_run)
        # The reused leaderboard was scored under that run's settings, not these.
        trainer.tags["rebuilt_from_run"] = from_run
    else:
        best, leaderboard = search(trainer, models, trials)
    final = trainer.register(best, leaderboard, register=register)
    print(leaderboard.round(4).to_string(index=False), flush=True)
    print(
        "TRAIN_COMPLETE",
        json.dumps({"model": best["model"], "params": best["params"], **final}),
        flush=True,
    )
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=list(BASELINES), default=list(BASELINES)
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Optuna trials per model"
    )
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--from-run", help="MLflow run id of a final run to rebuild without searching"
    )
    args = parser.parse_args()
    train(
        args.models,
        args.trials,
        register=not args.no_register,
        from_run=args.from_run,
    )


if __name__ == "__main__":
    main()
