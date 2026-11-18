"""SHAP explanations and drift detection (member C).

Two audiences, kept in separate sections below:

* ``explain_single`` is the only function member D's web front end calls. It
  takes one raw record (``schema.json`` column names) and returns a typed
  result: the contributions are in log-odds, which is exactly additive, and
  ``probability`` is what the model reports.
* ``global_shap`` / ``drift_report`` are report artefacts and are never called
  from the API.
"""

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from src.features import (
    INFO_GROUPS,
    INPUT_COLUMNS,
    NUMERIC_FEATURES,
    ROOT,
    complete_inputs,
)
from src.registry import load_registered_model
from src.triage import TriageClassifier, priorities

# Longest first, so "age_of_casualty" is not swallowed by a hypothetical "age".
SCHEMA_COLUMNS = sorted(
    set(INPUT_COLUMNS + NUMERIC_FEATURES),
    key=len,
    reverse=True,
)


# ---------------------------------------------------------------- public interface for D


@dataclass(frozen=True)
class FeatureContribution:
    """One raw field's push on the log-odds; positive means "more severe"."""

    feature: str
    value: str
    impact: float


@dataclass(frozen=True)
class SingleExplanation:
    """Everything the waterfall needs, already aggregated to raw field names."""

    base_value: float
    probability: float
    priority: int
    contributions: list[FeatureContribution]
    other_impact: float
    cutoffs: tuple[float, float] = (0.0, 1.0)


@lru_cache(maxsize=2)
def _explainer(model: TriageClassifier):
    """TreeExplainer construction takes a second; cache it per model instance."""
    import shap

    return shap.TreeExplainer(model.pipeline[-1])


def _tree_contributions(model, frame):
    """Explain the same encoded observations the prediction pipeline scores."""
    from xgboost import XGBClassifier

    preprocess = model.pipeline[:-1]
    encoded = preprocess.transform(frame)
    estimator = model.pipeline[-1]
    # An absent entry in XGBoost's CSR input follows the missing-value branch.
    # LightGBM's SHAP backend returns dense contributions for dense input.
    matrix = encoded
    if sparse.issparse(encoded) and not isinstance(estimator, XGBClassifier):
        matrix = encoded.toarray()
    explainer = _explainer(model)
    raw = explainer.shap_values(matrix)
    values = np.asarray(raw[1] if isinstance(raw, list) else raw)
    if values.ndim == 3:
        values = values[..., 1]
    base = float(np.ravel(explainer.expected_value)[-1])
    probability = estimator.predict_proba(encoded)[:, 1]
    if not np.allclose(
        expit(base + values.sum(axis=1)), probability, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("SHAP log-odds do not reproduce the pipeline probabilities")
    return matrix, values, base, probability


def collapse_to_source(
    feature_names, shap_row: np.ndarray, schema_columns=None
) -> dict[str, float]:
    """Sum one-hot columns back into their raw field.

    Without this the front end would show ``cat__vehicle_type_19`` instead of
    "车辆类型".
    """
    # Longest first no matter what order the caller passed in.
    columns = sorted(schema_columns or SCHEMA_COLUMNS, key=len, reverse=True)
    source: dict[str, float] = {}
    for name, value in zip(feature_names, shap_row):
        raw = str(name).split("__", 1)[-1].removeprefix("missingindicator_")
        match = next(
            (
                column
                for column in columns
                if raw == column or raw.startswith(f"{column}_")
            ),
            raw,
        )
        source[match] = source.get(match, 0.0) + float(value)
    return source


def explain_single(
    row: pd.DataFrame,
    model: TriageClassifier | None = None,
    top_n: int = 12,
) -> SingleExplanation:
    """Explain one collision record.

    ``row`` holds one row of raw inputs using the ``schema.json`` column names;
    ``base_value``, ``impact`` and ``other_impact`` are log-odds, and
    ``base + sum(impacts) + other_impact`` equals the model margin.
    """
    if len(row) != 1:
        raise ValueError("explain_single expects exactly one row.")
    if model is None:
        model, _ = load_registered_model()
    frame = complete_inputs(row)
    preprocess = model.pipeline[:-1]
    _, values, base, predicted = _tree_contributions(model, frame)
    shap_row = values[0]
    probability = float(predicted[0])
    priority = int(priorities(np.array([probability]), model.cutoffs)[0])

    aggregated = collapse_to_source(preprocess.get_feature_names_out(), shap_row)
    ordered = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)
    display = frame.iloc[0]

    contributions = [
        FeatureContribution(
            feature=name,
            value=_display(display.get(name)) if name in frame.columns else "派生特征",
            impact=float(impact),
        )
        for name, impact in ordered[:top_n]
    ]
    return SingleExplanation(
        base_value=base,
        probability=probability,
        priority=priority,
        contributions=contributions,
        other_impact=float(sum(impact for _, impact in ordered[top_n:])),
        cutoffs=(float(model.cutoffs[0]), float(model.cutoffs[1])),
    )


def _display(value) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


# ---------------------------------------------------------------- report artefacts


def global_shap(
    x: pd.DataFrame,
    model: TriageClassifier | None = None,
    sample_rows: int = 2000,
    out_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Beeswarm, bar chart and importance aggregated to seven information groups."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if model is None:
        model, _ = load_registered_model()
    out_dir = Path(out_dir or (ROOT / "artifacts/evaluation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(x) > sample_rows:
        x = x.sample(sample_rows, random_state=seed)

    preprocess = model.pipeline[:-1]
    matrix, values, _, _ = _tree_contributions(model, complete_inputs(x))
    if sparse.issparse(matrix):
        # Missing entries remain visibly missing in the XGBoost beeswarm.
        entries = matrix.tocoo()
        matrix = np.full(entries.shape, np.nan, dtype=float)
        matrix[entries.row, entries.col] = entries.data
    names = np.asarray(preprocess.get_feature_names_out())

    for plot_type, filename in (
        ("dot", "shap_beeswarm.png"),
        ("bar", "shap_importance.png"),
    ):
        shap.summary_plot(
            values,
            matrix,
            feature_names=names,
            plot_type=plot_type,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()

    importance = pd.Series(np.abs(values).mean(axis=0), index=names)
    groups: dict[str, float] = {}
    for name, value in importance.items():
        group = _information_group(str(name).split("__", 1)[-1])
        groups[group] = groups.get(group, 0.0) + float(value)
    total = sum(groups.values()) or 1.0
    summary = {
        key: round(value / total, 4)
        for key, value in sorted(groups.items(), key=lambda item: -item[1])
    }
    (out_dir / "shap_group_importance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "groups": summary,
        "top_features": importance.sort_values(ascending=False)
        .head(20)
        .round(4)
        .to_dict(),
    }


def _information_group(raw: str) -> str:
    raw = raw.removeprefix("missingindicator_")
    if raw.startswith("veh_"):
        return "vehicle"
    if raw.startswith("cas_"):
        return "casualty"
    if raw in INFO_GROUPS["weather"] or raw in (
        "rain_present",
        "wind_east_kmh",
        "wind_north_kmh",
    ):
        return "weather"
    if any(
        raw == column or raw.startswith(f"{column}_") for column in INFO_GROUPS["road"]
    ):
        return "road"
    if raw in ("latitude", "longitude"):
        return "location"
    if any(
        token in raw
        for token in ("hour", "weekday", "weekend", "month", "sun", "solar")
    ):
        return "time"
    return "other"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: Path | None = None,
) -> dict[str, object]:
    """``DataDriftPreset`` with training as reference and test as current.

    Drift is expected because the split is temporal; the question is which
    fields moved and whether that undermines the evaluation.
    """
    out_path = Path(out_path or (ROOT / "artifacts/evaluation/drift_report"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_path.with_suffix(".html")

    try:  # evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report(metrics=[DataDriftPreset()]).run(
            reference_data=reference, current_data=current
        )
        snapshot.save_html(str(html_path))
        return {"ok": True, "html": str(html_path)}
    except ImportError:
        pass
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        return _psi_fallback(reference, current, out_path, error)

    try:  # evidently <= 0.6
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(html_path))
        return {"ok": True, "html": str(html_path), "api": "legacy"}
    except (ImportError, TypeError, ValueError, RuntimeError, KeyError) as error:
        return _psi_fallback(reference, current, out_path, error)


def main() -> None:
    """Write the global SHAP artefacts the evaluation report refers to."""
    from src.database import load_splits

    parser = argparse.ArgumentParser(description="Generate global SHAP artefacts")
    parser.add_argument(
        "--model-version", help="pin a registered version (default: production alias)"
    )
    args = parser.parse_args()
    model, entry = load_registered_model(args.model_version)
    x_test, _ = load_splits()["test"]
    result = global_shap(x_test, model=model)
    print("MODEL", entry.name, entry.version, entry.run_id)
    print("EXPLAIN_COMPLETE", json.dumps(result["groups"], ensure_ascii=False))
    print("ARTIFACTS", str(ROOT / "artifacts/evaluation"), flush=True)


def _psi_fallback(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: Path,
    error: Exception,
    bins: int = 10,
) -> dict[str, object]:
    """Population stability index per numeric field when Evidently is unavailable."""
    rows = []
    for column in reference.select_dtypes(include="number").columns:
        ref = reference[column].dropna().to_numpy(dtype=float)
        cur = current[column].dropna().to_numpy(dtype=float)
        ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
        if len(ref) < 50 or len(cur) < 50:
            continue
        interior = np.quantile(ref, np.arange(1, bins) / bins)
        # Keep a point mass at the reference minimum in its own bucket.
        interior[interior == ref.min()] = np.nextafter(ref.min(), np.inf)
        if ref.min() == ref.max():
            interior = [ref[0], np.nextafter(ref[0], np.inf)]
        edges = np.r_[-np.inf, np.unique(interior), np.inf]
        ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
        cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
        ref_share = np.clip(ref_share, 1e-6, None)
        cur_share = np.clip(cur_share, 1e-6, None)
        psi = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
        rows.append({"feature": column, "psi": psi, "drifted": psi > 0.2})
    table = pd.DataFrame(rows, columns=["feature", "psi", "drifted"]).sort_values(
        "psi", ascending=False
    )
    table.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return {
        "ok": False,
        "fallback": "psi",
        "reason": str(error),
        "status": "computed" if len(table) else "insufficient_data",
        "evaluated_columns": len(table),
        "drifted_share": float(table["drifted"].mean()) if len(table) else None,
        "top": table.head(15).to_dict(orient="records"),
    }


if __name__ == "__main__":
    main()
