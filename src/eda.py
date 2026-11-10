"""Seven reproducible matplotlib figures; label exploration uses training rows only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import CATEGORICAL_INPUTS, RoadFeatureEngineer

BLUE, RED, LIGHT = "#24618E", "#C15A46", "#D8E5ED"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.size": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def grouped_rates(frame, grouping):
    groups = frame.groupby(grouping, observed=True, dropna=True).is_severe.agg(
        ["size", "sum", "mean"]
    )
    z = 1.96
    n, p = groups["size"], groups["mean"]
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    groups["lower"] = middle - half
    groups["upper"] = middle + half
    return groups


def _rate_chart(axis, table, title, xlabel):
    x = np.arange(len(table))
    errors = np.vstack([table["mean"] - table.lower, table.upper - table["mean"]]) * 100
    axis.bar(x, table["mean"] * 100, color=BLUE, width=0.7, yerr=errors, capsize=3)
    axis.set_xticks(x, [str(value) for value in table.index])
    axis.set(xlabel=xlabel, ylabel="Fatal or serious (%)", title=title)
    axis.grid(axis="y", alpha=0.16)


def generate_eda(data_dir=None):
    data_dir = (
        Path(data_dir)
        if data_dir is not None
        else Path(__file__).resolve().parents[1] / "data"
    )
    output = data_dir / "eda_figures"
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        data_dir / "splits/train.csv",
        dtype={**{c: object for c in CATEGORICAL_INPUTS}, "collision_index": str},
        low_memory=False,
    )
    train["collision_datetime_utc"] = pd.to_datetime(
        train.collision_datetime_utc, utc=True
    )
    local = train.collision_datetime_utc.dt.tz_convert("Europe/London")
    engineered = RoadFeatureEngineer().fit_transform(train)
    manifest = []

    def save(figure, number, name, description):
        path = output / f"{number:02d}_{name}.png"
        figure.tight_layout(pad=1.4)
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        manifest.append(
            {"file": path.name, "title": description, "population": "train_only"}
        )

    counts = train.is_severe.value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        ["Slight (0)", "Fatal or serious (1)"], counts, color=[BLUE, RED], width=0.6
    )
    ax.bar_label(
        bars, labels=[f"{n:,} ({n / len(train):.1%})" for n in counts], padding=6
    )
    ax.set(
        ylim=(0, counts.max() * 1.2),
        ylabel="Collisions",
        title="Severity balance in the training split",
    )
    save(fig, 1, "severity_balance", "训练集严重程度分布")

    monthly = (
        train.assign(month=local.dt.strftime("%Y-%m"))
        .groupby("month")
        .is_severe.agg(["size", "mean"])
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = np.arange(len(monthly))
    axes[0].bar(x, monthly["size"], color=BLUE)
    axes[0].set(ylabel="Collisions", title="Monthly coverage of the training split")
    axes[1].plot(x, monthly["mean"] * 100, color=RED, marker="o", markersize=3)
    axes[1].set(ylabel="Fatal or serious (%)", xlabel="Local calendar month")
    axes[1].set_xticks(x, monthly.index, rotation=45, ha="right")
    save(fig, 2, "monthly_pattern", "训练集逐月数量与严重事故比例")

    hourly = grouped_rates(train.assign(hour=local.dt.hour), "hour")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(hourly.index, hourly["size"], color=BLUE)
    axes[0].set(ylabel="Collisions", title="Collision hour in Europe/London")
    _rate_chart(
        axes[1], hourly, "Severity rate with 95% Wilson intervals", "Local hour"
    )
    save(fig, 3, "hourly_pattern", "事故小时分布与严重比例")

    speed = grouped_rates(train, "speed_limit")
    speed.index = speed.index.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(np.arange(len(speed)), speed["size"], color=BLUE)
    axes[0].set_xticks(np.arange(len(speed)), speed.index)
    axes[0].set(
        title="Training collisions by speed limit",
        xlabel="Speed limit (mph)",
        ylabel="Collisions",
    )
    _rate_chart(axes[1], speed, "Severity rate by speed limit", "Speed limit (mph)")
    save(fig, 4, "speed_limit", "限速与严重事故比例")

    missing = (
        train[list(engineered.columns.intersection(train.columns))]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(missing.index[::-1], missing.to_numpy()[::-1] * 100, color=BLUE)
    ax.set(
        xlabel="Missing cells (%)",
        title="Training input missingness before fitted imputation",
    )
    ax.grid(axis="x", alpha=0.16)
    save(fig, 5, "missingness", "训练输入缺失情况")

    rain_bin = pd.cut(
        train.prcp, [-0.001, 0, 1, 5, np.inf], labels=["0", "(0, 1]", "(1, 5]", ">5"]
    )
    rain = grouped_rates(train.assign(rain_group=rain_bin), "rain_group")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    samples = [
        train.loc[train.is_severe.eq(label), "temp"].dropna().to_numpy()
        for label in [0, 1]
    ]
    axes[0].boxplot(
        samples,
        tick_labels=["Slight", "Fatal/serious"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": LIGHT, "edgecolor": BLUE},
        medianprops={"color": RED},
    )
    axes[0].set(title="Temperature by severity", ylabel="Temperature (°C)")
    _rate_chart(
        axes[1],
        rain,
        "Severity rate by observed rainfall",
        "Observed 1h precipitation (mm)",
    )
    save(fig, 6, "weather_distribution", "气温和小时降水与严重程度")

    phases = [
        "day",
        "civil_twilight",
        "nautical_twilight",
        "astronomical_twilight",
        "night",
    ]
    phase = (
        grouped_rates(train.assign(solar_phase=engineered.solar_phase), "solar_phase")
        .reindex(phases)
        .dropna()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(np.arange(len(phase)), phase["size"], color=BLUE)
    phase_labels = {
        "day": "Day",
        "civil_twilight": "Civil",
        "nautical_twilight": "Nautical",
        "astronomical_twilight": "Astron.",
        "night": "Night",
    }
    axes[0].set_xticks(
        np.arange(len(phase)), [phase_labels[p] for p in phase.index], fontsize=12
    )
    axes[0].set(title="Solar phase at collision coordinates", ylabel="Collisions")
    _rate_chart(axes[1], phase, "Severity rate by computed solar phase", "Solar phase")
    axes[1].set_xticklabels([phase_labels[p] for p in phase.index], fontsize=12)
    save(fig, 7, "solar_phase", "按事故坐标计算的昼夜与曙暮光阶段")

    insights = {
        "population": "train_only",
        "rows": len(train),
        "positive_rate": float(train.is_severe.mean()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "monthly": monthly.reset_index().to_dict("records"),
        "hourly": hourly.reset_index().to_dict("records"),
        "speed_limit": speed.reset_index().to_dict("records"),
        "rainfall": rain.reset_index().to_dict("records"),
        "solar_phase": phase.reset_index().to_dict("records"),
        "missingness": missing.to_dict(),
        "figures": manifest,
        "interpretation": "Observed associations within recorded injury collisions; no exposure denominator and no causal claim.",
    }
    (output / "insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"EDA_COMPLETE: {len(manifest)} figures using {len(train):,} training rows",
        flush=True,
    )
    return insights


if __name__ == "__main__":
    generate_eda()
