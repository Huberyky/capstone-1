"""Plotting module: generates comparison charts for the three AI scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_SCENARIO_COLORS = {
    "low_ai": "#4CAF50",
    "medium_ai": "#FF9800",
    "high_ai": "#F44336",
}
_SCENARIO_LABELS = {
    "low_ai": "Low AI",
    "medium_ai": "Medium AI",
    "high_ai": "High AI",
}


def _save(fig: plt.Figure, path: Path, filename: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fp = path / filename
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def _scenario_means(
    df_summary: pd.DataFrame,
    metric: str,
) -> tuple[list[str], list[float], list[float]]:
    """Extract per-scenario mean ± std for a metric from the summary DataFrame."""
    scenarios = ["low_ai", "medium_ai", "high_ai"]
    means, stds = [], []
    for s in scenarios:
        row = df_summary[df_summary["scenario"] == s]
        if row.empty:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(float(row[f"{metric}_mean"].values[0]))
            stds.append(float(row.get(f"{metric}_std", pd.Series([0.0])).values[0]))
    return scenarios, means, stds


def _bar_chart(
    df_summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
) -> None:
    """Generic bar chart comparing three scenarios."""
    scenarios, means, stds = _scenario_means(df_summary, metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [_SCENARIO_COLORS[s] for s in scenarios]
    labels = [_SCENARIO_LABELS[s] for s in scenarios]
    x = np.arange(len(scenarios))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    _save(fig, output_dir, filename)


def plot_first_marriage_rate(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: first marriage rate (18–45) across scenarios."""
    _bar_chart(
        df_summary,
        "first_marriage_rate_18_45",
        "First Marriage Rate (Age 18–45)",
        "Proportion ever married",
        output_dir,
        "first_marriage_rate.png",
    )


def plot_first_birth_rate(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: first birth rate (18–45) across scenarios."""
    _bar_chart(
        df_summary,
        "first_birth_rate_18_45",
        "First Birth Rate (Age 18–45)",
        "Proportion ever had a child",
        output_dir,
        "first_birth_rate.png",
    )


def plot_avg_children(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: average number of children across scenarios."""
    _bar_chart(
        df_summary,
        "avg_children_adults",
        "Average Children per Adult (18–45)",
        "Mean number of children",
        output_dir,
        "avg_children.png",
    )


def plot_mechanism_variables(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: three mechanism variables across scenarios."""
    variables = [
        ("mean_employment_stability_pressure", "Employment\nStability Pressure"),
        ("mean_career_expectation_uncertainty", "Career Expectation\nUncertainty"),
        ("mean_relationship_time_compression", "Relationship Time\nCompression"),
    ]
    scenarios = ["low_ai", "medium_ai", "high_ai"]
    x = np.arange(len(variables))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, s in enumerate(scenarios):
        row = df_summary[df_summary["scenario"] == s]
        vals = []
        for var, _ in variables:
            if row.empty or f"{var}_mean" not in row.columns:
                vals.append(0.0)
            else:
                vals.append(float(row[f"{var}_mean"].values[0]))
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width,
            label=_SCENARIO_LABELS[s],
            color=_SCENARIO_COLORS[s],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([v[1] for v in variables], fontsize=10)
    ax.set_ylabel("Mean value (0–1)", fontsize=11)
    ax.set_title("Three Mechanism Variables by AI Scenario", fontsize=12)
    ax.legend(title="Scenario")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    _save(fig, output_dir, "mechanism_variables.png")


def plot_labour_situation_variables(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: labour situation variables across scenarios."""
    variables = [
        ("mean_task_restructuring_exposure", "Task\nRestructuring"),
        ("mean_entry_barrier_level", "Entry\nBarrier"),
        ("mean_employment_stability", "Employment\nStability"),
        ("mean_reskilling_need", "Reskilling\nNeed"),
        ("mean_career_predictability", "Career\nPredictability"),
        ("mean_work_life_boundary_blurring", "Work-Life\nBlurring"),
    ]
    scenarios = ["low_ai", "medium_ai", "high_ai"]
    x = np.arange(len(variables))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, s in enumerate(scenarios):
        row = df_summary[df_summary["scenario"] == s]
        vals = []
        for var, _ in variables:
            col = f"{var}_mean"
            if row.empty or col not in row.columns:
                vals.append(0.0)
            else:
                vals.append(float(row[col].values[0]))
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width,
            label=_SCENARIO_LABELS[s],
            color=_SCENARIO_COLORS[s],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([v[1] for v in variables], fontsize=9)
    ax.set_ylabel("Mean value (0–1)", fontsize=11)
    ax.set_title("Labour Situation Variables by AI Scenario", fontsize=12)
    ax.legend(title="Scenario")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    _save(fig, output_dir, "labour_situation_variables.png")


def plot_bdi_intention_distribution(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar chart: BDI intention distribution across scenarios."""
    intentions = [
        ("share_intention_search_partner", "Search Partner"),
        ("share_intention_marry_partner", "Marry Partner"),
        ("share_intention_postpone_marriage", "Postpone Marriage"),
        ("share_intention_have_child", "Have Child"),
        ("share_intention_postpone_childbearing", "Postpone Childbearing"),
        ("share_intention_stabilize_career_entry", "Stabilize Career Entry"),
        ("share_intention_reduce_career_uncertainty", "Reduce Career Uncertainty"),
        ("share_intention_protect_personal_time", "Protect Personal Time"),
    ]
    scenarios = ["low_ai", "medium_ai", "high_ai"]
    cmap = plt.cm.get_cmap("tab10", len(intentions))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scenarios))
    bottoms = np.zeros(len(scenarios))

    for i, (var, label) in enumerate(intentions):
        col = f"{var}_mean"
        vals = []
        for s in scenarios:
            row = df_summary[df_summary["scenario"] == s]
            if row.empty or col not in row.columns:
                vals.append(0.0)
            else:
                vals.append(float(row[col].values[0]))
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=label,
            color=cmap(i),
            edgecolor="white",
            linewidth=0.3,
        )
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([_SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel("Share of agents", fontsize=11)
    ax.set_title("BDI Intention Distribution by AI Scenario", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save(fig, output_dir, "bdi_intention_distribution.png")


def plot_education_assortative_mating(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar chart: education assortative mating across scenarios."""
    categories = [
        ("education_homogamy_share", "Homogamy"),
        ("female_hypergamy_share", "Female Hypergamy"),
        ("female_hypogamy_share", "Female Hypogamy"),
    ]
    scenarios = ["low_ai", "medium_ai", "high_ai"]
    colors = ["#2196F3", "#9C27B0", "#FF5722"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(scenarios))
    bottoms = np.zeros(len(scenarios))

    for (var, label), color in zip(categories, colors):
        col = f"{var}_mean"
        vals = []
        for s in scenarios:
            row = df_summary[df_summary["scenario"] == s]
            if row.empty or col not in row.columns:
                vals.append(0.0)
            else:
                vals.append(float(row[col].values[0]))
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([_SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel("Share of couples", fontsize=11)
    ax.set_title("Education Assortative Mating by AI Scenario", fontsize=12)
    ax.legend()
    _save(fig, output_dir, "education_assortative_mating.png")


def plot_all(
    df_summary: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate all plots and save to output_dir/figures/.

    Args:
        df_summary: Summary DataFrame from run_scenario_experiments.
        output_dir: Base output directory; figures go into output_dir/figures/.
    """
    if output_dir is None:
        output_dir = Path("outputs")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_first_marriage_rate(df_summary, figures_dir)
    plot_first_birth_rate(df_summary, figures_dir)
    plot_avg_children(df_summary, figures_dir)
    plot_mechanism_variables(df_summary, figures_dir)
    plot_labour_situation_variables(df_summary, figures_dir)
    plot_bdi_intention_distribution(df_summary, figures_dir)
    plot_education_assortative_mating(df_summary, figures_dir)
    print("All plots saved.\n")
