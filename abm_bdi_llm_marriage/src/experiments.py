"""Experiment runner: runs all three scenarios with multiple replications."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import ModelConfig, ScenarioConfig
from src.config import SCENARIOS
from src.model import MarriageFertilityABM

logger = logging.getLogger(__name__)


def run_single_replication(
    config: ModelConfig,
    scenario: ScenarioConfig,
    replication: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Run one replication of the model and return metrics + agent snapshot.

    Args:
        config: Model configuration.
        scenario: AI impact scenario.
        replication: Replication index (for logging).
        seed: Random seed for this replication.

    Returns:
        Tuple of (tick_metrics, agent_snapshot).
    """
    logger.info(
        "Starting %s | replication %d | seed %d",
        scenario.name, replication, seed,
    )
    model = MarriageFertilityABM(config, scenario, seed=seed)
    tick_metrics = model.run()
    agent_snapshot = model.get_agent_snapshot()

    # Tag each record
    for m in tick_metrics:
        m["replication"] = replication
    for r in agent_snapshot:
        r["replication"] = replication

    logger.info(
        "Finished %s | replication %d | "
        "first_marriage_rate=%.3f | avg_children=%.3f",
        scenario.name,
        replication,
        tick_metrics[-1].get("first_marriage_rate_18_45", 0.0),
        tick_metrics[-1].get("avg_children_adults", 0.0),
    )
    return tick_metrics, agent_snapshot


def run_scenario_experiments(
    config: ModelConfig,
    scenario_names: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Run low / medium / high AI scenarios with multiple replications.

    Args:
        config: Model configuration (controls population, years, replications).
        scenario_names: Subset of scenario names to run; None → all three.
        output_dir: Where to save CSV files; defaults to config.output_path.

    Returns:
        Dictionary with DataFrames:
        - "long": per-tick metrics for every replication
        - "summary": scenario-level mean ± std of final-tick metrics
        - "agent_snapshot": agent-level snapshot from the last replication
        - "tick_metrics": alias of "long"
    """
    if scenario_names is None:
        scenario_names = list(SCENARIOS.keys())

    out = output_dir or config.output_path
    out.mkdir(parents=True, exist_ok=True)

    all_ticks: list[dict] = []
    all_snapshots: list[dict] = []

    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]

        for rep in range(config.replications):
            seed = config.seed + rep * 1000 + hash(scenario_name) % 1000
            tick_metrics, agent_snapshot = run_single_replication(
                config, scenario, rep, seed
            )
            all_ticks.extend(tick_metrics)
            all_snapshots.extend(agent_snapshot)

    df_long = pd.DataFrame(all_ticks)
    df_agent = pd.DataFrame(all_snapshots)

    # Summary: take the last tick of each replication per scenario
    last_tick_idx = df_long.groupby(["scenario", "replication"])["tick"].idxmax()
    df_last = df_long.loc[last_tick_idx]

    numeric_cols = df_last.select_dtypes(include="number").columns.tolist()
    summary_records = []
    for scenario_name in scenario_names:
        sub = df_last[df_last["scenario"] == scenario_name][numeric_cols]
        row = {"scenario": scenario_name}
        for col in numeric_cols:
            if col in ("tick", "replication"):
                continue
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()
        summary_records.append(row)

    df_summary = pd.DataFrame(summary_records)

    # Save to CSV
    long_path = out / "scenario_results_long.csv"
    summary_path = out / "scenario_results_summary.csv"
    agent_path = out / "agent_snapshot_final.csv"
    tick_path = out / "tick_metrics.csv"

    df_long.to_csv(long_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_agent.to_csv(agent_path, index=False)
    df_long.to_csv(tick_path, index=False)  # alias

    logger.info("Results saved to %s", out)

    return {
        "long": df_long,
        "summary": df_summary,
        "agent_snapshot": df_agent,
        "tick_metrics": df_long,
    }
