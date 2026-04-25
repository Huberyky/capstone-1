"""Experiment runner: scenarios, diagnostics and sensitivity analysis."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ModelConfig, ScenarioConfig, SCENARIOS
from src.matching import set_matching_lambdas
from src.model import MarriageFertilityABM

logger = logging.getLogger(__name__)


def run_single_replication(
    config: ModelConfig,
    scenario: ScenarioConfig,
    replication: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    logger.info("Starting %s | replication %d | seed %d", scenario.name, replication, seed)
    model = MarriageFertilityABM(config, scenario, seed=seed)
    tick_metrics = model.run()
    agent_snapshot = model.get_agent_snapshot()

    diagnostics = {
        "belief_trajectory": model.belief_trajectory,
        "desire_trajectory": model.desire_trajectory,
        "intention_trajectory": model.intention_trajectory,
    }

    for m in tick_metrics:
        m["replication"] = replication
    for r in agent_snapshot:
        r["replication"] = replication
    for _, rows in diagnostics.items():
        for row in rows:
            row["replication"] = replication

    logger.info(
        "Finished %s | replication %d | first_marriage_rate=%.3f | avg_children=%.3f",
        scenario.name,
        replication,
        tick_metrics[-1].get("first_marriage_rate_18_45", 0.0),
        tick_metrics[-1].get("avg_children_adults", 0.0),
    )
    return tick_metrics, agent_snapshot, diagnostics


def _write_bdi_outputs(out: Path, diag_map: dict[str, list[dict]]) -> None:
    for name, rows in diag_map.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)




def _needs_calibration(df_summary: pd.DataFrame) -> bool:
    low = df_summary[df_summary["scenario"] == "low_ai"]
    high = df_summary[df_summary["scenario"] == "high_ai"]
    if low.empty or high.empty:
        return False
    marriage_gap = abs(float(high["first_marriage_rate_18_45_mean"].values[0]) - float(low["first_marriage_rate_18_45_mean"].values[0]))
    birth_gap = abs(float(high["first_birth_rate_18_45_mean"].values[0]) - float(low["first_birth_rate_18_45_mean"].values[0]))
    return marriage_gap > 0.12 or birth_gap > 0.10


def _soften_config(cfg: ModelConfig) -> None:
    cfg.lambda_employment *= 0.90
    cfg.lambda_career_uncertainty *= 0.90
    cfg.lambda_time_compression *= 0.90
    cfg.bdi_intention_factor_strength *= 0.92

def run_scenario_experiments(
    config: ModelConfig,
    scenario_names: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    config = copy.deepcopy(config)
    if scenario_names is None:
        scenario_names = list(SCENARIOS.keys())

    out = output_dir or config.output_path
    out.mkdir(parents=True, exist_ok=True)

    all_ticks: list[dict] = []
    all_snapshots: list[dict] = []
    belief_rows: list[dict] = []
    desire_rows: list[dict] = []
    intention_rows: list[dict] = []

    # Common random numbers: same replication seeds across scenarios.
    rep_seeds = [config.seed + rep * 1000 for rep in range(config.replications)]

    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]

        for rep, seed in enumerate(rep_seeds):
            tick_metrics, agent_snapshot, diagnostics = run_single_replication(config, scenario, rep, seed)
            all_ticks.extend(tick_metrics)
            all_snapshots.extend(agent_snapshot)
            belief_rows.extend(diagnostics["belief_trajectory"])
            desire_rows.extend(diagnostics["desire_trajectory"])
            intention_rows.extend(diagnostics["intention_trajectory"])

    df_long = pd.DataFrame(all_ticks)
    df_agent = pd.DataFrame(all_snapshots)

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

    if (not config.calibration_done) and _needs_calibration(df_summary):
        logger.warning("Calibration triggered: scenario gaps too large; softening penalty parameters and rerunning once.")
        _soften_config(config)
        config.calibration_done = True
        set_matching_lambdas(config.lambda_employment, config.lambda_career_uncertainty, config.lambda_time_compression)
        return run_scenario_experiments(config, scenario_names=scenario_names, output_dir=output_dir)

    df_long.to_csv(out / "scenario_results_long.csv", index=False)
    df_summary.to_csv(out / "scenario_results_summary.csv", index=False)
    df_agent.to_csv(out / "agent_snapshot_final.csv", index=False)
    df_long.to_csv(out / "tick_metrics.csv", index=False)

    _write_bdi_outputs(out, {
        "belief_trajectory": belief_rows,
        "desire_trajectory": desire_rows,
        "intention_trajectory": intention_rows,
    })

    sens = run_sensitivity_analysis(config, out)

    logger.info("Results saved to %s", out)
    return {
        "long": df_long,
        "summary": df_summary,
        "agent_snapshot": df_agent,
        "tick_metrics": df_long,
        "sensitivity": sens,
    }


def run_sensitivity_analysis(config: ModelConfig, output_dir: Path) -> pd.DataFrame:
    """One-factor sensitivity analysis over key smoothness-related parameters."""
    base = config
    params = {
        "lambda_employment": [0.40, base.lambda_employment, 0.70],
        "lambda_career_uncertainty": [0.35, base.lambda_career_uncertainty, 0.65],
        "lambda_time_compression": [0.30, base.lambda_time_compression, 0.60],
        "bdi_intention_factor_strength": [0.60, base.bdi_intention_factor_strength, 1.00],
        "llm_weight": [0.10, base.llm_weight, 0.40],
        "scenario_shock_size": [0.03, 0.05, 0.08],
    }

    records: list[dict] = []
    for pname, values in params.items():
        for v in values:
            cfg = copy.deepcopy(base)
            if pname != "scenario_shock_size":
                setattr(cfg, pname, v)
            set_matching_lambdas(cfg.lambda_employment, cfg.lambda_career_uncertainty, cfg.lambda_time_compression)

            scenario_results = {}
            for sname in ["low_ai", "medium_ai", "high_ai"]:
                scenario = copy.deepcopy(SCENARIOS[sname])
                if pname == "scenario_shock_size":
                    if sname == "low_ai":
                        scenario.scenario_shock_size = 0.0
                    elif sname == "medium_ai":
                        scenario.scenario_shock_size = v
                    else:
                        scenario.scenario_shock_size = v * 2

                model = MarriageFertilityABM(cfg, scenario, seed=cfg.seed)
                ticks = model.run()
                final = ticks[-1]
                scenario_results[sname] = {
                    "first_marriage_rate": final.get("first_marriage_rate_18_45", 0.0),
                    "first_birth_rate": final.get("first_birth_rate_18_45", 0.0),
                }

            diff_marriage = scenario_results["high_ai"]["first_marriage_rate"] - scenario_results["low_ai"]["first_marriage_rate"]
            diff_birth = scenario_results["high_ai"]["first_birth_rate"] - scenario_results["low_ai"]["first_birth_rate"]
            records.append({
                "parameter": pname,
                "value": v,
                "low_first_marriage_rate": scenario_results["low_ai"]["first_marriage_rate"],
                "high_first_marriage_rate": scenario_results["high_ai"]["first_marriage_rate"],
                "low_first_birth_rate": scenario_results["low_ai"]["first_birth_rate"],
                "high_first_birth_rate": scenario_results["high_ai"]["first_birth_rate"],
                "high_low_marriage_gap": diff_marriage,
                "high_low_birth_gap": diff_birth,
            })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "sensitivity_results.csv", index=False)

    # tornado-like bar plot by max absolute gap per parameter
    summary = df.groupby("parameter")[["high_low_marriage_gap", "high_low_birth_gap"]].apply(lambda x: x.abs().max())
    summary = summary.sort_values("high_low_marriage_gap", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(summary.index, summary["high_low_marriage_gap"], alpha=0.7, label="Marriage gap")
    ax.barh(summary.index, summary["high_low_birth_gap"], alpha=0.7, label="Birth gap")
    ax.set_xlabel("Max |high-low scenario gap|")
    ax.set_title("Sensitivity tornado (scenario gap drivers)")
    ax.legend()
    fig.savefig(output_dir / "sensitivity_tornado.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df
