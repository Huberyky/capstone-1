#!/usr/bin/env python3
"""CLI entry point for the ABM-BDI-LLM marriage simulation.

Examples
--------
No-LLM mode (default, runs without API key):
    python run_experiment.py --no-llm --population-size 500 --years 30 --replications 5 --plot

Use-LLM mode (requires DEEPSEEK_API_KEY):
    python run_experiment.py --use-llm --llm-sample-rate 0.05 --population-size 300 --years 20 --replications 2 --plot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv  # type: ignore

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AI-impact marriage/fertility ABM simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # LLM mode
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Enable DeepSeek LLM calls (requires DEEPSEEK_API_KEY).",
    )
    llm_group.add_argument(
        "--no-llm",
        action="store_true",
        default=True,
        help="Disable LLM; use heuristic rules only (default).",
    )

    # Simulation parameters
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--years", type=int, default=30)
    parser.add_argument("--ticks-per-year", type=int, default=4)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # LLM parameters
    parser.add_argument("--llm-sample-rate", type=float, default=0.05)
    parser.add_argument("--llm-model", type=str, default="deepseek-chat")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-timeout", type=float, default=30.0)

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate and save visualisation plots.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["low_ai", "medium_ai", "high_ai"],
        default=["low_ai", "medium_ai", "high_ai"],
        help="Which scenarios to run.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    use_llm = args.use_llm and not args.no_llm

    # Late imports (after path setup)
    from src.config import ModelConfig
    from src.experiments import run_scenario_experiments
    from src.plotting import plot_all

    config = ModelConfig(
        population_size=args.population_size,
        years=args.years,
        ticks_per_year=args.ticks_per_year,
        seed=args.seed,
        replications=args.replications,
        use_llm=use_llm,
        llm_sample_rate=args.llm_sample_rate,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_timeout=args.llm_timeout,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print(" ABM-BDI-LLM Marriage & Fertility Simulation")
    print("=" * 60)
    print(f"  Population : {config.population_size}")
    print(f"  Years      : {config.years} ({config.total_ticks} ticks)")
    print(f"  Replications: {config.replications}")
    print(f"  LLM mode   : {'ENABLED (DeepSeek)' if use_llm else 'DISABLED (heuristics)'}")
    print(f"  Scenarios  : {args.scenarios}")
    print(f"  Output dir : {config.output_dir}")
    print("=" * 60)

    output_path = Path(args.output_dir)
    results = run_scenario_experiments(
        config=config,
        scenario_names=args.scenarios,
        output_dir=output_path,
    )

    # Print summary table
    df_summary = results["summary"]
    key_metrics = [
        "first_marriage_rate_18_45",
        "first_birth_rate_18_45",
        "avg_children_adults",
        "mean_first_marriage_age",
        "mean_first_birth_age",
        "mean_employment_stability_pressure",
        "mean_career_expectation_uncertainty",
        "mean_relationship_time_compression",
    ]
    print("\n=== Summary Results ===")
    for _, row in df_summary.iterrows():
        print(f"\nScenario: {row['scenario']}")
        for metric in key_metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in row:
                mean_val = row[mean_col]
                std_val = row.get(std_col, 0.0)
                print(f"  {metric:<45} {mean_val:.4f} ± {std_val:.4f}")

    if args.plot:
        plot_all(df_summary, output_path)
    else:
        print("\n(Use --plot to generate visualisation figures.)")

    print(f"\nAll outputs saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
