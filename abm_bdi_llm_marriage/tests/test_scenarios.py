"""Integration tests: verify all three scenarios run end-to-end."""

from __future__ import annotations

import pytest

from src.config import ModelConfig, SCENARIOS
from src.experiments import run_scenario_experiments
from src.model import MarriageFertilityABM


# Small configuration for fast tests
def tiny_config(**kwargs) -> ModelConfig:
    defaults = dict(
        population_size=60,
        years=5,
        ticks_per_year=4,
        seed=7,
        replications=1,
        use_llm=False,
        llm_sample_rate=0.0,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestScenarioRuns:
    """Each scenario should run without errors and produce sensible metrics."""

    @pytest.mark.parametrize("scenario_name", ["low_ai", "medium_ai", "high_ai"])
    def test_scenario_completes(self, scenario_name, tmp_path):
        config = tiny_config(output_dir=str(tmp_path))
        scenario = SCENARIOS[scenario_name]
        model = MarriageFertilityABM(config, scenario, seed=42)
        metrics_list = model.run()
        assert len(metrics_list) == config.total_ticks

    @pytest.mark.parametrize("scenario_name", ["low_ai", "medium_ai", "high_ai"])
    def test_metrics_keys_present(self, scenario_name, tmp_path):
        config = tiny_config(output_dir=str(tmp_path))
        scenario = SCENARIOS[scenario_name]
        model = MarriageFertilityABM(config, scenario, seed=42)
        metrics_list = model.run()
        last = metrics_list[-1]
        required = [
            "first_marriage_rate_18_45",
            "first_birth_rate_18_45",
            "avg_children_adults",
            "mean_employment_stability_pressure",
            "mean_career_expectation_uncertainty",
            "mean_relationship_time_compression",
            "share_intention_search_partner",
            "llm_call_count",
        ]
        for key in required:
            assert key in last, f"Missing metric: {key}"

    @pytest.mark.parametrize("scenario_name", ["low_ai", "medium_ai", "high_ai"])
    def test_metrics_in_valid_range(self, scenario_name, tmp_path):
        config = tiny_config(output_dir=str(tmp_path))
        scenario = SCENARIOS[scenario_name]
        model = MarriageFertilityABM(config, scenario, seed=42)
        metrics_list = model.run()
        last = metrics_list[-1]
        rate_metrics = [
            "first_marriage_rate_18_45",
            "first_birth_rate_18_45",
            "avg_children_adults",
            "mean_employment_stability_pressure",
            "mean_career_expectation_uncertainty",
            "mean_relationship_time_compression",
        ]
        for key in rate_metrics:
            val = last[key]
            assert 0.0 <= val <= 10.0, f"{key}={val} unexpectedly large"

    def test_high_ai_has_higher_mechanism_pressures_than_low_ai(self, tmp_path):
        """High-AI scenario should produce higher mechanism variable means than low-AI."""
        config = tiny_config(output_dir=str(tmp_path))

        model_low = MarriageFertilityABM(config, SCENARIOS["low_ai"], seed=42)
        metrics_low = model_low.run()

        model_high = MarriageFertilityABM(config, SCENARIOS["high_ai"], seed=42)
        metrics_high = model_high.run()

        last_low = metrics_low[-1]
        last_high = metrics_high[-1]

        assert (
            last_high["mean_employment_stability_pressure"]
            > last_low["mean_employment_stability_pressure"]
        )
        assert (
            last_high["mean_career_expectation_uncertainty"]
            > last_low["mean_career_expectation_uncertainty"]
        )
        assert (
            last_high["mean_relationship_time_compression"]
            > last_low["mean_relationship_time_compression"]
        )


class TestExperimentRunner:
    def test_run_all_scenarios(self, tmp_path):
        config = tiny_config(replications=2, output_dir=str(tmp_path))
        results = run_scenario_experiments(config, output_dir=tmp_path)
        assert "long" in results
        assert "summary" in results
        assert "agent_snapshot" in results

    def test_csv_files_created(self, tmp_path):
        config = tiny_config(replications=1, output_dir=str(tmp_path))
        run_scenario_experiments(config, output_dir=tmp_path)
        assert (tmp_path / "scenario_results_long.csv").exists()
        assert (tmp_path / "scenario_results_summary.csv").exists()
        assert (tmp_path / "agent_snapshot_final.csv").exists()

    def test_summary_has_all_scenarios(self, tmp_path):
        config = tiny_config(replications=1, output_dir=str(tmp_path))
        results = run_scenario_experiments(config, output_dir=tmp_path)
        scenarios_in_summary = set(results["summary"]["scenario"].tolist())
        assert {"low_ai", "medium_ai", "high_ai"} == scenarios_in_summary

    def test_seed_reproducibility(self, tmp_path):
        """Same seed → same final metric."""
        config = tiny_config(replications=1, seed=99, output_dir=str(tmp_path))
        r1 = run_scenario_experiments(
            config, scenario_names=["low_ai"], output_dir=tmp_path
        )
        r2 = run_scenario_experiments(
            config, scenario_names=["low_ai"], output_dir=tmp_path
        )
        val1 = r1["summary"]["first_marriage_rate_18_45_mean"].values[0]
        val2 = r2["summary"]["first_marriage_rate_18_45_mean"].values[0]
        assert val1 == pytest.approx(val2, abs=1e-6)


class TestAgentSnapshot:
    def test_snapshot_has_expected_columns(self, tmp_path):
        config = tiny_config(output_dir=str(tmp_path))
        model = MarriageFertilityABM(config, SCENARIOS["medium_ai"], seed=1)
        model.run()
        snap = model.get_agent_snapshot()
        assert len(snap) > 0
        cols = set(snap[0].keys())
        for col in ("id", "sex", "age", "education", "num_children", "scenario"):
            assert col in cols, f"Missing column: {col}"

    def test_population_size_maintained(self, tmp_path):
        """Population should roughly stay near configured size."""
        config = tiny_config(population_size=60, output_dir=str(tmp_path))
        model = MarriageFertilityABM(config, SCENARIOS["low_ai"], seed=5)
        model.run()
        assert 40 <= len(model.agents) <= 80  # rough bounds
