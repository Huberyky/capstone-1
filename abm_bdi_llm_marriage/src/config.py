"""Configuration classes for scenarios and model parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.enums import AILevel


@dataclass
class ScenarioConfig:
    """Defines an AI-impact scenario with bounded incremental shocks over baseline."""

    name: str
    ai_level: AILevel
    scenario_shock_size: float
    task_restructuring_multiplier: float
    entry_barrier_multiplier: float
    employment_stability_multiplier: float
    reskilling_multiplier: float
    monitoring_multiplier: float
    remote_collaboration_multiplier: float
    work_life_blurring_multiplier: float
    positive_productivity_effect: float
    description: str


# ---------------------------------------------------------------------------
# Pre-defined scenarios (smoothed, baseline-centered)
# ---------------------------------------------------------------------------

BASE_MULTIPLIER = 1.00

LOW_AI_SCENARIO = ScenarioConfig(
    name="low_ai",
    ai_level=AILevel.LOW,
    scenario_shock_size=0.00,
    task_restructuring_multiplier=0.85 * BASE_MULTIPLIER,
    entry_barrier_multiplier=0.85 * BASE_MULTIPLIER,
    employment_stability_multiplier=0.85 * BASE_MULTIPLIER,
    reskilling_multiplier=0.85 * BASE_MULTIPLIER,
    monitoring_multiplier=0.85 * BASE_MULTIPLIER,
    remote_collaboration_multiplier=0.85 * BASE_MULTIPLIER,
    work_life_blurring_multiplier=0.85 * BASE_MULTIPLIER,
    positive_productivity_effect=0.08,
    description="Low AI diffusion: baseline labour structure with mild AI-related frictions.",
)

MEDIUM_AI_SCENARIO = ScenarioConfig(
    name="medium_ai",
    ai_level=AILevel.MEDIUM,
    scenario_shock_size=0.05,
    task_restructuring_multiplier=1.00 * BASE_MULTIPLIER,
    entry_barrier_multiplier=1.00 * BASE_MULTIPLIER,
    employment_stability_multiplier=1.00 * BASE_MULTIPLIER,
    reskilling_multiplier=1.00 * BASE_MULTIPLIER,
    monitoring_multiplier=1.00 * BASE_MULTIPLIER,
    remote_collaboration_multiplier=1.00 * BASE_MULTIPLIER,
    work_life_blurring_multiplier=1.00 * BASE_MULTIPLIER,
    positive_productivity_effect=0.10,
    description="Medium AI diffusion: baseline-aligned structural transition.",
)

HIGH_AI_SCENARIO = ScenarioConfig(
    name="high_ai",
    ai_level=AILevel.HIGH,
    scenario_shock_size=0.10,
    task_restructuring_multiplier=1.15 * BASE_MULTIPLIER,
    entry_barrier_multiplier=1.15 * BASE_MULTIPLIER,
    employment_stability_multiplier=1.15 * BASE_MULTIPLIER,
    reskilling_multiplier=1.15 * BASE_MULTIPLIER,
    monitoring_multiplier=1.15 * BASE_MULTIPLIER,
    remote_collaboration_multiplier=1.15 * BASE_MULTIPLIER,
    work_life_blurring_multiplier=1.15 * BASE_MULTIPLIER,
    positive_productivity_effect=0.12,
    description="High AI diffusion: stronger but still bounded incremental shock over baseline.",
)

SCENARIOS: dict[str, ScenarioConfig] = {
    "low_ai": LOW_AI_SCENARIO,
    "medium_ai": MEDIUM_AI_SCENARIO,
    "high_ai": HIGH_AI_SCENARIO,
}


@dataclass
class ModelConfig:
    """Controls global model parameters."""

    population_size: int = 500
    years: int = 30
    ticks_per_year: int = 4
    seed: int = 42
    replications: int = 3
    delta_school_meeting: float = 0.30
    beta_commitment: float = 0.05
    sigma_age_pressure_male: float = 30.0
    sigma_age_pressure_female: float = 27.0
    use_llm: bool = False
    api_key: str = ""
    llm_sample_rate: float = 0.05
    llm_refresh_interval: int = 8
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_timeout: float = 30.0
    output_dir: str = "outputs"

    # empirical baseline / smoothing / bounded transformation
    smoothing_alpha: float = 0.2
    mechanism_penalty_min: float = 0.70
    mechanism_penalty_max: float = 1.05
    lambda_employment: float = 0.55
    lambda_career_uncertainty: float = 0.50
    lambda_time_compression: float = 0.45
    bdi_intention_factor_strength: float = 0.80
    llm_weight: float = 0.30
    calibration_done: bool = False

    @property
    def total_ticks(self) -> int:
        return self.years * self.ticks_per_year

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
