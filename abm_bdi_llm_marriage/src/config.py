"""Configuration classes for scenarios and model parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.enums import AILevel


@dataclass
class ScenarioConfig:
    """Defines an AI-impact scenario with multipliers for each labour-market mechanism.

    Attributes:
        name: Scenario identifier.
        ai_level: Low / medium / high AI diffusion level.
        task_restructuring_multiplier: How strongly AI reshapes job-task composition.
        entry_barrier_multiplier: How strongly AI raises skill entry barriers.
        employment_stability_multiplier: How strongly AI erodes stable employment supply.
        reskilling_multiplier: How rapidly AI accelerates skill obsolescence.
        monitoring_multiplier: Intensity of AI-enabled performance monitoring.
        remote_collaboration_multiplier: Extent of AI-driven remote / platform work.
        work_life_blurring_multiplier: Degree to which AI blurs work-life boundaries.
        positive_productivity_effect: Offsetting positive productivity boost (0-1).
        description: Human-readable scenario description.
    """

    name: str
    ai_level: AILevel
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
# Pre-defined scenarios
# ---------------------------------------------------------------------------

LOW_AI_SCENARIO = ScenarioConfig(
    name="low_ai",
    ai_level=AILevel.LOW,
    task_restructuring_multiplier=0.15,
    entry_barrier_multiplier=0.10,
    employment_stability_multiplier=0.10,
    reskilling_multiplier=0.10,
    monitoring_multiplier=0.10,
    remote_collaboration_multiplier=0.10,
    work_life_blurring_multiplier=0.10,
    positive_productivity_effect=0.05,
    description=(
        "Low AI diffusion: AI has limited impact on job-task structure, "
        "career entry, time allocation, and work-life boundaries."
    ),
)

MEDIUM_AI_SCENARIO = ScenarioConfig(
    name="medium_ai",
    ai_level=AILevel.MEDIUM,
    task_restructuring_multiplier=0.45,
    entry_barrier_multiplier=0.40,
    employment_stability_multiplier=0.35,
    reskilling_multiplier=0.45,
    monitoring_multiplier=0.40,
    remote_collaboration_multiplier=0.45,
    work_life_blurring_multiplier=0.40,
    positive_productivity_effect=0.15,
    description=(
        "Medium AI diffusion: AI brings efficiency gains but also raises skill-update "
        "requirements, increases career uncertainty, and begins to blur work-life boundaries."
    ),
)

HIGH_AI_SCENARIO = ScenarioConfig(
    name="high_ai",
    ai_level=AILevel.HIGH,
    task_restructuring_multiplier=0.80,
    entry_barrier_multiplier=0.75,
    employment_stability_multiplier=0.70,
    reskilling_multiplier=0.80,
    monitoring_multiplier=0.75,
    remote_collaboration_multiplier=0.80,
    work_life_blurring_multiplier=0.78,
    positive_productivity_effect=0.20,
    description=(
        "High AI diffusion: AI is deeply embedded in labour processes; job-task "
        "restructuring, entry barriers, skill churn, platform-based monitoring, "
        "remote collaboration, and work-life boundary erosion are all pronounced."
    ),
)

SCENARIOS: dict[str, ScenarioConfig] = {
    "low_ai": LOW_AI_SCENARIO,
    "medium_ai": MEDIUM_AI_SCENARIO,
    "high_ai": HIGH_AI_SCENARIO,
}


@dataclass
class ModelConfig:
    """Controls global model parameters.

    Attributes:
        population_size: Number of agents in the simulation.
        years: Total simulation duration in years.
        ticks_per_year: Number of ticks per year (e.g. 4 = quarterly).
        seed: Random seed for reproducibility.
        replications: Number of independent runs per scenario.
        delta_school_meeting: Probability of education-homogamous encounter.
        beta_commitment: Scaling factor for relationship-to-marriage transition.
        sigma_age_pressure_male: Age at which male age-pressure peaks.
        sigma_age_pressure_female: Age at which female age-pressure peaks.
        use_llm: Whether to call the DeepSeek API.
        api_key: DeepSeek API key. Overrides the DEEPSEEK_API_KEY env variable
            when provided directly (e.g. via --api-key on the CLI).
        llm_sample_rate: Fraction of agents queried per tick (0-1).
        llm_refresh_interval: Ticks between LLM refreshes for the same agent.
        llm_model: DeepSeek model identifier.
        llm_temperature: Sampling temperature for the LLM.
        llm_timeout: API timeout in seconds.
        output_dir: Directory where outputs are saved.
    """

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
    api_key: str = ""          # CLI --api-key takes precedence over env var
    llm_sample_rate: float = 0.05
    llm_refresh_interval: int = 8
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_timeout: float = 30.0
    output_dir: str = "outputs"

    @property
    def total_ticks(self) -> int:
        return self.years * self.ticks_per_year

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
