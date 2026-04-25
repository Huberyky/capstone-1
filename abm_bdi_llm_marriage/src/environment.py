"""Environment module: updates labour-situation variables each tick."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.enums import EducationLevel, RelationshipStatus
from src.utils import clamp

if TYPE_CHECKING:
    from src.agent import YouthAgent
    from src.config import ModelConfig, ScenarioConfig


_EDUCATION_ENTRY_BUFFER: dict[str, float] = {
    EducationLevel.LOW: 0.0,
    EducationLevel.MEDIUM: 0.10,
    EducationLevel.HIGH: 0.20,
    EducationLevel.VERY_HIGH: 0.30,
}

_EDUCATION_RESKILL_CAPACITY: dict[str, float] = {
    EducationLevel.LOW: 0.0,
    EducationLevel.MEDIUM: 0.10,
    EducationLevel.HIGH: 0.25,
    EducationLevel.VERY_HIGH: 0.40,
}


def _smooth(current: float, prev: float, alpha: float) -> float:
    return clamp(alpha * current + (1 - alpha) * prev)


def update_labour_situation(
    agent: "YouthAgent",
    scenario: "ScenarioConfig",
    current_tick: int,
    model_config: "ModelConfig",
    ticks_per_year: int = 4,
) -> None:
    """Recompute all labour variables from empirical baseline + bounded AI increments."""
    alpha = model_config.smoothing_alpha
    ai = agent.ai_exposure
    edu = agent.education.value if hasattr(agent.education, "value") else agent.education

    agent.prev_employment_stability = agent.employment_stability
    agent.prev_career_predictability = agent.career_predictability

    # empirical baseline (common across scenarios)
    baseline_task_rest = clamp(ai * 0.28)
    baseline_entry_bar = clamp(ai * 0.22 - _EDUCATION_ENTRY_BUFFER.get(edu, 0.0))
    baseline_reskill = clamp(ai * 0.25 - _EDUCATION_RESKILL_CAPACITY.get(edu, 0.0) * 0.5)
    baseline_monitoring = clamp(ai * 0.22)
    baseline_remote = clamp(ai * 0.22)

    # incremental scenario shock only
    shock = scenario.scenario_shock_size
    task_rest_raw = clamp(baseline_task_rest * (1 + shock * scenario.task_restructuring_multiplier))
    entry_bar_raw = clamp(baseline_entry_bar * (1 + shock * scenario.entry_barrier_multiplier))

    age_years = agent.age_ticks / ticks_per_year
    youth_factor = clamp(1.0 - (age_years - 18.0) / 12.0) if age_years < 30 else 0.0
    entry_delay_raw = clamp((0.4 * task_rest_raw + 0.4 * entry_bar_raw + 0.2 * youth_factor) * 0.75)

    stab_raw = clamp(
        agent.income * 0.50
        + (1.0 - entry_delay_raw) * 0.30
        - entry_bar_raw * 0.22
        + scenario.positive_productivity_effect * 0.20
    )

    reskill_raw = clamp(baseline_reskill * (1 + shock * scenario.reskilling_multiplier))
    career_pred_raw = clamp((1.0 - reskill_raw) * 0.45 + (1.0 - task_rest_raw) * 0.25 + stab_raw * 0.30)

    monitoring_raw = clamp(baseline_monitoring * (1 + shock * scenario.monitoring_multiplier))
    remote_raw = clamp(baseline_remote * (1 + shock * scenario.remote_collaboration_multiplier))

    wlb_raw = clamp(
        (monitoring_raw * 0.35 + remote_raw * 0.35) * (0.60 + 0.40 * scenario.work_life_blurring_multiplier)
        + monitoring_raw * 0.12 + remote_raw * 0.12
    )

    rel_bonus = 0.05 if agent.relationship_status != RelationshipStatus.SINGLE else 0.0
    avail_time_raw = clamp(1.0 - wlb_raw * 0.45 - monitoring_raw * 0.25 + rel_bonus)

    # time smoothing for labour variables
    agent.task_restructuring_exposure = _smooth(task_rest_raw, agent.task_restructuring_exposure, alpha)
    agent.entry_barrier_level = _smooth(entry_bar_raw, agent.entry_barrier_level, alpha)
    agent.employment_entry_delay = _smooth(entry_delay_raw, agent.employment_entry_delay, alpha)
    agent.employment_stability = _smooth(stab_raw, agent.employment_stability, alpha)
    agent.reskilling_need = _smooth(reskill_raw, agent.reskilling_need, alpha)
    agent.career_predictability = _smooth(career_pred_raw, agent.career_predictability, alpha)
    agent.monitoring_intensity = _smooth(monitoring_raw, agent.monitoring_intensity, alpha)
    agent.remote_collaboration_intensity = _smooth(remote_raw, agent.remote_collaboration_intensity, alpha)
    agent.work_life_boundary_blurring = _smooth(wlb_raw, agent.work_life_boundary_blurring, alpha)
    agent.available_relationship_time = _smooth(avail_time_raw, agent.available_relationship_time, alpha)

    # bounded transformation for mechanism penalty (damping)
    mechanism_penalty = math.exp(
        -model_config.lambda_employment * agent.entry_barrier_level
        -model_config.lambda_career_uncertainty * (1.0 - agent.career_predictability)
        -model_config.lambda_time_compression * agent.work_life_boundary_blurring
    )
    mechanism_penalty = clamp(
        mechanism_penalty,
        model_config.mechanism_penalty_min,
        model_config.mechanism_penalty_max,
    )

    esp_raw = clamp(
        0.28 * agent.employment_entry_delay
        + 0.30 * (1.0 - agent.employment_stability)
        + 0.22 * agent.task_restructuring_exposure
        + 0.20 * agent.entry_barrier_level
    )
    ceu_raw = clamp(
        0.36 * agent.reskilling_need
        + 0.40 * (1.0 - agent.career_predictability)
        + 0.24 * ai
    )
    rtc_raw = clamp(
        0.33 * agent.work_life_boundary_blurring
        + 0.28 * agent.monitoring_intensity
        + 0.20 * agent.remote_collaboration_intensity
        + 0.19 * (1.0 - agent.available_relationship_time)
    )

    # damp mechanism shocks to avoid cliff effects
    esp_raw = clamp(esp_raw * (1.08 - mechanism_penalty))
    ceu_raw = clamp(ceu_raw * (1.08 - mechanism_penalty))
    rtc_raw = clamp(rtc_raw * (1.08 - mechanism_penalty))

    agent.employment_stability_pressure = _smooth(esp_raw, agent.employment_stability_pressure, alpha)
    agent.career_expectation_uncertainty = _smooth(ceu_raw, agent.career_expectation_uncertainty, alpha)
    agent.relationship_time_compression = _smooth(rtc_raw, agent.relationship_time_compression, alpha)

    agent.clamp_all()
