"""Environment module: updates agent labour-situation variables each tick.

This implements the causal chain:
  AI diffusion scenario + individual AI exposure
  → task restructuring, entry barriers, employment stability, reskilling,
    monitoring, remote collaboration, work-life blurring
  → three core mechanism variables:
      employment_stability_pressure
      career_expectation_uncertainty
      relationship_time_compression
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.enums import EducationLevel, RelationshipStatus
from src.utils import clamp

if TYPE_CHECKING:
    from src.agent import YouthAgent
    from src.config import ScenarioConfig


# Education level baseline resistance to AI-driven entry barriers
_EDUCATION_ENTRY_BUFFER: dict[str, float] = {
    EducationLevel.LOW: 0.0,
    EducationLevel.MEDIUM: 0.10,
    EducationLevel.HIGH: 0.20,
    EducationLevel.VERY_HIGH: 0.30,
}

# Education level reskilling capacity (higher edu → faster adaptation)
_EDUCATION_RESKILL_CAPACITY: dict[str, float] = {
    EducationLevel.LOW: 0.0,
    EducationLevel.MEDIUM: 0.10,
    EducationLevel.HIGH: 0.25,
    EducationLevel.VERY_HIGH: 0.40,
}


def update_labour_situation(
    agent: "YouthAgent",
    scenario: "ScenarioConfig",
    current_tick: int,
    ticks_per_year: int = 4,
) -> None:
    """Recompute all labour-situation variables and mechanism variables for one tick.

    The update follows a sequential causal logic:
    1. Task restructuring exposure
    2. Entry barrier level
    3. Employment entry delay
    4. Employment stability
    5. Reskilling need
    6. Career predictability
    7. Monitoring intensity
    8. Remote collaboration intensity
    9. Work-life boundary blurring
    10. Available relationship time
    11. Three mechanism variables

    Args:
        agent: The agent to update.
        scenario: Current AI scenario config.
        current_tick: Global tick counter.
        ticks_per_year: Ticks per simulated year.
    """
    ai = agent.ai_exposure
    edu = agent.education.value if hasattr(agent.education, "value") else agent.education

    # Store previous values for BDI change detection
    agent.prev_employment_stability = agent.employment_stability
    agent.prev_career_predictability = agent.career_predictability

    # ---- 1. Task restructuring exposure ----
    # How much AI has restructured this person's job tasks
    task_rest = clamp(ai * scenario.task_restructuring_multiplier)
    agent.task_restructuring_exposure = task_rest

    # ---- 2. Entry barrier level ----
    # Education partially buffers entry barriers
    edu_buffer = _EDUCATION_ENTRY_BUFFER.get(edu, 0.0)
    entry_bar = clamp(
        ai * scenario.entry_barrier_multiplier - edu_buffer
    )
    agent.entry_barrier_level = entry_bar

    # ---- 3. Employment entry delay ----
    # Young agents in high-barrier environments face longer school-to-work transitions
    age_years = agent.age_ticks / ticks_per_year
    youth_factor = clamp(1.0 - (age_years - 18.0) / 12.0) if age_years < 30 else 0.0
    entry_delay = clamp(
        (0.4 * task_rest + 0.4 * entry_bar + 0.2 * youth_factor) * 0.9
    )
    agent.employment_entry_delay = entry_delay

    # ---- 4. Employment stability ----
    # Income provides stability; entry delay and scenario multiplier erode it
    stab = clamp(
        agent.income * 0.5
        + (1.0 - entry_delay) * 0.3
        - entry_bar * scenario.employment_stability_multiplier * 0.4
        + scenario.positive_productivity_effect * 0.2
    )
    agent.employment_stability = stab

    # ---- 5. Reskilling need ----
    reskill_cap = _EDUCATION_RESKILL_CAPACITY.get(edu, 0.0)
    reskill = clamp(
        ai * scenario.reskilling_multiplier - reskill_cap * 0.5
    )
    agent.reskilling_need = reskill

    # ---- 6. Career predictability ----
    career_pred = clamp(
        (1.0 - reskill) * 0.4
        + (1.0 - task_rest) * 0.3
        + stab * 0.3
    )
    agent.career_predictability = career_pred

    # ---- 7. Monitoring intensity ----
    monitoring = clamp(ai * scenario.monitoring_multiplier)
    agent.monitoring_intensity = monitoring

    # ---- 8. Remote collaboration intensity ----
    remote = clamp(ai * scenario.remote_collaboration_multiplier)
    agent.remote_collaboration_intensity = remote

    # ---- 9. Work-life boundary blurring ----
    wlb = clamp(
        (monitoring * 0.4 + remote * 0.4)
        * scenario.work_life_blurring_multiplier
        + monitoring * 0.1
        + remote * 0.1
    )
    agent.work_life_boundary_blurring = wlb

    # ---- 10. Available relationship time ----
    # Being in a relationship slightly increases motivation to protect time
    rel_bonus = 0.05 if agent.relationship_status != RelationshipStatus.SINGLE else 0.0
    avail_time = clamp(
        1.0 - wlb * 0.5 - monitoring * 0.3 + rel_bonus
    )
    agent.available_relationship_time = avail_time

    # ---- 11. Three mechanism variables ----

    # Mechanism 1: employment stability pressure
    # Captures how difficult stable career entry is
    agent.employment_stability_pressure = clamp(
        entry_delay * 0.30
        + (1.0 - stab) * 0.30
        + task_rest * 0.20
        + entry_bar * 0.20
    )

    # Mechanism 2: career expectation uncertainty
    # Captures unpredictability of future career path
    agent.career_expectation_uncertainty = clamp(
        reskill * 0.40
        + (1.0 - career_pred) * 0.40
        + ai * 0.20
    )

    # Mechanism 3: relationship time compression
    # Captures shrinking of private / relational time
    agent.relationship_time_compression = clamp(
        wlb * 0.35
        + monitoring * 0.30
        + remote * 0.20
        + (1.0 - avail_time) * 0.15
    )

    agent.clamp_all()
