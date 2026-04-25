"""YouthAgent: the individual-level entity in the ABM."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.enums import (
    EducationLevel,
    Intention,
    RelationshipStatus,
    Sex,
)
from src.utils import clamp


# ---------------------------------------------------------------------------
# BDI state containers
# ---------------------------------------------------------------------------

@dataclass
class Beliefs:
    """Subjective perceptions held by an agent (not objective variables)."""

    employment_security: float = 0.5
    career_predictability: float = 0.5
    time_availability: float = 0.5
    marriage_affordability: float = 0.5
    childbearing_affordability: float = 0.5
    partner_quality: float = 0.5
    relationship_stability: float = 0.5
    age_pressure: float = 0.0
    social_norm_pressure: float = 0.3
    career_entry_confidence: float = 0.5
    long_term_life_planning_confidence: float = 0.5


@dataclass
class Desires:
    """Strength of each desire (0–1); desires may conflict."""

    seek_partner: float = 0.5
    maintain_relationship: float = 0.5
    marry: float = 0.3
    have_child: float = 0.3
    delay_marriage: float = 0.2
    delay_childbearing: float = 0.2
    stabilize_career_entry: float = 0.3
    reduce_career_uncertainty: float = 0.3
    protect_personal_time: float = 0.3
    personal_autonomy: float = 0.4


@dataclass
class IntentionState:
    """Current intention and its associated metadata."""

    current_intention: Optional[Intention] = None
    intention_duration: int = 0          # ticks since intention was set
    commitment_strength: float = 0.5
    intention_score: float = 0.0


@dataclass
class LLMState:
    """Subjective evaluation variables populated by LLM or heuristics."""

    marriage_value_score: float = 0.5
    fertility_value_score: float = 0.5
    marry_willingness: float = 0.5
    birth_willingness: float = 0.5
    fertility_intention: float = 0.5
    llm_reason: str = ""
    last_llm_tick: int = -999   # tick at which LLM was last called


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

@dataclass
class YouthAgent:
    """Represents a single young adult in the simulation.

    Parameters are set at initialisation by the model; all variables are
    updated each tick according to the environment, BDI engine, and
    matching / fertility modules.
    """

    # --- identity ---
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sex: Sex = Sex.MALE
    age_ticks: int = 72        # default 18 years × 4 ticks/year
    education: EducationLevel = EducationLevel.MEDIUM
    income: float = 0.4        # normalised 0–1
    occupation: str = "general"
    ai_exposure: float = 0.3   # individual-level AI exposure (0–1)
    school_status: bool = False  # True while in school
    alive: bool = True

    # --- relationship state ---
    relationship_status: RelationshipStatus = RelationshipStatus.SINGLE
    partner_id: Optional[str] = None
    relationship_duration: int = 0   # ticks in current relationship
    num_children: int = 0
    first_marriage_age: Optional[float] = None
    first_birth_age: Optional[float] = None

    # --- labour situation variables (updated each tick) ---
    task_restructuring_exposure: float = 0.2
    entry_barrier_level: float = 0.2
    employment_entry_delay: float = 0.1
    employment_stability: float = 0.6
    reskilling_need: float = 0.2
    career_predictability: float = 0.6
    monitoring_intensity: float = 0.2
    remote_collaboration_intensity: float = 0.2
    work_life_boundary_blurring: float = 0.2
    available_relationship_time: float = 0.7

    # --- three core mechanism variables ---
    employment_stability_pressure: float = 0.2
    career_expectation_uncertainty: float = 0.2
    relationship_time_compression: float = 0.2

    # --- BDI state ---
    beliefs: Beliefs = field(default_factory=Beliefs)
    desires: Desires = field(default_factory=Desires)
    intention_state: IntentionState = field(default_factory=IntentionState)

    # --- LLM subjective evaluation ---
    llm_state: LLMState = field(default_factory=LLMState)

    # --- internal meta ---
    belief_sensitivity: float = 0.25  # rho: how fast beliefs update
    prev_employment_stability: float = 0.6
    prev_career_predictability: float = 0.6

    @property
    def age(self) -> float:
        """Current age in years (float)."""
        return self.age_ticks / 4.0   # assuming 4 ticks per year

    @property
    def is_single(self) -> bool:
        return self.relationship_status == RelationshipStatus.SINGLE

    @property
    def is_dating(self) -> bool:
        return self.relationship_status == RelationshipStatus.DATING

    @property
    def is_married(self) -> bool:
        return self.relationship_status == RelationshipStatus.MARRIED

    def clamp_all(self) -> None:
        """Ensure all float fields in [0, 1]."""
        for attr in (
            "income", "ai_exposure",
            "task_restructuring_exposure", "entry_barrier_level",
            "employment_entry_delay", "employment_stability",
            "reskilling_need", "career_predictability",
            "monitoring_intensity", "remote_collaboration_intensity",
            "work_life_boundary_blurring", "available_relationship_time",
            "employment_stability_pressure",
            "career_expectation_uncertainty",
            "relationship_time_compression",
        ):
            setattr(self, attr, clamp(getattr(self, attr)))


def create_agent(
    rng: random.Random,
    agent_id: str,
    sex: Sex,
    age_ticks: int,
    ticks_per_year: int = 4,
) -> YouthAgent:
    """Factory function to create a randomly initialised YouthAgent.

    Args:
        rng: Seeded random number generator.
        agent_id: Unique agent identifier.
        sex: Agent sex.
        age_ticks: Starting age in ticks.
        ticks_per_year: Ticks per simulation year.

    Returns:
        A new YouthAgent with randomised baseline attributes.
    """
    edu_weights = [0.15, 0.35, 0.35, 0.15]
    education = rng.choices(list(EducationLevel), weights=edu_weights, k=1)[0]

    income = clamp(rng.gauss(0.45, 0.15))
    ai_exp = clamp(rng.gauss(0.35, 0.15))
    belief_sens = clamp(rng.gauss(0.25, 0.05), 0.05, 0.60)

    agent = YouthAgent(
        id=agent_id,
        sex=sex,
        age_ticks=age_ticks,
        education=education,
        income=income,
        ai_exposure=ai_exp,
        belief_sensitivity=belief_sens,
    )

    # School status: agents younger than 22 may still be in school
    age_years = age_ticks / ticks_per_year
    if age_years < 22 and education in (EducationLevel.HIGH, EducationLevel.VERY_HIGH):
        agent.school_status = rng.random() < 0.7
    elif age_years < 20:
        agent.school_status = rng.random() < 0.5

    return agent
