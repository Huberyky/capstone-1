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
    intention_duration: int = 0
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
    last_llm_tick: int = -999


@dataclass
class YouthAgent:
    """Represents a single young adult in the simulation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sex: Sex = Sex.MALE
    age_ticks: int = 72
    education: EducationLevel = EducationLevel.MEDIUM
    income: float = 0.4
    occupation: str = "general"
    ai_exposure: float = 0.3
    school_status: bool = False
    alive: bool = True

    relationship_status: RelationshipStatus = RelationshipStatus.SINGLE
    partner_id: Optional[str] = None
    relationship_duration: int = 0
    num_children: int = 0
    first_marriage_age: Optional[float] = None
    first_birth_age: Optional[float] = None

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

    employment_stability_pressure: float = 0.2
    career_expectation_uncertainty: float = 0.2
    relationship_time_compression: float = 0.2

    beliefs: Beliefs = field(default_factory=Beliefs)
    desires: Desires = field(default_factory=Desires)
    intention_state: IntentionState = field(default_factory=IntentionState)
    llm_state: LLMState = field(default_factory=LLMState)

    belief_sensitivity: float = 0.25
    prev_employment_stability: float = 0.6
    prev_career_predictability: float = 0.6

    # cognitive process extensions
    belief_memory: dict[str, list[float]] = field(default_factory=lambda: {
        "employment_stability_pressure": [],
        "career_expectation_uncertainty": [],
        "relationship_time_compression": [],
    })
    perceived_trend: float = 0.0
    uncertainty_tolerance: float = 0.5
    risk_preference: float = 0.5
    aspiration_level: float = 0.5
    decision_noise: float = 0.05
    last_decision_outcome: float = 0.0
    cognitive_consistency_score: float = 0.5
    subjective_interpretation: dict[str, float] = field(default_factory=lambda: {
        "economic_family_readiness": 0.5,
        "career_trajectory_predictability": 0.5,
        "time_for_relationship_investment": 0.5,
    })
    desire_conflict_index: float = 0.0
    intention_switch_count: int = 0
    intention_duration_history: list[int] = field(default_factory=list)

    @property
    def age(self) -> float:
        return self.age_ticks / 4.0

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
            "uncertainty_tolerance",
            "risk_preference",
            "aspiration_level",
            "decision_noise",
            "last_decision_outcome",
            "cognitive_consistency_score",
            "desire_conflict_index",
        ):
            setattr(self, attr, clamp(getattr(self, attr)))


def create_agent(
    rng: random.Random,
    agent_id: str,
    sex: Sex,
    age_ticks: int,
    ticks_per_year: int = 4,
) -> YouthAgent:
    """Factory function to create a randomly initialised YouthAgent."""
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
        uncertainty_tolerance=clamp(rng.gauss(0.5, 0.15)),
        risk_preference=clamp(rng.gauss(0.5, 0.2)),
        aspiration_level=clamp(rng.gauss(0.55, 0.15)),
        decision_noise=clamp(rng.gauss(0.06, 0.02), 0.01, 0.2),
    )

    age_years = age_ticks / ticks_per_year
    if age_years < 22 and education in (EducationLevel.HIGH, EducationLevel.VERY_HIGH):
        agent.school_status = rng.random() < 0.7
    elif age_years < 20:
        agent.school_status = rng.random() < 0.5

    return agent
