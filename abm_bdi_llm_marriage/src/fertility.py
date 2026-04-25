"""Fertility decision module.

Only married agents enter fertility decisions. The female agent is the
primary unit for the birth event, but children are recorded on both partners.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from src.utils import clamp, sigmoid

if TYPE_CHECKING:
    from src.agent import YouthAgent
    from src.bdi import BDIEngine


# Logit coefficients — calibrated to produce realistic-range birth rates
_THETA = {
    "intercept": -2.5,
    "relationship_duration": 0.06,  # per tick
    "couple_affordability": 2.0,
    "fertility_value_score": 1.5,
    "birth_willingness": 1.2,
    "available_time": 1.0,
    "employment_stability_pressure": -2.0,
    "career_expectation_uncertainty": -1.8,
    "relationship_time_compression": -1.5,
    "age_penalty": -0.15,   # per year above 32 (female)
    "existing_children": -1.2,  # per child
}

# Maximum number of children considered in the model
_MAX_CHILDREN = 3


def couple_affordability(
    female: "YouthAgent",
    male: "YouthAgent",
) -> float:
    """Compute joint financial capacity for childbearing.

    Args:
        female: Female agent.
        male: Male agent.

    Returns:
        Affordability score in [0, 1].
    """
    combined_income = (female.income + male.income) / 2.0
    combined_stability = (
        female.employment_stability + male.employment_stability
    ) / 2.0
    child_burden = min(female.num_children * 0.15, 0.45)
    return clamp(combined_income * 0.5 + combined_stability * 0.3 - child_burden)


def birth_probability(
    female: "YouthAgent",
    male: "YouthAgent",
    bdi: "BDIEngine",
    llm_weight: float = 0.0,
    ticks_per_year: int = 4,
) -> float:
    """Compute the probability of a birth event this tick for a married couple.

    Uses a sigmoid (logit) form combining structural, economic, subjective,
    and AI-mechanism variables.

    Only the female agent's age penalty applies; fertility_value_score and
    birth_willingness are weighted 70 % female / 30 % male.

    Args:
        female: The female partner.
        male: The male partner.
        bdi: BDI engine for intention factors.
        llm_weight: Weight placed on LLM-derived willingness scores.
        ticks_per_year: Ticks per simulation year.

    Returns:
        Birth probability in [0, 1].
    """
    if female.num_children >= _MAX_CHILDREN:
        return 0.0

    # ---- couple-level variables ----
    afford = couple_affordability(female, male)

    couple_esp = (
        female.employment_stability_pressure
        + male.employment_stability_pressure
    ) / 2.0
    couple_ceu = (
        female.career_expectation_uncertainty
        + male.career_expectation_uncertainty
    ) / 2.0
    couple_rtc = (
        female.relationship_time_compression
        + male.relationship_time_compression
    ) / 2.0

    # Fertility value and willingness: female-weighted
    fvs_rule = 0.7 * female.llm_state.fertility_value_score + 0.3 * male.llm_state.fertility_value_score
    bw_rule = 0.7 * female.llm_state.birth_willingness + 0.3 * male.llm_state.birth_willingness

    avail_time = (female.available_relationship_time + male.available_relationship_time) / 2.0

    # Age penalty: years above 32 for female
    female_age = female.age_ticks / ticks_per_year
    age_pen = max(0.0, female_age - 32.0)

    # ---- logit ----
    logit = (
        _THETA["intercept"]
        + _THETA["relationship_duration"] * female.relationship_duration
        + _THETA["couple_affordability"] * afford
        + _THETA["fertility_value_score"] * fvs_rule
        + _THETA["birth_willingness"] * bw_rule
        + _THETA["available_time"] * avail_time
        - abs(_THETA["employment_stability_pressure"]) * couple_esp
        - abs(_THETA["career_expectation_uncertainty"]) * couple_ceu
        - abs(_THETA["relationship_time_compression"]) * couple_rtc
        - abs(_THETA["age_penalty"]) * age_pen
        - abs(_THETA["existing_children"]) * female.num_children
    )

    rule_p = sigmoid(logit)

    # LLM blending using female birth_willingness as proxy
    llm_p = female.llm_state.birth_willingness
    blended = clamp((1 - llm_weight) * rule_p + llm_weight * llm_p)

    # BDI intention factors (use female as primary)
    bdi_f = bdi.intention_factor(female, "birth")
    bdi_m = bdi.intention_factor(male, "birth")
    bdi_factor = 0.7 * bdi_f + 0.3 * bdi_m
    blended *= bdi_factor

    mechanism_penalty = math.exp(
        -0.50 * couple_esp
        -0.45 * couple_ceu
        -0.40 * couple_rtc
    )
    mechanism_penalty = clamp(mechanism_penalty, 0.70, 1.05)
    blended *= mechanism_penalty
    blended *= 0.85

    return clamp(blended, 0.0001, 0.20)


def record_birth(
    female: "YouthAgent",
    male: "YouthAgent",
    current_tick: int,
    ticks_per_year: int = 4,
) -> None:
    """Record a birth event on both partners.

    Args:
        female: The mother.
        male: The father.
        current_tick: Current simulation tick.
        ticks_per_year: Ticks per simulation year.
    """
    female.num_children += 1
    male.num_children += 1

    current_age = current_tick / ticks_per_year
    # Track first birth ages
    if female.first_birth_age is None:
        female.first_birth_age = female.age_ticks / ticks_per_year
    if male.first_birth_age is None:
        male.first_birth_age = male.age_ticks / ticks_per_year
