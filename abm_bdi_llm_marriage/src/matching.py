"""Marriage market matching module.

Implements:
- mate_value(i, j): how attractive j is to i
- seek_probability(agent): probability of entering the market this tick
- encounter(agent, candidates): select a candidate to evaluate
- date_probability(i, j): probability of starting a relationship
- marry_probability(i, j): probability of transitioning to marriage
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

from src.enums import EDUCATION_RANK, EducationLevel, Intention, RelationshipStatus
from src.utils import clamp, sigmoid

if TYPE_CHECKING:
    from src.agent import YouthAgent
    from src.bdi import BDIEngine
    from src.config import ModelConfig


# Base weights for mate-value components
_BASE_INCOME_WEIGHT = 0.30
_BASE_EDUCATION_WEIGHT = 0.35
_BASE_AGE_WEIGHT = 0.20
_BASE_TIME_COMPAT_WEIGHT = 0.15

# Lambda scaling: how much each pressure amplifies the corresponding weight
_LAMBDA_EMPLOYMENT = 0.60   # employment_stability_pressure → income weight boost
_LAMBDA_CAREER_UNCT = 0.50  # career_expectation_uncertainty → education weight boost
_LAMBDA_TIME = 0.50         # relationship_time_compression → time-compat weight boost

# Age-matching preferred gap (in years)
_PREFERRED_AGE_GAP_MALE = 2.0    # male prefers slightly younger female
_PREFERRED_AGE_GAP_FEMALE = -2.0  # female prefers slightly older male


def mate_value(
    agent_i: "YouthAgent",
    agent_j: "YouthAgent",
) -> float:
    """Compute how attractive agent_j is to agent_i.

    The three core mechanism variables modulate which attributes agent_i
    prioritises:
    - employment_stability_pressure → weights j's income / stability more
    - career_expectation_uncertainty → weights educational match more
    - relationship_time_compression → weights lifestyle compatibility more

    Args:
        agent_i: The evaluating agent.
        agent_j: The candidate partner.

    Returns:
        A value in [0, 1] representing j's attractiveness to i.
    """
    # ---- effective weights (pressure-adjusted) ----
    w_income = _BASE_INCOME_WEIGHT * (
        1.0 + _LAMBDA_EMPLOYMENT * agent_i.employment_stability_pressure
    )
    w_edu = _BASE_EDUCATION_WEIGHT * (
        1.0 + _LAMBDA_CAREER_UNCT * agent_i.career_expectation_uncertainty
    )
    w_time = _BASE_TIME_COMPAT_WEIGHT * (
        1.0 + _LAMBDA_TIME * agent_i.relationship_time_compression
    )
    w_age = _BASE_AGE_WEIGHT

    total_w = w_income + w_edu + w_time + w_age
    w_income /= total_w
    w_edu /= total_w
    w_time /= total_w
    w_age /= total_w

    # ---- income value of j ----
    income_val = clamp(agent_j.income * (0.7 + 0.3 * agent_j.employment_stability))

    # ---- education match ----
    rank_i = EDUCATION_RANK.get(agent_i.education, 1)
    rank_j = EDUCATION_RANK.get(agent_j.education, 1)
    edu_match = clamp(1.0 - abs(rank_i - rank_j) / 3.0)
    # Slight preference toward upward match (hypergamy) for female evaluators
    if agent_i.sex.value == "female" and rank_j > rank_i:
        edu_match = clamp(edu_match + 0.05)

    # ---- age match ----
    age_diff = agent_j.age - agent_i.age  # positive if j is older
    if agent_i.sex.value == "male":
        preferred_gap = _PREFERRED_AGE_GAP_MALE
    else:
        preferred_gap = _PREFERRED_AGE_GAP_FEMALE
    age_match = clamp(1.0 - abs(age_diff - preferred_gap) / 10.0)

    # ---- time / lifestyle compatibility ----
    # Agents with similar time-compression levels match better
    time_diff = abs(
        agent_i.relationship_time_compression
        - agent_j.relationship_time_compression
    )
    time_compat = clamp(1.0 - time_diff * 1.5)

    mv = (
        w_income * income_val
        + w_edu * edu_match
        + w_age * age_match
        + w_time * time_compat
    )
    return clamp(mv)


def seek_probability(
    agent: "YouthAgent",
    bdi: "BDIEngine",
    time_seek_penalty: float = 0.4,
) -> float:
    """Probability that agent enters the marriage market this tick.

    Higher time-compression reduces search; career-stabilisation intent reduces
    search further. BDI intention factor provides final adjustment.

    Args:
        agent: The agent.
        bdi: BDI engine (for intention_factor).
        time_seek_penalty: How much relationship_time_compression suppresses search.

    Returns:
        Search probability in [0, 1].
    """
    # Baseline: lower for those in committed relationships
    if agent.relationship_status == RelationshipStatus.MARRIED:
        return 0.0

    if agent.relationship_status == RelationshipStatus.DATING:
        base = 0.10
    else:
        base = 0.35

    # Time compression penalty
    p = base * (1.0 - time_seek_penalty * agent.relationship_time_compression)

    # Career-focused intentions reduce dating search
    intention = agent.intention_state.current_intention
    if intention in (Intention.STABILIZE_CAREER_ENTRY, Intention.REDUCE_CAREER_UNCERTAINTY):
        p *= 0.7

    # BDI intention factor
    p *= bdi.intention_factor(agent, "search")

    return clamp(p)


def encounter(
    agent: "YouthAgent",
    candidates: list["YouthAgent"],
    rng: random.Random,
    delta_school_meeting: float = 0.30,
) -> Optional["YouthAgent"]:
    """Sample a candidate partner from available singles.

    With probability delta_school_meeting, restrict to same educational
    stratum (structural homophily via school / university environments).
    Otherwise draw from the full candidate pool.

    Args:
        agent: The searching agent.
        candidates: Pool of eligible opposite-sex single agents.
        rng: Seeded random number generator.
        delta_school_meeting: Probability of education-structured meeting.

    Returns:
        A candidate agent, or None if no candidates available.
    """
    if not candidates:
        return None

    if rng.random() < delta_school_meeting:
        # Education-homophilous encounter
        same_edu = [c for c in candidates if c.education == agent.education]
        pool = same_edu if same_edu else candidates
    else:
        pool = candidates

    return rng.choice(pool)


def date_probability(
    agent_i: "YouthAgent",
    agent_j: "YouthAgent",
    bdi: "BDIEngine",
    llm_weight: float = 0.0,
) -> float:
    """Probability that agent_i starts dating agent_j.

    Three mechanism pressures raise the bar for entering a relationship:
    - employment_stability_pressure: economic caution before committing
    - career_expectation_uncertainty: long-term planning hesitation
    - relationship_time_compression: reduced capacity to invest in relationships

    Args:
        agent_i: The initiating agent.
        agent_j: The candidate partner.
        bdi: BDI engine.
        llm_weight: Weight placed on LLM marriage_value_score.

    Returns:
        Probability in [0, 1].
    """
    mv = mate_value(agent_i, agent_j)

    # Age pressure boosts willingness
    age_boost = agent_i.beliefs.age_pressure * 0.15

    # Rule-based base probability
    rule_p = clamp(mv * 0.6 + age_boost)

    # LLM blending
    llm_p = agent_i.llm_state.marriage_value_score
    blended = clamp((1 - llm_weight) * rule_p + llm_weight * llm_p)

    # Mechanism penalties
    penalty = (
        0.20 * agent_i.employment_stability_pressure
        + 0.15 * agent_i.career_expectation_uncertainty
        + 0.15 * agent_i.relationship_time_compression
    )
    blended = clamp(blended - penalty)

    # BDI intention factor
    blended *= bdi.intention_factor(agent_i, "date")

    return clamp(blended)


def marry_probability(
    agent_i: "YouthAgent",
    agent_j: "YouthAgent",
    bdi: "BDIEngine",
    beta_commitment: float = 0.05,
    llm_weight: float = 0.0,
) -> float:
    """Probability that agent_i transitions from dating to marriage.

    Longer relationship duration raises probability (commitment effect).
    Three mechanism pressures lower it; BDI and LLM modulate the final value.

    Args:
        agent_i: Female or male agent in the dating pair.
        agent_j: The partner.
        bdi: BDI engine.
        beta_commitment: Per-tick commitment growth coefficient.
        llm_weight: Weight placed on LLM marry_willingness.

    Returns:
        Probability in [0, 1].
    """
    mv = mate_value(agent_i, agent_j)
    duration_bonus = clamp(agent_i.relationship_duration * beta_commitment)

    # Rule-based base
    rule_p = clamp(mv * 0.4 + duration_bonus + agent_i.beliefs.age_pressure * 0.1)

    # LLM blending
    llm_p = agent_i.llm_state.marry_willingness
    blended = clamp((1 - llm_weight) * rule_p + llm_weight * llm_p)

    # Mechanism penalties: economic base delayed + uncertain future + time compression
    penalty = (
        0.25 * agent_i.employment_stability_pressure
        + 0.20 * agent_i.career_expectation_uncertainty
        + 0.15 * agent_i.relationship_time_compression
    )
    blended = clamp(blended - penalty)

    # BDI intention factor
    blended *= bdi.intention_factor(agent_i, "marry")

    return clamp(blended)


def should_switch_partner(
    agent: "YouthAgent",
    current_partner: "YouthAgent",
    candidate: "YouthAgent",
    switch_threshold: float = 0.15,
) -> bool:
    """Return True if candidate is meaningfully better than current partner.

    Partner switching only occurs when the new candidate's mate value exceeds
    the current partner's by at least switch_threshold.

    Args:
        agent: The evaluating agent.
        current_partner: Current dating partner.
        candidate: New candidate met this tick.
        switch_threshold: Required mate-value surplus for switching.

    Returns:
        True if the agent should switch partners.
    """
    mv_current = mate_value(agent, current_partner)
    mv_candidate = mate_value(agent, candidate)
    return mv_candidate > mv_current + switch_threshold
