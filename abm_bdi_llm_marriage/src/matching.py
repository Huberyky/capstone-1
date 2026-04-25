"""Marriage market matching module."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Optional

from src.enums import EDUCATION_RANK, Intention, RelationshipStatus
from src.utils import clamp

if TYPE_CHECKING:
    from src.agent import YouthAgent
    from src.bdi import BDIEngine

_BASE_INCOME_WEIGHT = 0.30
_BASE_EDUCATION_WEIGHT = 0.35
_BASE_AGE_WEIGHT = 0.20
_BASE_TIME_COMPAT_WEIGHT = 0.15

_LAMBDA_EMPLOYMENT = 0.55
_LAMBDA_CAREER_UNCT = 0.50
_LAMBDA_TIME = 0.45

_PREFERRED_AGE_GAP_MALE = 2.0
_PREFERRED_AGE_GAP_FEMALE = -2.0


def set_matching_lambdas(le: float, lc: float, lt: float) -> None:
    global _LAMBDA_EMPLOYMENT, _LAMBDA_CAREER_UNCT, _LAMBDA_TIME
    _LAMBDA_EMPLOYMENT, _LAMBDA_CAREER_UNCT, _LAMBDA_TIME = le, lc, lt


def _mechanism_penalty(agent: "YouthAgent") -> float:
    raw = math.exp(
        -_LAMBDA_EMPLOYMENT * agent.employment_stability_pressure
        -_LAMBDA_CAREER_UNCT * agent.career_expectation_uncertainty
        -_LAMBDA_TIME * agent.relationship_time_compression
    )
    return clamp(raw, 0.70, 1.05)


def mate_value(agent_i: "YouthAgent", agent_j: "YouthAgent") -> float:
    w_income = _BASE_INCOME_WEIGHT * (1.0 + _LAMBDA_EMPLOYMENT * agent_i.employment_stability_pressure)
    w_edu = _BASE_EDUCATION_WEIGHT * (1.0 + _LAMBDA_CAREER_UNCT * agent_i.career_expectation_uncertainty)
    w_time = _BASE_TIME_COMPAT_WEIGHT * (1.0 + _LAMBDA_TIME * agent_i.relationship_time_compression)
    w_age = _BASE_AGE_WEIGHT

    total_w = w_income + w_edu + w_time + w_age
    w_income, w_edu, w_time, w_age = w_income / total_w, w_edu / total_w, w_time / total_w, w_age / total_w

    income_val = clamp(agent_j.income * (0.7 + 0.3 * agent_j.employment_stability))

    rank_i = EDUCATION_RANK.get(agent_i.education, 1)
    rank_j = EDUCATION_RANK.get(agent_j.education, 1)
    edu_match = clamp(1.0 - abs(rank_i - rank_j) / 3.0)
    if agent_i.sex.value == "female" and rank_j > rank_i:
        edu_match = clamp(edu_match + 0.05)

    age_diff = agent_j.age - agent_i.age
    preferred_gap = _PREFERRED_AGE_GAP_MALE if agent_i.sex.value == "male" else _PREFERRED_AGE_GAP_FEMALE
    age_match = clamp(1.0 - abs(age_diff - preferred_gap) / 10.0)

    time_diff = abs(agent_i.relationship_time_compression - agent_j.relationship_time_compression)
    time_compat = clamp(1.0 - time_diff * 1.5)

    return clamp(w_income * income_val + w_edu * edu_match + w_age * age_match + w_time * time_compat)


def seek_probability(agent: "YouthAgent", bdi: "BDIEngine", time_seek_penalty: float = 0.4) -> float:
    if agent.relationship_status == RelationshipStatus.MARRIED:
        return 0.0
    base = 0.10 if agent.relationship_status == RelationshipStatus.DATING else 0.35
    p = base * (1.0 - time_seek_penalty * agent.relationship_time_compression)

    intention = agent.intention_state.current_intention
    if intention in (Intention.STABILIZE_CAREER_ENTRY, Intention.REDUCE_CAREER_UNCERTAINTY):
        p *= 0.75

    p *= bdi.intention_factor(agent, "search")
    p *= _mechanism_penalty(agent)
    return clamp(p, 0.02, 0.80)


def encounter(agent: "YouthAgent", candidates: list["YouthAgent"], rng: random.Random, delta_school_meeting: float = 0.30) -> Optional["YouthAgent"]:
    if not candidates:
        return None
    if rng.random() < delta_school_meeting:
        same_edu = [c for c in candidates if c.education == agent.education]
        pool = same_edu if same_edu else candidates
    else:
        pool = candidates
    return rng.choice(pool)


def date_probability(agent_i: "YouthAgent", agent_j: "YouthAgent", bdi: "BDIEngine", llm_weight: float = 0.0) -> float:
    mv = mate_value(agent_i, agent_j)
    age_boost = agent_i.beliefs.age_pressure * 0.12
    rule_p = clamp(mv * 0.55 + age_boost)

    llm_p = agent_i.llm_state.marriage_value_score
    blended = clamp((1 - llm_weight) * rule_p + llm_weight * llm_p)

    blended *= _mechanism_penalty(agent_i)
    blended *= bdi.intention_factor(agent_i, "date")
    return clamp(blended, 0.01, 0.60)


def marry_probability(agent_i: "YouthAgent", agent_j: "YouthAgent", bdi: "BDIEngine", beta_commitment: float = 0.05, llm_weight: float = 0.0) -> float:
    mv = mate_value(agent_i, agent_j)
    duration_bonus = clamp(agent_i.relationship_duration * beta_commitment)
    rule_p = clamp(mv * 0.35 + duration_bonus + agent_i.beliefs.age_pressure * 0.08)

    llm_p = agent_i.llm_state.marry_willingness
    blended = clamp((1 - llm_weight) * rule_p + llm_weight * llm_p)

    blended *= _mechanism_penalty(agent_i)
    blended *= bdi.intention_factor(agent_i, "marry")
    return clamp(blended, 0.001, 0.30)


def should_switch_partner(agent: "YouthAgent", current_partner: "YouthAgent", candidate: "YouthAgent", switch_threshold: float = 0.15) -> bool:
    mv_current = mate_value(agent, current_partner)
    mv_candidate = mate_value(agent, candidate)
    return mv_candidate > mv_current + switch_threshold
