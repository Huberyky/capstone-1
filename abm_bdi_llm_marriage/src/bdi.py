"""BDI Engine: dynamic cognitive process with memory, conflict and bounded rationality."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from src.enums import Intention, RelationshipStatus
from src.utils import age_pressure_score, clamp

if TYPE_CHECKING:
    from src.agent import YouthAgent

_MAX_INTENTION_DURATION = 12
_MIN_INTENTION_SCORE = 0.15
_CHANGE_THRESHOLD = 0.10
_RECONSIDER_MARGIN = 0.08
_MEMORY_WINDOW = 5


class BDIEngine:
    """Implements the Belief-Desire-Intention reasoning cycle."""

    def __init__(self, intention_factor_strength: float = 0.8) -> None:
        self.intention_factor_strength = intention_factor_strength

    def _append_memory(self, agent: "YouthAgent", key: str, value: float) -> None:
        mem = agent.belief_memory.setdefault(key, [])
        mem.append(clamp(value))
        if len(mem) > _MEMORY_WINDOW:
            mem.pop(0)

    def _rolling_avg(self, agent: "YouthAgent", key: str) -> float:
        vals = agent.belief_memory.get(key, [])
        return sum(vals) / len(vals) if vals else 0.5

    def _interpret_signals(self, agent: "YouthAgent") -> None:
        """Cognitive interpretation layer: objective pressures -> subjective judgments."""
        esp_avg = self._rolling_avg(agent, "employment_stability_pressure")
        ceu_avg = self._rolling_avg(agent, "career_expectation_uncertainty")
        rtc_avg = self._rolling_avg(agent, "relationship_time_compression")

        eco_readiness = clamp(
            1.0 - 0.75 * esp_avg + 0.15 * agent.employment_stability + 0.10 * agent.aspiration_level
        )
        career_predict = clamp(
            1.0 - 0.70 * ceu_avg + 0.20 * agent.uncertainty_tolerance + 0.10 * (1.0 - agent.risk_preference)
        )
        relationship_time = clamp(
            1.0 - 0.70 * rtc_avg + 0.20 * agent.available_relationship_time + 0.10 * agent.last_decision_outcome
        )

        agent.subjective_interpretation["economic_family_readiness"] = eco_readiness
        agent.subjective_interpretation["career_trajectory_predictability"] = career_predict
        agent.subjective_interpretation["time_for_relationship_investment"] = relationship_time
        agent.perceived_trend = clamp((esp_avg + ceu_avg + rtc_avg) / 3.0)

    def update_beliefs(self, agent: "YouthAgent", model_context: dict) -> None:
        """Belief update with memory-based perceived signal and smoothing."""
        rho = agent.belief_sensitivity
        b = agent.beliefs

        self._append_memory(agent, "employment_stability_pressure", agent.employment_stability_pressure)
        self._append_memory(agent, "career_expectation_uncertainty", agent.career_expectation_uncertainty)
        self._append_memory(agent, "relationship_time_compression", agent.relationship_time_compression)
        self._interpret_signals(agent)

        sig_employment_security = agent.subjective_interpretation["economic_family_readiness"]
        sig_career_predictability = agent.subjective_interpretation["career_trajectory_predictability"]
        sig_time_availability = agent.subjective_interpretation["time_for_relationship_investment"]
        sig_career_entry_confidence = clamp(1.0 - self._rolling_avg(agent, "employment_stability_pressure"))
        sig_lt_planning_conf = clamp(
            1.0
            - 0.65 * self._rolling_avg(agent, "career_expectation_uncertainty")
            - 0.25 * self._rolling_avg(agent, "employment_stability_pressure")
            + 0.10 * agent.uncertainty_tolerance
        )

        sig_marriage_afford = clamp(
            agent.income * 0.45
            + agent.employment_stability * 0.30
            + 0.25 * sig_employment_security
        )
        sig_childbearing_afford = clamp(
            agent.income * 0.35
            + agent.employment_stability * 0.25
            + 0.25 * sig_employment_security
            + 0.15 * sig_time_availability
            - 0.10 * (1.0 - sig_career_predictability)
        )

        if agent.relationship_status != RelationshipStatus.SINGLE:
            sig_partner_quality = clamp(b.partner_quality + 0.10 * agent.last_decision_outcome)
            sig_rel_stability = clamp(
                0.25 + 0.45 * min(agent.relationship_duration / 20.0, 1.0) + 0.30 * sig_time_availability
            )
        else:
            sig_partner_quality = 0.0
            sig_rel_stability = 0.0

        peak = (
            model_context.get("sigma_age_pressure_female", 27.0)
            if agent.sex.value == "female"
            else model_context.get("sigma_age_pressure_male", 30.0)
        )
        sig_age_pressure = age_pressure_score(agent.age, peak, width=6.0)
        sig_social_norm = clamp(0.15 + 0.45 * sig_age_pressure + 0.10 * agent.aspiration_level)

        b.employment_security = clamp((1 - rho) * b.employment_security + rho * sig_employment_security)
        b.career_predictability = clamp((1 - rho) * b.career_predictability + rho * sig_career_predictability)
        b.time_availability = clamp((1 - rho) * b.time_availability + rho * sig_time_availability)
        b.marriage_affordability = clamp((1 - rho) * b.marriage_affordability + rho * sig_marriage_afford)
        b.childbearing_affordability = clamp((1 - rho) * b.childbearing_affordability + rho * sig_childbearing_afford)
        b.partner_quality = clamp((1 - rho) * b.partner_quality + rho * sig_partner_quality)
        b.relationship_stability = clamp((1 - rho) * b.relationship_stability + rho * sig_rel_stability)
        b.age_pressure = clamp((1 - rho) * b.age_pressure + rho * sig_age_pressure)
        b.social_norm_pressure = clamp((1 - rho) * b.social_norm_pressure + rho * sig_social_norm)
        b.career_entry_confidence = clamp((1 - rho) * b.career_entry_confidence + rho * sig_career_entry_confidence)
        b.long_term_life_planning_confidence = clamp(
            (1 - rho) * b.long_term_life_planning_confidence + rho * sig_lt_planning_conf
        )

        d = abs(b.career_predictability - sig_career_predictability)
        t = abs(b.time_availability - sig_time_availability)
        e = abs(b.employment_security - sig_employment_security)
        agent.cognitive_consistency_score = clamp(1.0 - (d + t + e) / 3.0)

    def generate_desires(self, agent: "YouthAgent") -> None:
        b = agent.beliefs
        d = agent.desires

        if agent.relationship_status == RelationshipStatus.SINGLE:
            d.seek_partner = clamp(0.28 + 0.25 * b.age_pressure + 0.20 * b.social_norm_pressure + 0.15 * b.time_availability)
        else:
            d.seek_partner = 0.0

        d.maintain_relationship = clamp(0.30 + 0.35 * b.relationship_stability + 0.20 * b.time_availability)
        d.marry = clamp(
            0.15 + 0.20 * b.relationship_stability + 0.18 * b.marriage_affordability + 0.12 * b.partner_quality
            + 0.12 * b.age_pressure + 0.13 * b.long_term_life_planning_confidence
            - 0.08 * agent.career_expectation_uncertainty + 0.08 * agent.uncertainty_tolerance
            - 0.08 * agent.risk_preference
        )
        d.have_child = clamp(
            0.12 + 0.16 * agent.llm_state.fertility_value_score + 0.16 * b.childbearing_affordability
            + 0.15 * b.relationship_stability + 0.09 * b.time_availability + 0.08 * b.age_pressure
            + 0.08 * agent.uncertainty_tolerance - 0.10 * agent.risk_preference
            - min(agent.num_children * 0.15, 0.45)
        )
        d.delay_marriage = clamp(
            0.10 + 0.26 * agent.employment_stability_pressure + 0.24 * agent.career_expectation_uncertainty
            + 0.18 * agent.relationship_time_compression + 0.12 * agent.risk_preference - 0.10 * b.age_pressure
        )
        d.delay_childbearing = clamp(
            0.10 + 0.24 * agent.employment_stability_pressure + 0.23 * agent.career_expectation_uncertainty
            + 0.20 * agent.relationship_time_compression + 0.10 * agent.risk_preference
            + min(agent.num_children * 0.10, 0.30) - 0.08 * b.age_pressure
        )
        d.stabilize_career_entry = clamp(0.12 + 0.45 * agent.employment_stability_pressure + 0.25 * agent.employment_entry_delay)
        d.reduce_career_uncertainty = clamp(0.12 + 0.44 * agent.career_expectation_uncertainty + 0.30 * agent.reskilling_need)
        d.protect_personal_time = clamp(0.12 + 0.45 * agent.relationship_time_compression + 0.30 * agent.work_life_boundary_blurring)
        d.personal_autonomy = clamp(0.25 + 0.15 * agent.relationship_time_compression - 0.10 * b.age_pressure)

        family_push = (d.marry + d.have_child) / 2.0
        delay_pull = (d.delay_marriage + d.delay_childbearing) / 2.0
        career_pull = (d.stabilize_career_entry + d.reduce_career_uncertainty + d.protect_personal_time) / 3.0
        agent.desire_conflict_index = clamp((abs(family_push - delay_pull) + abs(family_push - career_pull) + abs(delay_pull - career_pull)) / 1.5)

    def should_reconsider_intention(self, agent: "YouthAgent") -> bool:
        ist = agent.intention_state
        if ist.current_intention is None:
            return True
        if ist.intention_duration >= _MAX_INTENTION_DURATION:
            return True
        if ist.intention_score < _MIN_INTENTION_SCORE:
            return True

        if abs(agent.employment_stability - agent.prev_employment_stability) > _CHANGE_THRESHOLD:
            return True
        if abs(agent.career_predictability - agent.prev_career_predictability) > _CHANGE_THRESHOLD:
            return True

        current = agent.relationship_status
        intention = ist.current_intention
        if current == RelationshipStatus.SINGLE and intention in (Intention.MAINTAIN_RELATIONSHIP, Intention.MARRY_PARTNER, Intention.HAVE_CHILD):
            return True
        if current == RelationshipStatus.MARRIED and intention == Intention.SEARCH_PARTNER:
            return True

        return False

    def _score_candidates(self, agent: "YouthAgent") -> dict[Intention, float]:
        b = agent.beliefs
        d = agent.desires
        rel = agent.relationship_status
        llm = agent.llm_state
        scores: dict[Intention, float] = {}

        if rel == RelationshipStatus.SINGLE:
            scores[Intention.SEARCH_PARTNER] = clamp(d.seek_partner * b.time_availability - 0.12 * agent.relationship_time_compression)
            scores[Intention.START_DATING] = clamp(d.seek_partner * 0.75 * b.marriage_affordability)
        if rel != RelationshipStatus.SINGLE:
            scores[Intention.MAINTAIN_RELATIONSHIP] = clamp(d.maintain_relationship * b.relationship_stability)
        if rel == RelationshipStatus.DATING:
            scores[Intention.MARRY_PARTNER] = clamp(
                d.marry * b.relationship_stability * b.marriage_affordability * b.long_term_life_planning_confidence
                - 0.20 * agent.employment_stability_pressure - 0.15 * agent.career_expectation_uncertainty
                + 0.08 * llm.marry_willingness + 0.10 * agent.uncertainty_tolerance - 0.12 * agent.risk_preference
            )
            scores[Intention.POSTPONE_MARRIAGE] = clamp(d.delay_marriage + 0.08 * (1.0 - b.long_term_life_planning_confidence))
            scores[Intention.SWITCH_PARTNER] = clamp((1.0 - b.relationship_stability) * 0.35)
            scores[Intention.EXIT_RELATIONSHIP] = clamp((1.0 - b.relationship_stability) * 0.25 + 0.20 * d.delay_marriage)
        if rel in (RelationshipStatus.SINGLE, RelationshipStatus.DATING):
            scores.setdefault(Intention.POSTPONE_MARRIAGE, clamp(d.delay_marriage))
        if rel == RelationshipStatus.MARRIED:
            scores[Intention.HAVE_CHILD] = clamp(
                d.have_child * b.childbearing_affordability * b.relationship_stability * b.time_availability
                - 0.18 * agent.employment_stability_pressure - 0.15 * agent.career_expectation_uncertainty
                + 0.08 * llm.birth_willingness + 0.10 * agent.uncertainty_tolerance - 0.12 * agent.risk_preference
            )
            scores[Intention.POSTPONE_CHILDBEARING] = clamp(d.delay_childbearing)

        scores[Intention.STABILIZE_CAREER_ENTRY] = clamp(d.stabilize_career_entry + 0.10 * agent.risk_preference)
        scores[Intention.REDUCE_CAREER_UNCERTAINTY] = clamp(d.reduce_career_uncertainty + 0.08 * agent.risk_preference)
        scores[Intention.PROTECT_PERSONAL_TIME] = clamp(d.protect_personal_time)
        return scores

    def select_intention(self, agent: "YouthAgent") -> None:
        scores = self._score_candidates(agent)
        if not scores:
            return

        ist = agent.intention_state
        prev_intention = ist.current_intention

        for k in list(scores.keys()):
            scores[k] = clamp(scores[k] + (agent.decision_noise * (2 * random.random() - 1)))

        # inertia: reinforce current intention unless clear challenger appears
        if prev_intention in scores:
            scores[prev_intention] = clamp(scores[prev_intention] + 0.10 + 0.15 * ist.commitment_strength)

        best_intention = max(scores, key=lambda key: scores[key])
        best_score = scores[best_intention]

        if prev_intention is not None and prev_intention in scores:
            incumbent = scores[prev_intention]
            threshold = _RECONSIDER_MARGIN + 0.10 * agent.cognitive_consistency_score
            if best_intention != prev_intention and best_score < incumbent + threshold:
                best_intention = prev_intention
                best_score = incumbent

        if prev_intention != best_intention:
            if prev_intention is not None:
                agent.intention_duration_history.append(ist.intention_duration)
                agent.intention_switch_count += 1
            ist.current_intention = best_intention
            ist.intention_duration = 0
        else:
            ist.intention_duration += 1

        ist.intention_score = best_score
        ist.commitment_strength = clamp(0.6 * ist.commitment_strength + 0.4 * best_score)
        agent.last_decision_outcome = clamp(0.5 * agent.last_decision_outcome + 0.5 * (1.0 - agent.desire_conflict_index))

    def intention_factor(self, agent: "YouthAgent", action: str) -> float:
        intention = agent.intention_state.current_intention
        if intention is None:
            return 1.0
        s = self.intention_factor_strength

        if action == "search":
            if intention == Intention.SEARCH_PARTNER:
                return 1.0 + 0.30 * s
            if intention in (Intention.PROTECT_PERSONAL_TIME, Intention.STABILIZE_CAREER_ENTRY, Intention.REDUCE_CAREER_UNCERTAINTY):
                return 1.0 - 0.40 * s
            if intention == Intention.POSTPONE_MARRIAGE:
                return 1.0 - 0.20 * s
        elif action == "date":
            if intention == Intention.START_DATING:
                return 1.0 + 0.30 * s
            if intention in (Intention.PROTECT_PERSONAL_TIME, Intention.STABILIZE_CAREER_ENTRY):
                return 1.0 - 0.40 * s
            if intention == Intention.POSTPONE_MARRIAGE:
                return 1.0 - 0.25 * s
        elif action == "marry":
            if intention == Intention.MARRY_PARTNER:
                return 1.0 + 0.40 * s
            if intention == Intention.POSTPONE_MARRIAGE:
                return 1.0 - 0.60 * s
            if intention in (Intention.STABILIZE_CAREER_ENTRY, Intention.REDUCE_CAREER_UNCERTAINTY):
                return 1.0 - 0.40 * s
            if intention == Intention.PROTECT_PERSONAL_TIME:
                return 1.0 - 0.30 * s
        elif action == "birth":
            if intention == Intention.HAVE_CHILD:
                return 1.0 + 0.45 * s
            if intention == Intention.POSTPONE_CHILDBEARING:
                return 1.0 - 0.70 * s
            if intention in (Intention.PROTECT_PERSONAL_TIME, Intention.REDUCE_CAREER_UNCERTAINTY):
                return 1.0 - 0.45 * s
            if intention == Intention.STABILIZE_CAREER_ENTRY:
                return 1.0 - 0.40 * s
        return 1.0
