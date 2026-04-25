"""BDI Engine: Belief-Desire-Intention reasoning for YouthAgent.

Role in the three-layer architecture
--------------------------------------
ABM  → models how the social world operates (markets, encounters, demographics).
BDI  → models HOW agents think: beliefs formed from perceived signals,
        desires derived from beliefs, intentions selected from desires.
LLM  → models HOW agents interpret and narrate their own situation
        (subjective valuations fed back into beliefs/intentions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.enums import Intention, RelationshipStatus
from src.utils import clamp

if TYPE_CHECKING:
    from src.agent import YouthAgent


# Maximum ticks an intention persists before forced reconsideration
_MAX_INTENTION_DURATION = 12

# Minimum intention score below which reconsideration is triggered
_MIN_INTENTION_SCORE = 0.15

# Threshold for detecting "significant change" in labour variables
_CHANGE_THRESHOLD = 0.10


class BDIEngine:
    """Implements the Belief-Desire-Intention reasoning cycle.

    This engine is stateless beyond the agent objects it mutates; it can be
    shared across all agents in a model run.
    """

    # -----------------------------------------------------------------------
    # 1. Belief update
    # -----------------------------------------------------------------------

    def update_beliefs(
        self,
        agent: "YouthAgent",
        model_context: dict,
    ) -> None:
        """Update agent beliefs with memory inertia (Bayesian-style smoothing).

        Beliefs are subjective perceptions, not the objective variable values.
        new_belief = (1 - rho) * old_belief + rho * perceived_signal

        Args:
            agent: The agent whose beliefs are updated.
            model_context: Dictionary of global context (tick, scenario, etc.).
        """
        rho = agent.belief_sensitivity
        b = agent.beliefs

        # --- signals derived from objective mechanism variables ---
        sig_employment_security = clamp(
            1.0 - agent.employment_stability_pressure
        )
        sig_career_predictability = clamp(
            1.0 - agent.career_expectation_uncertainty
        )
        sig_time_availability = clamp(
            1.0 - agent.relationship_time_compression
        )
        sig_career_entry_confidence = clamp(
            1.0 - agent.employment_entry_delay
        )
        sig_lt_planning_conf = clamp(
            1.0 - agent.career_expectation_uncertainty * 0.8
            - agent.employment_stability_pressure * 0.2
        )

        # Marriage affordability: income + employment stability − pressure
        sig_marriage_afford = clamp(
            agent.income * 0.5
            + agent.employment_stability * 0.3
            - agent.employment_stability_pressure * 0.2
        )

        # Childbearing affordability: higher bar than marriage
        sig_childbearing_afford = clamp(
            agent.income * 0.4
            + agent.employment_stability * 0.3
            - agent.employment_stability_pressure * 0.2
            - agent.career_expectation_uncertainty * 0.1
        )

        # Partner quality: only meaningful if in a relationship
        if agent.relationship_status != RelationshipStatus.SINGLE:
            sig_partner_quality = clamp(b.partner_quality)  # stays sticky
        else:
            sig_partner_quality = 0.0

        # Relationship stability: function of duration and own time availability
        if agent.relationship_status in (
            RelationshipStatus.DATING, RelationshipStatus.MARRIED
        ):
            sig_rel_stability = clamp(
                0.3
                + 0.4 * min(agent.relationship_duration / 20.0, 1.0)
                + 0.3 * sig_time_availability
            )
        else:
            sig_rel_stability = 0.0

        # Age pressure: bell-shaped around peak ages
        tick_age = agent.age
        from src.utils import age_pressure_score
        peak = (
            model_context.get("sigma_age_pressure_female", 27.0)
            if agent.sex.value == "female"
            else model_context.get("sigma_age_pressure_male", 30.0)
        )
        sig_age_pressure = age_pressure_score(tick_age, peak, width=6.0)

        # Social norm pressure: proxy from age and relationship status
        if agent.relationship_status == RelationshipStatus.SINGLE:
            sig_social_norm = clamp(0.2 + 0.5 * sig_age_pressure)
        else:
            sig_social_norm = clamp(0.1 + 0.2 * sig_age_pressure)

        # --- apply inertia update ---
        b.employment_security = clamp(
            (1 - rho) * b.employment_security + rho * sig_employment_security
        )
        b.career_predictability = clamp(
            (1 - rho) * b.career_predictability + rho * sig_career_predictability
        )
        b.time_availability = clamp(
            (1 - rho) * b.time_availability + rho * sig_time_availability
        )
        b.marriage_affordability = clamp(
            (1 - rho) * b.marriage_affordability + rho * sig_marriage_afford
        )
        b.childbearing_affordability = clamp(
            (1 - rho) * b.childbearing_affordability + rho * sig_childbearing_afford
        )
        b.partner_quality = clamp(
            (1 - rho) * b.partner_quality + rho * sig_partner_quality
        )
        b.relationship_stability = clamp(
            (1 - rho) * b.relationship_stability + rho * sig_rel_stability
        )
        b.age_pressure = clamp(
            (1 - rho) * b.age_pressure + rho * sig_age_pressure
        )
        b.social_norm_pressure = clamp(
            (1 - rho) * b.social_norm_pressure + rho * sig_social_norm
        )
        b.career_entry_confidence = clamp(
            (1 - rho) * b.career_entry_confidence + rho * sig_career_entry_confidence
        )
        b.long_term_life_planning_confidence = clamp(
            (1 - rho) * b.long_term_life_planning_confidence + rho * sig_lt_planning_conf
        )

    # -----------------------------------------------------------------------
    # 2. Desire generation
    # -----------------------------------------------------------------------

    def generate_desires(self, agent: "YouthAgent") -> None:
        """Compute desire strengths from current beliefs.

        Desires may conflict (e.g. 'marry' and 'delay_marriage' can both be
        high). That conflict is resolved later in select_intention.

        Args:
            agent: The agent whose desires are updated in-place.
        """
        b = agent.beliefs
        d = agent.desires

        # seek_partner: driven by single status, age pressure, time availability
        if agent.relationship_status == RelationshipStatus.SINGLE:
            d.seek_partner = clamp(
                0.3
                + 0.3 * b.age_pressure
                + 0.2 * b.social_norm_pressure
                - 0.2 * agent.relationship_time_compression
            )
        else:
            d.seek_partner = 0.0

        # maintain_relationship: in a relationship
        if agent.relationship_status != RelationshipStatus.SINGLE:
            d.maintain_relationship = clamp(
                0.4 + 0.3 * b.relationship_stability + 0.3 * b.time_availability
            )
        else:
            d.maintain_relationship = 0.0

        # marry: positive from relationship quality, affordability, confidence
        d.marry = clamp(
            0.2
            + 0.25 * b.relationship_stability
            + 0.15 * b.marriage_affordability
            + 0.10 * b.partner_quality
            + 0.15 * b.age_pressure
            + 0.10 * b.long_term_life_planning_confidence
            - 0.20 * agent.career_expectation_uncertainty
            - 0.10 * agent.relationship_time_compression
        )

        # have_child: depends on relationship + affordability + time
        child_penalty = min(agent.num_children * 0.15, 0.45)
        d.have_child = clamp(
            0.15
            + 0.20 * agent.llm_state.fertility_value_score
            + 0.15 * b.childbearing_affordability
            + 0.15 * b.relationship_stability
            + 0.10 * b.age_pressure
            + 0.10 * b.time_availability
            - 0.15 * agent.employment_stability_pressure
            - 0.10 * agent.career_expectation_uncertainty
            - 0.10 * agent.relationship_time_compression
            - child_penalty
        )

        # delay_marriage: driven by AI-induced pressures
        d.delay_marriage = clamp(
            0.1
            + 0.35 * agent.employment_stability_pressure
            + 0.30 * agent.career_expectation_uncertainty
            + 0.20 * agent.relationship_time_compression
            - 0.15 * b.age_pressure
        )

        # delay_childbearing
        d.delay_childbearing = clamp(
            0.1
            + 0.30 * agent.employment_stability_pressure
            + 0.25 * agent.career_expectation_uncertainty
            + 0.20 * agent.relationship_time_compression
            + min(agent.num_children * 0.10, 0.30)
            - 0.10 * b.age_pressure
        )

        # stabilize_career_entry: driven by instability pressure
        d.stabilize_career_entry = clamp(
            0.15 + 0.55 * agent.employment_stability_pressure
            + 0.30 * agent.employment_entry_delay
        )

        # reduce_career_uncertainty: driven by uncertainty + reskilling
        d.reduce_career_uncertainty = clamp(
            0.15 + 0.50 * agent.career_expectation_uncertainty
            + 0.35 * agent.reskilling_need
        )

        # protect_personal_time: driven by time compression
        d.protect_personal_time = clamp(
            0.15 + 0.50 * agent.relationship_time_compression
            + 0.35 * agent.work_life_boundary_blurring
        )

        # personal_autonomy: background desire for independence
        d.personal_autonomy = clamp(
            0.3
            + 0.2 * agent.relationship_time_compression
            - 0.1 * b.age_pressure
        )

    # -----------------------------------------------------------------------
    # 3. Reconsideration check
    # -----------------------------------------------------------------------

    def should_reconsider_intention(self, agent: "YouthAgent") -> bool:
        """Return True when the agent should re-evaluate its current intention.

        Triggers: no intention, stale intention, low score, significant labour
        shock, completed intention, or relationship/status shift.

        Args:
            agent: The agent to evaluate.

        Returns:
            True if intention should be reconsidered.
        """
        ist = agent.intention_state

        if ist.current_intention is None:
            return True

        if ist.intention_duration >= _MAX_INTENTION_DURATION:
            return True

        if ist.intention_score < _MIN_INTENTION_SCORE:
            return True

        # Detect significant shift in labour situation
        if abs(agent.employment_stability - agent.prev_employment_stability) > _CHANGE_THRESHOLD:
            return True
        if abs(agent.career_predictability - agent.prev_career_predictability) > _CHANGE_THRESHOLD:
            return True

        # Relationship status shifts always require reconsideration
        current = agent.relationship_status
        intention = ist.current_intention
        if current == RelationshipStatus.SINGLE and intention in (
            Intention.MAINTAIN_RELATIONSHIP,
            Intention.MARRY_PARTNER,
            Intention.HAVE_CHILD,
        ):
            return True
        if current == RelationshipStatus.MARRIED and intention == Intention.SEARCH_PARTNER:
            return True

        return False

    # -----------------------------------------------------------------------
    # 4. Intention selection
    # -----------------------------------------------------------------------

    def select_intention(self, agent: "YouthAgent") -> None:
        """Choose the intention with the highest score.

        intention_score = desire_strength * feasibility − constraint_penalty

        Sets agent.intention_state in-place.

        Args:
            agent: The agent whose intention is updated.
        """
        b = agent.beliefs
        d = agent.desires
        rel = agent.relationship_status
        llm = agent.llm_state

        scores: dict[Intention, float] = {}

        # search_partner (only for single / dating agents)
        if rel == RelationshipStatus.SINGLE:
            scores[Intention.SEARCH_PARTNER] = clamp(
                d.seek_partner * b.time_availability
                - 0.2 * agent.relationship_time_compression
            )

        # maintain_relationship
        if rel != RelationshipStatus.SINGLE:
            scores[Intention.MAINTAIN_RELATIONSHIP] = clamp(
                d.maintain_relationship * b.relationship_stability
            )

        # start_dating (single only)
        if rel == RelationshipStatus.SINGLE:
            scores[Intention.START_DATING] = clamp(
                d.seek_partner * 0.8 * b.marriage_affordability
                - 0.1 * agent.employment_stability_pressure
            )

        # switch_partner (dating only)
        if rel == RelationshipStatus.DATING:
            scores[Intention.SWITCH_PARTNER] = clamp(
                (1.0 - b.relationship_stability) * 0.4
                - 0.2 * agent.relationship_time_compression
            )

        # marry_partner (dating only)
        if rel == RelationshipStatus.DATING:
            scores[Intention.MARRY_PARTNER] = clamp(
                d.marry
                * b.relationship_stability
                * b.marriage_affordability
                * b.long_term_life_planning_confidence
                - 0.3 * agent.employment_stability_pressure
                - 0.2 * agent.career_expectation_uncertainty
                - 0.15 * agent.relationship_time_compression
                + 0.1 * llm.marry_willingness
            )

        # postpone_marriage
        if rel in (RelationshipStatus.SINGLE, RelationshipStatus.DATING):
            scores[Intention.POSTPONE_MARRIAGE] = clamp(
                d.delay_marriage
                + 0.1 * (1.0 - b.long_term_life_planning_confidence)
            )

        # have_child (married only)
        if rel == RelationshipStatus.MARRIED:
            child_pen = min(agent.num_children * 0.2, 0.6)
            scores[Intention.HAVE_CHILD] = clamp(
                d.have_child
                * b.childbearing_affordability
                * b.relationship_stability
                * b.time_availability
                - 0.25 * agent.employment_stability_pressure
                - 0.20 * agent.career_expectation_uncertainty
                - 0.15 * agent.relationship_time_compression
                - child_pen
                + 0.1 * llm.birth_willingness
            )

        # postpone_childbearing (married only)
        if rel == RelationshipStatus.MARRIED:
            scores[Intention.POSTPONE_CHILDBEARING] = clamp(
                d.delay_childbearing
            )

        # stabilize_career_entry: universal career-related intention
        scores[Intention.STABILIZE_CAREER_ENTRY] = clamp(
            d.stabilize_career_entry
            * (0.5 + 0.5 * agent.employment_stability_pressure)
            + 0.2 * agent.employment_entry_delay
        )

        # reduce_career_uncertainty
        scores[Intention.REDUCE_CAREER_UNCERTAINTY] = clamp(
            d.reduce_career_uncertainty
            * (0.5 + 0.5 * agent.career_expectation_uncertainty)
            + 0.2 * agent.reskilling_need
        )

        # protect_personal_time
        scores[Intention.PROTECT_PERSONAL_TIME] = clamp(
            d.protect_personal_time
            * (0.5 + 0.5 * agent.relationship_time_compression)
            + 0.2 * agent.work_life_boundary_blurring
        )

        # exit_relationship (dating, low stability)
        if rel == RelationshipStatus.DATING:
            scores[Intention.EXIT_RELATIONSHIP] = clamp(
                (1.0 - b.relationship_stability) * 0.3
                + 0.2 * d.delay_marriage
            )

        if not scores:
            return

        best_intention = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intention]

        ist = agent.intention_state
        if ist.current_intention != best_intention:
            ist.current_intention = best_intention
            ist.intention_duration = 0
            ist.commitment_strength = clamp(best_score)
        else:
            ist.intention_duration += 1

        ist.intention_score = best_score

    # -----------------------------------------------------------------------
    # 5. Intention factor
    # -----------------------------------------------------------------------

    def intention_factor(self, agent: "YouthAgent", action: str) -> float:
        """Return a multiplier modifying ABM action probability based on intention.

        Values > 1 amplify the action; values < 1 suppress it.

        Args:
            agent: The acting agent.
            action: One of "search", "date", "marry", "birth".

        Returns:
            A positive float multiplier (typically in [0.3, 1.5]).
        """
        intention = agent.intention_state.current_intention
        if intention is None:
            return 1.0

        # --- search ---
        if action == "search":
            if intention == Intention.SEARCH_PARTNER:
                return 1.3
            if intention in (
                Intention.PROTECT_PERSONAL_TIME,
                Intention.STABILIZE_CAREER_ENTRY,
                Intention.REDUCE_CAREER_UNCERTAINTY,
            ):
                return 0.6
            if intention == Intention.POSTPONE_MARRIAGE:
                return 0.8

        # --- date (enter a new relationship) ---
        elif action == "date":
            if intention == Intention.START_DATING:
                return 1.3
            if intention in (
                Intention.PROTECT_PERSONAL_TIME,
                Intention.STABILIZE_CAREER_ENTRY,
            ):
                return 0.6
            if intention == Intention.POSTPONE_MARRIAGE:
                return 0.75

        # --- marry ---
        elif action == "marry":
            if intention == Intention.MARRY_PARTNER:
                return 1.4
            if intention == Intention.POSTPONE_MARRIAGE:
                return 0.4
            if intention in (
                Intention.STABILIZE_CAREER_ENTRY,
                Intention.REDUCE_CAREER_UNCERTAINTY,
            ):
                return 0.6
            if intention == Intention.PROTECT_PERSONAL_TIME:
                return 0.7

        # --- birth ---
        elif action == "birth":
            if intention == Intention.HAVE_CHILD:
                return 1.5
            if intention == Intention.POSTPONE_CHILDBEARING:
                return 0.3
            if intention in (
                Intention.PROTECT_PERSONAL_TIME,
                Intention.REDUCE_CAREER_UNCERTAINTY,
            ):
                return 0.5
            if intention == Intention.STABILIZE_CAREER_ENTRY:
                return 0.6

        return 1.0
