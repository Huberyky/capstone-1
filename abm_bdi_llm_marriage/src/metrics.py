"""Metrics module: compute population-level indicators from agent snapshots."""

from __future__ import annotations

from typing import Any

from src.enums import EDUCATION_RANK, RelationshipStatus, Sex


def compute_metrics(
    agents: list[Any],
    ticks_per_year: int,
    llm_stats: Any,
    current_tick: int,
) -> dict[str, float]:
    """Compute all simulation metrics from the current agent population.

    Args:
        agents: List of YouthAgent objects.
        ticks_per_year: Ticks per simulated year.
        llm_stats: LLMCallStats object.
        current_tick: Current global tick (used for intention share computation).

    Returns:
        Dictionary of metric name → float value.
    """
    alive = [a for a in agents if a.alive]
    adults = [a for a in alive if 18 <= a.age_ticks / ticks_per_year <= 45]
    women = [a for a in adults if a.sex == Sex.FEMALE]
    men = [a for a in adults if a.sex == Sex.MALE]

    metrics: dict[str, float] = {}

    # ---- Marriage and fertility behaviour ----

    # First marriage rate (ever married among 18–45)
    metrics["first_marriage_rate_18_45"] = (
        len([a for a in adults if a.first_marriage_age is not None]) / max(len(adults), 1)
    )

    # First birth rate (ever had a child, among 18–45)
    metrics["first_birth_rate_18_45"] = (
        len([a for a in adults if a.first_birth_age is not None]) / max(len(adults), 1)
    )

    # Female first birth rate
    metrics["female_first_birth_rate_18_45"] = (
        len([a for a in women if a.first_birth_age is not None]) / max(len(women), 1)
    )

    # Average children among all adults
    metrics["avg_children_adults"] = (
        sum(a.num_children for a in adults) / max(len(adults), 1)
    )

    # Average children among women 18–45
    metrics["avg_children_women_18_45"] = (
        sum(a.num_children for a in women) / max(len(women), 1)
    )

    # Mean first marriage age
    fma_list = [a.first_marriage_age for a in adults if a.first_marriage_age is not None]
    metrics["mean_first_marriage_age"] = sum(fma_list) / max(len(fma_list), 1)

    # Mean first birth age
    fba_list = [a.first_birth_age for a in adults if a.first_birth_age is not None]
    metrics["mean_first_birth_age"] = sum(fba_list) / max(len(fba_list), 1)

    # ---- Education assortative mating ----
    # Only look at currently married or dating couples where both are in adults pool
    couple_pairs: list[tuple[Any, Any]] = []
    seen_ids: set[str] = set()
    adults_by_id = {a.id: a for a in adults}

    for a in adults:
        if (
            a.relationship_status in (RelationshipStatus.DATING, RelationshipStatus.MARRIED)
            and a.partner_id is not None
            and a.partner_id in adults_by_id
            and a.id not in seen_ids
        ):
            partner = adults_by_id[a.partner_id]
            if partner.id not in seen_ids:
                if a.sex == Sex.FEMALE:
                    couple_pairs.append((a, partner))
                else:
                    couple_pairs.append((partner, a))
                seen_ids.add(a.id)
                seen_ids.add(partner.id)

    n_couples = max(len(couple_pairs), 1)

    homo = sum(
        1 for f, m in couple_pairs
        if EDUCATION_RANK.get(f.education, 1) == EDUCATION_RANK.get(m.education, 1)
    )
    hyper = sum(
        1 for f, m in couple_pairs
        if EDUCATION_RANK.get(f.education, 1) < EDUCATION_RANK.get(m.education, 1)
    )
    hypo = sum(
        1 for f, m in couple_pairs
        if EDUCATION_RANK.get(f.education, 1) > EDUCATION_RANK.get(m.education, 1)
    )

    metrics["education_homogamy_share"] = homo / n_couples
    metrics["female_hypergamy_share"] = hyper / n_couples
    metrics["female_hypogamy_share"] = hypo / n_couples

    # ---- Labour situation and mechanism variables ----
    _mean = lambda attr: sum(getattr(a, attr) for a in alive) / max(len(alive), 1)

    metrics["mean_ai_exposure"] = _mean("ai_exposure")
    metrics["mean_task_restructuring_exposure"] = _mean("task_restructuring_exposure")
    metrics["mean_entry_barrier_level"] = _mean("entry_barrier_level")
    metrics["mean_employment_entry_delay"] = _mean("employment_entry_delay")
    metrics["mean_employment_stability"] = _mean("employment_stability")
    metrics["mean_reskilling_need"] = _mean("reskilling_need")
    metrics["mean_career_predictability"] = _mean("career_predictability")
    metrics["mean_work_life_boundary_blurring"] = _mean("work_life_boundary_blurring")
    metrics["mean_available_relationship_time"] = _mean("available_relationship_time")
    metrics["mean_employment_stability_pressure"] = _mean("employment_stability_pressure")
    metrics["mean_career_expectation_uncertainty"] = _mean("career_expectation_uncertainty")
    metrics["mean_relationship_time_compression"] = _mean("relationship_time_compression")

    # ---- LLM subjective evaluation ----
    llm_mean = lambda attr: (
        sum(getattr(a.llm_state, attr) for a in alive) / max(len(alive), 1)
    )

    metrics["mean_marriage_value_score"] = llm_mean("marriage_value_score")
    metrics["mean_fertility_value_score"] = llm_mean("fertility_value_score")
    metrics["mean_marry_willingness"] = llm_mean("marry_willingness")
    metrics["mean_birth_willingness"] = llm_mean("birth_willingness")
    metrics["mean_fertility_intention"] = llm_mean("fertility_intention")

    # ---- BDI intention distribution ----
    intention_counts: dict[str, int] = {}
    for a in alive:
        intent = (
            a.intention_state.current_intention.value
            if a.intention_state.current_intention
            else "none"
        )
        intention_counts[intent] = intention_counts.get(intent, 0) + 1

    n_alive = max(len(alive), 1)
    metrics["share_intention_search_partner"] = intention_counts.get("search_partner", 0) / n_alive
    metrics["share_intention_marry_partner"] = intention_counts.get("marry_partner", 0) / n_alive
    metrics["share_intention_postpone_marriage"] = intention_counts.get("postpone_marriage", 0) / n_alive
    metrics["share_intention_have_child"] = intention_counts.get("have_child", 0) / n_alive
    metrics["share_intention_postpone_childbearing"] = intention_counts.get("postpone_childbearing", 0) / n_alive
    metrics["share_intention_stabilize_career_entry"] = intention_counts.get("stabilize_career_entry", 0) / n_alive
    metrics["share_intention_reduce_career_uncertainty"] = intention_counts.get("reduce_career_uncertainty", 0) / n_alive
    metrics["share_intention_protect_personal_time"] = intention_counts.get("protect_personal_time", 0) / n_alive

    # ---- LLM call statistics ----
    metrics["llm_call_count"] = float(llm_stats.call_count)
    metrics["llm_cache_hit_count"] = float(llm_stats.cache_hit_count)
    metrics["llm_fallback_count"] = float(llm_stats.fallback_count)
    metrics["llm_success_rate"] = llm_stats.success_rate

    return metrics
