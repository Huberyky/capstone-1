"""Main ABM model: MarriageFertilityABM.

Orchestrates the tick loop:
  age → schooling → labour situation → mechanism variables →
  BDI → LLM → matching → fertility → metrics → demographics
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

from src.agent import YouthAgent, create_agent
from src.bdi import BDIEngine
from src.config import ModelConfig, ScenarioConfig
from src.enums import RelationshipStatus, Sex
from src.environment import update_labour_situation
from src.fertility import birth_probability, record_birth
from src.llm_client import DeepSeekClient, LLMCallStats
from src.matching import (
    date_probability,
    encounter,
    marry_probability,
    seek_probability,
    should_switch_partner,
)
from src.metrics import compute_metrics
from src.utils import clamp

logger = logging.getLogger(__name__)


class MarriageFertilityABM:
    """Agent-Based Model for marriage and fertility dynamics under AI diffusion.

    The model integrates:
    - ABM: social structure, marriage market, demographic processes
    - BDI: belief-desire-intention reasoning for each agent
    - LLM: subjective self-assessment of career and marriage/fertility values

    Args:
        config: Model configuration.
        scenario: AI impact scenario.
        seed: Random seed (overrides config.seed if provided).
    """

    def __init__(
        self,
        config: ModelConfig,
        scenario: ScenarioConfig,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.seed = seed if seed is not None else config.seed
        self.rng = random.Random(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.bdi = BDIEngine(config.bdi_intention_factor_strength)
        self.llm_stats = LLMCallStats()
        self.llm_client = DeepSeekClient(config, self.llm_stats)

        self.agents: list[YouthAgent] = []
        self.tick_metrics: list[dict] = []
        self.current_tick: int = 0
        self.belief_trajectory: list[dict] = []
        self.desire_trajectory: list[dict] = []
        self.intention_trajectory: list[dict] = []

        self._next_id: int = 0
        self._init_population()

    # -----------------------------------------------------------------------
    # Population initialisation
    # -----------------------------------------------------------------------

    def _next_agent_id(self) -> str:
        aid = f"A{self._next_id:05d}"
        self._next_id += 1
        return aid

    def _init_population(self) -> None:
        """Create the initial population with balanced sex ratio and age spread."""
        n = self.config.population_size
        tpy = self.config.ticks_per_year

        for i in range(n):
            sex = Sex.MALE if i % 2 == 0 else Sex.FEMALE
            # Age 18–35 at start, uniform
            age_years = self.rng.uniform(18, 35)
            age_ticks = int(age_years * tpy)
            agent = create_agent(
                self.rng, self._next_agent_id(), sex, age_ticks, tpy
            )
            self.agents.append(agent)

        logger.info(
            "Population initialised: %d agents (seed=%d, scenario=%s)",
            n, self.seed, self.scenario.name,
        )

    # -----------------------------------------------------------------------
    # Main tick step
    # -----------------------------------------------------------------------

    def step(self) -> dict:
        """Execute one simulation tick and return this tick's metrics.

        Tick flow:
        1.  Age increment
        2.  Schooling update
        3.  Labour situation update (environment)
        4.  Mechanism variables already set in step 3
        5.  BDI belief update
        6.  Desire generation
        7.  Intention reconsideration check
        8.  Intention selection
        9.  LLM (sampled) / fallback
        10. Marriage-market search and encounter
        11. Dating formation
        12. Marriage conversion
        13. Fertility
        14. Relationship-duration increment
        15. Demographics (death / ageing-out, births to replenish)
        16. Metrics
        """
        tpy = self.config.ticks_per_year
        tick = self.current_tick

        model_context = {
            "tick": tick,
            "sigma_age_pressure_male": self.config.sigma_age_pressure_male,
            "sigma_age_pressure_female": self.config.sigma_age_pressure_female,
        }

        # Build lookup tables
        agents_by_id: dict[str, YouthAgent] = {a.id: a for a in self.agents if a.alive}
        # Separate eligible pools by sex and relationship status
        single_females = [
            a for a in agents_by_id.values()
            if a.sex == Sex.FEMALE and a.relationship_status == RelationshipStatus.SINGLE
        ]
        single_males = [
            a for a in agents_by_id.values()
            if a.sex == Sex.MALE and a.relationship_status == RelationshipStatus.SINGLE
        ]

        # Shuffle for fairness
        self.rng.shuffle(single_females)
        self.rng.shuffle(single_males)

        # ---- 1–4: Age, schooling, environment ----
        for agent in agents_by_id.values():
            agent.age_ticks += 1
            # Schooling: agents leave school by age 24
            if agent.age_ticks / tpy >= 24:
                agent.school_status = False
            update_labour_situation(agent, self.scenario, tick, self.config, tpy)

        # ---- 5–8: BDI cycle ----
        for agent in agents_by_id.values():
            self.bdi.update_beliefs(agent, model_context)
            self.bdi.generate_desires(agent)
            if self.bdi.should_reconsider_intention(agent):
                self.bdi.select_intention(agent)
            else:
                agent.intention_state.intention_duration += 1

        self._record_bdi_diagnostics(list(agents_by_id.values()), tick)

        # ---- 9: LLM (sampled) ----
        self._run_llm_step(list(agents_by_id.values()), tick)

        # ---- 10–12: Marriage market ----
        # Set of agents already matched this tick
        matched_this_tick: set[str] = set()

        for searcher in list(agents_by_id.values()):
            if not searcher.alive:
                continue
            if searcher.id in matched_this_tick:
                continue
            if searcher.relationship_status == RelationshipStatus.MARRIED:
                continue

            # Determine search probability
            p_seek = seek_probability(searcher, self.bdi)
            if self.rng.random() > p_seek:
                continue

            # Encounter
            if searcher.sex == Sex.MALE:
                candidate_pool = single_females
            else:
                candidate_pool = single_males

            eligible = [
                c for c in candidate_pool
                if c.id not in matched_this_tick and c.alive
                and c.relationship_status == RelationshipStatus.SINGLE
            ]
            candidate = encounter(
                searcher, eligible, self.rng, self.config.delta_school_meeting
            )
            if candidate is None:
                continue

            # ---- Dating formation (both single) ----
            if (
                searcher.relationship_status == RelationshipStatus.SINGLE
                and candidate.relationship_status == RelationshipStatus.SINGLE
            ):
                llm_w = self.config.llm_weight if self.config.use_llm else 0.0
                p_i = date_probability(searcher, candidate, self.bdi, llm_w)
                p_j = date_probability(candidate, searcher, self.bdi, llm_w)
                if self.rng.random() < p_i and self.rng.random() < p_j:
                    self._form_relationship(searcher, candidate)
                    matched_this_tick.add(searcher.id)
                    matched_this_tick.add(candidate.id)

            # ---- Partner switching (currently dating) ----
            elif (
                searcher.relationship_status == RelationshipStatus.DATING
                and searcher.partner_id is not None
            ):
                current_partner = agents_by_id.get(searcher.partner_id)
                if current_partner and should_switch_partner(
                    searcher, current_partner, candidate
                ):
                    # Break old relationship, form new
                    self._break_relationship(searcher, current_partner)
                    p_i = date_probability(searcher, candidate, self.bdi, 0.0)
                    if self.rng.random() < p_i:
                        self._form_relationship(searcher, candidate)
                        matched_this_tick.add(searcher.id)
                        matched_this_tick.add(candidate.id)

        # ---- 12: Marriage conversion (for dating couples) ----
        self._process_marriages(agents_by_id)

        # ---- 13: Fertility ----
        self._process_fertility(agents_by_id)

        # ---- 14: Relationship duration increment ----
        for agent in agents_by_id.values():
            if agent.relationship_status != RelationshipStatus.SINGLE:
                agent.relationship_duration += 1

        # ---- 15: Demographics ----
        self._process_demographics(tpy)

        # ---- 16: Metrics ----
        m = compute_metrics(self.agents, tpy, self.llm_stats, tick)
        m["tick"] = tick
        m["scenario"] = self.scenario.name
        self.tick_metrics.append(m)

        self.current_tick += 1

        if tick % (tpy * 5) == 0:  # log every 5 years
            logger.info(
                "Tick %d | married=%.2f | children=%.2f | %s",
                tick,
                m["first_marriage_rate_18_45"],
                m["avg_children_adults"],
                self.llm_stats.summary(),
            )

        return m

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _run_llm_step(
        self,
        agents: list[YouthAgent],
        tick: int,
    ) -> None:
        """Sample agents and call LLM (or fallback) to update subjective values.

        API calls are dispatched concurrently via ThreadPoolExecutor so that
        network latency for one agent does not block others.  The number of
        worker threads is capped at 8 to avoid flooding the rate limiter.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        llm_w = self.config.llm_weight if self.config.use_llm else 0.0

        # Select agents that need an LLM call this tick
        to_call: list[YouthAgent] = []
        for agent in agents:
            since_last = tick - agent.llm_state.last_llm_tick
            if since_last < self.config.llm_refresh_interval:
                continue
            if self.rng.random() > self.config.llm_sample_rate:
                continue
            to_call.append(agent)

        if to_call:
            # Concurrent API calls — each thread handles one agent
            max_workers = min(8, len(to_call))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_agent = {
                    pool.submit(self.llm_client.call, agent, self.scenario, tick): agent
                    for agent in to_call
                }
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.warning("LLM future failed for agent %s: %s", agent.id, exc)
                        result = self.llm_client.fallback_decision(agent)
                    self.llm_client.apply_to_agent(agent, result, llm_w)

        # Always run fallback for agents that have never been evaluated
        for agent in agents:
            if agent.llm_state.last_llm_tick == -999:
                result = self.llm_client.fallback_decision(agent)
                self.llm_client.apply_to_agent(agent, result, 0.15)
                agent.llm_state.last_llm_tick = tick

    def _form_relationship(
        self,
        a: YouthAgent,
        b: YouthAgent,
    ) -> None:
        """Link two agents in a dating relationship."""
        a.relationship_status = RelationshipStatus.DATING
        a.partner_id = b.id
        a.relationship_duration = 0
        b.relationship_status = RelationshipStatus.DATING
        b.partner_id = a.id
        b.relationship_duration = 0

    def _break_relationship(
        self,
        a: YouthAgent,
        b: YouthAgent,
    ) -> None:
        """Dissolve a dating (not marriage) relationship."""
        a.relationship_status = RelationshipStatus.SINGLE
        a.partner_id = None
        a.relationship_duration = 0
        b.relationship_status = RelationshipStatus.SINGLE
        b.partner_id = None
        b.relationship_duration = 0

    def _process_marriages(
        self,
        agents_by_id: dict[str, YouthAgent],
    ) -> None:
        """Convert eligible dating pairs to married couples."""
        processed: set[str] = set()
        tpy = self.config.ticks_per_year

        for agent in list(agents_by_id.values()):
            if agent.id in processed:
                continue
            if agent.relationship_status != RelationshipStatus.DATING:
                continue
            if agent.partner_id is None:
                continue

            partner = agents_by_id.get(agent.partner_id)
            if partner is None or partner.relationship_status != RelationshipStatus.DATING:
                continue

            llm_w = self.config.llm_weight if self.config.use_llm else 0.0
            p_i = marry_probability(
                agent, partner, self.bdi, self.config.beta_commitment, llm_w
            )
            p_j = marry_probability(
                partner, agent, self.bdi, self.config.beta_commitment, llm_w
            )

            if self.rng.random() < p_i and self.rng.random() < p_j:
                agent.relationship_status = RelationshipStatus.MARRIED
                partner.relationship_status = RelationshipStatus.MARRIED
                age_a = agent.age_ticks / tpy
                age_b = partner.age_ticks / tpy
                if agent.first_marriage_age is None:
                    agent.first_marriage_age = age_a
                if partner.first_marriage_age is None:
                    partner.first_marriage_age = age_b

            processed.add(agent.id)
            processed.add(partner.id)

    def _process_fertility(
        self,
        agents_by_id: dict[str, YouthAgent],
    ) -> None:
        """Evaluate birth events for all married couples."""
        processed: set[str] = set()
        tpy = self.config.ticks_per_year

        for agent in list(agents_by_id.values()):
            if agent.id in processed:
                continue
            if not (
                agent.relationship_status == RelationshipStatus.MARRIED
                and agent.sex == Sex.FEMALE
            ):
                continue

            partner = agents_by_id.get(agent.partner_id or "")
            if partner is None or partner.relationship_status != RelationshipStatus.MARRIED:
                continue

            female = agent
            male = partner

            # Age constraint: 18–45 for females
            f_age = female.age_ticks / tpy
            if not (18 <= f_age <= 45):
                processed.add(female.id)
                processed.add(male.id)
                continue

            llm_w = self.config.llm_weight if self.config.use_llm else 0.0
            p_birth = birth_probability(female, male, self.bdi, llm_w, tpy)

            if self.rng.random() < p_birth:
                record_birth(female, male, self.current_tick, tpy)

            processed.add(female.id)
            processed.add(male.id)

    def _process_demographics(self, tpy: int) -> None:
        """Remove agents over 65 and add replacement young agents."""
        to_remove = [
            a for a in self.agents
            if a.age_ticks / tpy > 65
        ]
        for old in to_remove:
            # Dissolve partner link
            if old.partner_id:
                for a in self.agents:
                    if a.id == old.partner_id:
                        a.partner_id = None
                        a.relationship_status = RelationshipStatus.SINGLE
                        a.relationship_duration = 0
            old.alive = False

        self.agents = [a for a in self.agents if a.alive]

        # Replenish with 18-year-olds to maintain approximate population size
        n_dead = len(to_remove)
        for i in range(n_dead):
            sex = Sex.MALE if i % 2 == 0 else Sex.FEMALE
            new_agent = create_agent(
                self.rng,
                self._next_agent_id(),
                sex,
                age_ticks=18 * tpy,
                ticks_per_year=tpy,
            )
            self.agents.append(new_agent)


    def _record_bdi_diagnostics(self, agents: list[YouthAgent], tick: int) -> None:
        if not agents:
            return
        n = max(len(agents), 1)
        self.belief_trajectory.append({
            "tick": tick,
            "scenario": self.scenario.name,
            "employment_security": sum(a.beliefs.employment_security for a in agents) / n,
            "career_predictability": sum(a.beliefs.career_predictability for a in agents) / n,
            "time_availability": sum(a.beliefs.time_availability for a in agents) / n,
            "belief_volatility": (
                sum(abs(a.beliefs.employment_security - a.subjective_interpretation["economic_family_readiness"]) for a in agents)
                + sum(abs(a.beliefs.career_predictability - a.subjective_interpretation["career_trajectory_predictability"]) for a in agents)
                + sum(abs(a.beliefs.time_availability - a.subjective_interpretation["time_for_relationship_investment"]) for a in agents)
            ) / (3 * n),
        })
        self.desire_trajectory.append({
            "tick": tick,
            "scenario": self.scenario.name,
            "desire_marry": sum(a.desires.marry for a in agents) / n,
            "desire_delay_marriage": sum(a.desires.delay_marriage for a in agents) / n,
            "desire_stabilize_career_entry": sum(a.desires.stabilize_career_entry for a in agents) / n,
            "desire_protect_personal_time": sum(a.desires.protect_personal_time for a in agents) / n,
            "desire_conflict_index": sum(a.desire_conflict_index for a in agents) / n,
        })
        intention_counts: dict[str, int] = {}
        for a in agents:
            key = a.intention_state.current_intention.value if a.intention_state.current_intention else "none"
            intention_counts[key] = intention_counts.get(key, 0) + 1
        row = {"tick": tick, "scenario": self.scenario.name}
        for k, v in intention_counts.items():
            row[f"share_{k}"] = v / n
        self.intention_trajectory.append(row)

    def get_bdi_diagnostics_summary(self) -> dict[str, float]:
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return {
                "intention_switch_count": 0.0,
                "mean_intention_duration": 0.0,
                "desire_conflict_index": 0.0,
                "belief_volatility": 0.0,
            }
        durations = [d for a in alive for d in a.intention_duration_history]
        mean_duration = (sum(durations) / len(durations)) if durations else 0.0
        belief_vol = sum(abs(a.beliefs.employment_security - a.subjective_interpretation["economic_family_readiness"]) for a in alive) / len(alive)
        return {
            "intention_switch_count": float(sum(a.intention_switch_count for a in alive)),
            "mean_intention_duration": float(mean_duration),
            "desire_conflict_index": float(sum(a.desire_conflict_index for a in alive) / len(alive)),
            "belief_volatility": float(belief_vol),
        }

    # -----------------------------------------------------------------------
    # Run full simulation
    # -----------------------------------------------------------------------

    def run(self) -> list[dict]:
        """Run the full simulation for config.total_ticks ticks.

        Returns:
            List of per-tick metric dictionaries.
        """
        for _ in range(self.config.total_ticks):
            self.step()

        if self.tick_metrics:
            self.tick_metrics[-1].update(self.get_bdi_diagnostics_summary())

        if self.config.use_llm:
            print(f"\n[LLM] {self.llm_stats.summary()}")

        return self.tick_metrics

    def get_agent_snapshot(self) -> list[dict]:
        """Return a snapshot of all agents as a list of dicts (for CSV export)."""
        tpy = self.config.ticks_per_year
        rows = []
        for a in self.agents:
            rows.append(
                {
                    "id": a.id,
                    "sex": a.sex.value,
                    "age": round(a.age_ticks / tpy, 1),
                    "education": a.education.value,
                    "income": round(a.income, 3),
                    "relationship_status": a.relationship_status.value,
                    "num_children": a.num_children,
                    "first_marriage_age": a.first_marriage_age,
                    "first_birth_age": a.first_birth_age,
                    "employment_stability_pressure": round(a.employment_stability_pressure, 3),
                    "career_expectation_uncertainty": round(a.career_expectation_uncertainty, 3),
                    "relationship_time_compression": round(a.relationship_time_compression, 3),
                    "current_intention": (
                        a.intention_state.current_intention.value
                        if a.intention_state.current_intention
                        else None
                    ),
                    "marriage_value_score": round(a.llm_state.marriage_value_score, 3),
                    "fertility_value_score": round(a.llm_state.fertility_value_score, 3),
                    "desire_conflict_index": round(a.desire_conflict_index, 3),
                    "intention_switch_count": a.intention_switch_count,
                    "scenario": self.scenario.name,
                }
            )
        return rows
