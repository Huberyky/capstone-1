"""DeepSeek LLM client with caching, retry, fallback, and JSON schema validation.

Role in the three-layer architecture
--------------------------------------
LLM  → generates subjective self-assessments: how does THIS agent perceive
       its own career situation, and what are its marriage/fertility valuations?
       The LLM does NOT decide behaviour — it outputs soft scores that feed
       into the BDI and ABM probability calculations.

Key features
------------
- sample_rate: only a fraction of agents are queried per tick.
- cache: reuses recent responses for agents with unchanged profiles.
- fallback: heuristic rules replace LLM when API is unavailable.
- strict JSON output with schema validation.
- retry + timeout with exponential back-off.
- logging of call / cache / fallback counts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional, TYPE_CHECKING

from src.enums import Intention
from src.utils import clamp

if TYPE_CHECKING:
    from src.agent import YouthAgent
    from src.config import ModelConfig, ScenarioConfig


logger = logging.getLogger(__name__)

# Valid intention strings for schema validation
_VALID_INTENTIONS: set[str] = {i.value for i in Intention}

# JSON schema field names expected in LLM response
_REQUIRED_TOP_KEYS = {
    "belief_adjustments",
    "marriage_value_score",
    "fertility_value_score",
    "marry_willingness",
    "birth_willingness",
    "fertility_intention",
    "suggested_intention",
    "confidence",
    "reason",
}
_REQUIRED_BELIEF_KEYS = {
    "employment_security",
    "career_predictability",
    "time_availability",
    "marriage_affordability",
    "childbearing_affordability",
    "career_entry_confidence",
    "long_term_life_planning_confidence",
}


class LLMCallStats:
    """Tracks LLM call statistics across a simulation run."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.cache_hit_count: int = 0
        self.fallback_count: int = 0
        self.success_count: int = 0

    @property
    def success_rate(self) -> float:
        attempts = self.call_count - self.cache_hit_count
        if attempts <= 0:
            return 0.0
        return self.success_count / attempts

    def summary(self) -> str:
        return (
            f"LLM calls={self.call_count} | "
            f"cache_hits={self.cache_hit_count} | "
            f"fallbacks={self.fallback_count} | "
            f"success_rate={self.success_rate:.2%}"
        )


class DeepSeekClient:
    """OpenAI-compatible client for the DeepSeek API.

    Wraps the openai SDK pointing at https://api.deepseek.com.
    All methods gracefully degrade to heuristic fallback when the API is
    unavailable, ensuring simulations can always proceed.

    Args:
        config: ModelConfig containing LLM settings.
        stats: Shared statistics tracker.
    """

    def __init__(
        self,
        config: "ModelConfig",
        stats: Optional[LLMCallStats] = None,
    ) -> None:
        self.config = config
        self.stats = stats or LLMCallStats()
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, int] = {}
        self._client: Any = None

        if config.use_llm:
            self._init_client()

    def _init_client(self) -> None:
        """Initialise the openai-compatible client.

        Key resolution order (first non-empty wins):
        1. config.api_key  — passed via --api-key on the CLI
        2. DEEPSEEK_API_KEY environment variable
        3. Falls back to heuristics with a warning
        """
        try:
            import openai  # type: ignore

            # Prefer explicitly passed key, then env var
            api_key = self.config.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                logger.warning(
                    "No API key found (set --api-key or DEEPSEEK_API_KEY) — "
                    "LLM calls will fall back to heuristics."
                )
            self._client = openai.OpenAI(
                api_key=api_key or "sk-dummy",
                base_url="https://api.deepseek.com",
                timeout=self.config.llm_timeout,
            )
            logger.info(
                "DeepSeek client initialised (model=%s, key_source=%s)",
                self.config.llm_model,
                "cli" if self.config.api_key else "env",
            )
        except ImportError:
            logger.warning("openai package not installed; falling back to heuristics.")
            self._client = None

    # -----------------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------------

    def _cache_key(self, agent: "YouthAgent") -> str:
        """Create a stable cache key from the agent's current profile snapshot."""
        profile = (
            f"{agent.sex.value}"
            f"|{agent.education.value}"
            f"|{round(agent.income, 1)}"
            f"|{round(agent.ai_exposure, 1)}"
            f"|{agent.relationship_status.value}"
            f"|{agent.num_children}"
            f"|{round(agent.employment_stability_pressure, 1)}"
            f"|{round(agent.career_expectation_uncertainty, 1)}"
            f"|{round(agent.relationship_time_compression, 1)}"
        )
        return hashlib.md5(profile.encode()).hexdigest()

    def _cache_valid(self, key: str, current_tick: int) -> bool:
        if key not in self._cache:
            return False
        age = current_tick - self._cache_timestamps.get(key, 0)
        return age < self.config.llm_refresh_interval

    # -----------------------------------------------------------------------
    # Prompt building
    # -----------------------------------------------------------------------

    def build_prompt(
        self,
        agent: "YouthAgent",
        scenario: "ScenarioConfig",
    ) -> str:
        """Build the system + user prompt for the agent's self-assessment.

        The prompt explicitly instructs the LLM NOT to decide behaviour, but
        to simulate the agent's subjective perceptions and valuations.

        Args:
            agent: The agent to simulate.
            scenario: Current AI scenario.

        Returns:
            A formatted prompt string.
        """
        age_y = agent.age_ticks / self.config.ticks_per_year
        beliefs_str = json.dumps(
            {
                k: round(getattr(agent.beliefs, k), 2)
                for k in vars(agent.beliefs)
                if not k.startswith("_")
            },
            ensure_ascii=False,
        )
        desires_str = json.dumps(
            {
                k: round(getattr(agent.desires, k), 2)
                for k in vars(agent.desires)
                if not k.startswith("_")
            },
            ensure_ascii=False,
        )
        intention_str = (
            agent.intention_state.current_intention.value
            if agent.intention_state.current_intention
            else "none"
        )

        prompt = f"""你是一名社会学研究辅助AI，用于模拟特定青年个体在当前处境下对婚育问题的主观判断。

【重要说明】
请不要直接决定该个体是否结婚或生育。你的任务是模拟该个体对自身职业处境、婚姻价值、生育价值和当前行动计划的主观评价，输出一组0到1之间的评分，用于调整其行为概率。

【个体画像】
- 性别：{agent.sex.value}
- 年龄：{age_y:.1f}岁
- 教育水平：{agent.education.value}
- 收入水平（标准化0-1）：{agent.income:.2f}
- 职业AI暴露度：{agent.ai_exposure:.2f}
- 当前关系状态：{agent.relationship_status.value}
- 子女数量：{agent.num_children}

【当前劳动处境与AI冲击】
情景描述：{scenario.description}
- 职业起点延迟与就业稳定性压力：{agent.employment_stability_pressure:.2f}（0=低压力，1=高压力）
- 职业预期不确定与长期规划审慎化：{agent.career_expectation_uncertainty:.2f}
- 工作—生活边界模糊与亲密关系时间资源压缩：{agent.relationship_time_compression:.2f}

【当前BDI状态】
信念：{beliefs_str}
愿望：{desires_str}
当前意图：{intention_str}

【任务】
请模拟该个体在上述处境下，对婚姻价值、生育价值、结婚意愿、生育意愿的主观评价，以及对其信念的微调建议。

请严格按照如下JSON格式输出，所有数值在0到1之间：

{{
  "belief_adjustments": {{
    "employment_security": <number>,
    "career_predictability": <number>,
    "time_availability": <number>,
    "marriage_affordability": <number>,
    "childbearing_affordability": <number>,
    "career_entry_confidence": <number>,
    "long_term_life_planning_confidence": <number>
  }},
  "marriage_value_score": <number>,
  "fertility_value_score": <number>,
  "marry_willingness": <number>,
  "birth_willingness": <number>,
  "fertility_intention": <number>,
  "suggested_intention": "<one of: search_partner, maintain_relationship, start_dating, switch_partner, marry_partner, postpone_marriage, have_child, postpone_childbearing, stabilize_career_entry, reduce_career_uncertainty, protect_personal_time, exit_relationship>",
  "confidence": <number>,
  "reason": "<简短中文解释，不超过50字>"
}}"""
        return prompt

    # -----------------------------------------------------------------------
    # API call
    # -----------------------------------------------------------------------

    def call(
        self,
        agent: "YouthAgent",
        scenario: "ScenarioConfig",
        current_tick: int,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Call the DeepSeek API with retry, caching, and fallback.

        Args:
            agent: Agent to evaluate.
            scenario: Current AI scenario.
            current_tick: Current simulation tick.
            max_retries: Maximum API retry attempts.

        Returns:
            Parsed and validated response dict.
        """
        self.stats.call_count += 1

        # Cache check
        key = self._cache_key(agent)
        if self._cache_valid(key, current_tick):
            self.stats.cache_hit_count += 1
            logger.debug("LLM cache hit for agent %s", agent.id)
            return self._cache[key]

        # No API client → fallback
        if self._client is None or not self.config.use_llm:
            self.stats.fallback_count += 1
            return self.fallback_decision(agent)

        prompt = self.build_prompt(agent, scenario)
        backoff = 2.0
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个社会学仿真辅助助手，专门模拟个体的主观婚育价值判断。"
                                "请严格按照要求的JSON格式输出，不要输出任何其他内容。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                parsed = self.parse_response(raw)
                # Cache successful result
                self._cache[key] = parsed
                self._cache_timestamps[key] = current_tick
                self.stats.success_count += 1
                agent.llm_state.last_llm_tick = current_tick
                logger.debug(
                    "LLM success for agent %s (attempt %d)", agent.id, attempt + 1
                )
                return parsed

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM call failed (agent=%s, attempt=%d/%d): %s",
                    agent.id, attempt + 1, max_retries, exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0

        # All retries exhausted → fallback
        logger.warning(
            "All LLM retries failed for agent %s; using fallback. Last error: %s",
            agent.id, last_error,
        )
        self.stats.fallback_count += 1
        return self.fallback_decision(agent)

    # -----------------------------------------------------------------------
    # JSON parsing + validation
    # -----------------------------------------------------------------------

    def parse_response(self, raw: str) -> dict[str, Any]:
        """Parse and validate the LLM JSON response.

        Raises ValueError for schema violations that cannot be auto-corrected.

        Args:
            raw: Raw string content from the API.

        Returns:
            Validated dict matching the expected schema.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        data = json.loads(text)

        # Top-level key validation
        missing = _REQUIRED_TOP_KEYS - data.keys()
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")

        # Belief adjustments key validation
        ba = data.get("belief_adjustments", {})
        missing_ba = _REQUIRED_BELIEF_KEYS - ba.keys()
        if missing_ba:
            raise ValueError(f"belief_adjustments missing: {missing_ba}")

        # Clamp all numeric values to [0, 1]
        for k in (
            "marriage_value_score", "fertility_value_score",
            "marry_willingness", "birth_willingness",
            "fertility_intention", "confidence",
        ):
            data[k] = clamp(float(data[k]))

        for k in _REQUIRED_BELIEF_KEYS:
            ba[k] = clamp(float(ba[k]))

        # Validate suggested_intention
        si = data.get("suggested_intention", "")
        if si not in _VALID_INTENTIONS:
            data["suggested_intention"] = "maintain_relationship"

        return data

    # -----------------------------------------------------------------------
    # Heuristic fallback
    # -----------------------------------------------------------------------

    def fallback_decision(self, agent: "YouthAgent") -> dict[str, Any]:
        """Generate heuristic equivalents of LLM outputs.

        This mirrors the subjective-evaluation role of the LLM without API
        access, using the agent's objective variables as proxies.

        Args:
            agent: The agent to evaluate.

        Returns:
            Dict matching the LLM schema.
        """
        esp = agent.employment_stability_pressure
        ceu = agent.career_expectation_uncertainty
        rtc = agent.relationship_time_compression

        mvs = clamp(
            0.5
            + 0.2 * agent.income
            + 0.1 * agent.beliefs.age_pressure
            - 0.25 * esp
            - 0.15 * ceu
            - 0.10 * rtc
        )
        fvs = clamp(
            0.4
            + 0.15 * agent.income
            + 0.10 * agent.beliefs.age_pressure
            - 0.20 * esp
            - 0.15 * ceu
            - 0.15 * rtc
            - 0.10 * agent.num_children
        )
        mw = clamp(mvs * 0.8 + agent.beliefs.age_pressure * 0.2)
        bw = clamp(fvs * 0.75 + agent.beliefs.age_pressure * 0.15)
        fi = clamp((mw + bw) / 2.0)

        # Determine suggested intention heuristically
        if esp > 0.6:
            sugg = "stabilize_career_entry"
        elif ceu > 0.6:
            sugg = "reduce_career_uncertainty"
        elif rtc > 0.6:
            sugg = "protect_personal_time"
        elif agent.relationship_status.value == "married" and agent.num_children == 0 and bw > 0.5:
            sugg = "have_child"
        elif agent.relationship_status.value == "dating" and mw > 0.55:
            sugg = "marry_partner"
        elif agent.relationship_status.value == "single":
            sugg = "search_partner"
        else:
            sugg = "maintain_relationship"

        return {
            "belief_adjustments": {
                "employment_security": clamp(1.0 - esp),
                "career_predictability": clamp(1.0 - ceu),
                "time_availability": clamp(1.0 - rtc),
                "marriage_affordability": clamp(agent.income * 0.6 + 0.2),
                "childbearing_affordability": clamp(agent.income * 0.5 + 0.1),
                "career_entry_confidence": clamp(1.0 - agent.employment_entry_delay),
                "long_term_life_planning_confidence": clamp(1.0 - ceu * 0.8),
            },
            "marriage_value_score": mvs,
            "fertility_value_score": fvs,
            "marry_willingness": mw,
            "birth_willingness": bw,
            "fertility_intention": fi,
            "suggested_intention": sugg,
            "confidence": 0.6,
            "reason": "heuristic fallback",
        }

    # -----------------------------------------------------------------------
    # Apply output to agent
    # -----------------------------------------------------------------------

    def apply_to_agent(
        self,
        agent: "YouthAgent",
        result: dict[str, Any],
        llm_weight: float = 0.3,
    ) -> None:
        """Blend LLM/fallback outputs into the agent's state.

        The LLM output is treated as a soft signal mixed with the agent's
        existing values; it never overrides them completely.

        Args:
            agent: The agent to update.
            result: Validated LLM/fallback output dict.
            llm_weight: Blending weight for LLM values.
        """
        lls = agent.llm_state
        bel = agent.beliefs

        w = llm_weight
        lls.marriage_value_score = clamp(
            (1 - w) * lls.marriage_value_score + w * result["marriage_value_score"]
        )
        lls.fertility_value_score = clamp(
            (1 - w) * lls.fertility_value_score + w * result["fertility_value_score"]
        )
        lls.marry_willingness = clamp(
            (1 - w) * lls.marry_willingness + w * result["marry_willingness"]
        )
        lls.birth_willingness = clamp(
            (1 - w) * lls.birth_willingness + w * result["birth_willingness"]
        )
        lls.fertility_intention = clamp(
            (1 - w) * lls.fertility_intention + w * result["fertility_intention"]
        )
        lls.llm_reason = result.get("reason", "")

        # Belief adjustments from LLM (light touch: 20 % weight)
        ba = result.get("belief_adjustments", {})
        bw = 0.20
        for attr, val in ba.items():
            if hasattr(bel, attr):
                old = getattr(bel, attr)
                setattr(bel, attr, clamp((1 - bw) * old + bw * val))
