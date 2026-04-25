"""Utility functions shared across the simulation modules."""

from __future__ import annotations

import math
import random
from typing import Any


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def sigmoid(x: float) -> float:
    """Logistic sigmoid function, numerically stable."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def weighted_mean(values: list[float], weights: list[float]) -> float:
    """Compute a weighted average; returns 0 if total weight is zero."""
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def blend(rule_value: float, llm_value: float, llm_weight: float) -> float:
    """Blend a rule-based value with an LLM-generated value.

    final = (1 - llm_weight) * rule_value + llm_weight * llm_value
    """
    return clamp((1.0 - llm_weight) * rule_value + llm_weight * llm_value)


def age_pressure_score(
    age: float,
    peak_age: float,
    width: float = 5.0,
) -> float:
    """Bell-shaped age-pressure score peaking at *peak_age*.

    Returns a value in [0, 1].  The score is highest near peak_age and
    falls off on both sides with scale *width*.
    """
    return clamp(math.exp(-0.5 * ((age - peak_age) / width) ** 2))


def education_distance(rank_i: int, rank_j: int) -> float:
    """Normalised education distance in [0, 1]; 0 means identical."""
    return clamp(abs(rank_i - rank_j) / 3.0)


def random_normal_clamp(
    rng: random.Random,
    mean: float,
    std: float,
    lo: float = 0.0,
    hi: float = 1.0,
) -> float:
    """Draw from N(mean, std) and clamp to [lo, hi]."""
    return clamp(rng.gauss(mean, std), lo, hi)


def dict_mean(records: list[dict[str, Any]], key: str) -> float:
    """Compute mean of *key* across a list of dicts, ignoring missing keys."""
    vals = [r[key] for r in records if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0
