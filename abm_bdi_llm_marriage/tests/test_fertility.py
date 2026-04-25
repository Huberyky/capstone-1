"""Tests for the fertility module: couple_affordability and birth_probability."""

from __future__ import annotations

import pytest

from src.agent import YouthAgent
from src.bdi import BDIEngine
from src.enums import Intention, RelationshipStatus, Sex
from src.fertility import birth_probability, couple_affordability, record_birth


def make_female(**kwargs) -> YouthAgent:
    defaults = dict(
        sex=Sex.FEMALE,
        age_ticks=108,   # 27 years
        income=0.5,
        relationship_status=RelationshipStatus.MARRIED,
        relationship_duration=8,
        num_children=0,
    )
    defaults.update(kwargs)
    return YouthAgent(**defaults)


def make_male(**kwargs) -> YouthAgent:
    defaults = dict(
        sex=Sex.MALE,
        age_ticks=112,   # 28 years
        income=0.5,
        relationship_status=RelationshipStatus.MARRIED,
        relationship_duration=8,
        num_children=0,
    )
    defaults.update(kwargs)
    return YouthAgent(**defaults)


def make_bdi() -> BDIEngine:
    return BDIEngine()


class TestCoupleAffordability:
    def test_in_range(self):
        f = make_female()
        m = make_male()
        aff = couple_affordability(f, m)
        assert 0.0 <= aff <= 1.0

    def test_high_income_increases_affordability(self):
        f_rich = make_female(income=0.9)
        m_rich = make_male(income=0.9)
        f_poor = make_female(income=0.1)
        m_poor = make_male(income=0.1)
        assert couple_affordability(f_rich, m_rich) > couple_affordability(f_poor, m_poor)

    def test_more_children_reduces_affordability(self):
        f0 = make_female(num_children=0)
        f2 = make_female(num_children=2)
        m = make_male()
        assert couple_affordability(f0, m) > couple_affordability(f2, m)


class TestBirthProbability:
    def test_in_range(self):
        bdi = make_bdi()
        f = make_female()
        m = make_male()
        p = birth_probability(f, m, bdi)
        assert 0.0 <= p <= 1.0

    def test_high_pressure_reduces_birth_prob(self):
        bdi = make_bdi()
        f_low = make_female(
            employment_stability_pressure=0.1,
            career_expectation_uncertainty=0.1,
            relationship_time_compression=0.1,
        )
        m_low = make_male(
            employment_stability_pressure=0.1,
            career_expectation_uncertainty=0.1,
            relationship_time_compression=0.1,
        )
        f_high = make_female(
            employment_stability_pressure=0.9,
            career_expectation_uncertainty=0.9,
            relationship_time_compression=0.9,
        )
        m_high = make_male(
            employment_stability_pressure=0.9,
            career_expectation_uncertainty=0.9,
            relationship_time_compression=0.9,
        )
        p_low = birth_probability(f_low, m_low, bdi)
        p_high = birth_probability(f_high, m_high, bdi)
        assert p_high < p_low

    def test_max_children_gives_zero_prob(self):
        bdi = make_bdi()
        f = make_female(num_children=3)
        m = make_male()
        assert birth_probability(f, m, bdi) == 0.0

    def test_older_female_has_lower_prob(self):
        bdi = make_bdi()
        f_young = make_female(age_ticks=28 * 4)   # 28 years
        f_old = make_female(age_ticks=42 * 4)     # 42 years
        m = make_male()
        p_young = birth_probability(f_young, m, bdi)
        p_old = birth_probability(f_old, m, bdi)
        assert p_young > p_old

    def test_have_child_intention_boosts_prob(self):
        bdi = make_bdi()
        f = make_female()
        m = make_male()
        f_intend = make_female()
        f_intend.intention_state.current_intention = Intention.HAVE_CHILD
        p_neutral = birth_probability(f, m, bdi)
        p_intend = birth_probability(f_intend, m, bdi)
        assert p_intend >= p_neutral

    def test_postpone_intention_reduces_prob(self):
        bdi = make_bdi()
        f = make_female()
        m = make_male()
        f_postpone = make_female()
        f_postpone.intention_state.current_intention = Intention.POSTPONE_CHILDBEARING
        p_neutral = birth_probability(f, m, bdi)
        p_postpone = birth_probability(f_postpone, m, bdi)
        assert p_postpone < p_neutral


class TestRecordBirth:
    def test_increments_children(self):
        f = make_female(num_children=0)
        m = make_male(num_children=0)
        record_birth(f, m, current_tick=100)
        assert f.num_children == 1
        assert m.num_children == 1

    def test_sets_first_birth_age(self):
        f = make_female(age_ticks=108, num_children=0)  # 27 years
        m = make_male(age_ticks=112, num_children=0)    # 28 years
        record_birth(f, m, current_tick=108, ticks_per_year=4)
        assert f.first_birth_age == pytest.approx(27.0)
        assert m.first_birth_age == pytest.approx(28.0)

    def test_does_not_overwrite_first_birth_age(self):
        f = make_female(num_children=1)
        f.first_birth_age = 25.0
        m = make_male(num_children=1)
        m.first_birth_age = 26.0
        record_birth(f, m, current_tick=120)
        assert f.first_birth_age == 25.0
        assert m.first_birth_age == 26.0
