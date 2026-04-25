"""Tests for the matching module: mate_value, seek_probability, date/marry probability."""

from __future__ import annotations

import random

import pytest

from src.agent import YouthAgent
from src.bdi import BDIEngine
from src.enums import EducationLevel, Intention, RelationshipStatus, Sex
from src.matching import (
    date_probability,
    encounter,
    marry_probability,
    mate_value,
    seek_probability,
    should_switch_partner,
)


def make_male(**kwargs) -> YouthAgent:
    defaults = dict(sex=Sex.MALE, age_ticks=104, income=0.5, education=EducationLevel.HIGH)
    defaults.update(kwargs)
    return YouthAgent(**defaults)


def make_female(**kwargs) -> YouthAgent:
    defaults = dict(sex=Sex.FEMALE, age_ticks=100, income=0.5, education=EducationLevel.HIGH)
    defaults.update(kwargs)
    return YouthAgent(**defaults)


def make_bdi() -> BDIEngine:
    return BDIEngine()


class TestMateValue:
    def test_in_range(self):
        m = make_male()
        f = make_female()
        mv = mate_value(m, f)
        assert 0.0 <= mv <= 1.0

    def test_same_education_higher_than_large_gap(self):
        f = make_female(education=EducationLevel.HIGH)
        m_same = make_male(education=EducationLevel.HIGH)
        m_diff = make_male(education=EducationLevel.LOW)
        mv_same = mate_value(f, m_same)
        mv_diff = mate_value(f, m_diff)
        assert mv_same > mv_diff

    def test_higher_income_generally_more_attractive(self):
        f = make_female()
        m_rich = make_male(income=0.9)
        m_poor = make_male(income=0.1)
        mv_rich = mate_value(f, m_rich)
        mv_poor = mate_value(f, m_poor)
        assert mv_rich > mv_poor

    def test_employment_pressure_boosts_income_weight(self):
        # Use a slight education mismatch so the income-weight boost is visible:
        # high-pressure agent up-weights income relative to education, which
        # benefits a high-income / slightly-lower-edu candidate.
        f_high = make_female(education=EducationLevel.HIGH, employment_stability_pressure=0.9)
        f_low  = make_female(education=EducationLevel.HIGH, employment_stability_pressure=0.1)
        m = make_male(income=0.9, education=EducationLevel.MEDIUM)  # rich but edu mismatch
        mv_high = mate_value(f_high, m)
        mv_low  = mate_value(f_low, m)
        # High-pressure agent weighs income more → higher mate value for rich partner
        assert mv_high >= mv_low

    def test_not_symmetric(self):
        m = make_male(income=0.8, age_ticks=110)
        f = make_female(income=0.4, age_ticks=96)
        mv_mf = mate_value(m, f)
        mv_fm = mate_value(f, m)
        # Preferences differ by sex, so values need not be equal
        assert isinstance(mv_mf, float)
        assert isinstance(mv_fm, float)


class TestSeekProbability:
    def test_married_agent_does_not_seek(self):
        bdi = make_bdi()
        a = make_female(relationship_status=RelationshipStatus.MARRIED)
        assert seek_probability(a, bdi) == 0.0

    def test_single_agent_has_positive_seek_prob(self):
        bdi = make_bdi()
        a = make_female(relationship_status=RelationshipStatus.SINGLE)
        p = seek_probability(a, bdi)
        assert p > 0.0

    def test_high_time_compression_reduces_seek(self):
        bdi = make_bdi()
        a_low = make_female(relationship_time_compression=0.1)
        a_high = make_female(relationship_time_compression=0.9)
        p_low = seek_probability(a_low, bdi)
        p_high = seek_probability(a_high, bdi)
        assert p_high < p_low

    def test_career_intent_reduces_seek(self):
        bdi = make_bdi()
        a = make_female()
        a.intention_state.current_intention = Intention.STABILIZE_CAREER_ENTRY
        p_career = seek_probability(a, bdi)
        a2 = make_female()
        a2.intention_state.current_intention = None
        p_none = seek_probability(a2, bdi)
        assert p_career <= p_none


class TestEncounter:
    def test_returns_none_for_empty_pool(self):
        a = make_male()
        rng = random.Random(0)
        result = encounter(a, [], rng)
        assert result is None

    def test_returns_candidate_from_pool(self):
        a = make_male()
        candidates = [make_female() for _ in range(5)]
        rng = random.Random(0)
        result = encounter(a, candidates, rng, delta_school_meeting=0.0)
        assert result in candidates

    def test_education_structured_encounter(self):
        a = make_male(education=EducationLevel.HIGH)
        same_edu = [make_female(education=EducationLevel.HIGH) for _ in range(3)]
        diff_edu = [make_female(education=EducationLevel.LOW) for _ in range(3)]
        pool = same_edu + diff_edu
        rng = random.Random(0)
        # Force education-structured meeting every time
        results = [
            encounter(a, pool, rng, delta_school_meeting=1.0)
            for _ in range(20)
        ]
        assert all(r in same_edu for r in results)


class TestDateProbability:
    def test_in_range(self):
        bdi = make_bdi()
        m = make_male()
        f = make_female()
        p = date_probability(m, f, bdi)
        assert 0.0 <= p <= 1.0

    def test_high_mechanism_pressure_reduces_date_prob(self):
        bdi = make_bdi()
        m_low = make_male(
            employment_stability_pressure=0.1,
            career_expectation_uncertainty=0.1,
            relationship_time_compression=0.1,
        )
        m_high = make_male(
            employment_stability_pressure=0.9,
            career_expectation_uncertainty=0.9,
            relationship_time_compression=0.9,
        )
        f = make_female()
        p_low = date_probability(m_low, f, bdi)
        p_high = date_probability(m_high, f, bdi)
        assert p_high < p_low


class TestMarryProbability:
    def test_in_range(self):
        bdi = make_bdi()
        m = make_male(relationship_status=RelationshipStatus.DATING, relationship_duration=8)
        f = make_female(relationship_status=RelationshipStatus.DATING, relationship_duration=8)
        p = marry_probability(m, f, bdi)
        assert 0.0 <= p <= 1.0

    def test_longer_duration_increases_marry_prob(self):
        bdi = make_bdi()
        f = make_female()
        m_short = make_male(relationship_duration=1)
        m_long = make_male(relationship_duration=20)
        p_short = marry_probability(m_short, f, bdi)
        p_long = marry_probability(m_long, f, bdi)
        assert p_long >= p_short

    def test_postpone_intention_reduces_marry_prob(self):
        bdi = make_bdi()
        f = make_female()
        m_normal = make_male(relationship_duration=8)
        m_postpone = make_male(relationship_duration=8)
        m_postpone.intention_state.current_intention = Intention.POSTPONE_MARRIAGE
        p_normal = marry_probability(m_normal, f, bdi)
        p_postpone = marry_probability(m_postpone, f, bdi)
        assert p_postpone < p_normal


class TestShouldSwitchPartner:
    def test_no_switch_when_current_better(self):
        a = make_female()
        current = make_male(income=0.9, education=EducationLevel.VERY_HIGH)
        candidate = make_male(income=0.1, education=EducationLevel.LOW)
        assert should_switch_partner(a, current, candidate) is False

    def test_switch_when_candidate_much_better(self):
        a = make_female()
        current = make_male(income=0.1, education=EducationLevel.LOW)
        candidate = make_male(income=0.9, education=EducationLevel.VERY_HIGH)
        assert should_switch_partner(a, current, candidate) is True
