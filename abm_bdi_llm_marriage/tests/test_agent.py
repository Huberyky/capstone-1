"""Tests for YouthAgent initialisation and basic properties."""

from __future__ import annotations

import random

import pytest

from src.agent import YouthAgent, create_agent
from src.enums import EducationLevel, RelationshipStatus, Sex


def make_rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


class TestYouthAgentDefaults:
    def test_default_agent_alive(self):
        a = YouthAgent()
        assert a.alive is True

    def test_default_relationship_single(self):
        a = YouthAgent()
        assert a.relationship_status == RelationshipStatus.SINGLE

    def test_default_no_children(self):
        a = YouthAgent()
        assert a.num_children == 0

    def test_default_no_partner(self):
        a = YouthAgent()
        assert a.partner_id is None

    def test_age_property(self):
        a = YouthAgent(age_ticks=80)  # 20 years at 4 ticks/year
        assert a.age == pytest.approx(20.0)

    def test_is_single_flag(self):
        a = YouthAgent()
        assert a.is_single is True
        assert a.is_dating is False
        assert a.is_married is False

    def test_is_married_flag(self):
        a = YouthAgent(relationship_status=RelationshipStatus.MARRIED)
        assert a.is_married is True
        assert a.is_single is False

    def test_clamp_all_keeps_values_in_range(self):
        a = YouthAgent()
        a.income = 1.5
        a.employment_stability_pressure = -0.3
        a.clamp_all()
        assert a.income == 1.0
        assert a.employment_stability_pressure == 0.0

    def test_belief_sensitivity_in_range(self):
        a = YouthAgent()
        assert 0 <= a.belief_sensitivity <= 1


class TestCreateAgentFactory:
    def test_creates_male_agent(self):
        rng = make_rng(1)
        a = create_agent(rng, "T001", Sex.MALE, age_ticks=80)
        assert a.sex == Sex.MALE

    def test_creates_female_agent(self):
        rng = make_rng(2)
        a = create_agent(rng, "T002", Sex.FEMALE, age_ticks=100)
        assert a.sex == Sex.FEMALE

    def test_income_in_range(self):
        rng = make_rng(3)
        for _ in range(20):
            a = create_agent(rng, "X", Sex.MALE, 80)
            assert 0.0 <= a.income <= 1.0

    def test_ai_exposure_in_range(self):
        rng = make_rng(4)
        for _ in range(20):
            a = create_agent(rng, "X", Sex.FEMALE, 80)
            assert 0.0 <= a.ai_exposure <= 1.0

    def test_education_is_valid(self):
        rng = make_rng(5)
        valid_edu = set(EducationLevel)
        for _ in range(30):
            a = create_agent(rng, "X", Sex.MALE, 80)
            assert a.education in valid_edu

    def test_reproducibility(self):
        rng1 = make_rng(99)
        rng2 = make_rng(99)
        a1 = create_agent(rng1, "R1", Sex.MALE, 80)
        a2 = create_agent(rng2, "R1", Sex.MALE, 80)
        assert a1.income == pytest.approx(a2.income)
        assert a1.education == a2.education

    def test_school_status_for_young_high_edu(self):
        # Young high-education agents may be in school
        rng = make_rng(10)
        statuses = []
        for _ in range(50):
            a = create_agent(rng, "S", Sex.FEMALE, age_ticks=20 * 4)  # 20 yrs
            a.education = EducationLevel.HIGH
            statuses.append(a.school_status)
        # Not asserting exact value, just that it's bool
        assert all(isinstance(s, bool) for s in statuses)
