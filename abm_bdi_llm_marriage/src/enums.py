"""Enumerations for the ABM-BDI-LLM marriage simulation."""

from enum import Enum


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class EducationLevel(str, Enum):
    LOW = "low"          # less than high school
    MEDIUM = "medium"    # high school / vocational
    HIGH = "high"        # bachelor's degree
    VERY_HIGH = "very_high"  # master's or above


class RelationshipStatus(str, Enum):
    SINGLE = "single"
    DATING = "dating"
    MARRIED = "married"


class Intention(str, Enum):
    SEARCH_PARTNER = "search_partner"
    MAINTAIN_RELATIONSHIP = "maintain_relationship"
    START_DATING = "start_dating"
    SWITCH_PARTNER = "switch_partner"
    MARRY_PARTNER = "marry_partner"
    POSTPONE_MARRIAGE = "postpone_marriage"
    HAVE_CHILD = "have_child"
    POSTPONE_CHILDBEARING = "postpone_childbearing"
    STABILIZE_CAREER_ENTRY = "stabilize_career_entry"
    REDUCE_CAREER_UNCERTAINTY = "reduce_career_uncertainty"
    PROTECT_PERSONAL_TIME = "protect_personal_time"
    EXIT_RELATIONSHIP = "exit_relationship"


class AILevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Education level to numeric rank mapping (used in matching)
EDUCATION_RANK: dict[str, int] = {
    EducationLevel.LOW: 0,
    EducationLevel.MEDIUM: 1,
    EducationLevel.HIGH: 2,
    EducationLevel.VERY_HIGH: 3,
}
