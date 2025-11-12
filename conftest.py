"""Shared pytest fixtures for all tests."""
import pytest
from settings import SimConfig
from seeds import CorpSeed, PersonSeed


@pytest.fixture
def default_sim_config():
    """Default simulation configuration for tests."""
    return SimConfig(
        nr_of_ticks=100,
        nr_of_people=10,
        nr_of_corps=3,
        min_base_salary=50,
        max_base_salary=100,
    )


@pytest.fixture
def default_corp_seed():
    """Default corporation seed for tests."""
    return CorpSeed(
        balance=10000,
        price=10,
        salary=50,
        nr_of_employees=5,
        upe=10,
    )


@pytest.fixture
def default_person_seed():
    """Default person seed for tests."""
    return PersonSeed(
        mpc=0.5,
        balance=0,
    )
