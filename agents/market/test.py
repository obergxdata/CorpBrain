from agents.market.agent import Market
from agents.corporations.agent import Corporation
from agents.people.agent import Person
from settings import SimConfig
from seeds import CorpSeed, PersonSeed
import pytest
import random


def test_market_share(default_sim_config, default_corp_seed, default_person_seed):
    m = Market(default_sim_config, default_corp_seed, default_person_seed)
    m.people = [Person() for _ in range(0, 10)]
    corp = Corporation()

    corp.history.record(sales=5)
    assert m.market_share(corp) == 0.5

    corp.history.record(sales=6)
    assert m.market_share(corp) == 0.6


def test_init_corps():
    sim_config = SimConfig(
        nr_of_ticks=100,
        nr_of_people=10,
        nr_of_corps=10,
        min_base_salary=50,
        max_base_salary=100,
    )
    corp_seed = CorpSeed(
        balance=100,
        price=10,
        salary=50,
        nr_of_employees=5,
        upe=10,
    )
    person_seed = PersonSeed(mpc=0.5, balance=0)
    m = Market(sim_config, corp_seed, person_seed)
    m.init_corps()

    assert len(m.corporations) == 10
    assert random.choice(m.corporations).balance == 100


def test_init_people():
    sim_config = SimConfig(
        nr_of_ticks=100,
        nr_of_people=100,
        nr_of_corps=3,
        min_base_salary=45,
        max_base_salary=100,
    )
    corp_seed = CorpSeed(
        balance=10000,
        price=10,
        salary=50,
        nr_of_employees=5,
        upe=10,
    )
    person_seed = PersonSeed(mpc=0.5, balance=0)
    m = Market(sim_config, corp_seed, person_seed)
    m.init_people()

    assert len(m.people) == 100
    assert max([p.salary for p in m.people]) == pytest.approx(100, 5)
    assert min([p.salary for p in m.people]) == pytest.approx(45, 50)


def test_person_corp_list(default_sim_config, default_corp_seed, default_person_seed):
    m = Market(default_sim_config, default_corp_seed, default_person_seed)
    for i in range(3, 10):
        if i % 3:
            c = Corporation(price=i - 1)
        else:
            c = Corporation(price=i)

        m.corporations.append(c)

    # Note: person_corp_list now just returns alive_corps
    # The sampling and sorting happens in Person.step()
    corps_list = m.alive_corps

    # Just verify we get all the corps back
    assert len(corps_list) == 7
