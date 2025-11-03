from agents.market.agent import Market
from agents.corporations.agent import Corporation
from agents.people.agent import Person
import pytest
import random


def test_market_share():
    m = Market()
    m.people = [Person() for _ in range(0, 10)]
    corp = Corporation()

    corp.history.record(sales=5)
    assert m.market_share(corp) == 0.5

    corp.history.record(sales=6)
    assert m.market_share(corp) == 0.6


def test_init_corps(mocker):
    mock_conf = mocker.patch("agents.market.agent.SIM_CONFIG")
    mock_seed = mocker.patch("agents.market.agent.CORP_SEED")
    mock_conf.nr_of_corps = 10
    mock_seed.balance = 100
    m = Market()
    m.init_corps()

    assert len(m.corporations) == 10
    assert random.choice(m.corporations).balance == 100


def test_init_people(mocker):
    mock_conf = mocker.patch("agents.market.agent.SIM_CONFIG")
    mock_conf.nr_of_people = 5
    m = Market()
    m.init_people()

    assert len(m.people) == 5


def test_person_corp_list():
    m = Market()
    for i in range(3, 10):
        if i % 3:
            c = Corporation(price=i - 1)
        else:
            c = Corporation(price=i)

        m.corporations.append(c)

    random.shuffle(m.corporations)
    pcl = m.person_corp_list()

    assert pcl[0].price < pcl[-1].price
    assert pcl[0].price == pcl[1].price

    number_ones = []
    for _ in range(0, 10):
        first_place = m.person_corp_list()[0]
        number_ones.append(first_place)

    assert pcl[1] in number_ones
