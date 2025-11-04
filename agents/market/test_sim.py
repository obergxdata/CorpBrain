from agents.market.agent import Market

import pytest
import random


def test_run_10(mocker):

    mock_sim_conf = mocker.patch("agents.market.agent.SIM_CONFIG")
    mock_corp_seed = mocker.patch("agents.market.agent.CORP_SEED")
    mock_person_seed = mocker.patch("agents.market.agent.PERSON_SEED")

    # Sim conf
    mock_sim_conf.nr_of_people = 1500
    mock_sim_conf.nr_of_corps = 3
    mock_sim_conf.min_base_salary = 50
    mock_sim_conf.max_base_salary = 125
    mock_sim_conf.nr_of_ticks = 100

    # Corp seed
    mock_corp_seed.balance = 50000
    mock_corp_seed.price = 35
    mock_corp_seed.salary = 100
    mock_corp_seed.nr_of_employees = 10
    mock_corp_seed.upe = 8
    # Person seed
    mock_person_seed.mpc = 0.5
    mock_person_seed.balance = 0

    m = Market()
    m.init_corps()
    m.init_people()

    for _ in range(0, 10000):
        if m.stop:
            return

        m.step()

    m.history.chart(keys=["avg_profit"], name="avg_profit")
    m.history.chart(keys=["avg_price"], name="avg_price")
    m.history.chart(keys=["avg_nr_employees"], name="avg_nr_employees")
    rand_corp = random.choice(m.corporations)
    raise Exception(rand_corp.MDP.format_env())
