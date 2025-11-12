from agents.corporations.RL.MDP import StateDisc, BellmanMDP
from agents.corporations.agent import Corporation
from agents.market.agent import Market
from settings import SimConfig
from seeds import CorpSeed, PersonSeed
from unittest.mock import PropertyMock
import json
import pytest


@pytest.mark.parametrize(
    "prices,expected",
    [
        ([10, 100, 50], 0.4),
        ([10, 20, 200], 1.0),
        ([10, 10, 10], 0.5),
        ([50, 44, 22], 0.0),
    ],
)
def test_price_bucket(prices, expected, default_sim_config, default_corp_seed, default_person_seed):
    m = Market(default_sim_config, default_corp_seed, default_person_seed)
    for price in prices:
        corp = Corporation(price=price)
        m.corporations.append(corp)

    # Last corp
    state = StateDisc(market=m)
    assert state.bucket_price(corp=corp) == expected


def test_hash_state(mocker, default_sim_config, default_corp_seed, default_person_seed):
    m = Market(default_sim_config, default_corp_seed, default_person_seed)
    corp = Corporation()
    state = StateDisc(market=m)
    v = {"price_bucket": "mid"}
    mocker.patch.object(StateDisc, "get_state", return_value=v)

    expected = "ce5094725deddf0673d4f1bc77f9f7ad03585309"
    assert state.hash_state(corp) == expected


def test_mdp_env_eval_action(mocker):
    m = Market
    corp = Corporation()
    state = StateDisc(market=m)
    hash_state = "abc1"
    mocker.patch.object(StateDisc, "hash_state", return_value=hash_state)
    mdp = BellmanMDP(state_disc=state, corp=corp)

    # Perform price increase
    mdp.record_action("price_increase", state=hash_state)
    assert mdp.eval_queue == (hash_state, "price_increase")
    # Evaluate action from queue
    mdp._eval_action(reward=10)
    assert dict(dict(mdp.records)[hash_state]) == {"price_increase": [("abc1", 10)]}
    # Perform evaluate without action in queue
    mdp._eval_action(reward=20)
    assert dict(dict(mdp.records)[hash_state]) == {"price_increase": [("abc1", 10)]}
    # Perform hire in same state
    mdp.record_action("hire", state=hash_state)
    mdp._eval_action(reward=15)
    assert dict(dict(mdp.records)[hash_state]) == {
        "price_increase": [("abc1", 10)],
        "hire": [("abc1", 15)],
    }
    # Add a new price_increase with different reward
    mdp.record_action("price_increase", state=hash_state)
    mdp._eval_action(reward=12)
    assert dict(dict(mdp.records)[hash_state]) == {
        "price_increase": [("abc1", 10), ("abc1", 12)],
        "hire": [("abc1", 15)],
    }
    # Add new state
    hash_state = "abc2"
    mocker.patch.object(StateDisc, "hash_state", return_value=hash_state)
    mdp.record_action("hire", state=hash_state)
    mdp._eval_action(reward=5)
    assert dict(dict(mdp.records)[hash_state]) == {"hire": [("abc2", 5)]}


def test_mdp_update_env():

    m = Market
    corp = Corporation()
    state = StateDisc(market=m)
    mdp = BellmanMDP(state_disc=state, corp=corp)
    mdp.records = {
        "HUNGRY": {
            "EAT": [
                ("FULL", 10),
                ("FULL", 5),
                ("TIRED", -5),
            ],
            "SLEEP": [
                ("HUNGRY", -5),
                ("TIRED", -5),
                ("RESTED", 5),
            ],
        },
        "TIRED": {
            "EAT": [
                ("HUNGRY", -5),
                ("HUNGRY", -5),
            ],
            "SLEEP": [
                ("RESTED", 5),
                ("RESTED", 10),
            ],
        },
    }
    assert mdp._update_env("HUNGRY", "EAT") == [
        (0.67, "FULL", 7.5),
        (0.33, "TIRED", -5.0),
    ]

    assert mdp._update_env("TIRED", "SLEEP") == [
        (1.0, "RESTED", 7.5),
    ]


def test_mdp_value_iter():
    m = Market
    corp = Corporation()
    state = StateDisc(market=m)
    mdp = BellmanMDP(state_disc=state, corp=corp)
    mdp.env = {
        "HUNGRY": {
            "EAT": [(0.8, "FULL", +5), (0.2, "HUNGRY", +5)],
            "WAIT": [(1.0, "HUNGRY", -1)],
            "TV": [(0.7, "HUNGRY", +2), (0.3, "BED", +2)],
        },
        "FULL": {
            "WORK": [(1.0, "HUNGRY", +3)],
            "SLEEP": [(1.0, "BED", +1)],
            "TV": [(0.5, "BED", +2), (0.5, "HUNGRY", +2)],
        },
        "BED": {"DONE": [(1.0, "BED", 0)]},
    }

    assert dict(mdp.value_iter()) == {
        "HUNGRY": 41.6,
        "FULL": 40.44,
        "BED": 0.0,
    }


def test_mdp_build_policy():
    m = Market
    corp = Corporation()
    state = StateDisc(market=m)
    mdp = BellmanMDP(state_disc=state, corp=corp)

    mdp.env = {
        "HUNGRY": {
            "EAT": [(0.8, "FULL", +5), (0.2, "HUNGRY", +5)],
            "WAIT": [(1.0, "HUNGRY", -1)],
            "TV": [(0.7, "HUNGRY", +2), (0.3, "BED", +2)],
        },
        "FULL": {
            "WORK": [(1.0, "HUNGRY", +3)],
            "SLEEP": [(1.0, "BED", +1)],
            "TV": [(0.5, "BED", +2), (0.5, "HUNGRY", +2)],
        },
        "BED": {"DONE": [(1.0, "BED", 0)]},
    }

    mdp.V = {
        "HUNGRY": 41.6,
        "FULL": 40.44,
        "BED": 0.0,
    }

    assert mdp.build_policy() == {
        "BED": ("DONE", 0.0),
        "FULL": ("WORK", 40.44),
        "HUNGRY": ("EAT", 41.6),
    }
