from __future__ import annotations

from stats.history import History
from settings import SimConfig
from seeds import PersonSeed, CorpSeed
from agents.corporations.agent import Corporation
from agents.corporations.RL.MDP import BellmanMDP, StateDisc
from agents.people.agent import Person
from agents.agents_logger import logger
import random

SIM_CONFIG = SimConfig()
CORP_SEED = CorpSeed()
PERSON_SEED = PersonSeed()


class Market:

    def __init__(self):
        self.corporations: list[Corporation] = []
        self.people: list[Person] = []
        self.history: History = History(window=20000)
        self.tick: int = 0
        self.stop: bool = False

    def run(self):
        pass

    def person_corp_list(self):
        "Sort corps based on price, secondary sort by random"
        return sorted(self.corporations, key=lambda c: (c.price, random.random()))

    def step(self):

        self.tick += 1
        logger.info(f"Tick {self.tick} started")

        if not len([corp for corp in self.corporations if corp.alive]):
            logger.warning(f"{self.tick}: All companies are dead")

        for corp in self.corporations:
            corp.step(self.tick)
        for person in self.people:
            person.get_paid()
            person.step(corps=self.person_corp_list())

        self.record()

    # --- Control --- #
    def record(self):
        self.history.record(avg_price=self.avg_price)
        self.history.record(avg_profit=self.avg_profit)
        self.history.record(avg_nr_employees=self.avg_nr_employees)

    # --- Derived --- #
    @property
    def avg_price(self):
        return sum([corp.price for corp in self.corporations if corp.alive]) / len(
            self.corporations
        )

    @property
    def avg_profit(self):
        return sum(
            [corp.history.last("profit") for corp in self.corporations if corp.alive]
        ) / len(self.corporations)

    @property
    def avg_nr_employees(self):
        return sum([corp.employees for corp in self.corporations if corp.alive]) / len(
            self.corporations
        )

    @property
    def min_max_price(self):
        prices = [corp.price for corp in self.corporations if corp.alive]
        return min(prices), max(prices)

    @property
    def min_max_market_share(self):
        market_share = [
            self.market_share(corp) for corp in self.corporations if corp.alive
        ]
        return min(market_share), max(market_share)

    @property
    def avg_mpc(self):
        return sum([p.mpc for p in self.people]) / len(self.people)

    def market_share(self, corp: Corporation):
        return corp.history.last("sales", default=0) / len(self.people)

    # -- Init --- #
    def init_corps(self):
        state_disc = StateDisc(self)
        for _ in range(0, SIM_CONFIG.nr_of_corps):
            corp = Corporation(
                balance=CORP_SEED.balance,
                employees=CORP_SEED.nr_of_employees,
                upe=CORP_SEED.upe,
                salary=CORP_SEED.salary,
                price=CORP_SEED.price,
                max_tick=SIM_CONFIG.nr_of_ticks,
            )
            corp.MDP = BellmanMDP(state_disc=state_disc, corp=corp)
            self.corporations.append(corp)

    def init_people(self):
        for _ in range(0, SIM_CONFIG.nr_of_people):
            salary = random.randint(
                SIM_CONFIG.min_base_salary, SIM_CONFIG.max_base_salary
            )
            person = Person(
                mpc=PERSON_SEED.mpc,
                salary=salary,
                balance=salary * 3,
                latest_pay=salary,
            )
            self.people.append(person)
