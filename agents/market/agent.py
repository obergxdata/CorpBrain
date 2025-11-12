from __future__ import annotations

from stats.history import History
from settings import SimConfig
from seeds import PersonSeed, CorpSeed
from agents.corporations.agent import Corporation
from agents.corporations.RL.MDP import BellmanMDP, StateDisc
from agents.people.agent import Person
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)


class Market:

    def __init__(
        self,
        sim_config: SimConfig,
        corp_seed: CorpSeed,
        person_seed: PersonSeed,
    ):
        self.sim_config = sim_config
        self.corp_seed = corp_seed
        self.person_seed = person_seed
        self.corporations: list[Corporation] = []
        self.people: list[Person] = []
        self.history: History = History(window=20000)
        self.tick: int = 0
        self.stop: bool = False

    def run(self):
        self.init_corps()
        self.init_people()

        logging.info(f"Running {self.sim_config.nr_of_ticks} ticks")

        for _ in range(0, self.sim_config.nr_of_ticks):
            if not self.stop:
                self.step()
                if not self.tick % 250:
                    logging.info(f"Tick {self.tick}")

        logging.info("ticks completed")

    def generate_charts(self):
        self.history.chart(
            keys=["avg_price"],
            name="avg_price",
            subfolder="market",
        )
        self.history.chart(
            keys=["avg_profit"],
            name="avg_profit",
            subfolder="market",
        )

        self.history.chart(
            keys=["avg_sales"],
            name="avg_sales",
            subfolder="market",
        )

        # Corp prices
        self.history.chart(
            keys=[f"price_{corp.name}" for corp in self.alive_corps],
            name="corp_prices",
            subfolder="corp",
            smooth=50,  # 50-tick moving average for smoother lines
        )

        self.history.chart(
            keys=[f"profit_{corp.name}" for corp in self.alive_corps],
            name="corp_profit",
            subfolder="corp",
            smooth=50,  # 50-tick moving average for smoother lines
        )

    @property
    def alive_corps(self):
        return [corp for corp in self.corporations if corp.alive]

    def step(self):

        self.tick += 1

        if not len(self.alive_corps):
            logging.warning("All companies went out of business")
            self.stop = True

        for corp in self.alive_corps:
            corp.step(self.tick)
        for person in self.people:
            person.get_paid()
            person.step(corps=self.alive_corps)

        for corp in self.alive_corps:
            corp.finish_step()

        self.record()

    # --- Control --- #
    def record(self):
        self.history.record(avg_price=self.avg_price)
        self.history.record(avg_profit=self.avg_profit)
        self.history.record(avg_sales=self.avg_sales)

        for corp in self.alive_corps:
            self.history.record(**{f"price_{corp.name}": corp.price})
            self.history.record(**{f"profit_{corp.name}": corp.history.last("profit")})

    # --- Derived --- #
    @property
    def avg_price(self):
        return sum([corp.price for corp in self.alive_corps if corp.alive]) / len(
            self.alive_corps
        )

    @property
    def avg_profit(self):
        return sum(
            [corp.history.last("profit") for corp in self.alive_corps if corp.alive]
        ) / len(self.alive_corps)

    @property
    def avg_stock(self):
        return sum([corp.stock for corp in self.alive_corps]) / len(self.alive_corps)

    @property
    def avg_nr_employees(self):
        return sum([corp.employees for corp in self.alive_corps if corp.alive]) / len(
            self.alive_corps
        )

    @property
    def min_max_price(self):
        prices = [corp.price for corp in self.alive_corps if corp.alive]
        return min(prices), max(prices)

    @property
    def min_max_market_share(self):
        market_share = [
            self.market_share(corp) for corp in self.alive_corps if corp.alive
        ]
        return min(market_share), max(market_share)

    @property
    def avg_salary(self):
        return sum([p.salary for p in self.people]) / len(self.people)

    @property
    def avg_sales(self):
        avg_sales = sum(
            [corp.history.last("sales") for corp in self.alive_corps]
        ) / len(self.alive_corps)
        return avg_sales

    @property
    def min_max_salary(self):
        salaries = [p.salary for p in self.people]
        return min(salaries), max(salaries)

    @property
    def avg_mpc(self):
        return sum([p.mpc for p in self.people]) / len(self.people)

    def market_share(self, corp: Corporation):
        return corp.history.last("sales", default=0) / len(self.people)

    # -- Init --- #
    def init_corps(self):
        logging.info(f"Initiating {self.sim_config.nr_of_corps} corps")
        state_disc = StateDisc(self)
        for _ in range(0, self.sim_config.nr_of_corps):
            corp = Corporation(
                balance=self.corp_seed.balance,
                employees=self.corp_seed.nr_of_employees,
                upe=self.corp_seed.upe,
                salary=self.corp_seed.salary,
                price=self.corp_seed.price,
                max_tick=self.sim_config.nr_of_ticks,
            )
            corp.MDP = BellmanMDP(state_disc=state_disc, corp=corp)
            self.corporations.append(corp)

    def init_people(self):
        logging.info(f"Initiating {self.sim_config.nr_of_people} people")
        salaries = self.generate_income_distribution(
            min_income=self.sim_config.min_base_salary,
            max_income=self.sim_config.max_base_salary,
            n=self.sim_config.nr_of_people,
        )
        for i in range(0, self.sim_config.nr_of_people):
            salary = salaries[i]
            person = Person(
                mpc=self.person_seed.mpc,
                salary=salary,
                balance=salary * 3,
                latest_pay=salary,
            )
            self.people.append(person)

    def generate_income_distribution(self, min_income, max_income, n) -> list:

        mean = (min_income + max_income) / 2
        std = (max_income - min_income) / 6  # ~99.7% of values within range

        incomes = np.random.normal(mean, std, n)
        incomes = np.clip(incomes, min_income, max_income)

        return incomes
