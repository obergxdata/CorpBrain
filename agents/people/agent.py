from dataclasses import dataclass, field
from agents.corporations.agent import Corporation
import uuid
import random


@dataclass(slots=True)
class Person:

    mpc: float = 0.0
    balance: float = 0.0
    latest_pay: float = 0.0
    tick: int = 0
    purchases: int = 0
    salary: float = 0
    name: str = field(default_factory=lambda: str(uuid.uuid4()))

    # --- Actions --- #
    def buy(self, budget: float, corps: list[Corporation]):
        purchased_this_round = False

        # Try to buy from each corp once in this iteration
        for corp in corps:
            if corp.price <= budget:
                purchase = corp.sell(1)
                # It might be out of stock
                if purchase:
                    budget -= corp.price
                    self.purchases += 1
                    purchased_this_round = True

        # If we made purchases and still have budget, recurse with leftover budget
        if budget > 0 and purchased_this_round:
            self.buy(budget=budget, corps=corps)

        return

    def get_paid(self):
        self.balance += self.salary
        self.latest_pay += self.salary

    # --- Control --- #
    def step(self, corps: list[Corporation]):
        self.tick += 1

        # Imperfect information: each person only sees 50% of available corporations
        sample_size = max(1, len(corps) // 2)
        visible_corps = random.sample(corps, sample_size)
        # Sort by price (cheapest first) - stable sort preserves random sample order for ties
        visible_corps.sort(key=lambda c: c.price)

        budget = self.latest_pay * self.mpc
        self.buy(budget=budget, corps=visible_corps)
        self.latest_pay = 0

    def clean(self):
        self.latest_pay = 0
