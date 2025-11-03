from dataclasses import dataclass
from agents.corporations.agent import Corporation


@dataclass(slots=True)
class Person:

    mpc: float = 0.0
    balance: float = 0.0
    latest_pay: float = 0.0
    tick: int = 0
    purchases: int = 0

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

    def get_paid(self, amount: float):
        self.balance += amount
        self.latest_pay += amount

    # --- Control --- #
    def step(self, corps: list[Corporation]):
        self.tick += 1

        budget = self.latest_pay * self.mpc
        self.buy(budget=budget, corps=corps)
        self.latest_pay = 0

    def clean(self):
        self.latest_pay = 0
