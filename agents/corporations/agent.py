from dataclasses import dataclass, field
from stats.history import History
from typing import TYPE_CHECKING
import uuid
import random
import logging

logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    from agents.corporations.RL.MDP import BellmanMDP


@dataclass(slots=True)
class Corporation:

    tick: int = 0
    max_tick: int = 0
    employees: int = 0
    balance: float = 0.0
    cost: float = 0.0
    stock: int = 0
    upe: int = 0
    salary: float = 0.0
    sales: int = 0
    demand: int = 0
    produced: int = 0
    price: float = 0.0
    alive: bool = True
    name: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: History = field(default_factory=lambda: History(window=5000))
    MDP: "BellmanMDP | None" = None

    def actions(self):
        actions = {
            "change_price": (
                self.change_price,
                [-0.10, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.1],
            ),
        }

        return actions

    # --- Actions --- #
    def sell(self, orders: int):
        # We register demand either way
        self.demand += orders
        if self.stock >= orders:

            self.stock -= orders
            self.sales += orders
            self.balance += self.price * orders
            return True
        return False

    def change_price(self, pct: float):
        self.price *= 1 + pct

    def change_employees(self, q: int):
        self.employees += q

    def pay_saleries(self):
        self.cost = self.salary * self.employees
        self.balance -= self.cost
        if self.balance < 0:
            logging.warning(f"corp {self.name} is dead")
            self.alive = False

    def produce(self):

        if self.tick == 1:
            qty = self.capacity
        else:
            qty = min(self.capacity - self.stock, self.history.last("demand"))
        self.produced = qty
        self.stock += qty

    # --- Control --- #
    def step(self, tick: int):
        if not self.alive:
            return

        self.tick = tick
        # Take Action
        if tick > 1 and random.randint(1, 3) == 2:
            self.MDP.choose_action()
        self.produce()
        self.pay_saleries()

    def finish_step(self):
        self.record()
        # Take MDP step
        self.MDP.step()
        # Reset values
        self.clean()

    def record(self):
        self.history.record(
            employees=self.employees,
            balance=self.balance,
            stock=self.stock,
            upe=self.upe,
            salary=self.salary,
            sales=self.sales,
            demand=self.demand,
            price=self.price,
            profit=self.profit,
            cost=self.cost,
            produced=self.produced,
        )

    def clean(self):
        self.sales = 0
        self.demand = 0
        self.produced = 0

    # --- Derived --- #
    @property
    def capacity(self):
        return self.employees * self.upe

    @property
    def revenue(self):
        return self.price * self.sales

    @property
    def costs(self):
        return self.employees * self.salary

    @property
    def profit(self):
        return self.revenue - self.costs

    @property
    def unit_margin(self):
        cost = self.salary / self.upe
        return self.price - cost

    @property
    def burn(self):
        return max(0.0, self.costs - self.revenue)

    @property
    def runway(self):
        return self.balance / self.burn

    @property
    def reward(self):

        profit_reward = self.history.delta("profit") / 10.0
        return profit_reward
