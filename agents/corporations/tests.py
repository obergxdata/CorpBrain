import pytest
from unittest.mock import patch

from agents.corporations.agent import Corporation
from agents.market.agent import Market


def test_properties():
    corp = Corporation()
    corp.employees = 20
    corp.balance = 5000
    corp.upe = 5
    corp.salary = 100
    corp.sales = 25
    corp.demand = 50
    corp.price = 5

    assert corp.capacity == corp.employees * corp.upe
    assert corp.revenue == corp.price * corp.sales
    assert corp.costs == corp.employees * corp.salary
    assert corp.profit == corp.revenue - corp.costs
    assert corp.unit_margin == corp.price - (corp.salary / corp.upe)
    assert corp.burn == corp.costs - corp.revenue
    assert corp.runway == corp.balance / corp.burn


def test_history():
    corp = Corporation()
    corp.employees = 20
    corp.upe = 5

    corp.record()

    assert corp.history.last("employees") == 20
    assert corp.history.last("upe") == 5

    corp.employees = 25
    corp.upe = 10

    corp.record()

    assert corp.history.last("employees") == 25
    assert corp.history.last("upe") == 10

    assert corp.history.tail("employees", 2) == [20, 25]
    assert corp.history.tail("upe", 2) == [5, 10]


def test_sell():
    corp = Corporation()
    corp.sales = 10
    corp.demand = 100
    corp.stock = 50
    corp.balance = 0
    corp.price = 10

    orders = 25
    corp.sell(orders=orders)

    assert corp.sales == 35
    assert corp.demand == 125
    assert corp.stock == 25
    assert corp.balance == 10 * 25


def test_profit_trend():
    corp = Corporation()
    corp.history.record(profit=50)
    corp.history.record(profit=100)
    corp.history.record(profit=150)

    # Test default n=2 (last 2 values: 100->150)
    # (150-100)/100 = 0.5
    assert corp.profit_trend == 0.5

    # Test with n=3 (all 3 values: 50->100, 100->150)
    # Average of: (100-50)/50=1.0 and (150-100)/100=0.5 = 0.75
    assert corp.history.trend("profit", n=3) == 0.75

    # Test with just 2 values (100->200)
    # (200-100)/100 = 1.0
    corp2 = Corporation()
    corp2.history.record(profit=100)
    corp2.history.record(profit=200)
    assert corp2.profit_trend == 1.0


def test_reward():
    corp = Corporation()

    corp.history.record(profit=100)
    corp.history.record(profit=200)
    corp.history.record(balance=1000)
    corp.history.record(balance=1000)

    assert corp.reward == 10

    corp.history.record(profit=100)
    corp.history.record(profit=200)
    corp.history.record(balance=1000)
    corp.history.record(balance=2000)

    assert corp.reward == 15
