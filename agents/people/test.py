import pytest
from agents.people.agent import Person
from agents.corporations.agent import Corporation


def make_corp(price: float, stock: int) -> Corporation:
    """Helper to create a corporation with given price and stock."""
    return Corporation(price=price, stock=stock)


@pytest.mark.parametrize(
    "budget,corp_prices,corp_stocks,expected_purchases",
    [
        # Simple case: buy one item from cheapest corp
        (10.0, [5.0, 15.0, 20.0], [10, 10, 10], 2),
        # Buy from each affordable corp, then recurse with leftover budget
        (25.0, [5.0, 15.0, 20.0], [10, 10, 10], 3),
        # Budget exactly matches total price
        (15.0, [5.0, 10.0], [5, 5], 2),
        # Out of stock scenario
        (100.0, [10.0, 20.0], [2, 1], 3),
        # All corps too expensive
        (5.0, [10.0, 20.0, 30.0], [10, 10, 10], 0),
        # Empty corp list
        (100.0, [], [], 0),
    ],
)
def test_buy(budget, corp_prices, corp_stocks, expected_purchases):
    """Test that buy function correctly purchases from cheapest corps first."""
    person = Person()

    # Create corps sorted by price (as the function expects)
    corps = [make_corp(price, stock) for price, stock in zip(corp_prices, corp_stocks)]

    person.buy(budget=budget, corps=corps)

    assert person.purchases == expected_purchases
