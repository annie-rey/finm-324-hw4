import pytest
from somewhat_smart_order_router import best_price_improvement


def test_best_price_improvement_base_case():
    """
    Base case: ensure the function runs end-to-end with real models
    and returns a valid exchange + numeric price improvement.
    """

    best_exchange, best_value = best_price_improvement(
        symbol="AAPL",
        side="B",
        quantity=100,
        limit_price=101.0,
        bid_price=100.0,
        ask_price=100.5,
        bid_size=200,
        ask_size=300,
    )

    # Validate exchange format (IDxxxx)
    assert best_exchange in {"ID1516", "ID29608"}
    # Validate numeric result
    assert isinstance(best_value, float)


def test_best_price_improvement_invalid_bid_ask():
    """
    Corner case: bid_price > ask_price should raise ValueError.
    """

    with pytest.raises(ValueError, match="bid_price cannot be greater than ask_price"):
        best_price_improvement(
            symbol="AAPL",
            side="S",
            quantity=50,
            limit_price=99.0,
            bid_price=105.0, # invalid: bid > ask
            ask_price=100.0,
            bid_size=100,
            ask_size=150,
        )

def test_best_price_improvement_invalid_side():
    """
    Corner case: side must be 'B' or 'S'. Any other value should raise ValueError.
    """
    with pytest.raises(ValueError, match="side must be 'B' \\(buy\\) or 'S' \\(sell\\)."):
        best_price_improvement(
            symbol="AAPL",
            side="X", # invalid side
            quantity=100,
            limit_price=101.0,
            bid_price=100.0,
            ask_price=100.5,
            bid_size=200,
            ask_size=300,
        )

def test_quantity_positive_integer_errors():
    """
    Corner case: test that quantity fails for:
    nonpositive integer (0)
    non-integer type (1.5)
    This tests all the positive integer fields which have the same logic
    """

    with pytest.raises(ValueError, match="quantity must be a positive integer."):
        best_price_improvement(
            symbol="AAPL",
            side="B",
            quantity=0, # invalid: nonpositive
            limit_price=100.0,
            bid_price=99.0,
            ask_price=100.0,
            bid_size=200,
            ask_size=300,
        )

    with pytest.raises(ValueError, match="quantity must be a positive integer."):
        best_price_improvement(
            symbol="AAPL",
            side="B",
            quantity=1.5, # invalid: not an integer
            limit_price=100.0,
            bid_price=99.0,
            ask_price=100.0,
            bid_size=200,
            ask_size=300,
        )
