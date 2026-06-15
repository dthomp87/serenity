import pytest

from serenity.signal.calc import simple_moving_average, true_range, average_true_range


def test_simple_moving_average_basic():
    assert simple_moving_average([1, 2, 3, 4]) == pytest.approx(2.5)
    assert simple_moving_average([5]) == pytest.approx(5.0)


def test_simple_moving_average_empty_raises():
    with pytest.raises(ValueError):
        simple_moving_average([])


def test_true_range_uses_high_low_when_no_gap():
    # high-low range dominates when the previous close sits inside the bar
    assert true_range(high=10.0, low=9.0, prev_close=9.5) == pytest.approx(1.0)


def test_true_range_captures_gap_up():
    # gap up: |high - prev_close| dominates the plain high-low range
    assert true_range(high=12.0, low=11.5, prev_close=10.0) == pytest.approx(2.0)


def test_true_range_captures_gap_down():
    # gap down: |low - prev_close| dominates
    assert true_range(high=9.0, low=8.0, prev_close=10.0) == pytest.approx(2.0)


def test_average_true_range_matches_manual_calculation():
    # (high, low, close); first candle only seeds the previous close
    candles = [
        (10.0, 9.0, 9.5),
        (11.0, 9.8, 10.8),   # TR = max(1.2, |11-9.5|=1.5, |9.8-9.5|=0.3) = 1.5
        (10.9, 10.1, 10.2),  # TR = max(0.8, |10.9-10.8|=0.1, |10.1-10.8|=0.7) = 0.8
        (12.0, 10.5, 11.9),  # TR = max(1.5, |12-10.2|=1.8, |10.5-10.2|=0.3) = 1.8
    ]
    assert average_true_range(candles) == pytest.approx((1.5 + 0.8 + 1.8) / 3)


def test_average_true_range_requires_two_candles():
    with pytest.raises(ValueError):
        average_true_range([(10.0, 9.0, 9.5)])
