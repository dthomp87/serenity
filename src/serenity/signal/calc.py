"""
Pure, dependency-free quantitative helper functions.

These functions deliberately avoid importing tau, numpy or any other heavyweight
dependency so they can be unit tested in isolation and reused outside the reactive
signal graph (e.g. in batch research notebooks). The reactive ``Function`` wrappers in
:mod:`serenity.signal.indicators` delegate their math to the helpers here.
"""
from typing import Sequence, Tuple


def simple_moving_average(values: Sequence[float]) -> float:
    """
    Arithmetic mean of ``values``. Raises ValueError on an empty sequence so callers do
    not silently divide by zero.
    """
    if not values:
        raise ValueError('simple_moving_average requires at least one value')
    return sum(values) / len(values)


def true_range(high: float, low: float, prev_close: float) -> float:
    """
    Wilder's True Range for a single bar: the greatest of the current high-low range and
    the gaps from the previous close. This captures overnight/gap moves that a plain
    high-low range would miss.
    """
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def average_true_range(candles: Sequence[Tuple[float, float, float]]) -> float:
    """
    Average True Range over a series of ``(high, low, close)`` candles. The first candle is
    used only to seed the previous close, so True Range is computed for ``candles[1:]`` and
    at least two candles are required.
    """
    if len(candles) < 2:
        raise ValueError('average_true_range requires at least two candles')
    true_ranges = []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1][2]
        high, low, _close = candles[i]
        true_ranges.append(true_range(high, low, prev_close))
    return simple_moving_average(true_ranges)
