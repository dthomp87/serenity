import numpy as np

from tau.core import Network, Signal
from tau.signal import Function, WindowWithCount


class MovingAverageCrossover:
    """
    Value class holding the fast and slow simple moving averages used by a dual
    moving-average crossover trend-following strategy.
    """
    def __init__(self, fast: float, slow: float):
        self.fast = fast
        self.slow = slow

    def spread(self) -> float:
        """
        Signed distance between the fast and slow averages. Positive values indicate an
        up-trend (fast above slow); negative values a down-trend.
        """
        return self.fast - self.slow

    def __str__(self) -> str:
        return f'MACross(fast={self.fast}, slow={self.slow}, spread={self.spread()})'


class ComputeMovingAverageCrossover(Function):
    """
    Computes a fast and a slow simple moving average from a single price stream. The
    indicator only becomes valid once enough prices have accumulated to fill the slow
    window, so both averages cover a full window before any crossover is acted upon.
    """
    def __init__(self, network: Network, prices: Signal, fast_window: int, slow_window: int):
        if fast_window >= slow_window:
            raise ValueError(f'fast_window ({fast_window}) must be less than slow_window ({slow_window})')
        super().__init__(network, [prices])
        self.prices = prices
        self.fast_buffer = WindowWithCount(network, prices, fast_window)
        self.slow_buffer = WindowWithCount(network, prices, slow_window)

    def _call(self):
        if self.fast_buffer.is_valid() and self.slow_buffer.is_valid():
            fast = np.array(self.fast_buffer.get_value()).mean()
            slow = np.array(self.slow_buffer.get_value()).mean()
            self._update(MovingAverageCrossover(fast, slow))


class ComputeAverageTrueRange(Function):
    """
    Computes the Average True Range (ATR) from a stream of OHLC candles. ATR is a classic
    volatility measure: the mean of the True Range over a window, where True Range for each
    candle is the greatest of (high - low), |high - prev_close| and |low - prev_close|.

    The indicator buffers ``window + 1`` candles so it always has a previous close available
    for every True Range in the window, and only becomes valid once that buffer is full.
    """
    def __init__(self, network: Network, ohlc: Signal, window: int):
        super().__init__(network, [ohlc])
        self.ohlc = ohlc
        self.window = window
        self.buffer = WindowWithCount(network, ohlc, window + 1)

    def _call(self):
        if self.buffer.is_valid():
            candles = self.buffer.get_value()
            true_ranges = []
            for i in range(1, len(candles)):
                prev_close = candles[i - 1].close_px
                cur = candles[i]
                true_range = max(cur.high_px - cur.low_px,
                                 abs(cur.high_px - prev_close),
                                 abs(cur.low_px - prev_close))
                true_ranges.append(true_range)
            if true_ranges:
                self._update(float(np.mean(true_ranges)))


class BollingerBands:
    def __init__(self, sma, upper, lower):
        self.sma = sma
        self.upper = upper
        self.lower = lower

    def __str__(self) -> str:
        return f'BBands(SMA={self.sma}, upper={self.upper}, lower={self.lower})'


class ComputeBollingerBands(Function):
    def __init__(self, network: Network, prices: Signal, window: int, num_std: int):
        super().__init__(network, [prices])
        self.prices = prices
        self.buffer = WindowWithCount(network, prices, window)
        self.num_std = num_std

    def _call(self):
        if self.buffer.is_valid():
            np_array = np.array(self.buffer.get_value())
            sma = np_array.mean()
            sd = np_array.std()
            upper = sma + (self.num_std * sd)
            lower = sma - (self.num_std * sd)
            self._update(BollingerBands(sma, upper, lower))
