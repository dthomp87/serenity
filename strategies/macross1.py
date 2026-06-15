import logging
from enum import Enum, auto

import pandas as pd

from datetime import timedelta

from tau.core import Event, NetworkScheduler
from tau.event import Do
from tau.signal import Map, BufferWithTime

from serenity.algo.api import Strategy, StrategyContext
from serenity.signal.indicators import ComputeMovingAverageCrossover
from serenity.signal.marketdata import ComputeOHLC
from serenity.trading.api import Side, OrderStatus, ExecutionReport, Reject
from serenity.trading.oms import OrderPlacerService


class MovingAverageCrossoverStrategy1(Strategy):
    """
    A classic dual moving-average crossover trend-following strategy. It bins trades into
    OHLC candles, computes a fast and a slow simple moving average of the close price, and:

      * enters a long position on a "golden cross" (fast crosses above slow), and
      * flattens the position on a "death cross" (fast crosses below slow).

    A protective stop is placed a configurable percentage below the entry price so a single
    adverse move is bounded. This is a well-understood, sound strategy -- it harvests
    sustained trends and pays small premiums in choppy, range-bound markets. It is not a
    money printer: expect a modest win rate with winners larger than losers.
    """

    logger = logging.getLogger(__name__)

    def init(self, ctx: StrategyContext):
        scheduler = ctx.get_scheduler()
        network = scheduler.get_network()

        contract_qty = int(ctx.getenv('MACROSS_QTY', 1))
        fast_window = int(ctx.getenv('MACROSS_FAST_WINDOW', 10))
        slow_window = int(ctx.getenv('MACROSS_SLOW_WINDOW', 30))
        stop_pct = float(ctx.getenv('MACROSS_STOP_PCT', 2.0)) / 100.0
        bin_minutes = int(ctx.getenv('MACROSS_BIN_MINUTES', 5))
        cooling_period_seconds = int(ctx.getenv('MACROSS_COOL_SECONDS', 15))
        exchange_code, instrument_code = ctx.getenv('TRADING_INSTRUMENT').split(':')
        instrument = ctx.get_instrument_cache().get_crypto_exchange_instrument(exchange_code, instrument_code)
        trades = ctx.get_marketdata_service().get_trades(instrument)
        trades_bin = BufferWithTime(scheduler, trades, timedelta(minutes=bin_minutes))
        prices = ComputeOHLC(network, trades_bin)
        close_prices = Map(network, prices, lambda x: x.close_px)
        macross = ComputeMovingAverageCrossover(network, close_prices, fast_window, slow_window)

        op_service = ctx.get_order_placer_service()
        oms = op_service.get_order_manager_service()
        dcs = ctx.get_data_capture_service()

        exchange_id = ctx.getenv('EXCHANGE_ID', 'phemex')
        exchange_instance = ctx.getenv('EXCHANGE_INSTANCE', 'prod')
        account = ctx.getenv('EXCHANGE_ACCOUNT')
        op_uri = f'{exchange_id}:{exchange_instance}'

        # subscribe to marks, position updates and exchange position updates
        marks = ctx.get_mark_service().get_marks(instrument)
        Do(scheduler.get_network(), marks, lambda: self.logger.debug(marks.get_value()))

        position = ctx.get_position_service().get_position(account, instrument)
        Do(scheduler.get_network(), position, lambda: self.logger.info(position.get_value()))

        exch_position = ctx.get_exchange_position_service().get_exchange_positions()
        Do(scheduler.get_network(), exch_position, lambda: self.logger.info(exch_position.get_value()))

        # capture position and moving-average crossover data
        Do(scheduler.get_network(), position, lambda: dcs.capture('Position', {
            'time': pd.to_datetime(scheduler.get_time(), unit='ms'),
            'position': position.get_value().get_qty()
        }))
        Do(scheduler.get_network(), macross, lambda: dcs.capture('MovingAverageCrossover', {
            'time': pd.to_datetime(scheduler.get_time(), unit='ms'),
            'fast': macross.get_value().fast,
            'slow': macross.get_value().slow,
            'spread': macross.get_value().spread()
        }))

        # debug log basic marketdata
        Do(scheduler.get_network(), prices, lambda: self.logger.debug(prices.get_value()))

        class TraderState(Enum):
            GOING_LONG = auto()
            LONG = auto()
            FLATTENING = auto()
            FLAT = auto()

        # order placement logic
        class CrossoverTrader(Event):
            # noinspection PyShadowingNames
            def __init__(self, scheduler: NetworkScheduler, op_service: OrderPlacerService,
                         strategy: MovingAverageCrossoverStrategy1):
                self.scheduler = scheduler
                self.op = op_service.get_order_placer(f'{op_uri}')
                self.strategy = strategy
                self.last_entry = 0
                self.last_exit = 0
                self.cum_pnl = 0
                self.stop = None
                self.trader_state = TraderState.FLAT
                self.last_trade_time = 0
                # tracks the sign of (fast - slow) on the previous update so we can detect a
                # crossover rather than merely an up- or down-trend
                self.last_spread_sign = 0

                self.scheduler.get_network().connect(oms.get_order_events(), self)
                self.scheduler.get_network().connect(position, self)

            def on_activate(self) -> bool:
                if self.scheduler.get_network().has_activated(oms.get_order_events()):
                    order_event = oms.get_order_events().get_value()
                    if isinstance(order_event, ExecutionReport) and order_event.is_fill():
                        order_type = 'stop order' if self.stop is not None and order_event.get_order_id() == \
                                                     self.stop.order_id else 'market order'
                        self.strategy.logger.info(f'Received fill event for {order_type}: {order_event}')
                        if self.trader_state == TraderState.GOING_LONG:
                            self.last_entry = order_event.get_last_px()
                            if order_event.get_order_status() == OrderStatus.FILLED:
                                self.strategy.logger.info(f'Entered long position: entry price={self.last_entry}')
                                self.trader_state = TraderState.LONG
                        elif self.trader_state in (TraderState.FLATTENING, TraderState.LONG) and \
                                order_event.get_order_status() == OrderStatus.FILLED:
                            if order_type == 'stop order':
                                self.strategy.logger.info(f'stop loss filled at {order_event.get_last_px()}')
                                self.stop = None

                            trade_pnl = (order_event.get_last_px() - self.last_entry) * \
                                        (contract_qty / self.last_entry)
                            self.cum_pnl += trade_pnl
                            self.strategy.logger.info(f'Trade P&L={trade_pnl}; cumulative P&L={self.cum_pnl}')

                            dcs.capture('PnL', {
                                'time': pd.to_datetime(scheduler.get_time(), unit='ms'),
                                'trade_pnl': trade_pnl,
                                'cum_pnl': self.cum_pnl
                            })
                            self.trader_state = TraderState.FLAT
                    elif isinstance(order_event, Reject):
                        self.strategy.logger.error(f'Order rejected: {order_event.get_message()}')
                        self.trader_state = TraderState.FLAT
                else:
                    spread = macross.get_value().spread()
                    spread_sign = 1 if spread > 0 else (-1 if spread < 0 else 0)
                    golden_cross = self.last_spread_sign <= 0 < spread_sign
                    death_cross = self.last_spread_sign >= 0 > spread_sign
                    self.last_spread_sign = spread_sign

                    if self.trader_state == TraderState.FLAT and golden_cross:
                        if self.last_trade_time != 0 and (scheduler.get_time() - self.last_trade_time) < \
                                cooling_period_seconds * 1000:
                            self.strategy.logger.info('Cooling off -- not trading again on rapidly repeated signal')
                            return False

                        last_px = close_prices.get_value()
                        stop_px = last_px * (1 - stop_pct)
                        self.strategy.logger.info(f'Golden cross at {scheduler.get_clock().get_time()}, '
                                                  f'enter long: last_px = {last_px}, stop_px = {stop_px}')

                        order = self.op.get_order_factory().create_market_order(Side.BUY, contract_qty, instrument)
                        self.stop = self.op.get_order_factory().create_stop_order(Side.SELL, contract_qty, stop_px,
                                                                                  instrument)
                        self.op.submit(order)
                        self.op.submit(self.stop)

                        self.last_trade_time = scheduler.get_time()
                        self.trader_state = TraderState.GOING_LONG
                    elif self.trader_state == TraderState.LONG and death_cross:
                        self.strategy.logger.info(f'Death cross at {scheduler.get_clock().get_time()}, '
                                                  f'exiting long position')

                        order = self.op.get_order_factory().create_market_order(Side.SELL, contract_qty, instrument)
                        self.op.submit(order)
                        if self.stop is not None:
                            self.op.cancel(self.stop)
                            self.stop = None

                        self.trader_state = TraderState.FLATTENING
                return False

        network.connect(macross, CrossoverTrader(scheduler, op_service, self))
