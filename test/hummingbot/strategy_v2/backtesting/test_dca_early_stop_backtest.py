"""Controller-driven DCA exit must settle at the stop candle, without future prices."""
import unittest
from decimal import Decimal

import pandas as pd

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig, DCAMode
from hummingbot.strategy_v2.models.executor_actions import StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType


class TestDCAEarlyStopBacktest(unittest.TestCase):
    def test_stop_uses_current_price_and_filled_so_levels_without_lookahead(self):
        start = 1700000000.0
        for side, levels, closes in (
            (TradeType.BUY, [100, 99, 98], [100, 99, 98, 96, 120]),
            (TradeType.SELL, [100, 101, 102], [100, 101, 102, 104, 80]),
        ):
            with self.subTest(side=side.name):
                times = [start + i * 60 for i in range(len(closes))]
                candles = pd.DataFrame({
                    "timestamp": times, "close": closes, "high": closes, "low": closes,
                })
                candles.index = candles["timestamp"]
                config = DCAExecutorConfig(
                    timestamp=start, connector_name="binance_perpetual",
                    trading_pair="BTC-USDT", side=side, mode=DCAMode.MAKER,
                    leverage=3, prices=[Decimal(p) for p in levels],
                    amounts_quote=[Decimal(180)] * 3,
                    take_profit=None, stop_loss=None, time_limit=3600,
                )
                engine = BacktestingEngineBase()
                full = engine.dca_executor_simulator.simulate(candles, config, trade_cost=0.0002)
                self.assertGreater(full.get_executor_info_at_timestamp(times[-1]).net_pnl_quote, 0)
                engine.active_executor_simulations.append(full)

                stop_time = times[3]  # Completed candle, before the future favorable bounce.
                expected = engine.dca_executor_simulator.simulate(
                    candles.loc[:stop_time], config, trade_cost=0.0002
                ).get_executor_info_at_timestamp(stop_time)
                self.assertLess(expected.net_pnl_quote, 0)
                engine.handle_stop_action(
                    StopExecutorAction(controller_id="test", executor_id=config.id),
                    timestamp=stop_time, market_data=candles, trade_cost=0.0002,
                )
                ledger = engine.collect_executors_ledger(stop_time)
                self.assertEqual(1, len(ledger))
                actual = ledger[0]
                self.assertEqual(CloseType.EARLY_STOP, actual.close_type)
                self.assertEqual(stop_time, actual.close_timestamp)
                self.assertEqual(expected.net_pnl_quote, actual.net_pnl_quote)
                self.assertEqual(expected.cum_fees_quote, actual.cum_fees_quote)
                self.assertEqual(expected.filled_amount_quote, actual.filled_amount_quote)
                self.assertEqual(expected.custom_info["current_position_average_price"],
                                 actual.custom_info["current_position_average_price"])
                self.assertEqual(closes[3], actual.custom_info["close_price"])
                self.assertEqual(Decimal("1080"), actual.filled_amount_quote)
                self.assertEqual(Decimal("0.216"), actual.cum_fees_quote)
                self.assertEqual(float(actual.net_pnl_quote), engine._executor_realized_pnl)
                self.assertEqual([], engine.active_executor_simulations)

    def test_filled_dca_stop_requires_causal_data_and_fee(self):
        start = 1700000000.0
        candles = pd.DataFrame({
            "timestamp": [start, start + 60, start + 120],
            "close": [100.0, 99.0, 98.0],
            "high": [100.0, 99.0, 98.0],
            "low": [100.0, 99.0, 98.0],
        })
        candles.index = candles["timestamp"]
        config = DCAExecutorConfig(
            timestamp=start, connector_name="binance_perpetual",
            trading_pair="BTC-USDT", side=TradeType.BUY, mode=DCAMode.MAKER,
            leverage=3, prices=[Decimal(100), Decimal(99), Decimal(98)],
            amounts_quote=[Decimal(180)] * 3, time_limit=3600,
            take_profit=None, stop_loss=None,
        )
        engine = BacktestingEngineBase()
        engine.active_executor_simulations.append(
            engine.dca_executor_simulator.simulate(candles, config, trade_cost=0.0002)
        )
        action = StopExecutorAction(controller_id="test", executor_id=config.id)
        with self.assertRaisesRegex(ValueError, "candle data and the backtest trade cost"):
            engine.handle_stop_action(action, timestamp=start + 60)
        self.assertEqual(1, len(engine.active_executor_simulations))
        self.assertEqual([], engine.stopped_executors_info)


if __name__ == "__main__":
    unittest.main()
