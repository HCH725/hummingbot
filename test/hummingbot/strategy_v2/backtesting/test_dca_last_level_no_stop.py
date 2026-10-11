"""Regression: a fully filled DCA ladder with no stop-loss must not index past its last level."""
import unittest
from decimal import Decimal

import pandas as pd

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.backtesting.executors_simulator.dca_executor_simulator import DCAExecutorSimulator
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig, DCAMode
from hummingbot.strategy_v2.models.executors import CloseType


class TestDCALastLevelNoStop(unittest.TestCase):
    def test_last_stage_without_stop_loss_buy_and_sell(self):
        for side, prices, closes in (
            (TradeType.BUY, [100, 99, 98], [100, 99, 97, 96]),
            (TradeType.SELL, [100, 101, 102], [100, 101, 103, 104]),
        ):
            with self.subTest(side=side.name):
                timestamp = 1700000000.0
                frame = pd.DataFrame({
                    "timestamp": [timestamp + i * 60 for i in range(4)],
                    "close": [float(c) for c in closes],
                    "low": [float(c) for c in closes],
                    "high": [float(c) for c in closes],
                })
                frame.index = frame["timestamp"]
                config = DCAExecutorConfig(
                    timestamp=timestamp,
                    connector_name="binance_perpetual", trading_pair="BTC-USDT",
                    side=side, mode=DCAMode.MAKER, leverage=3,
                    prices=[Decimal(p) for p in prices],
                    amounts_quote=[Decimal(180)] * 3,
                    take_profit=None, stop_loss=None, time_limit=180,
                )
                result = DCAExecutorSimulator().simulate(frame, config, trade_cost=0.0002)
                self.assertEqual(CloseType.TIME_LIMIT, result.close_type)
                ledger = result.executor_simulation
                self.assertEqual(4, len(ledger))
                self.assertEqual(180, ledger["filled_amount_quote_2"].iloc[-1])
                self.assertTrue(ledger["net_pnl_quote"].notna().all())


if __name__ == "__main__":
    unittest.main()
