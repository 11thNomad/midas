from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.simulator import FillSimulator
from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class OneShotStrategy(BaseStrategy):
    def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
        ts = market_data["timestamp"]
        instrument = self.config.get("instrument", "NIFTY")
        if self.state.current_position is None:
            self.state.current_position = {"symbol": instrument, "quantity": 1}
            return Signal(
                signal_type=SignalType.ENTRY_LONG,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[{"symbol": instrument, "action": "BUY", "quantity": 1}],
                regime=regime,
            )
        if ts == self.config["exit_ts"]:
            self.state.current_position = None
            return Signal(
                signal_type=SignalType.EXIT,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[{"symbol": instrument, "action": "SELL", "quantity": 1}],
                regime=regime,
            )
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
        )

    def get_exit_conditions(self, market_data: dict) -> Signal | None:
        return None

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return 1


def test_engine_runs_and_produces_equity_and_fills():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
        }
    )
    strategy = OneShotStrategy(
        name="oneshot",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value], "exit_ts": datetime(2026, 1, 3)},
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    )

    result = engine.run(candles=candles)

    assert len(result.equity_curve) == 5
    assert len(result.fills) >= 2
    assert "final_equity" in result.metrics
