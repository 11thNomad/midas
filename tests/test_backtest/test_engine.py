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


class ChainCaptureStrategy(BaseStrategy):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.captured_lengths: list[int] = []

    def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
        chain = market_data.get("option_chain")
        self.captured_lengths.append(0 if chain is None else len(chain))
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=self.config.get("instrument", "NIFTY"),
            timestamp=market_data["timestamp"],
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
        config={
            "instrument": "NIFTY",
            "active_regimes": [RegimeState.LOW_VOL_RANGING.value],
            "exit_ts": datetime(2026, 1, 3),
        },
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


def test_engine_passes_latest_option_chain_snapshot_to_strategy():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
        }
    )
    chain = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-03 00:00:00",
                    "2026-01-03 00:00:00",
                ]
            ),
            "option_type": ["CE", "PE", "CE", "PE"],
            "strike": [22000, 22000, 22100, 21900],
        }
    )
    strategy = ChainCaptureStrategy(
        name="chain_capture",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    )
    engine.run(candles=candles, option_chain_df=chain)
    assert strategy.captured_lengths[0] == 0
    assert strategy.captured_lengths[-1] == 2


def test_engine_avoids_lookahead_in_candle_history():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="D"),
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 103],
        }
    )

    class HistoryLengthStrategy(BaseStrategy):
        def __init__(self, name: str, config: dict):
            super().__init__(name, config)
            self.lengths: list[int] = []

        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            self.lengths.append(len(market_data["candles"]))
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                strategy_name=self.name,
                instrument=self.config.get("instrument", "NIFTY"),
                timestamp=market_data["timestamp"],
                regime=regime,
            )

        def get_exit_conditions(self, market_data: dict) -> Signal | None:
            return None

        def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
            return 1

    strategy = HistoryLengthStrategy(
        name="history_len",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    )
    engine.run(candles=candles)
    assert strategy.lengths[0] == 0
    assert strategy.lengths[1] == 1


def test_engine_passes_realized_and_unrealized_pnl_to_breaker():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
        }
    )
    strategy = OneShotStrategy(
        name="oneshot",
        config={
            "instrument": "NIFTY",
            "active_regimes": [RegimeState.LOW_VOL_RANGING.value],
            "exit_ts": datetime(2026, 1, 3),
        },
    )

    class CaptureBreaker:
        def __init__(self):
            self.updates: list[dict] = []

        def can_trade(self) -> bool:
            return True

        def update(self, **kwargs):
            self.updates.append(kwargs)

    breaker = CaptureBreaker()
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(
            slippage_pct=0.0,
            commission_per_order=0.0,
            stt_pct=0.0,
            exchange_txn_charges_pct=0.0,
            gst_pct=0.0,
            sebi_fee_pct=0.0,
            stamp_duty_pct=0.0,
        ),
        initial_capital=1000.0,
        circuit_breaker=breaker,  # type: ignore[arg-type]
    )
    engine.run(candles=candles)

    assert breaker.updates
    final = breaker.updates[-1]
    assert final["realized_pnl_today"] == 2.0
    assert final["unrealized_pnl"] == 0.0


def test_engine_infers_buy_for_exit_without_orders_when_short():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "open": [100, 100, 100],
            "high": [100, 100, 100],
            "low": [100, 100, 100],
            "close": [100, 100, 100],
        }
    )

    class ShortThenExitStrategy(BaseStrategy):
        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            ts = market_data["timestamp"]
            if self.state.current_position is None:
                self.state.current_position = {"qty": -1}
                return Signal(
                    signal_type=SignalType.ENTRY_SHORT,
                    strategy_name=self.name,
                    instrument="NIFTY",
                    timestamp=ts,
                    regime=regime,
                )
            if ts == datetime(2026, 1, 3):
                self.state.current_position = None
                return Signal(
                    signal_type=SignalType.EXIT,
                    strategy_name=self.name,
                    instrument="NIFTY",
                    timestamp=ts,
                    regime=regime,
                )
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                strategy_name=self.name,
                instrument="NIFTY",
                timestamp=ts,
                regime=regime,
            )

        def get_exit_conditions(self, market_data: dict) -> Signal | None:
            return None

        def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
            return 1

    strategy = ShortThenExitStrategy(
        name="short_then_exit",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    )
    result = engine.run(candles=candles)
    sides = result.fills["side"].tolist()
    assert sides == ["SELL", "BUY"]


def test_engine_analysis_start_skips_pre_window_trading_and_outputs():
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
        config={
            "instrument": "NIFTY",
            "active_regimes": [RegimeState.LOW_VOL_RANGING.value],
            "exit_ts": datetime(2026, 1, 5),
        },
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    )

    result = engine.run(candles=candles, analysis_start=datetime(2026, 1, 3))

    assert len(result.equity_curve) == 3
    assert len(result.regimes) == 3
    fill_ts = pd.to_datetime(result.fills["timestamp"], errors="coerce")
    assert fill_ts.min() >= pd.Timestamp("2026-01-03")


def test_engine_analysis_start_keeps_warmup_candle_history_context():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="D"),
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 103],
        }
    )

    class HistoryLengthStrategy(BaseStrategy):
        def __init__(self, name: str, config: dict):
            super().__init__(name, config)
            self.lengths: list[int] = []

        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            self.lengths.append(len(market_data["candles"]))
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                strategy_name=self.name,
                instrument=self.config.get("instrument", "NIFTY"),
                timestamp=market_data["timestamp"],
                regime=regime,
            )

        def get_exit_conditions(self, market_data: dict) -> Signal | None:
            return None

        def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
            return 1

    strategy = HistoryLengthStrategy(
        name="history_len_warm",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    )
    engine.run(candles=candles, analysis_start=datetime(2026, 1, 3))
    assert strategy.lengths == [2, 3]


def test_engine_allows_exit_when_circuit_breaker_blocks_new_trades():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=1, freq="D"),
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
        }
    )

    class ExitOnlyStrategy(BaseStrategy):
        def __init__(self, name: str, config: dict):
            super().__init__(name, config)
            self.state.current_position = {"symbol": "NIFTY", "quantity": 1}

        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                strategy_name=self.name,
                instrument="NIFTY",
                timestamp=market_data["timestamp"],
                regime=regime,
            )

        def get_exit_conditions(self, market_data: dict) -> Signal | None:
            return Signal(
                signal_type=SignalType.EXIT,
                strategy_name=self.name,
                instrument="NIFTY",
                timestamp=market_data["timestamp"],
                orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 1}],
                regime=RegimeState.LOW_VOL_RANGING,
            )

        def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
            return 1

    class HaltedBreaker:
        def can_trade(self) -> bool:
            return False

        def update(self, **kwargs):  # noqa: ANN003 - simple test double
            return None

    strategy = ExitOnlyStrategy(
        name="exit_only",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
        circuit_breaker=HaltedBreaker(),  # type: ignore[arg-type]
    )

    result = engine.run(candles=candles)
    assert len(result.fills) == 1
    assert result.fills.iloc[0]["side"] == "SELL"
