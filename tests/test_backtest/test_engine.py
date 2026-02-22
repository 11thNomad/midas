from __future__ import annotations

from datetime import datetime

import pandas as pd
from pandas.testing import assert_frame_equal

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
    assert len(result.signal_snapshots) == 5
    assert set(result.signal_snapshots["symbol"]) == {"NIFTY"}
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


def test_engine_does_not_generate_new_entries_when_circuit_breaker_halts():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="D"),
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
        }
    )

    class EntryMutatingStrategy(BaseStrategy):
        def __init__(self, name: str, config: dict):
            super().__init__(name, config)
            self.generate_calls = 0

        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            self.generate_calls += 1
            self.state.current_position = {"symbol": "NIFTY", "quantity": 1}
            return Signal(
                signal_type=SignalType.ENTRY_LONG,
                strategy_name=self.name,
                instrument="NIFTY",
                timestamp=market_data["timestamp"],
                orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1}],
                regime=regime,
            )

        def get_exit_conditions(self, market_data: dict) -> Signal | None:
            return None

        def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
            return 1

    class HaltedBreaker:
        def can_trade(self) -> bool:
            return False

        def update(self, **kwargs):  # noqa: ANN003 - simple test double
            return None

    strategy = EntryMutatingStrategy(
        name="entry_mutating",
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
    assert len(result.fills) == 0
    assert strategy.generate_calls == 0
    assert strategy.state.current_position is None


def test_precomputed_context_matches_dynamic_run_outputs():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="D"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
        }
    )
    chain = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-02 00:00:00",
                    "2026-01-02 00:00:00",
                    "2026-01-04 00:00:00",
                    "2026-01-04 00:00:00",
                ]
            ),
            "option_type": ["CE", "PE", "CE", "PE", "CE", "PE"],
            "strike": [22000, 22000, 22100, 21900, 22200, 21800],
            "symbol": [
                "NIFTY_20260108_22000CE",
                "NIFTY_20260108_22000PE",
                "NIFTY_20260108_22100CE",
                "NIFTY_20260108_21900PE",
                "NIFTY_20260108_22200CE",
                "NIFTY_20260108_21800PE",
            ],
            "ltp": [110.0, 100.0, 95.0, 90.0, 80.0, 70.0],
            "underlying_price": [22050.0, 22050.0, 22100.0, 22100.0, 22200.0, 22200.0],
            "expiry": pd.to_datetime(["2026-01-08"] * 6),
            "iv": [12.0, 13.0, 11.5, 12.5, 11.0, 12.0],
        }
    )
    precomputed = BacktestEngine.prepare_precomputed_data(
        candles=candles,
        thresholds=RegimeThresholds(),
        symbol="NIFTY",
        timeframe="1d",
        option_chain_df=chain,
    )

    strategy_dynamic = OneShotStrategy(
        name="oneshot",
        config={
            "instrument": "NIFTY",
            "active_regimes": [RegimeState.LOW_VOL_RANGING.value],
            "exit_ts": datetime(2026, 1, 4),
        },
    )
    dynamic = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy_dynamic,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    ).run(candles=candles, option_chain_df=chain)

    strategy_precomputed = OneShotStrategy(
        name="oneshot",
        config={
            "instrument": "NIFTY",
            "active_regimes": [RegimeState.LOW_VOL_RANGING.value],
            "exit_ts": datetime(2026, 1, 4),
        },
    )
    cached = BacktestEngine(
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        strategy=strategy_precomputed,
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
        initial_capital=1000.0,
    ).run(candles=candles, option_chain_df=chain, precomputed_data=precomputed)

    assert_frame_equal(dynamic.equity_curve, cached.equity_curve, check_dtype=False)
    assert_frame_equal(dynamic.fills, cached.fills, check_dtype=False)
    assert_frame_equal(dynamic.regimes, cached.regimes, check_dtype=False)
    assert_frame_equal(dynamic.signal_snapshots, cached.signal_snapshots, check_dtype=False)
    assert dynamic.metrics == cached.metrics


def test_engine_forces_liquidation_at_window_end_and_reports_integrity():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
        }
    )

    class EnterAndHoldStrategy(BaseStrategy):
        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            ts = market_data["timestamp"]
            if self.state.current_position is None:
                self.state.current_position = {"symbol": "NIFTY", "quantity": 1}
                return Signal(
                    signal_type=SignalType.ENTRY_LONG,
                    strategy_name=self.name,
                    instrument="NIFTY",
                    timestamp=ts,
                    orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1}],
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

    strategy = EnterAndHoldStrategy(
        name="enter_hold",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
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
        number_of_symbols_in_run=1,
    )

    result = engine.run(candles=candles)
    assert "run_integrity" in result.metrics
    integrity = result.metrics["run_integrity"]
    assert isinstance(integrity, dict)
    forced = integrity["forced_liquidations"]
    assert forced["count"] == 1
    assert forced["flag"] is False
    assert forced["threshold"] == 1
    assert forced["symbols"] == ["NIFTY"]

    # Entry + forced exit
    assert len(result.fills) == 2
    assert result.fills.iloc[-1]["reason"] == "backtest_window_end"

    assert "early_exit_opportunity" in result.decisions.columns
    assert "earliest_exit_day" in result.decisions.columns
    assert "actual_exit_pnl" in result.decisions.columns
