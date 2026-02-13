from __future__ import annotations

from datetime import datetime

from src.execution.paper_executor import PaperExecutionEngine
from src.strategies.base import RegimeState, Signal, SignalType


def test_paper_executor_executes_actionable_signal_and_persists(tmp_path):
    engine = PaperExecutionEngine(base_dir=str(tmp_path / "cache"), slippage_bps=5.0, commission_per_order=20.0)
    signals = [
        Signal(
            signal_type=SignalType.ENTRY_LONG,
            strategy_name="dummy",
            instrument="NIFTY",
            timestamp=datetime(2026, 1, 2, 9, 15),
            orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 2, "price": 100.0}],
            regime=RegimeState.LOW_VOL_TRENDING,
        )
    ]

    fills = engine.execute_signals(signals, market_data={"symbol": "NIFTY", "vix": 12.0})
    assert len(fills) == 1
    assert fills[0]["strategy_name"] == "dummy"
    assert fills[0]["quantity"] == 2
    assert fills[0]["price"] > 100.0

    persisted = engine.read_fills(symbol="NIFTY")
    assert len(persisted) == 1
    assert persisted.loc[0, "instrument"] == "NIFTY"


def test_paper_executor_skips_no_signal(tmp_path):
    engine = PaperExecutionEngine(base_dir=str(tmp_path / "cache"))
    signals = [
        Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name="dummy",
            instrument="NIFTY",
            timestamp=datetime(2026, 1, 2, 9, 15),
            regime=RegimeState.UNKNOWN,
        )
    ]
    fills = engine.execute_signals(signals, market_data={"symbol": "NIFTY"})
    assert fills == []


def test_paper_executor_infers_buy_to_cover_for_short_exit(tmp_path):
    engine = PaperExecutionEngine(base_dir=str(tmp_path / "cache"), slippage_bps=0.0, commission_per_order=0.0)
    entry = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        regime=RegimeState.LOW_VOL_RANGING,
    )
    exit_signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 30),
        regime=RegimeState.LOW_VOL_RANGING,
    )
    entry_fills = engine.execute_signals([entry], market_data={"symbol": "NIFTY", "last_price": 100.0})
    exit_fills = engine.execute_signals([exit_signal], market_data={"symbol": "NIFTY", "last_price": 95.0})
    assert entry_fills[0]["side"] == "SELL"
    assert exit_fills[0]["side"] == "BUY"


def test_paper_executor_infers_exit_quantity_from_open_position(tmp_path):
    engine = PaperExecutionEngine(base_dir=str(tmp_path / "cache"), slippage_bps=0.0, commission_per_order=0.0)
    entry = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 2, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    exit_signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 45),
        regime=RegimeState.LOW_VOL_RANGING,
    )
    engine.execute_signals([entry], market_data={"symbol": "NIFTY", "last_price": 100.0})
    exit_fills = engine.execute_signals([exit_signal], market_data={"symbol": "NIFTY", "last_price": 105.0})
    assert exit_fills[0]["side"] == "SELL"
    assert exit_fills[0]["quantity"] == 2
