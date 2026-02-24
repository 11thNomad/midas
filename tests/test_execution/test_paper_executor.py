from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd
import pytest

from src.execution.paper_executor import PaperExecutionEngine
from src.strategies.base import RegimeState, Signal, SignalType


def test_paper_executor_executes_actionable_signal_and_persists(tmp_path):
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=5.0,
        commission_per_order=20.0,
    )
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
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
    )
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
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
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
    entry_fills = engine.execute_signals(
        [entry], market_data={"symbol": "NIFTY", "last_price": 100.0}
    )
    exit_fills = engine.execute_signals(
        [exit_signal], market_data={"symbol": "NIFTY", "last_price": 95.0}
    )
    assert entry_fills[0]["side"] == "SELL"
    assert exit_fills[0]["side"] == "BUY"


def test_paper_executor_infers_exit_quantity_from_open_position(tmp_path):
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
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
    exit_fills = engine.execute_signals(
        [exit_signal], market_data={"symbol": "NIFTY", "last_price": 105.0}
    )
    assert exit_fills[0]["side"] == "SELL"
    assert exit_fills[0]["quantity"] == 2


def test_paper_executor_aborts_entry_short_on_insufficient_margin(tmp_path):
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
        paper_capital=100.0,
        margin_buffer_pct=15.0,
    )
    entry = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 75, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    fills = engine.execute_signals([entry], market_data={"symbol": "NIFTY", "last_price": 100.0})
    assert fills == []


def test_paper_executor_uses_resolver_before_paper_capital(tmp_path):
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
        paper_capital=500_000.0,
        margin_buffer_pct=15.0,
        available_cash_resolver=lambda: 100.0,
    )
    entry = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 75, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    fills = engine.execute_signals([entry], market_data={"symbol": "NIFTY", "last_price": 100.0})
    assert fills == []


def test_paper_executor_applies_slippage_multiplier_to_buy_and_sell(tmp_path):
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=5.0,  # 0.05%
        slippage_multiplier=1.5,  # effective 0.075%
        commission_per_order=0.0,
        paper_capital=200_000.0,
    )
    buy_signal = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    sell_signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 16),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 1, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )

    buy_fills = engine.execute_signals(
        [buy_signal],
        market_data={"symbol": "NIFTY", "last_price": 100.0},
    )
    sell_fills = engine.execute_signals(
        [sell_signal],
        market_data={"symbol": "NIFTY", "last_price": 100.0, "close_price": 100.0},
    )

    assert buy_fills[0]["price"] == pytest.approx(100.075, rel=1e-9)
    assert sell_fills[0]["price"] == pytest.approx(99.925, rel=1e-9)


def test_paper_executor_writes_daily_fill_and_summary_csv(tmp_path):
    log_dir = tmp_path / "paper"
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(log_dir),
        slippage_bps=5.0,
        slippage_multiplier=1.5,
        commission_per_order=10.0,
        paper_capital=200_000.0,
    )
    signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[
            {
                "symbol": "NIFTY_20260108_22000CE",
                "action": "SELL",
                "quantity": 75,
                "price": 100.0,
                "strike": 22000.0,
                "expiry": "2026-01-08",
                "option_type": "CE",
            }
        ],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    engine.execute_signals(
        [signal],
        market_data={
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "symbol": "NIFTY",
            "close_price": 22000.0,
            "vix": 14.0,
            "adx": 22.0,
        },
    )

    fills_path = log_dir / "fills_20260102.csv"
    summary_path = log_dir / "daily_summary_20260102.csv"
    assert fills_path.exists()
    assert summary_path.exists()

    fills = pd.read_csv(fills_path)
    summary = pd.read_csv(summary_path)
    assert set(
        [
            "timestamp",
            "trade_id",
            "signal_type",
            "instrument",
            "leg",
            "action",
            "strike",
            "expiry",
            "quantity",
            "fill_price",
            "mid_price",
            "slippage_applied",
            "fees_estimated",
            "regime",
            "vix",
            "adx",
        ]
    ).issubset(set(fills.columns))
    assert fills.loc[0, "leg"] == "call_short"
    assert fills.loc[0, "action"] == "SELL"
    assert float(fills.loc[0, "slippage_applied"]) < 0.0

    assert set(
        [
            "date",
            "entries",
            "exits",
            "open_positions",
            "gross_pnl",
            "fees",
            "net_pnl",
            "realized_balance",
            "mtm_balance",
            "unrealized_pnl",
            "cash_balance",
            "margin_utilisation_pct",
        ]
    ).issubset(set(summary.columns))
    assert int(summary.loc[0, "entries"]) == 1


def test_paper_executor_daily_counters_reset_on_day_change(tmp_path):
    log_dir = tmp_path / "paper"
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
        paper_capital=200_000.0,
    )
    signal_day1 = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[
            {
                "symbol": "NIFTY_20260108_22000CE",
                "action": "SELL",
                "quantity": 75,
                "price": 100.0,
            }
        ],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    engine.execute_signals(
        [signal_day1],
        market_data={
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "symbol": "NIFTY",
            "close_price": 22000.0,
        },
    )

    signal_day2 = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 3, 9, 15),
        orders=[
            {
                "symbol": "NIFTY_20260115_22000CE",
                "action": "SELL",
                "quantity": 75,
                "price": 100.0,
            }
        ],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    engine.execute_signals(
        [signal_day2],
        market_data={
            "timestamp": datetime(2026, 1, 3, 9, 15),
            "symbol": "NIFTY",
            "close_price": 22000.0,
        },
    )

    day1_summary = pd.read_csv(log_dir / "daily_summary_20260102.csv")
    day2_summary = pd.read_csv(log_dir / "daily_summary_20260103.csv")
    assert int(day1_summary.loc[0, "entries"]) == 1
    assert int(day2_summary.loc[0, "entries"]) == 1


def test_paper_executor_fill_logging_handles_same_timestamp_signals(tmp_path):
    log_dir = tmp_path / "paper"
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
        paper_capital=200_000.0,
    )
    ts = datetime(2026, 1, 2, 9, 15)
    entry_signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=ts,
        orders=[
            {"symbol": "NIFTY_20260108_22000CE", "action": "SELL", "quantity": 75, "price": 100.0},
            {"symbol": "NIFTY_20260108_21900PE", "action": "SELL", "quantity": 75, "price": 100.0},
            {"symbol": "NIFTY_20260108_22100CE", "action": "BUY", "quantity": 75, "price": 10.0},
            {"symbol": "NIFTY_20260108_21800PE", "action": "BUY", "quantity": 75, "price": 10.0},
        ],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    exit_signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=ts,
        orders=[
            {"symbol": "NIFTY_20260108_22000CE", "action": "BUY", "quantity": 75, "price": 101.0},
            {"symbol": "NIFTY_20260108_21900PE", "action": "BUY", "quantity": 75, "price": 101.0},
        ],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    engine.execute_signals(
        [entry_signal, exit_signal],
        market_data={"timestamp": ts, "symbol": "NIFTY", "close_price": 22000.0},
    )

    fills = pd.read_csv(log_dir / "fills_20260102.csv")
    assert len(fills) == 6
    assert set(fills["signal_type"].str.lower()) == {"entry_short", "exit"}


def test_paper_executor_recovers_positions_and_sequence_from_sqlite(tmp_path):
    log_dir = tmp_path / "paper"
    cache_dir = tmp_path / "cache"
    engine = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    entry_signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 2, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    entry_fills = engine.execute_signals(
        [entry_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )
    assert entry_fills[0]["fill_id"] == "PAPER-00000001"

    restarted = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    exit_signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 30),
        regime=RegimeState.LOW_VOL_RANGING,
    )
    exit_fills = restarted.execute_signals(
        [exit_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 30),
            "last_price": 95.0,
        },
    )
    assert exit_fills[0]["fill_id"] == "PAPER-00000002"
    assert exit_fills[0]["side"] == "BUY"
    assert exit_fills[0]["quantity"] == 2


def test_paper_executor_rebuilds_from_fill_ledger_when_snapshot_missing(tmp_path):
    log_dir = tmp_path / "paper"
    cache_dir = tmp_path / "cache"
    engine = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    entry_signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 2, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    engine.execute_signals(
        [entry_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )

    db_path = log_dir / "paper_state.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM paper_state_snapshot WHERE id = 1")

    restarted = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    exit_signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 30),
        regime=RegimeState.LOW_VOL_RANGING,
    )
    exit_fills = restarted.execute_signals(
        [exit_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 30),
            "last_price": 95.0,
        },
    )
    assert exit_fills[0]["fill_id"] == "PAPER-00000002"
    assert exit_fills[0]["side"] == "BUY"
    assert exit_fills[0]["quantity"] == 2


def test_paper_executor_rebuilds_multi_day_daily_counters_correctly(tmp_path):
    log_dir = tmp_path / "paper"
    cache_dir = tmp_path / "cache"
    engine = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=1.0,
        paper_capital=1_000.0,
    )

    day1_entry = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    day1_exit = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 15, 20),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 1, "price": 110.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    day2_entry = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 3, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1, "price": 120.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    day2_exit = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 3, 15, 20),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 1, "price": 130.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )

    engine.execute_signals(
        [day1_entry],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )
    engine.execute_signals(
        [day1_exit],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 15, 20),
            "last_price": 110.0,
        },
    )
    engine.execute_signals(
        [day2_entry],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 3, 9, 15),
            "last_price": 120.0,
        },
    )
    engine.execute_signals(
        [day2_exit],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 3, 15, 20),
            "last_price": 130.0,
        },
    )

    db_path = log_dir / "paper_state.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM paper_state_snapshot WHERE id = 1")

    restarted = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=1.0,
        paper_capital=1_000.0,
    )

    assert restarted._summary_day_key == "20260103"
    assert restarted._entries_today == 1
    assert restarted._exits_today == 1
    assert restarted._gross_realized_pnl_today == pytest.approx(10.0, rel=1e-9)
    assert restarted._fees_paid_today == pytest.approx(2.0, rel=1e-9)
    assert restarted._realized_pnl_today == pytest.approx(8.0, rel=1e-9)
    available_cash, source = restarted._resolve_available_cash()
    assert source == "paper_balance"
    assert available_cash == pytest.approx(1_016.0, rel=1e-9)


def test_paper_executor_tracks_mtm_balance_and_persists_across_restart(tmp_path):
    log_dir = tmp_path / "paper"
    cache_dir = tmp_path / "cache"
    engine = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
        initial_capital=1_000.0,
        paper_capital=1_000.0,
    )
    entry = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    engine.execute_signals(
        [entry],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )

    engine.execute_signals(
        [],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 10, 0),
            "last_price": 130.0,
            "instrument_prices": {"NIFTY": 130.0},
        },
    )
    assert engine._current_balance == pytest.approx(1_000.0, rel=1e-9)
    assert engine._unrealized_pnl_current == pytest.approx(30.0, rel=1e-9)
    assert engine._mtm_balance == pytest.approx(1_030.0, rel=1e-9)

    restarted = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
        initial_capital=1_000.0,
        paper_capital=1_000.0,
    )
    assert restarted._current_balance == pytest.approx(1_000.0, rel=1e-9)
    assert restarted._unrealized_pnl_current == pytest.approx(30.0, rel=1e-9)
    assert restarted._mtm_balance == pytest.approx(1_030.0, rel=1e-9)


def test_paper_executor_uses_single_starting_base_for_all_balances(tmp_path):
    log_dir = tmp_path / "paper"
    cache_dir = tmp_path / "cache"
    engine = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
        initial_capital=150_000.0,
        paper_capital=1_000.0,
    )

    assert engine.paper_capital == pytest.approx(1_000.0, rel=1e-9)
    assert engine._cash == pytest.approx(1_000.0, rel=1e-9)
    assert engine._current_balance == pytest.approx(1_000.0, rel=1e-9)
    assert engine._mtm_balance == pytest.approx(1_000.0, rel=1e-9)

    engine.execute_signals(
        [],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )

    restarted = PaperExecutionEngine(
        base_dir=str(cache_dir),
        paper_log_dir=str(log_dir),
        slippage_bps=0.0,
        commission_per_order=0.0,
        initial_capital=150_000.0,
        paper_capital=1_000.0,
    )
    assert restarted._cash == pytest.approx(1_000.0, rel=1e-9)
    assert restarted._current_balance == pytest.approx(1_000.0, rel=1e-9)
    assert restarted._mtm_balance == pytest.approx(1_000.0, rel=1e-9)


def test_paper_executor_exit_bypasses_circuit_breaker_block(tmp_path):
    class _BreakerAlwaysBlocked:
        @staticmethod
        def can_trade() -> bool:
            return False

        @staticmethod
        def update(**_kwargs) -> None:
            return None

    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
        circuit_breaker=_BreakerAlwaysBlocked(),
    )
    engine._positions["NIFTY"] = -2
    engine._avg_cost_by_instrument["NIFTY"] = 100.0
    engine._last_price_by_instrument["NIFTY"] = 100.0

    exit_signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 10, 0),
        regime=RegimeState.LOW_VOL_RANGING,
    )
    fills = engine.execute_signals(
        [exit_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 10, 0),
            "last_price": 95.0,
        },
    )
    assert len(fills) == 1
    assert fills[0]["side"] == "BUY"
    assert fills[0]["quantity"] == 2


def test_paper_executor_option_exit_uses_side_aware_quote(tmp_path):
    instrument = "NIFTY_20260108_22000CE"
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    entry_signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[
            {
                "symbol": instrument,
                "action": "SELL",
                "quantity": 50,
                "price": 100.0,
                "expiry": "2026-01-08",
                "strike": 22000.0,
                "option_type": "CE",
            }
        ],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    engine.execute_signals(
        [entry_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )

    exit_signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 10, 0),
        orders=[{"symbol": instrument, "action": "BUY", "quantity": 50, "option_type": "CE"}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    fills = engine.execute_signals(
        [exit_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 10, 0),
            "close_price": 22_000.0,
            "option_quotes": {instrument: {"bid": 90.0, "ask": 111.0, "ltp": 95.0}},
        },
    )
    assert len(fills) == 1
    assert fills[0]["side"] == "BUY"
    assert fills[0]["price"] == pytest.approx(111.0, rel=1e-9)


def test_paper_executor_enforces_option_lot_size_guard(tmp_path):
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[
            {
                "symbol": "NIFTY_20260108_22000CE",
                "action": "SELL",
                "quantity": 75,
                "price": 100.0,
                "option_type": "CE",
            }
        ],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    fills = engine.execute_signals(
        [signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
            "lot_size_by_underlying": {"NIFTY": 50},
        },
    )
    assert fills == []


def test_paper_executor_settles_expired_option_positions(tmp_path):
    instrument = "NIFTY_20260108_22000CE"
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    entry_signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="iron_condor",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[
            {
                "symbol": instrument,
                "action": "SELL",
                "quantity": 50,
                "price": 100.0,
                "expiry": "2026-01-08",
                "strike": 22000.0,
                "option_type": "CE",
            }
        ],
        regime=RegimeState.LOW_VOL_RANGING,
        indicators={"call_wing": 100.0},
    )
    engine.execute_signals(
        [entry_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )

    fills = engine.execute_signals(
        [],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 9, 15, 25),
            "close_price": 22_150.0,
            "underlying_prices": {"NIFTY": 22_150.0},
        },
    )
    settlement_fills = [fill for fill in fills if fill.get("strategy_name") == "expiry_settlement"]
    assert len(settlement_fills) == 1
    settlement = settlement_fills[0]
    assert settlement["side"] == "BUY"
    assert settlement["price"] == pytest.approx(150.0, rel=1e-9)
    assert settlement["reason"] == "expiry_settlement_itm"
    assert engine._positions[instrument] == 0


def test_paper_executor_watchdog_force_exit_closes_open_positions(tmp_path):
    engine = PaperExecutionEngine(
        base_dir=str(tmp_path / "cache"),
        paper_log_dir=str(tmp_path / "paper"),
        slippage_bps=0.0,
        commission_per_order=0.0,
    )
    entry_signal = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 2, "price": 100.0}],
        regime=RegimeState.LOW_VOL_RANGING,
    )
    engine.execute_signals(
        [entry_signal],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 2, 9, 15),
            "last_price": 100.0,
        },
    )

    fills = engine.execute_signals(
        [],
        market_data={
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 1, 7, 15, 25),
            "close_price": 95.0,
            "force_exit_all": True,
            "force_exit_reason": "watchdog_wed_1525_force_exit",
        },
    )
    watchdog_fills = [fill for fill in fills if fill.get("strategy_name") == "watchdog"]
    assert len(watchdog_fills) == 1
    assert watchdog_fills[0]["side"] == "SELL"
    assert watchdog_fills[0]["reason"] == "watchdog_wed_1525_force_exit"
    assert engine._positions["NIFTY"] == 0
