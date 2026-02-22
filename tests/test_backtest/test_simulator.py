from __future__ import annotations

from datetime import datetime

from src.backtest.simulator import FillSimulator
from src.strategies.base import RegimeState, Signal, SignalType


def test_simulator_applies_slippage_and_commission():
    sim = FillSimulator(
        slippage_pct=0.1,
        commission_per_order=10.0,
        stt_pct=0.0,
        exchange_txn_charges_pct=0.0,
        gst_pct=0.0,
        sebi_fee_pct=0.0,
        stamp_duty_pct=0.0,
    )
    signal = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="x",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1, "price": 100.0}],
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    fills = sim.simulate(signal, close_price=100.0, timestamp=signal.timestamp)
    assert len(fills) == 1
    assert fills[0]["price"] > 100.0
    assert fills[0]["fees"] == 10.0


def test_simulator_models_full_fee_stack():
    sim = FillSimulator(
        slippage_pct=0.0,
        commission_per_order=20.0,
        stt_pct=0.0125,
        exchange_txn_charges_pct=0.053,
        gst_pct=18.0,
        sebi_fee_pct=0.0001,
        stamp_duty_pct=0.003,
    )
    signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="x",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "SELL", "quantity": 1, "price": 100.0}],
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    fills = sim.simulate(signal, close_price=100.0, timestamp=signal.timestamp)
    assert len(fills) == 1
    assert fills[0]["fees"] > 20.0
    assert fills[0]["stt"] > 0.0


def test_simulator_drops_option_fills_without_option_price_lookup():
    sim = FillSimulator(slippage_pct=0.0, commission_per_order=0.0)
    signal = Signal(
        signal_type=SignalType.ENTRY_SHORT,
        strategy_name="x",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 15),
        orders=[{"symbol": "NIFTY_20260115_22000CE", "action": "SELL", "quantity": 1}],
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    fills = sim.simulate(signal, close_price=100.0, timestamp=signal.timestamp, price_lookup={})
    assert fills == []


def test_simulator_resolves_option_price_via_canonical_key():
    sim = FillSimulator(slippage_pct=0.0, commission_per_order=0.0)
    signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="x",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 15),
        orders=[{"symbol": "NIFTY_20260115_22000CE", "action": "BUY", "quantity": 1}],
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    fills = sim.simulate(
        signal,
        close_price=100.0,
        timestamp=signal.timestamp,
        price_lookup={"OPT::20260115_22000_CE": 42.0},
    )
    assert len(fills) == 1
    assert fills[0]["price"] == 42.0


def test_simulator_resolves_option_price_via_compact_symbol_alias():
    sim = FillSimulator(slippage_pct=0.0, commission_per_order=0.0)
    signal = Signal(
        signal_type=SignalType.EXIT,
        strategy_name="x",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 15),
        orders=[{"symbol": "NIFTY_20260115_22000CE", "action": "BUY", "quantity": 1}],
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    fills = sim.simulate(
        signal,
        close_price=100.0,
        timestamp=signal.timestamp,
        price_lookup={"NIFTY26011522000CE": 43.0},
    )
    assert len(fills) == 1
    assert fills[0]["price"] == 43.0
