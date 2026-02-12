from __future__ import annotations

from datetime import datetime

from src.backtest.simulator import FillSimulator
from src.strategies.base import RegimeState, Signal, SignalType


def test_simulator_applies_slippage_and_commission():
    sim = FillSimulator(slippage_pct=0.1, commission_per_order=10.0)
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
