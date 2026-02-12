from __future__ import annotations

from datetime import datetime

from src.strategies.base import RegimeState, SignalType
from src.strategies.iron_condor import IronCondorStrategy


def test_iron_condor_enters_when_no_position():
    strategy = IronCondorStrategy(name="iron_condor", config={"instrument": "NIFTY", "max_lots": 1})
    signal = strategy.generate_signal(
        market_data={"timestamp": datetime(2026, 1, 1, 9, 15)},
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.ENTRY_SHORT
    assert len(signal.orders) == 4


def test_iron_condor_exits_on_unfavorable_regime_change():
    strategy = IronCondorStrategy(name="iron_condor", config={"instrument": "NIFTY", "max_lots": 1})
    strategy.state.current_position = {"quantity": 1}
    signal = strategy.on_regime_change(RegimeState.LOW_VOL_RANGING, RegimeState.HIGH_VOL_CHOPPY)
    assert signal is not None
    assert signal.signal_type == SignalType.EXIT
