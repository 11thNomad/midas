from __future__ import annotations

from datetime import datetime

import pandas as pd

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


def test_iron_condor_uses_chain_delta_targets_for_leg_selection():
    strategy = IronCondorStrategy(
        name="iron_condor",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "call_delta": 0.15,
            "put_delta": -0.15,
            "wing_width": 100,
        },
    )
    chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 09:15:00")] * 8,
            "symbol": [
                "NIFTY_CE_22100",
                "NIFTY_CE_22200",
                "NIFTY_CE_22300",
                "NIFTY_CE_22400",
                "NIFTY_PE_21900",
                "NIFTY_PE_21800",
                "NIFTY_PE_21700",
                "NIFTY_PE_21600",
            ],
            "option_type": ["CE", "CE", "CE", "CE", "PE", "PE", "PE", "PE"],
            "strike": [22100, 22200, 22300, 22400, 21900, 21800, 21700, 21600],
            "delta": [0.22, 0.15, 0.11, 0.08, -0.20, -0.15, -0.10, -0.06],
        }
    )
    signal = strategy.generate_signal(
        market_data={
            "timestamp": datetime(2026, 1, 1, 9, 15),
            "option_chain": chain,
            "underlying_price": 22000.0,
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.ENTRY_SHORT
    symbols = [o["symbol"] for o in signal.orders]
    assert "NIFTY_CE_22200" in symbols
    assert "NIFTY_PE_21800" in symbols
