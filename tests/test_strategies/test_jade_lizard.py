from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.strategies.base import RegimeState, SignalType
from src.strategies.jade_lizard import JadeLizardStrategy


def _sample_chain() -> pd.DataFrame:
    return pd.DataFrame(
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
            "ltp": [120.0, 100.0, 85.0, 70.0, 95.0, 82.0, 66.0, 54.0],
            "expiry": [pd.Timestamp("2026-01-08")] * 8,
        }
    )


def test_jade_lizard_bullish_entry_has_three_legs():
    strategy = JadeLizardStrategy(
        name="jade_lizard",
        config={"instrument": "NIFTY", "max_lots": 1, "variant": "bullish", "spread_width": 100},
    )
    signal = strategy.generate_signal(
        market_data={
            "timestamp": datetime(2026, 1, 1, 9, 15),
            "option_chain": _sample_chain(),
            "underlying_price": 22000.0,
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.ENTRY_SHORT
    assert len(signal.orders) == 3
    assert signal.orders[0]["action"] == "SELL"
    assert signal.orders[1]["action"] == "SELL"
    assert signal.orders[2]["action"] == "BUY"
    assert signal.orders[2]["symbol"].endswith("CE_22300")


def test_jade_lizard_bearish_entry_uses_put_hedge():
    strategy = JadeLizardStrategy(
        name="jade_lizard",
        config={"instrument": "NIFTY", "max_lots": 1, "variant": "bearish", "spread_width": 100},
    )
    signal = strategy.generate_signal(
        market_data={
            "timestamp": datetime(2026, 1, 1, 9, 15),
            "option_chain": _sample_chain(),
            "underlying_price": 22000.0,
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.ENTRY_SHORT
    assert len(signal.orders) == 3
    assert signal.orders[2]["symbol"].endswith("PE_21700")


def test_jade_lizard_neutral_variant_uses_bias():
    strategy = JadeLizardStrategy(
        name="jade_lizard",
        config={"instrument": "NIFTY", "max_lots": 1, "variant": "neutral", "spread_width": 100},
    )
    signal = strategy.generate_signal(
        market_data={
            "timestamp": datetime(2026, 1, 1, 9, 15),
            "option_chain": _sample_chain(),
            "underlying_price": 22000.0,
            "bias": -0.5,
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.ENTRY_SHORT
    assert signal.reason.startswith("Jade lizard entry (bearish)")


def test_jade_lizard_exits_on_unfavorable_regime_change():
    strategy = JadeLizardStrategy(name="jade_lizard", config={"instrument": "NIFTY", "max_lots": 1})
    strategy.state.current_position = {"quantity": 1, "legs": [{"symbol": "A", "action": "SELL"}]}
    signal = strategy.on_regime_change(RegimeState.LOW_VOL_RANGING, RegimeState.HIGH_VOL_CHOPPY)
    assert signal is not None
    assert signal.signal_type == SignalType.EXIT


def test_jade_lizard_exit_conditions_profit_target():
    strategy = JadeLizardStrategy(
        name="jade_lizard",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "profit_target_pct": 50,
            "stop_loss_pct": 100,
        },
    )
    entry_chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 09:15:00")] * 3,
            "symbol": ["PUT_SHORT", "CALL_SHORT", "CALL_HEDGE"],
            "option_type": ["PE", "CE", "CE"],
            "strike": [21800, 22200, 22300],
            "delta": [-0.15, 0.15, 0.10],
            "ltp": [100.0, 100.0, 20.0],
        }
    )
    strategy.generate_signal(
        market_data={
            "timestamp": datetime(2026, 1, 1, 9, 15),
            "option_chain": entry_chain,
            "underlying_price": 22000.0,
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )

    exit_chain = entry_chain.copy()
    exit_chain["ltp"] = [50.0, 50.0, 20.0]
    exit_signal = strategy.get_exit_conditions(
        {"timestamp": datetime(2026, 1, 1, 12, 0), "option_chain": exit_chain}
    )
    assert exit_signal is not None
    assert exit_signal.signal_type == SignalType.EXIT
