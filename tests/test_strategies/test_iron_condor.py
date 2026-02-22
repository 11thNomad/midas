from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.strategies.base import RegimeState, SignalType
from src.strategies.iron_condor import IronCondorStrategy


def _sample_chain(ts: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(ts)] * 8,
            "symbol": [
                "NIFTY_20260115_22100CE",
                "NIFTY_20260115_22200CE",
                "NIFTY_20260115_22300CE",
                "NIFTY_20260115_22400CE",
                "NIFTY_20260115_21900PE",
                "NIFTY_20260115_21800PE",
                "NIFTY_20260115_21700PE",
                "NIFTY_20260115_21600PE",
            ],
            "option_type": ["CE", "CE", "CE", "CE", "PE", "PE", "PE", "PE"],
            "strike": [22100, 22200, 22300, 22400, 21900, 21800, 21700, 21600],
            "ltp": [160, 120, 90, 65, 150, 110, 80, 55],
            "delta": [0.0] * 8,  # upstream placeholder values must be ignored.
            "expiry": [pd.Timestamp("2026-01-15")] * 8,
        }
    )


def test_iron_condor_enters_with_valid_chain_and_gates():
    strategy = IronCondorStrategy(
        name="iron_condor",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "lot_size": 50,
            "entry_days": [0],  # Monday
            "min_entry_vix": 9.0,
            "max_entry_vix": 30.0,
            "wing_width": 100,
            "dte_min": 5,
            "dte_max": 20,
            "min_premium": 1.0,
            "enable_time_exit": False,
        },
    )
    ts = datetime(2026, 1, 5, 9, 15)  # Monday
    signal = strategy.generate_signal(
        market_data={
            "timestamp": ts,
            "option_chain": _sample_chain(ts),
            "underlying_price": 22000.0,
            "vix": 14.0,
            "candles": pd.DataFrame(),
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.ENTRY_SHORT
    assert len(signal.orders) == 4
    assert strategy.state.current_position is not None


def test_iron_condor_rejects_invalid_weekday():
    strategy = IronCondorStrategy(
        name="iron_condor",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "lot_size": 50,
            "entry_days": [0],  # Monday only
            "min_entry_vix": 9.0,
            "max_entry_vix": 30.0,
            "wing_width": 100,
            "dte_min": 5,
            "dte_max": 20,
            "min_premium": 1.0,
            "enable_time_exit": False,
        },
    )
    ts = datetime(2026, 1, 6, 9, 15)  # Tuesday
    signal = strategy.generate_signal(
        market_data={
            "timestamp": ts,
            "option_chain": _sample_chain(ts),
            "underlying_price": 22000.0,
            "vix": 14.0,
            "candles": pd.DataFrame(),
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.NO_SIGNAL
    assert "weekday" in signal.reason.lower()


def test_iron_condor_ignores_upstream_delta_placeholders():
    strategy = IronCondorStrategy(
        name="iron_condor",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "lot_size": 50,
            "entry_days": [0],  # Monday
            "min_entry_vix": 9.0,
            "max_entry_vix": 30.0,
            "wing_width": 100,
            "dte_min": 5,
            "dte_max": 20,
            "min_premium": 1.0,
            "enable_time_exit": False,
        },
    )
    ts = datetime(2026, 1, 5, 9, 15)
    signal = strategy.generate_signal(
        market_data={
            "timestamp": ts,
            "option_chain": _sample_chain(ts),
            "underlying_price": 22000.0,
            "vix": 14.0,
            "candles": pd.DataFrame(),
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.ENTRY_SHORT
    details = signal.indicators
    assert details["call_short_strike"] > details["spot"]
    assert details["put_short_strike"] < details["spot"]
    assert details["call_wing"] > 0
    assert details["put_wing"] > 0


def test_iron_condor_exit_conditions_profit_target():
    strategy = IronCondorStrategy(
        name="iron_condor",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "lot_size": 1,
            "entry_days": [0],  # Monday
            "min_entry_vix": 9.0,
            "max_entry_vix": 30.0,
            "profit_target_pct": 50,
            "stop_loss_pct": 100,
            "wing_width": 100,
            "min_premium": 0.0,
            "enable_time_exit": False,
        },
    )
    ts = datetime(2026, 1, 5, 9, 15)
    entry_chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(ts)] * 4,
            "symbol": ["CE_SHORT", "PE_SHORT", "CE_HEDGE", "PE_HEDGE"],
            "option_type": ["CE", "PE", "CE", "PE"],
            "strike": [22200, 21800, 22300, 21700],
            "ltp": [100.0, 100.0, 20.0, 20.0],
            "expiry": [pd.Timestamp("2026-01-15")] * 4,
        }
    )
    strategy.state.current_position = {
        "structure": "iron_condor",
        "quantity": 1,
        "entry_time": ts,
        "legs": [
            {"symbol": "CE_SHORT", "action": "SELL", "quantity": 1, "price": 100.0},
            {"symbol": "PE_SHORT", "action": "SELL", "quantity": 1, "price": 100.0},
            {"symbol": "CE_HEDGE", "action": "BUY", "quantity": 1, "price": 20.0},
            {"symbol": "PE_HEDGE", "action": "BUY", "quantity": 1, "price": 20.0},
        ],
        "entry_credit": 160.0,
    }

    # close_debit=80 satisfies 50% target on 160 credit.
    exit_chain = entry_chain.copy()
    exit_chain["ltp"] = [60.0, 60.0, 20.0, 20.0]
    exit_signal = strategy.get_exit_conditions(
        {"timestamp": datetime(2026, 1, 5, 12, 0), "option_chain": exit_chain}
    )
    assert exit_signal is not None
    assert exit_signal.signal_type == SignalType.EXIT


def test_iron_condor_exit_conditions_resolve_compact_chain_symbols():
    strategy = IronCondorStrategy(
        name="iron_condor",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "lot_size": 1,
            "entry_days": [0],
            "min_entry_vix": 9.0,
            "max_entry_vix": 30.0,
            "profit_target_pct": 50,
            "stop_loss_pct": 100,
            "wing_width": 100,
            "min_premium": 0.0,
            "enable_time_exit": False,
        },
    )
    ts = datetime(2024, 7, 8, 9, 15)
    strategy.state.current_position = {
        "structure": "iron_condor",
        "quantity": 1,
        "entry_time": ts,
        "entry_regime": RegimeState.HIGH_VOL_CHOPPY.value,
        "legs": [
            {
                "symbol": "NIFTY_20240718_24700CE",
                "action": "SELL",
                "quantity": 1,
                "price": 56.0,
                "expiry": pd.Timestamp("2024-07-18"),
                "strike": 24700.0,
                "option_type": "CE",
            },
            {
                "symbol": "NIFTY_20240718_23950PE",
                "action": "SELL",
                "quantity": 1,
                "price": 54.0,
                "expiry": pd.Timestamp("2024-07-18"),
                "strike": 23950.0,
                "option_type": "PE",
            },
            {
                "symbol": "NIFTY_20240718_24800CE",
                "action": "BUY",
                "quantity": 1,
                "price": 37.0,
                "expiry": pd.Timestamp("2024-07-18"),
                "strike": 24800.0,
                "option_type": "CE",
            },
            {
                "symbol": "NIFTY_20240718_23850PE",
                "action": "BUY",
                "quantity": 1,
                "price": 39.0,
                "expiry": pd.Timestamp("2024-07-18"),
                "strike": 23850.0,
                "option_type": "PE",
            },
        ],
        "entry_credit": 34.0,
    }

    compact_chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-07-09 18:30:00")] * 4,
            "symbol": [
                "NIFTY2471824700CE",
                "NIFTY2471823950PE",
                "NIFTY2471824800CE",
                "NIFTY2471823850PE",
            ],
            "option_type": ["CE", "PE", "CE", "PE"],
            "strike": [24700, 23950, 24800, 23850],
            "ltp": [10.0, 10.0, 5.0, 5.0],
            "expiry": [pd.Timestamp("2024-07-18")] * 4,
        }
    )

    exit_signal = strategy.get_exit_conditions(
        {"timestamp": datetime(2024, 7, 10, 9, 15), "option_chain": compact_chain}
    )

    assert exit_signal is not None
    assert exit_signal.signal_type == SignalType.EXIT
    assert "Profit target hit" in exit_signal.reason


def test_iron_condor_skips_entry_when_no_expiry_in_dte_window():
    strategy = IronCondorStrategy(
        name="iron_condor",
        config={
            "instrument": "NIFTY",
            "max_lots": 1,
            "entry_days": [0],  # Monday
            "min_entry_vix": 9.0,
            "max_entry_vix": 30.0,
            "dte_min": 5,
            "dte_max": 14,
            "enable_time_exit": False,
        },
    )
    ts = datetime(2026, 1, 5, 9, 15)
    chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(ts)] * 2,
            "symbol": ["NIFTY_CE_22200", "NIFTY_PE_21800"],
            "option_type": ["CE", "PE"],
            "strike": [22200, 21800],
            "ltp": [100, 100],
            "expiry": [pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-07")],
        }
    )
    signal = strategy.generate_signal(
        market_data={
            "timestamp": ts,
            "option_chain": chain,
            "underlying_price": 22000.0,
            "vix": 14.0,
            "candles": pd.DataFrame(),
        },
        regime=RegimeState.LOW_VOL_RANGING,
    )
    assert signal.signal_type == SignalType.NO_SIGNAL
    assert "DTE bounds" in signal.reason
