from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.strategies.base import RegimeState, SignalType
from src.strategies.baseline_trend import BaselineTrendStrategy


def _candles(n: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": [100 + i for i in range(n)],
            "high": [101 + i for i in range(n)],
            "low": [99 + i for i in range(n)],
            "close": [100 + i for i in range(n)],
        }
    )


def test_baseline_trend_enters_when_gate_open():
    strategy = BaselineTrendStrategy(
        name="baseline_trend",
        config={
            "instrument": "NIFTY",
            "active_regimes": ["low_vol_trending", "high_vol_trending"],
            "adx_min": 5.0,
            "max_lots": 1,
        },
    )
    signal = strategy.generate_signal(
        market_data={"timestamp": datetime(2026, 3, 1), "candles": _candles(), "vix": 14.0},
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    assert signal.signal_type == SignalType.ENTRY_LONG


def test_baseline_trend_exits_when_gate_closes():
    strategy = BaselineTrendStrategy(
        name="baseline_trend",
        config={
            "instrument": "NIFTY",
            "active_regimes": ["low_vol_trending", "high_vol_trending"],
            "adx_min": 5.0,
            "max_lots": 1,
        },
    )
    strategy.state.current_position = {"side": "LONG", "quantity": 1}
    signal = strategy.generate_signal(
        market_data={"timestamp": datetime(2026, 3, 2), "candles": _candles(), "vix": 14.0},
        regime=RegimeState.HIGH_VOL_CHOPPY,
    )
    assert signal.signal_type == SignalType.EXIT


def test_baseline_trend_respects_vix_cap():
    strategy = BaselineTrendStrategy(
        name="baseline_trend",
        config={
            "instrument": "NIFTY",
            "active_regimes": ["low_vol_trending", "high_vol_trending"],
            "adx_min": 5.0,
            "vix_max": 16.0,
            "max_lots": 1,
        },
    )
    signal = strategy.generate_signal(
        market_data={"timestamp": datetime(2026, 3, 1), "candles": _candles(), "vix": 19.0},
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    assert signal.signal_type == SignalType.NO_SIGNAL

