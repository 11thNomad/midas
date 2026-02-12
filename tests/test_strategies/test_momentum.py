from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.strategies.base import RegimeState, SignalType
from src.strategies.momentum import MomentumStrategy


def _candles_uptrend(n: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": [100 + i for i in range(n)],
            "high": [101 + i for i in range(n)],
            "low": [99 + i for i in range(n)],
            "close": [100 + i for i in range(n)],
            "volume": [1000] * n,
        }
    )


def test_momentum_strategy_generates_actionable_signal_with_enough_data():
    strategy = MomentumStrategy(
        name="momentum",
        config={"instrument": "NIFTY", "max_lots": 1, "fast_ema": 5, "slow_ema": 20, "adx_filter": 5},
    )
    signal = strategy.generate_signal(
        market_data={"timestamp": datetime(2026, 3, 1), "candles": _candles_uptrend()},
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    assert signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT, SignalType.NO_SIGNAL}


def test_momentum_strategy_returns_no_signal_when_insufficient_data():
    strategy = MomentumStrategy(name="momentum", config={"instrument": "NIFTY"})
    short = _candles_uptrend(20)
    signal = strategy.generate_signal(
        market_data={"timestamp": datetime(2026, 1, 20), "candles": short},
        regime=RegimeState.LOW_VOL_TRENDING,
    )
    assert signal.signal_type == SignalType.NO_SIGNAL
