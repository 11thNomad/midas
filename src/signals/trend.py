"""Trend signals."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def ema_crossover(series: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    signal = pd.Series(np.where(fast_ema > slow_ema, 1, -1), index=series.index)
    return signal


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    indicator = ADXIndicator(high=high, low=low, close=close, window=period)
    return cast(pd.Series, indicator.adx())


def dma_position(close: pd.Series, period: int) -> pd.Series:
    ma = sma(close, period=period)
    return (close > ma).astype(int)


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Return supertrend line and direction (+1/-1)."""
    atr = AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
    hl2 = (high + low) / 2.0

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = pd.Series(1, index=close.index, dtype="int64")

    for i in range(1, len(close)):
        if close.iloc[i - 1] <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i - 1])
        else:
            final_upper.iloc[i] = upperband.iloc[i]

        if close.iloc[i - 1] >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i - 1])
        else:
            final_lower.iloc[i] = lowerband.iloc[i]

        if close.iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    line = pd.Series(np.where(direction > 0, final_lower, final_upper), index=close.index)
    return pd.DataFrame({"supertrend": line, "direction": direction}, index=close.index)
