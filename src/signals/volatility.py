"""Volatility signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    a = atr(high, low, close, period=period)
    return (a / close.replace(0, pd.NA)) * 100.0


def historical_volatility(close: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=window).std(ddof=0) * np.sqrt(annualization)


def vix_change(vix: pd.Series, periods: int = 5) -> pd.Series:
    return vix.diff(periods=periods)


def iv_rank(iv: pd.Series, lookback: int = 252) -> pd.Series:
    rolling_low = iv.rolling(lookback).min()
    rolling_high = iv.rolling(lookback).max()
    return ((iv - rolling_low) / (rolling_high - rolling_low).replace(0, pd.NA)) * 100.0


def iv_skew(put_iv: pd.Series, call_iv: pd.Series) -> pd.Series:
    return put_iv - call_iv
