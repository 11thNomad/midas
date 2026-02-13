"""Mean reversion signals."""

from __future__ import annotations

from typing import cast

import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return cast(pd.Series, RSIIndicator(close=close, window=period).rsi())


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    bb = BollingerBands(close=close, window=period, window_dev=std_dev)
    return pd.DataFrame(
        {
            "middle": bb.bollinger_mavg(),
            "upper": bb.bollinger_hband(),
            "lower": bb.bollinger_lband(),
        },
        index=close.index,
    )


def bollinger_percent_b(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    bb = BollingerBands(close=close, window=period, window_dev=std_dev)
    return cast(pd.Series, bb.bollinger_pband())


def zscore(series: pd.Series, lookback: int = 20) -> pd.Series:
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0, pd.NA)


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3.0
    cum_pv = (typical * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, pd.NA)
    return cum_pv / cum_vol


def vwap_deviation_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    atr_period: int = 14,
) -> pd.Series:
    vw = vwap(high, low, close, volume)
    atr = AverageTrueRange(high=high, low=low, close=close, window=atr_period).average_true_range()
    return cast(pd.Series, (close - vw) / atr.replace(0, pd.NA))
