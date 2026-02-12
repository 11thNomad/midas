"""Volume and flow signals."""

from __future__ import annotations

import pandas as pd
from ta.volume import OnBalanceVolumeIndicator


def volume_spike(volume: pd.Series, lookback: int = 20) -> pd.Series:
    avg = volume.rolling(lookback).mean().replace(0, pd.NA)
    return volume / avg


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    return OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()


def fii_net_rolling(fii_net: pd.Series, window: int = 3) -> pd.Series:
    return fii_net.rolling(window).sum()


def fii_direction(fii_net: pd.Series, window: int = 3) -> pd.Series:
    return fii_net_rolling(fii_net, window=window).apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
