"""Composite multi-signal setups."""

from __future__ import annotations

import pandas as pd


def boring_alpha_setup(adx: pd.Series, vix: pd.Series, pcr: pd.Series) -> pd.Series:
    return ((adx < 20) & (vix < 14) & (pcr.between(0.8, 1.2))).astype(int)


def breakout_confirmation(
    close: pd.Series,
    volume_spike: pd.Series,
    adx: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    breakout = close > close.rolling(lookback).max().shift(1)
    return (breakout & (volume_spike > 1.5) & (adx > adx.shift(1))).astype(int)


def regime_transition_warning(vix_change: pd.Series, fii_net_3d: pd.Series) -> pd.Series:
    return ((vix_change > 3.0) & (fii_net_3d < 0)).astype(int)
