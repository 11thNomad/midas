"""Regime-oriented filters and banding helpers."""

from __future__ import annotations

import pandas as pd


def vix_regime_band(vix: pd.Series, low: float = 14.0, high: float = 18.0) -> pd.Series:
    def band(value: float) -> str:
        if value < low:
            return "low"
        if value >= high:
            return "high"
        return "transition"

    return vix.apply(band)


def adx_regime_band(
    adx_values: pd.Series, ranging: float = 20.0, trending: float = 25.0
) -> pd.Series:
    def band(value: float) -> str:
        if value < ranging:
            return "ranging"
        if value >= trending:
            return "trending"
        return "transition"

    return adx_values.apply(band)


def rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int = 20) -> pd.Series:
    return series_a.rolling(window).corr(series_b)


def composite_regime_score(
    vix_z: pd.Series,
    adx_z: pd.Series,
    fii_z: pd.Series | None = None,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> pd.Series:
    if fii_z is None:
        return (weights[0] * vix_z) + (weights[1] * adx_z)
    return (weights[0] * vix_z) + (weights[1] * adx_z) + (weights[2] * fii_z)
