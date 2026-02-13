"""Helpers to assemble RegimeSignals from feature inputs."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.regime.classifier import RegimeSignals
from src.signals import options_signals, trend, volatility


def build_regime_signals(
    *,
    timestamp: datetime,
    candles: pd.DataFrame,
    vix_value: float,
    vix_series: pd.Series | None = None,
    chain_df: pd.DataFrame | None = None,
    previous_chain_df: pd.DataFrame | None = None,
    fii_net_3d: float = 0.0,
    nifty_banknifty_corr: float = 0.95,
) -> RegimeSignals:
    """Build RegimeSignals from latest candles/chain snapshot.

    TODO: Integrate dedicated signal cache to avoid recomputing full ADX on each tick.
    """
    if candles.empty:
        adx_14 = 0.0
        above_50dma = True
        above_200dma = True
    else:
        close = candles["close"]
        high = candles["high"]
        low = candles["low"]
        if len(candles) >= 28:
            adx_series = trend.adx(high=high, low=low, close=close, period=14)
            adx_14 = float(adx_series.dropna().iloc[-1]) if not adx_series.dropna().empty else 0.0
        else:
            adx_14 = 0.0

        dma50 = trend.sma(close, period=50)
        dma200 = trend.sma(close, period=200)
        latest_close = float(close.iloc[-1])
        above_50dma = (
            bool(latest_close > float(dma50.dropna().iloc[-1]))
            if not dma50.dropna().empty
            else True
        )
        above_200dma = (
            bool(latest_close > float(dma200.dropna().iloc[-1]))
            if not dma200.dropna().empty
            else True
        )

    pcr = options_signals.put_call_ratio(chain_df) if chain_df is not None else 0.0
    vix_change_5d = 0.0
    if vix_series is not None and not vix_series.empty:
        change = volatility.vix_change(vix_series.astype("float64"), periods=5).dropna()
        if not change.empty:
            vix_change_5d = float(change.iloc[-1])

    iv_surface_parallel_shift = 0.0
    iv_surface_tilt_change = 0.0
    if chain_df is not None and previous_chain_df is not None:
        iv_surface_parallel_shift = options_signals.iv_surface_parallel_shift(
            previous_chain_df, chain_df
        )
        iv_surface_tilt_change = options_signals.iv_surface_tilt_change(previous_chain_df, chain_df)

    return RegimeSignals(
        timestamp=timestamp,
        india_vix=float(vix_value),
        vix_change_5d=float(vix_change_5d),
        iv_surface_parallel_shift=float(iv_surface_parallel_shift),
        iv_surface_tilt_change=float(iv_surface_tilt_change),
        adx_14=adx_14,
        pcr=float(pcr),
        fii_net_3d=float(fii_net_3d),
        nifty_above_50dma=above_50dma,
        nifty_above_200dma=above_200dma,
        nifty_banknifty_corr=float(nifty_banknifty_corr),
    )
