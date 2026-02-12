"""Helpers to assemble RegimeSignals from feature inputs."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.regime.classifier import RegimeSignals
from src.signals import options_signals, trend


def build_regime_signals(
    *,
    timestamp: datetime,
    candles: pd.DataFrame,
    vix_value: float,
    chain_df: pd.DataFrame | None = None,
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
        adx_series = trend.adx(high=high, low=low, close=close, period=14)
        adx_14 = float(adx_series.dropna().iloc[-1]) if not adx_series.dropna().empty else 0.0

        dma50 = trend.sma(close, period=50)
        dma200 = trend.sma(close, period=200)
        latest_close = float(close.iloc[-1])
        above_50dma = bool(latest_close > float(dma50.dropna().iloc[-1])) if not dma50.dropna().empty else True
        above_200dma = bool(latest_close > float(dma200.dropna().iloc[-1])) if not dma200.dropna().empty else True

    pcr = options_signals.put_call_ratio(chain_df) if chain_df is not None else 0.0

    return RegimeSignals(
        timestamp=timestamp,
        india_vix=float(vix_value),
        adx_14=adx_14,
        pcr=float(pcr),
        fii_net_3d=float(fii_net_3d),
        nifty_above_50dma=above_50dma,
        nifty_above_200dma=above_200dma,
        nifty_banknifty_corr=float(nifty_banknifty_corr),
    )
