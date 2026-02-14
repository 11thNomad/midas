"""Unified feature pipeline that emits both regime signals and frozen DTO snapshots."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.regime.classifier import RegimeSignals, RegimeThresholds
from src.signals import mean_reversion, options_signals, trend, volatility, volume_flow
from src.signals.contracts import SignalSnapshotDTO, signal_snapshot_from_mapping
from src.signals.greeks import mibian_greeks


def build_feature_context(
    *,
    timestamp: datetime,
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    vix_value: float,
    vix_series: pd.Series | None = None,
    chain_df: pd.DataFrame | None = None,
    previous_chain_df: pd.DataFrame | None = None,
    fii_df: pd.DataFrame | None = None,
    usdinr_close: pd.Series | None = None,
    regime: str = "unknown",
    thresholds: RegimeThresholds | None = None,
    source: str = "signal_pipeline",
) -> tuple[SignalSnapshotDTO, RegimeSignals]:
    adx_14 = 0.0
    atr_14 = 0.0
    rsi_14 = 0.0
    boll_width = 0.0
    above_50dma = True
    above_200dma = True
    if not candles.empty:
        close = candles["close"].astype("float64")
        high = candles["high"].astype("float64")
        low = candles["low"].astype("float64")

        if len(candles) >= 28:
            adx_14 = _latest(trend.adx(high=high, low=low, close=close, period=14))
            atr_14 = _latest(volatility.atr(high=high, low=low, close=close, period=14))
            rsi_14 = _latest(mean_reversion.rsi(close=close, period=14))
            boll_width = _latest(
                mean_reversion.bollinger_band_width(close=close, period=20, std_dev=2.0)
            )

        dma50 = trend.sma(close, period=50)
        dma200 = trend.sma(close, period=200)
        latest_close = float(close.iloc[-1]) if not close.empty else 0.0
        if not dma50.dropna().empty:
            above_50dma = bool(latest_close > float(dma50.dropna().iloc[-1]))
        if not dma200.dropna().empty:
            above_200dma = bool(latest_close > float(dma200.dropna().iloc[-1]))

    pcr_oi = options_signals.put_call_ratio(chain_df) if chain_df is not None else 0.0
    oi_support, oi_resistance = (
        options_signals.oi_support_resistance(chain_df) if chain_df is not None else (0.0, 0.0)
    )

    vix_roc_5d = 0.0
    if vix_series is not None and not vix_series.empty:
        change = volatility.vix_change(vix_series.astype("float64"), periods=5).dropna()
        if not change.empty:
            vix_roc_5d = float(change.iloc[-1])

    fii_net_3d = 0.0
    fii_net_5d = 0.0
    if fii_df is not None and not fii_df.empty and "fii_net" in fii_df.columns:
        fii_series = pd.to_numeric(fii_df["fii_net"], errors="coerce").dropna()
        if not fii_series.empty:
            fii_net_3d = _latest(volume_flow.fii_net_rolling(fii_series, window=3))
            fii_net_5d = _latest(volume_flow.fii_net_rolling(fii_series, window=5))

    usdinr_roc_1d = 0.0
    usdinr_roc_3d = 0.0
    if usdinr_close is not None and not usdinr_close.empty:
        usd = pd.to_numeric(usdinr_close, errors="coerce").dropna()
        if not usd.empty:
            usdinr_roc_1d = _latest(usd.pct_change(periods=1) * 100.0)
            usdinr_roc_3d = _latest(usd.pct_change(periods=3) * 100.0)

    iv_surface_parallel_shift = 0.0
    iv_surface_tilt_change = 0.0
    if chain_df is not None and previous_chain_df is not None:
        iv_surface_parallel_shift = options_signals.iv_surface_parallel_shift(
            previous_chain_df, chain_df
        )
        iv_surface_tilt_change = options_signals.iv_surface_tilt_change(previous_chain_df, chain_df)

    greeks = _atm_greeks_from_chain(
        chain_df=chain_df,
        asof=timestamp,
        fallback_spot=_latest(pd.to_numeric(candles["close"], errors="coerce"))
        if not candles.empty
        else 0.0,
    )

    confidence = _regime_confidence(
        vix_level=float(vix_value),
        adx_14=float(adx_14),
        thresholds=thresholds,
    )
    snapshot = signal_snapshot_from_mapping(
        {
            "timestamp": timestamp,
            "symbol": symbol,
            "timeframe": timeframe,
            "vix_level": float(vix_value),
            "vix_roc_5d": float(vix_roc_5d),
            "adx_14": float(adx_14),
            "atr_14": float(atr_14),
            "pcr_oi": float(pcr_oi),
            "fii_net_3d": float(fii_net_3d),
            "fii_net_5d": float(fii_net_5d),
            "usdinr_roc_1d": float(usdinr_roc_1d),
            "usdinr_roc_3d": float(usdinr_roc_3d),
            "rsi_14": float(rsi_14),
            "bollinger_width_20_2": float(boll_width),
            "oi_support": float(oi_support),
            "oi_resistance": float(oi_resistance),
            "iv_surface_parallel_shift": float(iv_surface_parallel_shift),
            "iv_surface_tilt_change": float(iv_surface_tilt_change),
            "atm_call_delta": float(greeks["call_delta"]),
            "atm_put_delta": float(greeks["put_delta"]),
            "atm_gamma": float(greeks["gamma"]),
            "atm_theta": float(greeks["theta"]),
            "atm_vega": float(greeks["vega"]),
            "atm_rho": float(greeks["rho"]),
            "regime": regime,
            "regime_confidence": float(confidence),
            "source": source,
        }
    )

    regime_signals = RegimeSignals(
        timestamp=timestamp,
        india_vix=float(vix_value),
        vix_change_5d=float(vix_roc_5d),
        iv_surface_parallel_shift=float(iv_surface_parallel_shift),
        iv_surface_tilt_change=float(iv_surface_tilt_change),
        adx_14=float(adx_14),
        pcr=float(pcr_oi),
        fii_net_3d=float(fii_net_3d),
        nifty_above_50dma=above_50dma,
        nifty_above_200dma=above_200dma,
        nifty_banknifty_corr=0.95,
    )
    return snapshot, regime_signals


def _latest(series: pd.Series) -> float:
    out = pd.to_numeric(series, errors="coerce").dropna()
    if out.empty:
        return 0.0
    return float(out.iloc[-1])


def _regime_confidence(
    *,
    vix_level: float,
    adx_14: float,
    thresholds: RegimeThresholds | None,
) -> float:
    if thresholds is None:
        return 0.0

    vix_band = max(thresholds.vix_high - thresholds.vix_low, 1e-6)
    if vix_level <= thresholds.vix_low or vix_level >= thresholds.vix_high:
        vix_score = 1.0
    else:
        midpoint = (thresholds.vix_high + thresholds.vix_low) / 2.0
        vix_score = min(abs(vix_level - midpoint) / (vix_band / 2.0), 1.0)

    adx_margin = max(thresholds.adx_trending - thresholds.adx_ranging, 1e-6)
    if adx_14 >= thresholds.adx_trending or adx_14 <= thresholds.adx_ranging:
        adx_score = 1.0
    else:
        adx_mid = (thresholds.adx_trending + thresholds.adx_ranging) / 2.0
        adx_score = min(abs(adx_14 - adx_mid) / (adx_margin / 2.0), 1.0)

    return float(max(0.0, min((vix_score + adx_score) / 2.0, 1.0)))


def _atm_greeks_from_chain(
    *,
    chain_df: pd.DataFrame | None,
    asof: datetime,
    fallback_spot: float,
) -> dict[str, float]:
    empty = {
        "call_delta": 0.0,
        "put_delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
        "rho": 0.0,
    }
    if chain_df is None or chain_df.empty:
        return empty
    required = {"option_type", "strike"}
    if not required.issubset(chain_df.columns):
        return empty

    out = chain_df.copy()
    out["option_type"] = out["option_type"].astype(str).str.upper()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["iv"] = pd.to_numeric(out.get("iv", 0.0), errors="coerce")
    if "expiry" in out.columns:
        out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    out = out.dropna(subset=["strike"])
    if out.empty:
        return empty

    spot = float(
        pd.to_numeric(out.get("underlying_price", pd.Series(dtype="float64")), errors="coerce")
        .dropna()
        .tail(1)
        .mean()
    )
    if spot <= 0.0:
        spot = float(fallback_spot)
    if spot <= 0.0:
        return empty

    out["_dist"] = (out["strike"] - spot).abs()
    calls = out.loc[out["option_type"] == "CE"].sort_values("_dist")
    puts = out.loc[out["option_type"] == "PE"].sort_values("_dist")
    if calls.empty or puts.empty:
        return empty

    call_row = calls.iloc[0]
    put_row = puts.iloc[0]

    call_iv = float(call_row.get("iv", 0.0) or 0.0)
    put_iv = float(put_row.get("iv", 0.0) or 0.0)
    if 0 < call_iv <= 1.0:
        call_iv *= 100.0
    if 0 < put_iv <= 1.0:
        put_iv *= 100.0
    if call_iv <= 0.0:
        call_iv = 20.0
    if put_iv <= 0.0:
        put_iv = 20.0

    expiry_raw = call_row.get("expiry")
    expiry = pd.to_datetime(expiry_raw, errors="coerce") if expiry_raw is not None else pd.NaT
    days_to_expiry = 1
    if pd.notna(expiry):
        dte = int((expiry.normalize() - pd.Timestamp(asof).normalize()).days)
        days_to_expiry = max(1, dte)

    rate_pct = 8.0
    call_g = mibian_greeks(
        spot=spot,
        strike=float(call_row["strike"]),
        rate_pct=rate_pct,
        days_to_expiry=days_to_expiry,
        iv_pct=call_iv,
        option_type="CE",
    )
    put_g = mibian_greeks(
        spot=spot,
        strike=float(put_row["strike"]),
        rate_pct=rate_pct,
        days_to_expiry=days_to_expiry,
        iv_pct=put_iv,
        option_type="PE",
    )

    return {
        "call_delta": float(call_g["delta"]),
        "put_delta": float(put_g["delta"]),
        "gamma": float((call_g["gamma"] + put_g["gamma"]) / 2.0),
        "theta": float((call_g["theta"] + put_g["theta"]) / 2.0),
        "vega": float((call_g["vega"] + put_g["vega"]) / 2.0),
        "rho": float((call_g["rho"] + put_g["rho"]) / 2.0),
    }
