"""Historical regime replay utilities (strict no-lookahead processing)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.regime.classifier import RegimeClassifier
from src.signals.regime import build_regime_signals


@dataclass
class ReplayResult:
    snapshots: pd.DataFrame
    transitions: pd.DataFrame


def replay_regimes_no_lookahead(
    *,
    candles: pd.DataFrame,
    classifier: RegimeClassifier,
    vix_df: pd.DataFrame | None = None,
    fii_df: pd.DataFrame | None = None,
    chain_by_timestamp: dict[datetime, pd.DataFrame] | None = None,
    analysis_start: datetime | None = None,
) -> ReplayResult:
    """Replay regime labels in chronological order using only data available at each timestamp."""
    if candles.empty:
        return ReplayResult(snapshots=pd.DataFrame(), transitions=pd.DataFrame())

    candles_sorted = candles.copy()
    candles_sorted["timestamp"] = pd.to_datetime(candles_sorted["timestamp"], errors="coerce")
    candles_sorted = (
        candles_sorted.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    )

    vix_frame = _prep_vix(vix_df)
    fii_frame = _prep_fii(fii_df)

    snapshots: list[dict[str, Any]] = []
    prev_chain: pd.DataFrame | None = None

    for i in range(len(candles_sorted)):
        current_ts = pd.Timestamp(candles_sorted.iloc[i]["timestamp"]).to_pydatetime()
        candles_hist = candles_sorted.iloc[: i + 1]

        vix_hist = _slice_by_ts(vix_frame, current_ts, "timestamp")
        vix_series = vix_hist["close"].astype("float64") if not vix_hist.empty else None
        vix_value = (
            float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else 0.0
        )

        fii_hist = _slice_by_ts(fii_frame, current_ts, "date")
        fii_net_3d = float(fii_hist["fii_net"].tail(3).sum()) if not fii_hist.empty else 0.0

        chain_curr = chain_by_timestamp.get(current_ts) if chain_by_timestamp else None
        signals = build_regime_signals(
            timestamp=current_ts,
            candles=candles_hist,
            vix_value=vix_value,
            vix_series=vix_series,
            chain_df=chain_curr,
            previous_chain_df=prev_chain,
            fii_net_3d=fii_net_3d,
        )
        regime = classifier.classify(signals)
        snapshots.append(classifier.snapshot(signals=signals, regime=regime))

        if chain_curr is not None:
            prev_chain = chain_curr

    transitions = pd.DataFrame(classifier.history)
    snapshots_df = pd.DataFrame(snapshots)
    snapshots_df = _filter_from_analysis_start(
        snapshots_df,
        timestamp_col="timestamp",
        analysis_start=analysis_start,
    )
    transitions = _filter_from_analysis_start(
        transitions,
        timestamp_col="timestamp",
        analysis_start=analysis_start,
    )
    return ReplayResult(snapshots=snapshots_df, transitions=transitions)


def _prep_vix(vix_df: pd.DataFrame | None) -> pd.DataFrame:
    if vix_df is None or vix_df.empty:
        return pd.DataFrame(columns=["timestamp", "close"])
    out = vix_df.copy()
    if "timestamp" not in out.columns or "close" not in out.columns:
        return pd.DataFrame(columns=["timestamp", "close"])
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    return out


def _prep_fii(fii_df: pd.DataFrame | None) -> pd.DataFrame:
    if fii_df is None or fii_df.empty:
        return pd.DataFrame(columns=["date", "fii_net"])
    out = fii_df.copy()
    if "date" not in out.columns or "fii_net" not in out.columns:
        return pd.DataFrame(columns=["date", "fii_net"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["fii_net"] = pd.to_numeric(out["fii_net"], errors="coerce")
    out = out.dropna(subset=["date", "fii_net"]).sort_values("date").reset_index(drop=True)
    return out


def _slice_by_ts(df: pd.DataFrame, ts: datetime, timestamp_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp(ts)
    return df.loc[df[timestamp_col] <= cutoff].reset_index(drop=True)


def _filter_from_analysis_start(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    analysis_start: datetime | None,
) -> pd.DataFrame:
    if analysis_start is None or frame.empty or timestamp_col not in frame.columns:
        return frame
    cutoff = pd.Timestamp(analysis_start)
    ts = pd.to_datetime(frame[timestamp_col], errors="coerce")
    return frame.loc[ts >= cutoff].reset_index(drop=True)
