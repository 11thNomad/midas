"""Typed contracts for mode-agnostic feature snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


class SignalContractError(ValueError):
    """Raised when a payload cannot be mapped to the signal snapshot contract."""


@dataclass(frozen=True)
class SignalSnapshotDTO:
    """Single normalized feature snapshot used across backtest/paper/live."""

    timestamp: datetime
    symbol: str
    timeframe: str
    schema_version: str = "1.0.0"
    vix_level: float = 0.0
    vix_roc_5d: float = 0.0
    adx_14: float = 0.0
    atr_14: float = 0.0
    pcr_oi: float = 0.0
    fii_net_3d: float = 0.0
    fii_net_5d: float = 0.0
    usdinr_roc_1d: float = 0.0
    usdinr_roc_3d: float = 0.0
    rsi_14: float = 0.0
    bollinger_width_20_2: float = 0.0
    oi_support: float = 0.0
    oi_resistance: float = 0.0
    iv_surface_parallel_shift: float = 0.0
    iv_surface_tilt_change: float = 0.0
    atm_call_delta: float = 0.0
    atm_put_delta: float = 0.0
    atm_gamma: float = 0.0
    atm_theta: float = 0.0
    atm_vega: float = 0.0
    atm_rho: float = 0.0
    regime: str = "unknown"
    regime_confidence: float = 0.0
    source: str = "engine"


def signal_snapshot_from_mapping(payload: Mapping[str, Any]) -> SignalSnapshotDTO:
    """Parse and validate a raw mapping into a typed snapshot DTO."""
    missing = {"timestamp", "symbol", "timeframe"} - payload.keys()
    if missing:
        raise SignalContractError(f"Signal snapshot missing keys: {sorted(missing)}")

    timestamp = _to_datetime(payload["timestamp"])
    symbol = str(payload["symbol"]).strip()
    timeframe = str(payload["timeframe"]).strip()
    if not symbol:
        raise SignalContractError("Signal snapshot symbol cannot be empty")
    if not timeframe:
        raise SignalContractError("Signal snapshot timeframe cannot be empty")

    confidence = _to_float(payload.get("regime_confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise SignalContractError("regime_confidence must be between 0.0 and 1.0")

    return SignalSnapshotDTO(
        schema_version=str(payload.get("schema_version", "1.0.0")),
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        vix_level=_to_float(payload.get("vix_level", 0.0)),
        vix_roc_5d=_to_float(payload.get("vix_roc_5d", 0.0)),
        adx_14=_to_float(payload.get("adx_14", 0.0)),
        atr_14=_to_float(payload.get("atr_14", 0.0)),
        pcr_oi=_to_float(payload.get("pcr_oi", 0.0)),
        fii_net_3d=_to_float(payload.get("fii_net_3d", 0.0)),
        fii_net_5d=_to_float(payload.get("fii_net_5d", 0.0)),
        usdinr_roc_1d=_to_float(payload.get("usdinr_roc_1d", 0.0)),
        usdinr_roc_3d=_to_float(payload.get("usdinr_roc_3d", 0.0)),
        rsi_14=_to_float(payload.get("rsi_14", 0.0)),
        bollinger_width_20_2=_to_float(payload.get("bollinger_width_20_2", 0.0)),
        oi_support=_to_float(payload.get("oi_support", 0.0)),
        oi_resistance=_to_float(payload.get("oi_resistance", 0.0)),
        iv_surface_parallel_shift=_to_float(payload.get("iv_surface_parallel_shift", 0.0)),
        iv_surface_tilt_change=_to_float(payload.get("iv_surface_tilt_change", 0.0)),
        atm_call_delta=_to_float(payload.get("atm_call_delta", 0.0)),
        atm_put_delta=_to_float(payload.get("atm_put_delta", 0.0)),
        atm_gamma=_to_float(payload.get("atm_gamma", 0.0)),
        atm_theta=_to_float(payload.get("atm_theta", 0.0)),
        atm_vega=_to_float(payload.get("atm_vega", 0.0)),
        atm_rho=_to_float(payload.get("atm_rho", 0.0)),
        regime=str(payload.get("regime", "unknown")),
        regime_confidence=confidence,
        source=str(payload.get("source", "engine")),
    )


def signal_snapshots_from_frame(df: pd.DataFrame) -> list[SignalSnapshotDTO]:
    """Map a DataFrame to validated signal snapshot DTOs."""
    if df.empty:
        return []

    out: list[SignalSnapshotDTO] = []
    for row in df.to_dict(orient="records"):
        payload = {str(key): value for key, value in row.items()}
        out.append(signal_snapshot_from_mapping(payload))
    return out


def frame_from_signal_snapshots(dtos: list[SignalSnapshotDTO]) -> pd.DataFrame:
    """Map DTO snapshots to a stable DataFrame schema."""
    columns = [
        "schema_version",
        "timestamp",
        "symbol",
        "timeframe",
        "vix_level",
        "vix_roc_5d",
        "adx_14",
        "atr_14",
        "pcr_oi",
        "fii_net_3d",
        "fii_net_5d",
        "usdinr_roc_1d",
        "usdinr_roc_3d",
        "rsi_14",
        "bollinger_width_20_2",
        "oi_support",
        "oi_resistance",
        "iv_surface_parallel_shift",
        "iv_surface_tilt_change",
        "atm_call_delta",
        "atm_put_delta",
        "atm_gamma",
        "atm_theta",
        "atm_vega",
        "atm_rho",
        "regime",
        "regime_confidence",
        "source",
    ]
    if not dtos:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(
            [
                {
                    "schema_version": dto.schema_version,
                    "timestamp": dto.timestamp,
                    "symbol": dto.symbol,
                    "timeframe": dto.timeframe,
                    "vix_level": dto.vix_level,
                    "vix_roc_5d": dto.vix_roc_5d,
                    "adx_14": dto.adx_14,
                    "atr_14": dto.atr_14,
                    "pcr_oi": dto.pcr_oi,
                    "fii_net_3d": dto.fii_net_3d,
                    "fii_net_5d": dto.fii_net_5d,
                    "usdinr_roc_1d": dto.usdinr_roc_1d,
                    "usdinr_roc_3d": dto.usdinr_roc_3d,
                    "rsi_14": dto.rsi_14,
                    "bollinger_width_20_2": dto.bollinger_width_20_2,
                    "oi_support": dto.oi_support,
                    "oi_resistance": dto.oi_resistance,
                    "iv_surface_parallel_shift": dto.iv_surface_parallel_shift,
                    "iv_surface_tilt_change": dto.iv_surface_tilt_change,
                    "atm_call_delta": dto.atm_call_delta,
                    "atm_put_delta": dto.atm_put_delta,
                    "atm_gamma": dto.atm_gamma,
                    "atm_theta": dto.atm_theta,
                    "atm_vega": dto.atm_vega,
                    "atm_rho": dto.atm_rho,
                    "regime": dto.regime,
                    "regime_confidence": dto.regime_confidence,
                    "source": dto.source,
                }
                for dto in dtos
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _to_datetime(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise SignalContractError(f"Invalid timestamp: {value!r}")
    return ts.to_pydatetime()


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
