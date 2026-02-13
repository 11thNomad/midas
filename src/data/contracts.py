"""Normalized DTO contracts for provider-agnostic data mapping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class NormalizedCandleDTO:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    oi: float = 0.0
    source: str = "unknown"
    symbol: str = ""
    timeframe: str = ""


@dataclass(frozen=True)
class NormalizedOptionRowDTO:
    timestamp: datetime
    underlying: str
    expiry: datetime
    strike: float
    option_type: str  # CE or PE
    ltp: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    oi: float = 0.0
    change_in_oi: float = 0.0
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    source: str = "unknown"


@dataclass(frozen=True)
class NormalizedFiiDiiDTO:
    date: datetime
    fii_buy: float = 0.0
    fii_sell: float = 0.0
    fii_net: float = 0.0
    dii_buy: float = 0.0
    dii_sell: float = 0.0
    dii_net: float = 0.0
    source: str = "unknown"


class DTOValidationError(ValueError):
    """Raised when provider payload cannot be mapped to a normalized DTO contract."""


def candle_dtos_from_frame(
    df: pd.DataFrame,
    *,
    source: str,
    symbol: str,
    timeframe: str,
) -> list[NormalizedCandleDTO]:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise DTOValidationError(f"Candle frame missing columns: {sorted(missing)}")

    out: list[NormalizedCandleDTO] = []
    for row in df.to_dict(orient="records"):
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        out.append(
            NormalizedCandleDTO(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0) or 0.0),
                oi=float(row.get("oi", 0.0) or 0.0),
                source=source,
                symbol=symbol,
                timeframe=timeframe,
            )
        )
    return out


def frame_from_candle_dtos(dtos: list[NormalizedCandleDTO]) -> pd.DataFrame:
    if not dtos:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])

    return (
        pd.DataFrame(
            [
                {
                    "timestamp": dto.timestamp,
                    "open": dto.open,
                    "high": dto.high,
                    "low": dto.low,
                    "close": dto.close,
                    "volume": dto.volume,
                    "oi": dto.oi,
                }
                for dto in dtos
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def option_dtos_from_chain(
    chain,
    *,
    source: str,
) -> list[NormalizedOptionRowDTO]:
    out: list[NormalizedOptionRowDTO] = []
    for contract in chain.contracts:
        out.append(
            NormalizedOptionRowDTO(
                timestamp=chain.timestamp,
                underlying=chain.underlying,
                expiry=contract.expiry,
                strike=float(contract.strike),
                option_type=str(contract.instrument_type.value),
                ltp=float(contract.ltp),
                bid=float(contract.bid),
                ask=float(contract.ask),
                volume=float(contract.volume),
                oi=float(contract.oi),
                change_in_oi=float(contract.change_in_oi),
                iv=float(contract.greeks.iv),
                delta=float(contract.greeks.delta),
                gamma=float(contract.greeks.gamma),
                theta=float(contract.greeks.theta),
                vega=float(contract.greeks.vega),
                rho=float(contract.greeks.rho),
                source=source,
            )
        )
    return out


def fii_dtos_from_frame(df: pd.DataFrame, *, source: str) -> list[NormalizedFiiDiiDTO]:
    required = {"date"}
    missing = required.difference(df.columns)
    if missing:
        raise DTOValidationError(f"FII frame missing columns: {sorted(missing)}")

    out: list[NormalizedFiiDiiDTO] = []
    for row in df.to_dict(orient="records"):
        out.append(
            NormalizedFiiDiiDTO(
                date=pd.Timestamp(row["date"]).to_pydatetime(),
                fii_buy=float(row.get("fii_buy", 0.0) or 0.0),
                fii_sell=float(row.get("fii_sell", 0.0) or 0.0),
                fii_net=float(row.get("fii_net", 0.0) or 0.0),
                dii_buy=float(row.get("dii_buy", 0.0) or 0.0),
                dii_sell=float(row.get("dii_sell", 0.0) or 0.0),
                dii_net=float(row.get("dii_net", 0.0) or 0.0),
                source=source,
            )
        )
    return out


def frame_from_fii_dtos(dtos: list[NormalizedFiiDiiDTO]) -> pd.DataFrame:
    if not dtos:
        return pd.DataFrame(
            columns=["date", "fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]
        )

    return (
        pd.DataFrame(
            [
                {
                    "date": dto.date,
                    "fii_buy": dto.fii_buy,
                    "fii_sell": dto.fii_sell,
                    "fii_net": dto.fii_net,
                    "dii_buy": dto.dii_buy,
                    "dii_sell": dto.dii_sell,
                    "dii_net": dto.dii_net,
                }
                for dto in dtos
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
