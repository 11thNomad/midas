"""Normalized DTO contracts for provider-agnostic data mapping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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


# TODO: wire these DTOs into feed normalization paths (KiteFeed/TrueDataFeed/FreeFeed)
# before storage writes so all datasets are validated against a single contract.
