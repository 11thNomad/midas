"""Data feed interfaces for historical and live market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from src.data.schemas import OptionChain


@dataclass(frozen=True)
class CandleRequest:
    """Normalized candle request contract across all data providers."""

    symbol: str
    timeframe: str
    start: datetime
    end: datetime


class DataFeed(Protocol):
    """Unified data provider contract used by backtests and live components."""

    name: str

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return OHLCV candles with at least timestamp/open/high/low/close columns."""

    def get_option_chain(
        self,
        symbol: str,
        expiry: datetime,
        timestamp: datetime | None = None,
    ) -> OptionChain:
        """Return option chain for an expiry snapshot."""

    def get_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return India VIX series for the requested range."""

    def get_fii_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return FII/DII flow time series for the requested range."""
