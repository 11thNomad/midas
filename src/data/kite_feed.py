"""Kite data feed stub for supplementary historical access."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.feed import DataFeed
from src.data.schemas import OptionChain


class KiteFeed(DataFeed):
    """Supplementary broker feed interface (placeholder)."""

    name = "kite"

    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("Kite historical ingestion is not yet implemented.")

    def get_option_chain(
        self,
        symbol: str,
        expiry: datetime,
        timestamp: datetime | None = None,
    ) -> OptionChain:
        raise NotImplementedError("Kite option chain retrieval is not yet implemented.")

    def get_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError("Use NSE/TrueData for VIX history.")

    def get_fii_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError("FII flow data is not available from Kite feed.")
