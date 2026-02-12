"""TrueData feed stub.

Implemented in Phase 1 as a contract placeholder so downstream modules can
program against a stable interface.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.feed import DataFeed
from src.data.schemas import OptionChain


class TrueDataFeed(DataFeed):
    """Primary paid feed interface (placeholder)."""

    name = "truedata"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("TrueData candle ingestion to be implemented in the next increment.")

    def get_option_chain(
        self,
        symbol: str,
        expiry: datetime,
        timestamp: datetime | None = None,
    ) -> OptionChain:
        raise NotImplementedError("TrueData option chain ingestion to be implemented in the next increment.")

    def get_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError("TrueData VIX ingestion to be implemented in the next increment.")

    def get_fii_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError("FII flow ingestion is expected via NSE dataset pipeline.")
