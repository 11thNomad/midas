"""Persistence utilities for regime classification snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.data.store import DataStore


@dataclass
class RegimeSnapshotStore:
    """Write/read regime snapshots using the shared parquet DataStore."""

    base_dir: str = "data/cache"
    dataset: str = "regime_snapshots"
    _store: DataStore = field(init=False, repr=False)

    def __post_init__(self):
        self._store = DataStore(base_dir=self.base_dir)

    def persist_snapshot(self, snapshot: dict, *, symbol: str = "NIFTY", source: str = "regime_classifier") -> int:
        frame = pd.DataFrame([snapshot])
        if "timestamp" not in frame.columns:
            raise ValueError("Snapshot requires 'timestamp' field.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            return 0
        return self._store.write_time_series(
            self.dataset,
            frame,
            symbol=symbol,
            timestamp_col="timestamp",
            source=source,
        )

    def persist_snapshots(
        self,
        snapshots: list[dict],
        *,
        symbol: str = "NIFTY",
        source: str = "regime_classifier",
    ) -> int:
        if not snapshots:
            return 0
        frame = pd.DataFrame(snapshots)
        if "timestamp" not in frame.columns:
            raise ValueError("Snapshots require 'timestamp' field.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            return 0
        return self._store.write_time_series(
            self.dataset,
            frame,
            symbol=symbol,
            timestamp_col="timestamp",
            source=source,
        )

    def read_snapshots(
        self,
        *,
        symbol: str = "NIFTY",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return self._store.read_time_series(
            self.dataset,
            symbol=symbol,
            start=start,
            end=end,
            timestamp_col="timestamp",
        )
