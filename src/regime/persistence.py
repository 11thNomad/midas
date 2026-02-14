"""Persistence utilities for regime classification snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.store import DataStore
from src.signals.contracts import SignalSnapshotDTO, frame_from_signal_snapshots


@dataclass
class RegimeSnapshotStore:
    """Write/read regime snapshots using the shared parquet DataStore."""

    base_dir: str = "data/cache"
    dataset: str = "regime_snapshots"
    _store: DataStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = DataStore(base_dir=self.base_dir)

    def persist_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        symbol: str = "NIFTY",
        source: str = "regime_classifier",
    ) -> int:
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
        snapshots: list[dict[str, Any]],
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


@dataclass
class StrategyTransitionStore:
    """Write/read strategy-router activation transitions."""

    base_dir: str = "data/cache"
    dataset: str = "strategy_transitions"
    _store: DataStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = DataStore(base_dir=self.base_dir)

    def persist_transitions(
        self,
        transitions: list[dict[str, Any]],
        *,
        symbol: str = "NIFTY",
        source: str = "strategy_router",
    ) -> int:
        if not transitions:
            return 0
        frame = pd.DataFrame(transitions)
        if "timestamp" not in frame.columns:
            raise ValueError("Transitions require 'timestamp' field.")
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

    def read_transitions(
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


@dataclass
class SignalSnapshotStore:
    """Write/read normalized signal snapshots for backtest/paper/live parity."""

    base_dir: str = "data/cache"
    dataset: str = "signal_snapshots"
    _store: DataStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = DataStore(base_dir=self.base_dir)

    def persist_snapshot(
        self,
        snapshot: SignalSnapshotDTO,
        *,
        symbol: str = "NIFTY",
        timeframe: str = "1d",
        source: str = "signal_contract",
    ) -> int:
        frame = frame_from_signal_snapshots([snapshot])
        if frame.empty:
            return 0
        return self._store.write_time_series(
            self.dataset,
            frame,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_col="timestamp",
            source=source,
        )

    def persist_snapshots(
        self,
        snapshots: list[SignalSnapshotDTO],
        *,
        symbol: str = "NIFTY",
        timeframe: str = "1d",
        source: str = "signal_contract",
    ) -> int:
        if not snapshots:
            return 0
        frame = frame_from_signal_snapshots(snapshots)
        if frame.empty:
            return 0
        return self._store.write_time_series(
            self.dataset,
            frame,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_col="timestamp",
            source=source,
        )

    def read_snapshots(
        self,
        *,
        symbol: str = "NIFTY",
        timeframe: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return self._store.read_time_series(
            self.dataset,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            timestamp_col="timestamp",
        )
