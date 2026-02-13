"""Parquet-backed storage and retrieval utilities for market datasets."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - fallback when optional engine is unavailable.
    pq = None


@dataclass
class DataStore:
    """Simple partitioned parquet store with metadata tracking."""

    base_dir: str = "data/cache"

    def __post_init__(self):
        self.root = Path(self.base_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "metadata.json"
        self.meta_lock_path = self.root / ".metadata.lock"
        if not self.meta_path.exists():
            self._save_metadata({"datasets": {}})

    def _load_metadata(self) -> dict:
        if not self.meta_path.exists():
            return {"datasets": {}}
        return json.loads(self.meta_path.read_text())

    def _save_metadata(self, data: dict):
        payload = json.dumps(data, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.root),
            prefix="metadata.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self.meta_path)

    def _dataset_dir(
        self,
        dataset: str,
        symbol: str | None = None,
        timeframe: str | None = None,
        *,
        create: bool = True,
    ) -> Path:
        parts = [self.root, dataset]
        if symbol:
            parts.append(symbol.upper())
        if timeframe:
            parts.append(timeframe.lower())
        path = Path(*parts)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _ensure_ts(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        if timestamp_col not in df.columns:
            raise ValueError(f"'{timestamp_col}' column is required for time partitioning.")
        out = df.copy()
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
        out = out.dropna(subset=[timestamp_col])
        out = out.sort_values(timestamp_col)
        return out

    def write_time_series(
        self,
        dataset: str,
        df: pd.DataFrame,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        timestamp_col: str = "timestamp",
        dedup_cols: list[str] | None = None,
        source: str = "unknown",
    ) -> int:
        """Write/merge a timeseries dataset as yearly parquet partitions.

        Args:
            dedup_cols:
                Optional uniqueness key columns used for upsert deduplication.
                Defaults to [timestamp_col]. When provided, timestamp_col is always included.

        Returns:
            Number of net-new rows added after deduplication/upsert.
        """
        if df.empty:
            return 0

        normalized = self._ensure_ts(df, timestamp_col)
        dedup_subset = self._resolve_dedup_subset(
            df=normalized,
            timestamp_col=timestamp_col,
            dedup_cols=dedup_cols,
        )
        normalized["_year"] = normalized[timestamp_col].dt.year

        target_dir = self._dataset_dir(dataset, symbol, timeframe, create=True)
        rows_upserted = 0

        for year, chunk in normalized.groupby("_year"):
            part = chunk.drop(columns=["_year"]).copy()
            part_path = target_dir / f"{int(year)}.parquet"
            if part_path.exists():
                existing = pd.read_parquet(part_path)
                existing[timestamp_col] = pd.to_datetime(existing[timestamp_col], errors="coerce")
                existing = existing.dropna(subset=[timestamp_col]).drop_duplicates(subset=dedup_subset, keep="last")
                existing_len = len(existing)
                combined = pd.concat([existing, part], ignore_index=True)
                combined[timestamp_col] = pd.to_datetime(combined[timestamp_col], errors="coerce")
                combined = combined.dropna(subset=[timestamp_col])
                combined = combined.sort_values(timestamp_col).drop_duplicates(subset=dedup_subset, keep="last")
            else:
                existing_len = 0
                combined = part.sort_values(timestamp_col).drop_duplicates(subset=dedup_subset, keep="last")

            combined.to_parquet(part_path, index=False)
            rows_upserted += max(len(combined) - existing_len, 0)

        total_rows = self._count_rows(target_dir)
        self._update_metadata(
            dataset=dataset,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            timestamp_col=timestamp_col,
            rows=total_rows,
            min_ts=normalized[timestamp_col].min(),
            max_ts=normalized[timestamp_col].max(),
        )

        return rows_upserted

    @staticmethod
    def _resolve_dedup_subset(
        *,
        df: pd.DataFrame,
        timestamp_col: str,
        dedup_cols: Iterable[str] | None,
    ) -> list[str]:
        requested = [col.strip() for col in (dedup_cols or [timestamp_col]) if str(col).strip()]
        if timestamp_col not in requested:
            requested.append(timestamp_col)

        seen: set[str] = set()
        subset: list[str] = []
        for col in requested:
            if col in seen:
                continue
            seen.add(col)
            subset.append(col)

        missing = [col for col in subset if col not in df.columns]
        if missing:
            raise ValueError(f"Dedup columns missing from dataset: {missing}")
        return subset

    def read_time_series(
        self,
        dataset: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Read and filter a partitioned timeseries dataset."""
        directory = self._dataset_dir(dataset, symbol, timeframe, create=False)
        if not directory.exists():
            return pd.DataFrame()

        files = self._partition_files_for_range(directory=directory, start=start, end=end)
        if not files:
            return pd.DataFrame()

        frames = [pd.read_parquet(path) for path in files]
        out = pd.concat(frames, ignore_index=True)

        if timestamp_col in out.columns:
            out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
            out = out.dropna(subset=[timestamp_col])
            if start is not None:
                out = out[out[timestamp_col] >= pd.Timestamp(start)]
            if end is not None:
                out = out[out[timestamp_col] <= pd.Timestamp(end)]
            out = out.sort_values(timestamp_col)

        return out.reset_index(drop=True)

    @staticmethod
    def _partition_files_for_range(
        *,
        directory: Path,
        start: datetime | None,
        end: datetime | None,
    ) -> list[Path]:
        files = sorted(directory.glob("*.parquet"))
        if not files:
            return []
        if start is None and end is None:
            return files

        start_year = pd.Timestamp(start).year if start is not None else None
        end_year = pd.Timestamp(end).year if end is not None else None

        selected: list[Path] = []
        for path in files:
            stem = path.stem
            if not stem.isdigit():
                selected.append(path)
                continue
            year = int(stem)
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            selected.append(path)
        return selected

    def _update_metadata(
        self,
        *,
        dataset: str,
        symbol: str | None,
        timeframe: str | None,
        source: str,
        timestamp_col: str,
        rows: int,
        min_ts: pd.Timestamp,
        max_ts: pd.Timestamp,
    ):
        with self._metadata_lock():
            metadata = self._load_metadata()
            key_parts = [dataset]
            if symbol:
                key_parts.append(symbol.upper())
            if timeframe:
                key_parts.append(timeframe.lower())
            key = ":".join(key_parts)

            metadata.setdefault("datasets", {})[key] = {
                "dataset": dataset,
                "symbol": symbol,
                "timeframe": timeframe,
                "source": source,
                "timestamp_col": timestamp_col,
                "rows": int(rows),
                "min_timestamp": pd.Timestamp(min_ts).isoformat(),
                "max_timestamp": pd.Timestamp(max_ts).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
            self._save_metadata(metadata)

    @staticmethod
    def _count_rows(directory: Path) -> int:
        total = 0
        for path in directory.glob("*.parquet"):
            if pq is not None:
                total += int(pq.ParquetFile(path).metadata.num_rows)
            else:
                total += len(pd.read_parquet(path, columns=[]))
        return total

    def get_metadata(self) -> dict:
        return self._load_metadata()

    @contextmanager
    def _metadata_lock(self):
        self.meta_lock_path.touch(exist_ok=True)
        with self.meta_lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
