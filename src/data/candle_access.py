"""Helpers to load candles with curated-cache preference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.store import DataStore


@dataclass(frozen=True)
class CandleStores:
    raw: DataStore
    curated: DataStore | None
    prefer_curated: bool


def build_candle_stores(*, settings: dict, repo_root: Path) -> CandleStores:
    data_cfg = settings.get("data", {})
    raw_cache_dir = repo_root / str(data_cfg.get("cache_dir", "data/cache"))
    curated_cache_dir = repo_root / str(data_cfg.get("curated_cache_dir", "data/curated_cache"))
    prefer_curated = bool(data_cfg.get("prefer_curated_candles", True))
    curated_store = DataStore(base_dir=str(curated_cache_dir)) if prefer_curated else None
    return CandleStores(
        raw=DataStore(base_dir=str(raw_cache_dir)),
        curated=curated_store,
        prefer_curated=prefer_curated,
    )


def read_candles(
    *,
    stores: CandleStores,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[pd.DataFrame, str]:
    if stores.prefer_curated and stores.curated is not None:
        curated = stores.curated.read_time_series(
            "candles",
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        if not curated.empty:
            return curated, "curated"

    raw = stores.raw.read_time_series(
        "candles",
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    return raw, "raw"
