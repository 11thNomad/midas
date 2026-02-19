"""Data cleanup helpers for curated candle datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CleanupStats:
    """Summary metrics for a cleanup run."""

    input_rows: int
    output_rows: int
    duplicate_trade_dates: int
    dropped_rows: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "duplicate_trade_dates": self.duplicate_trade_dates,
            "dropped_rows": self.dropped_rows,
        }


def derive_trade_date_ist(timestamp: pd.Series) -> pd.Series:
    """Derive NSE trade date in IST from mixed timestamp conventions.

    Historical daily bars in this repository can arrive as:
    - "calendar midnight" (e.g. 2026-01-28 00:00:00)
    - "previous day 18:30" (UTC-naive representation of IST midnight)
    Both map to the same trade date after +05:30 normalization.
    """
    ts = pd.to_datetime(timestamp, errors="coerce")
    trade_date = (ts + pd.Timedelta(hours=5, minutes=30)).dt.date
    return trade_date


def clean_daily_candles(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanupStats]:
    """Deduplicate daily candles by IST trade date with deterministic best-row selection."""
    if df.empty:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        return empty, CleanupStats(0, 0, 0, 0)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"daily candle cleanup missing required columns: {sorted(missing)}")

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "oi"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    if "oi" not in frame.columns:
        frame["oi"] = 0.0

    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(
        drop=True
    )
    if frame.empty:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        return empty, CleanupStats(len(df), 0, 0, len(df))

    frame["trade_date_ist"] = derive_trade_date_ist(frame["timestamp"])
    frame = frame.dropna(subset=["trade_date_ist"]).reset_index(drop=True)
    if frame.empty:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        return empty, CleanupStats(len(df), 0, 0, len(df))

    rows: list[pd.Series] = []
    duplicate_days = 0

    for trade_date, group in frame.groupby("trade_date_ist", sort=True):
        if len(group) > 1:
            duplicate_days += 1
        best = _pick_best_daily_row(group)
        best = best.copy()
        best["trade_date_ist"] = trade_date
        rows.append(best)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["timestamp"] = out["trade_date_ist"].map(_midnight_timestamp)
    out = out.sort_values("timestamp").reset_index(drop=True)

    keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    out = out[keep_cols]

    stats = CleanupStats(
        input_rows=int(len(frame)),
        output_rows=int(len(out)),
        duplicate_trade_dates=int(duplicate_days),
        dropped_rows=int(len(frame) - len(out)),
    )
    return out, stats


def clean_intraday_candles(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanupStats]:
    """Deduplicate intraday candles by exact timestamp (keep latest after sort)."""
    if df.empty:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        return empty, CleanupStats(0, 0, 0, 0)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"intraday candle cleanup missing required columns: {sorted(missing)}")

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "oi"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    if "oi" not in frame.columns:
        frame["oi"] = 0.0

    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(
        drop=True
    )
    if frame.empty:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        return empty, CleanupStats(len(df), 0, 0, len(df))

    dup_rows = int(frame["timestamp"].duplicated(keep=False).sum())
    out = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out = out.sort_values("timestamp").reset_index(drop=True)
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    out = out[keep_cols]

    stats = CleanupStats(
        input_rows=int(len(frame)),
        output_rows=int(len(out)),
        duplicate_trade_dates=int(dup_rows),
        dropped_rows=int(len(frame) - len(out)),
    )
    return out, stats


def _pick_best_daily_row(group: pd.DataFrame) -> pd.Series:
    candidate = group.copy()

    candidate["_valid_ohlc"] = (
        (candidate["high"] >= candidate[["open", "close"]].max(axis=1))
        & (candidate["low"] <= candidate[["open", "close"]].min(axis=1))
    ).astype(int)
    candidate["_nonnull_count"] = (
        candidate[["open", "high", "low", "close", "volume", "oi"]].notna().sum(axis=1)
    )
    candidate["_info_score"] = (
        (candidate["volume"].fillna(0.0) > 0).astype(int)
        + (candidate["oi"].fillna(0.0) > 0).astype(int)
    )
    # Prefer rows with finer numeric precision when all else is tied.
    candidate["_precision_score"] = (
        candidate["open"].map(_fractional_precision)
        + candidate["high"].map(_fractional_precision)
        + candidate["low"].map(_fractional_precision)
        + candidate["close"].map(_fractional_precision)
    )

    ranked = candidate.sort_values(
        [
            "_valid_ohlc",
            "_nonnull_count",
            "_info_score",
            "_precision_score",
            "timestamp",
        ],
        ascending=[False, False, False, False, False],
    )
    return ranked.iloc[0]


def _fractional_precision(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    text = f"{float(value):.8f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".")[-1])


def _midnight_timestamp(day: date) -> pd.Timestamp:
    return pd.Timestamp(day)
