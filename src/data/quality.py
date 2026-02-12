"""Data quality checks for cached market datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class CandleQualityReport:
    rows: int
    missing_timestamps: int
    duplicate_timestamps: int
    invalid_ohlc_rows: int
    negative_or_zero_price_rows: int
    non_monotonic_timestamps: int
    missing_business_days: int | None
    largest_gap_minutes: float | None

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "missing_timestamps": self.missing_timestamps,
            "duplicate_timestamps": self.duplicate_timestamps,
            "invalid_ohlc_rows": self.invalid_ohlc_rows,
            "negative_or_zero_price_rows": self.negative_or_zero_price_rows,
            "non_monotonic_timestamps": self.non_monotonic_timestamps,
            "missing_business_days": self.missing_business_days,
            "largest_gap_minutes": self.largest_gap_minutes,
        }


def _safe_minutes(delta: pd.Timedelta | None) -> float | None:
    if delta is None or pd.isna(delta):
        return None
    return round(float(delta.total_seconds()) / 60.0, 2)


def assess_candle_quality(df: pd.DataFrame, timeframe: str) -> CandleQualityReport:
    if df.empty:
        return CandleQualityReport(
            rows=0,
            missing_timestamps=0,
            duplicate_timestamps=0,
            invalid_ohlc_rows=0,
            negative_or_zero_price_rows=0,
            non_monotonic_timestamps=0,
            missing_business_days=None,
            largest_gap_minutes=None,
        )

    data = df.copy()
    rows = len(data)

    if "timestamp" not in data.columns:
        raise ValueError("Candle dataset requires 'timestamp' column for quality checks.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    missing_timestamps = int(data["timestamp"].isna().sum())
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")

    duplicate_timestamps = int(data["timestamp"].duplicated(keep=False).sum())

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required OHLC column: {col}")

    invalid_ohlc_rows = int(((data["high"] < data[["open", "close"]].max(axis=1)) | (data["low"] > data[["open", "close"]].min(axis=1))).sum())

    negative_or_zero_price_rows = int((data[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())

    # If sorting changes order materially, original data was non-monotonic.
    non_monotonic_timestamps = int((pd.to_datetime(df["timestamp"], errors="coerce").diff() < pd.Timedelta(0)).sum())

    # Gap checks
    deltas = data["timestamp"].diff().dropna()
    largest_gap_minutes = _safe_minutes(deltas.max() if not deltas.empty else None)

    missing_business_days: int | None = None
    if timeframe.lower() == "1d" and len(data) > 1:
        start = data["timestamp"].min().normalize()
        end = data["timestamp"].max().normalize()
        expected = pd.bdate_range(start=start, end=end)
        observed = pd.DatetimeIndex(data["timestamp"].dt.normalize().unique())
        missing_business_days = int(len(expected.difference(observed)))

    return CandleQualityReport(
        rows=rows,
        missing_timestamps=missing_timestamps,
        duplicate_timestamps=duplicate_timestamps,
        invalid_ohlc_rows=invalid_ohlc_rows,
        negative_or_zero_price_rows=negative_or_zero_price_rows,
        non_monotonic_timestamps=non_monotonic_timestamps,
        missing_business_days=missing_business_days,
        largest_gap_minutes=largest_gap_minutes,
    )


def summarize_issue_count(report: CandleQualityReport) -> int:
    fields = [
        report.missing_timestamps,
        report.duplicate_timestamps,
        report.invalid_ohlc_rows,
        report.negative_or_zero_price_rows,
        report.non_monotonic_timestamps,
        report.missing_business_days or 0,
    ]
    return int(sum(fields))
