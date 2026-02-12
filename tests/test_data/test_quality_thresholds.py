from __future__ import annotations

import pandas as pd

from src.data.quality import (
    CandleQualityThresholds,
    assess_candle_quality,
    evaluate_quality_gate,
)


def test_holiday_is_not_counted_as_missing_trading_day():
    # 2026-01-26 is a configured NSE holiday in calendar.py
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-23", "2026-01-27"]),
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100.5, 101.5],
            "volume": [10, 10],
        }
    )

    report = assess_candle_quality(df, timeframe="1d")
    assert report.missing_trading_days == 0


def test_quality_gate_fails_when_threshold_exceeded():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"]),
            "open": [100, 100, 100],
            "high": [101, 101, 101],
            "low": [99, 99, 99],
            "close": [100, 100, 100],
            "volume": [10, 10, 10],
        }
    )
    report = assess_candle_quality(df, timeframe="1d")

    gate = evaluate_quality_gate(
        report,
        CandleQualityThresholds(max_duplicate_timestamps=0, max_largest_gap_minutes=10000),
    )

    assert gate.status == "failed_thresholds"
    assert any("duplicate_timestamps" in v for v in gate.violations)
