from __future__ import annotations

import pandas as pd

from src.data.cleanup import clean_daily_candles


def test_clean_daily_candles_merges_mixed_timestamp_conventions():
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-27 18:30:00",  # maps to IST trade date 2026-01-28
                "2026-01-28 00:00:00",  # same IST trade date
                "2026-01-29 00:00:00",
            ],
            "open": [100.0, 100.0001, 102.0],
            "high": [105.0, 105.0001, 106.0],
            "low": [99.0, 99.0001, 101.0],
            "close": [104.0, 104.0001, 103.0],
            "volume": [0.0, 500.0, 300.0],
            "oi": [0.0, 0.0, 0.0],
        }
    )

    out, stats = clean_daily_candles(raw)

    assert len(out) == 2
    assert stats.input_rows == 3
    assert stats.output_rows == 2
    assert stats.duplicate_trade_dates == 1
    assert stats.dropped_rows == 1

    first = out.iloc[0]
    assert pd.Timestamp(first["timestamp"]) == pd.Timestamp("2026-01-28 00:00:00")
    # Prefer row with stronger informational content (volume > 0).
    assert float(first["close"]) == 104.0001

    second = out.iloc[1]
    assert pd.Timestamp(second["timestamp"]) == pd.Timestamp("2026-01-29 00:00:00")


def test_clean_daily_candles_prefers_valid_ohlc_even_if_less_informative():
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2026-02-03 18:30:00",
                "2026-02-04 00:00:00",
            ],
            "open": [100.0, 100.0],
            "high": [99.0, 105.0],  # first row invalid (high < max(open, close))
            "low": [98.0, 95.0],
            "close": [101.0, 101.0],
            "volume": [1000.0, 0.0],
            "oi": [0.0, 0.0],
        }
    )

    out, stats = clean_daily_candles(raw)

    assert len(out) == 1
    assert stats.duplicate_trade_dates == 1

    row = out.iloc[0]
    # Valid OHLC row must win over invalid row.
    assert float(row["high"]) == 105.0
    assert float(row["low"]) == 95.0
