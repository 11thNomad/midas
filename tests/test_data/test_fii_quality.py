from __future__ import annotations

import pandas as pd

from src.data.fii import normalize_fii_payload
from src.data.quality import assess_candle_quality, summarize_issue_count


def test_normalize_fii_payload_pivots_categories():
    payload = [
        {
            "category": "DII",
            "date": "12-Feb-2026",
            "buyValue": "17213.85",
            "sellValue": "16937",
            "netValue": "276.85",
        },
        {
            "category": "FII/FPI",
            "date": "12-Feb-2026",
            "buyValue": "17949.52",
            "sellValue": "17841.1",
            "netValue": "108.42",
        },
    ]

    df = normalize_fii_payload(payload)
    assert len(df) == 1
    assert set(df.columns) == {
        "date",
        "fii_buy",
        "fii_sell",
        "fii_net",
        "dii_buy",
        "dii_sell",
        "dii_net",
    }
    assert float(df.loc[0, "fii_net"]) == 108.42
    assert float(df.loc[0, "dii_net"]) == 276.85


def test_assess_candle_quality_detects_issues():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",  # duplicate
                    "2026-01-01",  # non-monotonic vs input order
                ]
            ),
            "open": [100, 100, 100, 100],
            "high": [99, 101, 101, 101],  # invalid first row (high < open)
            "low": [98, 99, 99, -1],  # invalid negative low on last row
            "close": [100, 100, 100, 100],
            "volume": [10, 10, 10, 10],
        }
    )

    report = assess_candle_quality(df, timeframe="1d")
    assert report.rows == 4
    assert report.duplicate_timestamps >= 2
    assert report.invalid_ohlc_rows >= 1
    assert report.negative_or_zero_price_rows >= 1
    assert report.non_monotonic_timestamps >= 1
    assert summarize_issue_count(report) > 0
