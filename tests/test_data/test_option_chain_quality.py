from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.option_chain_quality import (
    OptionChainQualityThresholds,
    assess_option_chain_quality,
    evaluate_option_chain_quality,
)


def test_option_chain_quality_no_data_returns_no_data_status():
    report = assess_option_chain_quality(None)
    result = evaluate_option_chain_quality(report, OptionChainQualityThresholds())
    assert report.rows == 0
    assert result.status == "no_data"
    assert result.issue_count == 0


def test_option_chain_quality_valid_chain_passes_gate():
    chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-02-10 09:30:00")] * 6,
            "expiry": [pd.Timestamp("2026-02-26")] * 6,
            "option_type": ["CE", "CE", "CE", "PE", "PE", "PE"],
            "strike": [22000, 22100, 22200, 22000, 21900, 21800],
            "ltp": [150.0, 120.0, 95.0, 130.0, 145.0, 170.0],
            "oi": [1000, 900, 700, 1100, 950, 800],
        }
    )
    thresholds = OptionChainQualityThresholds(min_rows=6, min_unique_strikes_per_side=3)
    report = assess_option_chain_quality(chain, asof=datetime(2026, 2, 10), dte_min=1, dte_max=30)
    result = evaluate_option_chain_quality(report, thresholds)
    assert report.rows == 6
    assert report.unique_call_strikes == 3
    assert report.unique_put_strikes == 3
    assert result.status == "ok"
    assert result.issue_count == 0


def test_option_chain_quality_detects_duplicate_and_missing_side():
    chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-02-10 09:30:00")] * 2,
            "expiry": [pd.Timestamp("2026-02-26"), pd.Timestamp("2026-02-26")],
            "option_type": ["CE", "CE"],
            "strike": [22000, 22000],
            "ltp": [150.0, 150.0],
            "oi": [1000, 1000],
        }
    )
    thresholds = OptionChainQualityThresholds(min_rows=2, min_unique_strikes_per_side=1)
    report = assess_option_chain_quality(chain)
    result = evaluate_option_chain_quality(report, thresholds)
    assert report.duplicate_contract_rows == 2
    assert result.status == "failed_thresholds"
    assert result.issue_count > 0
    assert any("duplicate_contract_rows" in v for v in result.violations)
    assert any(v == "missing_option_side" for v in result.violations)
