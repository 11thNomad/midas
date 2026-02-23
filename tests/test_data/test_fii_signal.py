from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.fii_signal import get_fii_signal


def _write_cache(path: Path, rows: list[tuple[str, float]]) -> str:
    df = pd.DataFrame(rows, columns=["date", "fii_equity_net_cr"])
    df.to_csv(path, index=False)
    return str(path)


def _settings() -> dict:
    return {
        "regime": {
            "fii_bearish_daily_threshold": -1000.0,
            "fii_bullish_daily_threshold": 1000.0,
            "fii_consecutive_days": 3,
        }
    }


def test_bearish_when_all_three_days_below_threshold(tmp_path: Path):
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-01-03", -1300.0), ("2025-01-06", -1400.0), ("2025-01-07", -1200.0)],
    )
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "bearish"
    assert out["fii_consecutive_negative"] is True
    assert out["fii_consecutive_positive"] is False
    assert out["data_complete"] is True


def test_neutral_when_only_two_of_three_days_below_threshold(tmp_path: Path):
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-01-03", -1300.0), ("2025-01-06", -900.0), ("2025-01-07", -1200.0)],
    )
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "neutral"
    assert out["fii_consecutive_negative"] is False
    assert out["data_complete"] is True


def test_neutral_when_middle_day_reverses_positive(tmp_path: Path):
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-01-03", -1300.0), ("2025-01-06", 500.0), ("2025-01-07", -1400.0)],
    )
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "neutral"
    assert out["fii_consecutive_negative"] is False
    assert out["fii_consecutive_positive"] is False


def test_bullish_when_all_three_days_above_threshold(tmp_path: Path):
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-01-03", 1100.0), ("2025-01-06", 1200.0), ("2025-01-07", 1300.0)],
    )
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "bullish"
    assert out["fii_consecutive_positive"] is True
    assert out["fii_consecutive_negative"] is False
    assert out["data_complete"] is True


def test_neutral_for_mixed_positive_and_negative(tmp_path: Path):
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-01-03", 1500.0), ("2025-01-06", -2000.0), ("2025-01-07", 500.0)],
    )
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "neutral"
    assert out["data_complete"] is True


def test_missing_t1_data_returns_neutral_incomplete(tmp_path: Path):
    cache = _write_cache(tmp_path / "fii.csv", [("2025-01-03", -1300.0), ("2025-01-06", -1200.0)])
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "neutral"
    assert out["data_complete"] is False
    assert math.isnan(float(out["fii_t1"]))


def test_holiday_t1_steps_back_to_previous_trading_day(tmp_path: Path):
    # 2025-03-14 is a holiday in calendar; as_of 2025-03-17 should use 13/12/11.
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-03-11", -1300.0), ("2025-03-12", -1200.0), ("2025-03-13", -1400.0)],
    )
    out = get_fii_signal(date(2025, 3, 17), cache, _settings())
    assert out["data_complete"] is True
    assert out["fii_t1"] == -1400.0
    assert out["fii_t2"] == -1200.0
    assert out["fii_t3"] == -1300.0
    assert out["fii_signal"] == "bearish"


def test_exactly_at_threshold_is_not_bearish(tmp_path: Path):
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-01-03", -1000.0), ("2025-01-06", -1200.0), ("2025-01-07", -1300.0)],
    )
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "neutral"
    assert out["fii_consecutive_negative"] is False


def test_fewer_than_three_prior_days_returns_neutral_incomplete(tmp_path: Path):
    cache = _write_cache(tmp_path / "fii.csv", [("2025-01-03", -1300.0)])
    out = get_fii_signal(date(2025, 1, 6), cache, _settings())
    assert out["fii_signal"] == "neutral"
    assert out["data_complete"] is False
    assert math.isnan(float(out["fii_t2"]))
    assert math.isnan(float(out["fii_t3"]))


def test_anomalous_values_are_ignored_if_present_in_cache(tmp_path: Path):
    cache = _write_cache(
        tmp_path / "fii.csv",
        [("2025-01-03", -1300.0), ("2025-01-06", -25000.0), ("2025-01-07", -1200.0)],
    )
    out = get_fii_signal(date(2025, 1, 8), cache, _settings())
    assert out["fii_signal"] == "neutral"
    assert out["data_complete"] is False
    assert math.isnan(float(out["fii_t2"]))
