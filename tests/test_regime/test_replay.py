from __future__ import annotations

import pandas as pd

from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.regime.replay import replay_regimes_no_lookahead


def _candles(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": [100 + i for i in range(n)],
            "high": [101 + i for i in range(n)],
            "low": [99 + i for i in range(n)],
            "close": [100 + i for i in range(n)],
            "volume": [1000 for _ in range(n)],
        }
    )


def test_replay_returns_snapshot_for_each_bar():
    candles = _candles(15)
    vix = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=15, freq="D"),
            "close": [12.0 + 0.1 * i for i in range(15)],
        }
    )
    fii = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=15, freq="D"),
            "fii_net": [100.0 for _ in range(15)],
        }
    )

    result = replay_regimes_no_lookahead(
        candles=candles,
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        vix_df=vix,
        fii_df=fii,
    )

    assert len(result.snapshots) == len(candles)
    assert "regime" in result.snapshots.columns


def test_replay_is_no_lookahead_for_prefix_window():
    candles_full = _candles(12)
    vix_full = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="D"),
            "close": [12.0] * 11 + [30.0],  # extreme future spike
        }
    )

    full = replay_regimes_no_lookahead(
        candles=candles_full,
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        vix_df=vix_full,
    )

    candles_prefix = candles_full.iloc[:11].copy()
    vix_prefix = vix_full.iloc[:11].copy()
    prefix = replay_regimes_no_lookahead(
        candles=candles_prefix,
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        vix_df=vix_prefix,
    )

    # Future VIX spike at bar 12 must not change labels for bars 1..11.
    assert list(full.snapshots["regime"].iloc[:11]) == list(prefix.snapshots["regime"])


def test_replay_with_empty_candles_returns_empty_frames():
    result = replay_regimes_no_lookahead(
        candles=pd.DataFrame(),
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
    )
    assert result.snapshots.empty
    assert result.transitions.empty


def test_replay_analysis_start_filters_output_rows():
    candles = _candles(10)
    vix = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
            "close": [12.0 + 0.1 * i for i in range(10)],
        }
    )
    cutoff = pd.Timestamp("2026-01-06")

    result = replay_regimes_no_lookahead(
        candles=candles,
        classifier=RegimeClassifier(thresholds=RegimeThresholds()),
        vix_df=vix,
        analysis_start=cutoff.to_pydatetime(),
    )

    assert len(result.snapshots) == 5
    snapshot_ts = pd.to_datetime(result.snapshots["timestamp"], errors="coerce")
    assert snapshot_ts.min() >= cutoff
    if not result.transitions.empty:
        transition_ts = pd.to_datetime(result.transitions["timestamp"], errors="coerce")
        assert transition_ts.min() >= cutoff
