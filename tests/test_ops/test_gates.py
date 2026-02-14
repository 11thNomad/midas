from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.data.store import DataStore
from src.ops.gates import (
    FreshnessGate,
    build_default_open_gates,
    check_freshness_gate,
    evaluate_freshness_gates,
    summarize_gate_results,
)


def test_check_freshness_gate_passes_when_recent(tmp_path):
    base = tmp_path / "cache"
    store = DataStore(base_dir=str(base))
    now = datetime.now(UTC).replace(tzinfo=None)
    frame = pd.DataFrame(
        {"timestamp": [pd.Timestamp(now - timedelta(minutes=30))], "close": [1.0]}
    )
    store.write_time_series("candles", frame, symbol="NIFTY", timeframe="1d")
    gate = FreshnessGate(
        name="candles_recent",
        dataset="candles",
        symbol="NIFTY",
        timeframe="1d",
        max_age_minutes=60,
    )
    result = check_freshness_gate(store, gate, now=now)
    assert result.ok is True
    assert result.age_minutes is not None and result.age_minutes <= 60


def test_check_freshness_gate_fails_when_stale(tmp_path):
    base = tmp_path / "cache"
    store = DataStore(base_dir=str(base))
    now = datetime.now(UTC).replace(tzinfo=None)
    frame = pd.DataFrame(
        {"timestamp": [pd.Timestamp(now - timedelta(days=4))], "close": [1.0]}
    )
    store.write_time_series("vix", frame, symbol="INDIAVIX", timeframe="1d")
    gate = FreshnessGate(
        name="vix_stale",
        dataset="vix",
        symbol="INDIAVIX",
        timeframe="1d",
        max_age_minutes=24 * 60,
    )
    result = check_freshness_gate(store, gate, now=now)
    assert result.ok is False
    assert result.message == "Data stale"


def test_summarize_gate_results_counts_hard_and_warning_failures(tmp_path):
    base = tmp_path / "cache"
    store = DataStore(base_dir=str(base))
    now = datetime.now(UTC).replace(tzinfo=None)

    gates = [
        FreshnessGate(
            name="required_missing",
            dataset="candles",
            symbol="NIFTY",
            timeframe="1d",
            max_age_minutes=60,
            required=True,
            severity="error",
        ),
        FreshnessGate(
            name="optional_missing",
            dataset="signal_snapshots",
            symbol="NIFTY",
            timeframe="1d",
            max_age_minutes=60,
            required=False,
            severity="warning",
        ),
    ]
    results = evaluate_freshness_gates(store, gates, now=now)
    summary = summarize_gate_results(results)
    assert summary["hard_failures"] == 1
    assert summary["warning_failures"] == 1
    assert summary["ok"] is False


def test_build_default_open_gates_includes_usdinr_symbol():
    settings = {"market": {"usdinr_symbol": "USDINR"}}
    gates = build_default_open_gates(settings, symbol="NIFTY", timeframe="1d")
    names = {g.name for g in gates}
    assert "usdinr_daily" in names
