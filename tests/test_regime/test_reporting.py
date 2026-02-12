from __future__ import annotations

import pandas as pd

from src.regime.reporting import (
    summarize_regime_daily,
    summarize_transitions_by_strategy,
    summarize_transitions_daily,
)


def test_summarize_regime_daily_counts_rows():
    snapshots = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02 09:15:00", "2026-01-02 09:20:00", "2026-01-03 09:15:00"]
            ),
            "regime": ["low_vol_trending", "low_vol_trending", "high_vol_choppy"],
        }
    )
    out = summarize_regime_daily(snapshots)
    assert len(out) == 2
    assert int(out.loc[out["regime"] == "low_vol_trending", "snapshots"].iloc[0]) == 2


def test_summarize_transitions_daily_counts_activation_and_deactivation():
    transitions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02 09:15:00", "2026-01-02 10:00:00", "2026-01-03 09:15:00"]
            ),
            "strategy": ["a", "a", "b"],
            "from_active": [False, True, False],
            "to_active": [True, False, True],
            "regime": ["low_vol_trending", "high_vol_choppy", "low_vol_trending"],
        }
    )
    out = summarize_transitions_daily(transitions)
    assert len(out) == 2
    day1 = out.iloc[0]
    assert int(day1["activations"]) == 1
    assert int(day1["deactivations"]) == 1


def test_summarize_transitions_by_strategy_aggregates_counts():
    transitions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02 09:15:00", "2026-01-02 10:00:00", "2026-01-03 09:15:00"]
            ),
            "strategy": ["a", "a", "b"],
            "from_active": [False, True, False],
            "to_active": [True, False, True],
        }
    )
    out = summarize_transitions_by_strategy(transitions)
    assert len(out) == 2
    row_a = out.loc[out["strategy"] == "a"].iloc[0]
    assert int(row_a["events"]) == 2
