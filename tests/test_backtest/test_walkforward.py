from __future__ import annotations

from datetime import datetime

from src.backtest.walkforward import aggregate_walk_forward_metrics, generate_walk_forward_windows


def test_generate_walk_forward_windows_produces_ordered_windows():
    windows = generate_walk_forward_windows(
        start=datetime(2022, 1, 1),
        end=datetime(2023, 1, 1),
        train_months=6,
        test_months=3,
        step_months=3,
    )
    assert len(windows) >= 1
    assert windows[0].train_start < windows[0].train_end
    assert windows[0].test_start == windows[0].train_end
    assert windows[0].test_end <= datetime(2023, 1, 1)


def test_aggregate_walk_forward_metrics_returns_summary_stats():
    summary = aggregate_walk_forward_metrics(
        [
            {"total_return_pct": 2.0, "max_drawdown_pct": 1.0},
            {"total_return_pct": 4.0, "max_drawdown_pct": 3.0},
        ]
    )
    assert summary["folds"] == 2.0
    assert summary["total_return_pct_mean"] == 3.0
    assert summary["max_drawdown_pct_median"] == 2.0
