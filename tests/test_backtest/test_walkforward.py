from __future__ import annotations

import math
from datetime import datetime

from src.backtest.walkforward import (
    aggregate_cross_instrument_results,
    aggregate_walk_forward_metrics,
    build_sensitivity_variants,
    generate_walk_forward_windows,
    summarize_sensitivity_results,
)


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


def test_aggregate_walk_forward_metrics_ignores_infinite_values():
    summary = aggregate_walk_forward_metrics(
        [
            {"total_return_pct": 2.0, "sharpe_ratio": float("inf")},
            {"total_return_pct": 4.0, "sharpe_ratio": 1.0},
        ]
    )
    assert summary["total_return_pct_mean"] == 3.0
    assert math.isclose(summary["sharpe_ratio_mean"], 1.0)


def test_build_sensitivity_variants_creates_numeric_perturbations():
    variants = build_sensitivity_variants(
        base_config={"fast_ema": 20, "slow_ema": 50, "name": "x"},
        params=["fast_ema", "slow_ema"],
        multipliers=[0.8, 1.0, 1.2],
    )
    assert len(variants) == 4
    ids = {v["variant_id"] for v in variants}
    assert "fast_ema_x0.80" in ids
    assert "slow_ema_x1.20" in ids


def test_summarize_sensitivity_results_calculates_robustness():
    summary = summarize_sensitivity_results(
        variant_rows=[
            {"total_return_pct": 8.0},
            {"total_return_pct": 6.0},
            {"total_return_pct": -2.0},
        ],
        base_total_return_pct=10.0,
    )
    assert summary["variant_count"] == 3.0
    assert 0.0 <= summary["robustness_score"] <= 1.0
    assert summary["positive_return_share"] == (2 / 3)


def test_aggregate_cross_instrument_results_groups_by_strategy():
    summary = aggregate_cross_instrument_results(
        [
            {
                "strategy": "iron_condor",
                "symbol": "NIFTY",
                "total_return_pct": 10.0,
                "sharpe_ratio": 1.2,
            },
            {
                "strategy": "iron_condor",
                "symbol": "BANKNIFTY",
                "total_return_pct": 5.0,
                "sharpe_ratio": 0.8,
            },
            {
                "strategy": "momentum",
                "symbol": "NIFTY",
                "total_return_pct": -2.0,
                "sharpe_ratio": 0.2,
            },
        ]
    )
    assert len(summary) == 2
    ic = summary.loc[summary["strategy"] == "iron_condor"].iloc[0]
    assert ic["symbols_tested"] == 2
    assert ic["symbols_positive_return"] == 2
