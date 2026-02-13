"""Backtest layer exports."""

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import (
    monte_carlo_permutation_pvalue,
    regime_segmented_returns,
    summarize_backtest,
)
from src.backtest.report import write_backtest_report, write_walkforward_report
from src.backtest.simulator import FillSimulator
from src.backtest.walkforward import (
    WalkForwardWindow,
    aggregate_cross_instrument_results,
    aggregate_walk_forward_metrics,
    build_sensitivity_variants,
    generate_walk_forward_windows,
    summarize_sensitivity_results,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "FillSimulator",
    "WalkForwardWindow",
    "generate_walk_forward_windows",
    "aggregate_walk_forward_metrics",
    "build_sensitivity_variants",
    "summarize_sensitivity_results",
    "aggregate_cross_instrument_results",
    "regime_segmented_returns",
    "monte_carlo_permutation_pvalue",
    "summarize_backtest",
    "write_backtest_report",
    "write_walkforward_report",
]
