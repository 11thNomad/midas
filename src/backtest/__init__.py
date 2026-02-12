"""Backtest layer exports."""

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import summarize_backtest
from src.backtest.report import write_backtest_report
from src.backtest.simulator import FillSimulator
from src.backtest.walkforward import aggregate_walk_forward_metrics, generate_walk_forward_windows, WalkForwardWindow

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "FillSimulator",
    "WalkForwardWindow",
    "generate_walk_forward_windows",
    "aggregate_walk_forward_metrics",
    "summarize_backtest",
    "write_backtest_report",
]
