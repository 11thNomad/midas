"""Performance metrics for backtest outputs."""

from __future__ import annotations

import math

import pandas as pd


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, pd.NA)
    return float(abs(dd.min()) * 100.0) if not dd.dropna().empty else 0.0


def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=0))
    if std == 0:
        return 0.0
    return float((returns.mean() / std) * math.sqrt(periods_per_year))


def summarize_backtest(
    *,
    equity_curve: pd.DataFrame,
    fills: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 252,
) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "fill_count": 0.0,
            "fees_paid": 0.0,
        }

    equity = equity_curve["equity"].astype("float64")
    final_equity = float(equity.iloc[-1])
    total_return_pct = ((final_equity - float(initial_capital)) / float(initial_capital)) * 100.0
    fees_paid = float(fills["fees"].sum()) if not fills.empty and "fees" in fills.columns else 0.0

    return {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_drawdown_pct(equity)),
        "sharpe_ratio": float(sharpe_ratio(equity, periods_per_year=periods_per_year)),
        "fill_count": float(len(fills)),
        "fees_paid": fees_paid,
    }
