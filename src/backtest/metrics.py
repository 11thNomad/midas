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


def regime_segmented_returns(equity_curve: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Compute return profile grouped by regime labels."""
    required_eq = {"timestamp", "equity"}
    required_reg = {"timestamp", "regime"}
    if equity_curve.empty or regimes.empty:
        return pd.DataFrame(columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"])
    if not required_eq.issubset(equity_curve.columns) or not required_reg.issubset(regimes.columns):
        return pd.DataFrame(columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"])

    eq = equity_curve.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if eq.empty:
        return pd.DataFrame(columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"])
    eq["bar_return"] = eq["equity"].pct_change().fillna(0.0)

    reg = regimes.copy()
    reg["timestamp"] = pd.to_datetime(reg["timestamp"], errors="coerce")
    reg = reg.dropna(subset=["timestamp"]).sort_values("timestamp")
    if reg.empty:
        return pd.DataFrame(columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"])

    merged = eq.merge(reg[["timestamp", "regime"]], on="timestamp", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"])

    rows: list[dict] = []
    for regime, grp in merged.groupby("regime"):
        returns = grp["bar_return"].astype("float64")
        cumulative = (returns + 1.0).prod() - 1.0
        rows.append(
            {
                "regime": str(regime),
                "bars": int(len(grp)),
                "mean_bar_return_pct": float(returns.mean() * 100.0),
                "cumulative_return_pct": float(cumulative * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
