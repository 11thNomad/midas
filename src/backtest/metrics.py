"""Performance metrics for backtest outputs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, pd.NA)
    return float(abs(dd.min()) * 100.0) if not dd.dropna().empty else 0.0


def sharpe_ratio(
    equity: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.07,
) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0
    rf_per_period = ((1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year)) - 1.0
    excess = returns - rf_per_period
    std = float(excess.std(ddof=1))
    if std == 0 or pd.isna(std):
        return 0.0
    return float((excess.mean() / std) * math.sqrt(periods_per_year))


def sortino_ratio(
    equity: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.07,
) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0
    rf_per_period = ((1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year)) - 1.0
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    downside_std = float(downside.std(ddof=1))
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0
    return float((excess.mean() / downside_std) * math.sqrt(periods_per_year))


def annualized_return_pct(equity: pd.Series, *, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((((end / start) ** (1.0 / years)) - 1.0) * 100.0)


def summarize_backtest(
    *,
    equity_curve: pd.DataFrame,
    fills: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.07,
    monte_carlo_permutations: int = 200,
    minimum_trade_count: int = 50,
) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "fill_count": 0.0,
            "trade_count_estimate": 0.0,
            "min_trade_count_required": float(minimum_trade_count),
            "min_trade_count_pass": 0.0,
            "monte_carlo_permutation_pvalue": 1.0,
            "anti_overfit_pass": 0.0,
            "fees_paid": 0.0,
        }

    equity = equity_curve["equity"].astype("float64")
    final_equity = float(equity.iloc[-1])
    total_return_pct = ((final_equity - float(initial_capital)) / float(initial_capital)) * 100.0
    fees_paid = float(fills["fees"].sum()) if not fills.empty and "fees" in fills.columns else 0.0
    trade_count_estimate = _estimate_trade_count(fills)
    min_trade_count_pass = 1.0 if trade_count_estimate >= int(minimum_trade_count) else 0.0
    permutation_pvalue = monte_carlo_permutation_pvalue(
        equity=equity,
        periods_per_year=periods_per_year,
        permutations=monte_carlo_permutations,
    )
    anti_overfit_pass = 1.0 if (min_trade_count_pass == 1.0 and permutation_pvalue <= 0.10) else 0.0

    return {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "annualized_return_pct": float(annualized_return_pct(equity, periods_per_year=periods_per_year)),
        "max_drawdown_pct": float(max_drawdown_pct(equity)),
        "sharpe_ratio": float(
            sharpe_ratio(
                equity,
                periods_per_year=periods_per_year,
                risk_free_rate_annual=risk_free_rate_annual,
            )
        ),
        "sortino_ratio": float(
            sortino_ratio(
                equity,
                periods_per_year=periods_per_year,
                risk_free_rate_annual=risk_free_rate_annual,
            )
        ),
        "calmar_ratio": float(
            (annualized_return_pct(equity, periods_per_year=periods_per_year) / max_drawdown_pct(equity))
            if max_drawdown_pct(equity) > 0
            else 0.0
        ),
        "win_rate_pct": float(_win_rate_pct(fills)),
        "profit_factor": float(_profit_factor(fills)),
        "fill_count": float(len(fills)),
        "trade_count_estimate": float(trade_count_estimate),
        "min_trade_count_required": float(minimum_trade_count),
        "min_trade_count_pass": float(min_trade_count_pass),
        "monte_carlo_permutation_pvalue": float(permutation_pvalue),
        "anti_overfit_pass": float(anti_overfit_pass),
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


def _fill_cashflow(fills: pd.DataFrame) -> pd.Series:
    if fills.empty:
        return pd.Series(dtype="float64")
    if "side" not in fills.columns:
        return pd.Series(dtype="float64")
    side = fills["side"].astype(str).str.upper()
    notional = pd.to_numeric(fills.get("notional", 0.0), errors="coerce").fillna(0.0)
    fees = pd.to_numeric(fills.get("fees", 0.0), errors="coerce").fillna(0.0)
    signed = notional.where(side == "SELL", -notional) - fees
    return signed


def _win_rate_pct(fills: pd.DataFrame) -> float:
    if fills.empty:
        return 0.0
    pnl = _fill_cashflow(fills)
    wins = float((pnl > 0).sum())
    total = float(len(pnl))
    if total == 0:
        return 0.0
    return (wins / total) * 100.0


def _profit_factor(fills: pd.DataFrame) -> float:
    if fills.empty:
        return 0.0
    pnl = _fill_cashflow(fills)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return 0.0
    return gross_profit / gross_loss


def _estimate_trade_count(fills: pd.DataFrame) -> int:
    if fills.empty:
        return 0
    required = {"timestamp", "strategy_name", "signal_type"}
    if required.issubset(fills.columns):
        signal_type = fills["signal_type"].astype(str).str.lower()
        entry_rows = fills.loc[signal_type.isin(["entry_long", "entry_short"])].copy()
        if not entry_rows.empty:
            entry_rows["timestamp"] = pd.to_datetime(entry_rows["timestamp"], errors="coerce")
            entry_rows = entry_rows.dropna(subset=["timestamp"])
            if not entry_rows.empty:
                return int(len(entry_rows[["timestamp", "strategy_name"]].drop_duplicates()))
    return int(len(fills))


def monte_carlo_permutation_pvalue(
    *,
    equity: pd.Series,
    periods_per_year: int = 252,
    permutations: int = 200,
    seed: int = 42,
) -> float:
    if len(equity) < 4:
        return 1.0
    returns = equity.pct_change().dropna().astype("float64")
    if returns.empty:
        return 1.0
    observed_calmar = _calmar_from_returns(returns, periods_per_year=periods_per_year)
    if observed_calmar <= 0:
        return 1.0

    n = max(int(permutations), 1)
    rng = np.random.default_rng(seed)
    values = returns.to_numpy(copy=True)
    at_or_above = 0
    for _ in range(n):
        shuffled = values[rng.permutation(len(values))]
        candidate = _calmar_from_returns(pd.Series(shuffled), periods_per_year=periods_per_year)
        if candidate >= observed_calmar:
            at_or_above += 1
    return float((at_or_above + 1) / (n + 1))


def _calmar_from_returns(returns: pd.Series, *, periods_per_year: int) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    ann = annualized_return_pct(equity, periods_per_year=periods_per_year) / 100.0
    mdd = max_drawdown_pct(equity) / 100.0
    if mdd <= 0:
        return 0.0
    return float(ann / mdd)
