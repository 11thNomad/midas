"""Performance metrics for backtest outputs."""

from __future__ import annotations

import math
from typing import Any

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
    return _sharpe_from_returns(
        returns=returns,
        periods_per_year=float(periods_per_year),
        risk_free_rate_annual=float(risk_free_rate_annual),
    )


def sortino_ratio(
    equity: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.07,
) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    return _sortino_from_returns(
        returns=returns,
        periods_per_year=float(periods_per_year),
        risk_free_rate_annual=float(risk_free_rate_annual),
    )


def _sharpe_from_returns(
    *,
    returns: pd.Series,
    periods_per_year: float,
    risk_free_rate_annual: float,
) -> float:
    if returns.empty:
        return 0.0
    if periods_per_year <= 0:
        return 0.0
    rf_per_period = ((1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year)) - 1.0
    excess = returns - rf_per_period
    std = float(excess.std(ddof=1))
    if std == 0 or pd.isna(std):
        return 0.0
    return float((excess.mean() / std) * math.sqrt(periods_per_year))


def _sortino_from_returns(
    *,
    returns: pd.Series,
    periods_per_year: float,
    risk_free_rate_annual: float,
) -> float:
    if returns.empty:
        return 0.0
    if periods_per_year <= 0:
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
    run_integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if equity_curve.empty:
        metrics = {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,  # alias: daily RF=7%
            "sharpe_daily_rf7": 0.0,
            "sharpe_daily_rf0": 0.0,
            "sharpe_trade_rf7": 0.0,
            "sharpe_trade_rf0": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "fill_win_rate_pct": 0.0,
            "fill_profit_factor": 0.0,
            "trade_win_rate_pct": 0.0,
            "trade_profit_factor": 0.0,
            # Backward-compatible aliases (fill-level)
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "fill_count": 0.0,
            "trade_count_closed": 0.0,
            "trade_count_estimate": 0.0,
            "min_trade_count_required": float(minimum_trade_count),
            "min_trade_count_pass": 0.0,
            "monte_carlo_permutation_pvalue": 1.0,
            "anti_overfit_pass": 0.0,
            "fees_paid": 0.0,
        }
        if run_integrity is not None:
            metrics["run_integrity"] = run_integrity
        return metrics

    equity = equity_curve["equity"].astype("float64")
    final_equity = float(equity.iloc[-1])
    total_return_pct = ((final_equity - float(initial_capital)) / float(initial_capital)) * 100.0
    fees_paid = float(fills["fees"].sum()) if not fills.empty and "fees" in fills.columns else 0.0
    daily_returns = equity.pct_change().dropna().astype("float64")
    fill_win_rate = float(_win_rate_pct(fills))
    fill_pf = float(_profit_factor(fills))
    trade_pnls = _paired_trade_pnls(fills)
    trade_returns = (
        (trade_pnls / float(initial_capital)).astype("float64")
        if not trade_pnls.empty and float(initial_capital) > 0
        else pd.Series(dtype="float64")
    )
    years = len(equity) / max(float(periods_per_year), 1.0)
    trade_periods_per_year = (
        max(float(len(trade_returns)) / years, 1.0)
        if years > 0 and not trade_returns.empty
        else float(periods_per_year)
    )
    sharpe_daily_rf7 = _sharpe_from_returns(
        returns=daily_returns,
        periods_per_year=float(periods_per_year),
        risk_free_rate_annual=float(risk_free_rate_annual),
    )
    sharpe_daily_rf0 = _sharpe_from_returns(
        returns=daily_returns,
        periods_per_year=float(periods_per_year),
        risk_free_rate_annual=0.0,
    )
    sharpe_trade_rf7 = _sharpe_from_returns(
        returns=trade_returns,
        periods_per_year=trade_periods_per_year,
        risk_free_rate_annual=float(risk_free_rate_annual),
    )
    sharpe_trade_rf0 = _sharpe_from_returns(
        returns=trade_returns,
        periods_per_year=trade_periods_per_year,
        risk_free_rate_annual=0.0,
    )
    trade_count_estimate = _estimate_trade_count(fills)
    min_trade_count_pass = 1.0 if trade_count_estimate >= int(minimum_trade_count) else 0.0
    permutation_pvalue = monte_carlo_permutation_pvalue(
        equity=equity,
        periods_per_year=periods_per_year,
        permutations=monte_carlo_permutations,
    )
    anti_overfit_pass = 1.0 if (min_trade_count_pass == 1.0 and permutation_pvalue <= 0.10) else 0.0

    metrics = {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "annualized_return_pct": float(
            annualized_return_pct(equity, periods_per_year=periods_per_year)
        ),
        "max_drawdown_pct": float(max_drawdown_pct(equity)),
        "sharpe_ratio": float(sharpe_daily_rf7),  # legacy alias
        "sharpe_daily_rf7": float(sharpe_daily_rf7),
        "sharpe_daily_rf0": float(sharpe_daily_rf0),
        "sharpe_trade_rf7": float(sharpe_trade_rf7),
        "sharpe_trade_rf0": float(sharpe_trade_rf0),
        "sortino_ratio": float(
            sortino_ratio(
                equity,
                periods_per_year=periods_per_year,
                risk_free_rate_annual=risk_free_rate_annual,
            )
        ),
        "calmar_ratio": float(
            (
                annualized_return_pct(equity, periods_per_year=periods_per_year)
                / max_drawdown_pct(equity)
            )
            if max_drawdown_pct(equity) > 0
            else 0.0
        ),
        "fill_win_rate_pct": fill_win_rate,
        "fill_profit_factor": fill_pf,
        "trade_win_rate_pct": float(_trade_win_rate_pct(trade_pnls)),
        "trade_profit_factor": float(_trade_profit_factor(trade_pnls)),
        "win_rate_pct": fill_win_rate,  # legacy alias
        "profit_factor": fill_pf,  # legacy alias
        "fill_count": float(len(fills)),
        "trade_count_closed": float(len(trade_pnls)),
        "trade_count_estimate": float(trade_count_estimate),
        "min_trade_count_required": float(minimum_trade_count),
        "min_trade_count_pass": float(min_trade_count_pass),
        "monte_carlo_permutation_pvalue": float(permutation_pvalue),
        "anti_overfit_pass": float(anti_overfit_pass),
        "fees_paid": fees_paid,
    }
    if run_integrity is not None:
        metrics["run_integrity"] = run_integrity
    return metrics


def regime_segmented_returns(equity_curve: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Compute return profile grouped by regime labels."""
    required_eq = {"timestamp", "equity"}
    required_reg = {"timestamp", "regime"}
    if equity_curve.empty or regimes.empty:
        return pd.DataFrame(
            columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"]
        )
    if not required_eq.issubset(equity_curve.columns) or not required_reg.issubset(regimes.columns):
        return pd.DataFrame(
            columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"]
        )

    eq = equity_curve.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if eq.empty:
        return pd.DataFrame(
            columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"]
        )
    eq["bar_return"] = eq["equity"].pct_change().fillna(0.0)

    reg = regimes.copy()
    reg["timestamp"] = pd.to_datetime(reg["timestamp"], errors="coerce")
    reg = reg.dropna(subset=["timestamp"]).sort_values("timestamp")
    if reg.empty:
        return pd.DataFrame(
            columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"]
        )

    merged = eq.merge(reg[["timestamp", "regime"]], on="timestamp", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["regime", "bars", "mean_bar_return_pct", "cumulative_return_pct"]
        )

    rows: list[dict[str, Any]] = []
    for regime, grp in merged.groupby("regime"):
        returns = pd.Series(
            pd.to_numeric(grp["bar_return"], errors="coerce"),
            index=grp.index,
            dtype="float64",
        ).dropna()
        if returns.empty:
            cumulative = 0.0
            mean_bar_return_pct = 0.0
        else:
            values = returns.to_numpy(dtype="float64", copy=False)
            cumulative = float(np.prod(values + 1.0) - 1.0)
            mean_bar_return_pct = float(values.mean() * 100.0)
        rows.append(
            {
                "regime": str(regime),
                "bars": int(len(grp)),
                "mean_bar_return_pct": mean_bar_return_pct,
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
    notional_source = (
        fills["notional"] if "notional" in fills.columns else pd.Series(0.0, index=fills.index)
    )
    fees_source = fills["fees"] if "fees" in fills.columns else pd.Series(0.0, index=fills.index)
    notional = pd.to_numeric(notional_source, errors="coerce").fillna(0.0)
    fees = pd.to_numeric(fees_source, errors="coerce").fillna(0.0)
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


def _paired_trade_pnls(fills: pd.DataFrame) -> pd.Series:
    if fills.empty:
        return pd.Series(dtype="float64")
    required = {"timestamp", "signal_type"}
    if not required.issubset(fills.columns):
        return pd.Series(dtype="float64")

    out = fills.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    if out.empty:
        return pd.Series(dtype="float64")

    out["_cashflow"] = _fill_cashflow(out)
    signal_type = out["signal_type"].astype(str).str.lower()
    entry = (
        out.loc[signal_type.isin(["entry_long", "entry_short"])]
        .groupby("timestamp", as_index=False)["_cashflow"]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    exit_rows = (
        out.loc[signal_type == "exit"]
        .groupby("timestamp", as_index=False)["_cashflow"]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if entry.empty or exit_rows.empty:
        return pd.Series(dtype="float64")

    n = min(len(entry), len(exit_rows))
    if n <= 0:
        return pd.Series(dtype="float64")
    paired = (
        pd.to_numeric(entry.iloc[:n]["_cashflow"], errors="coerce").fillna(0.0).to_numpy()
        + pd.to_numeric(exit_rows.iloc[:n]["_cashflow"], errors="coerce").fillna(0.0).to_numpy()
    )
    return pd.Series(paired, dtype="float64")


def _trade_win_rate_pct(trade_pnls: pd.Series) -> float:
    if trade_pnls.empty:
        return 0.0
    wins = float((trade_pnls > 0).sum())
    total = float(len(trade_pnls))
    if total == 0:
        return 0.0
    return (wins / total) * 100.0


def _trade_profit_factor(trade_pnls: pd.Series) -> float:
    if trade_pnls.empty:
        return 0.0
    gross_profit = float(trade_pnls[trade_pnls > 0].sum())
    gross_loss = abs(float(trade_pnls[trade_pnls < 0].sum()))
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
