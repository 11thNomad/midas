from __future__ import annotations

import pandas as pd

from src.backtest.metrics import summarize_backtest


def test_summarize_backtest_returns_core_fields():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "equity": [1000.0, 1010.0, 990.0],
        }
    )
    fills = pd.DataFrame({"fees": [20.0, 20.0]})
    m = summarize_backtest(equity_curve=equity, fills=fills, initial_capital=1000.0)
    assert "total_return_pct" in m
    assert "max_drawdown_pct" in m
    assert "sharpe_daily_rf7" in m
    assert "sharpe_trade_rf0" in m
    assert "trade_win_rate_pct" in m
    assert "trade_profit_factor" in m
    assert m["fees_paid"] == 40.0


def test_summarize_backtest_uses_configurable_risk_free_rate():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="D"),
            "equity": [1000.0, 1010.0, 1005.0, 1020.0],
        }
    )
    fills = pd.DataFrame({"fees": [0.0]})
    low_rf = summarize_backtest(
        equity_curve=equity, fills=fills, initial_capital=1000.0, risk_free_rate_annual=0.0
    )
    high_rf = summarize_backtest(
        equity_curve=equity, fills=fills, initial_capital=1000.0, risk_free_rate_annual=0.20
    )
    assert low_rf["sharpe_ratio"] != high_rf["sharpe_ratio"]


def test_summarize_backtest_includes_anti_overfit_metrics():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="D"),
            "equity": [1000.0, 1020.0, 1010.0, 1035.0, 1025.0, 1040.0],
        }
    )
    fills = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05"]),
            "strategy_name": ["x", "x", "x"],
            "signal_type": ["entry_long", "entry_long", "entry_long"],
            "side": ["BUY", "BUY", "BUY"],
            "notional": [100.0, 100.0, 100.0],
            "fees": [0.0, 0.0, 0.0],
        }
    )
    m = summarize_backtest(
        equity_curve=equity,
        fills=fills,
        initial_capital=1000.0,
        monte_carlo_permutations=50,
        minimum_trade_count=2,
    )
    assert "monte_carlo_permutation_pvalue" in m
    assert 0.0 <= m["monte_carlo_permutation_pvalue"] <= 1.0
    assert m["trade_count_estimate"] == 3.0
    assert m["min_trade_count_pass"] == 1.0


def test_summarize_backtest_includes_run_integrity_payload():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="D"),
            "equity": [1000.0, 1005.0],
        }
    )
    fills = pd.DataFrame({"fees": [1.0]})
    run_integrity = {
        "forced_liquidations": {"count": 0, "symbols": [], "threshold": 1, "flag": False},
        "unfilled_exits": {"attempted": 0, "filled": 0, "unfilled": 0, "failure_reasons": {}},
    }
    m = summarize_backtest(
        equity_curve=equity,
        fills=fills,
        initial_capital=1000.0,
        run_integrity=run_integrity,
    )
    assert "run_integrity" in m
    assert m["run_integrity"] == run_integrity
