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
    assert m["fees_paid"] == 40.0


def test_summarize_backtest_uses_configurable_risk_free_rate():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="D"),
            "equity": [1000.0, 1010.0, 1005.0, 1020.0],
        }
    )
    fills = pd.DataFrame({"fees": [0.0]})
    low_rf = summarize_backtest(equity_curve=equity, fills=fills, initial_capital=1000.0, risk_free_rate_annual=0.0)
    high_rf = summarize_backtest(equity_curve=equity, fills=fills, initial_capital=1000.0, risk_free_rate_annual=0.20)
    assert low_rf["sharpe_ratio"] != high_rf["sharpe_ratio"]
