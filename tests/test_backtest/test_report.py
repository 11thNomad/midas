from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.report import write_backtest_report, write_walkforward_report


def test_write_backtest_report_creates_files(tmp_path):
    result = BacktestResult(
        equity_curve=pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=1), "equity": [1000.0]}
        ),
        fills=pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=1), "fees": [20.0]}),
        regimes=pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=1), "regime": ["low_vol_ranging"]}
        ),
        metrics={"final_equity": 1000.0},
    )
    paths = write_backtest_report(result, output_dir=str(tmp_path), run_name="demo")
    for _, p in paths.items():
        assert Path(p).exists()


def test_write_walkforward_report_creates_files(tmp_path):
    folds = pd.DataFrame(
        [
            {"fold": 1, "total_return_pct": 1.0, "dominant_regime": "low_vol_trending"},
            {"fold": 2, "total_return_pct": 2.0, "dominant_regime": "high_vol_choppy"},
        ]
    )
    summary = {"folds": 2.0, "total_return_pct_mean": 1.5}
    regime_table = pd.DataFrame(
        [
            {
                "regime": "low_vol_trending",
                "folds": 2,
                "bars": 100,
                "mean_bar_return_pct": 0.1,
                "cumulative_return_pct": 2.5,
            }
        ]
    )
    paths = write_walkforward_report(
        folds=folds,
        summary=summary,
        regime_table=regime_table,
        output_dir=str(tmp_path),
        run_name="wf",
    )
    for _, p in paths.items():
        assert Path(p).exists()
    html = Path(paths["html_report"]).read_text()
    assert "Regime-Segmented Returns" in html
