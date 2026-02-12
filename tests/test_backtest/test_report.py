from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.report import write_backtest_report


def test_write_backtest_report_creates_files(tmp_path):
    result = BacktestResult(
        equity_curve=pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=1), "equity": [1000.0]}),
        fills=pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=1), "fees": [20.0]}),
        regimes=pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=1), "regime": ["low_vol_ranging"]}),
        metrics={"final_equity": 1000.0},
    )
    paths = write_backtest_report(result, output_dir=str(tmp_path), run_name="demo")
    for _, p in paths.items():
        assert Path(p).exists()
