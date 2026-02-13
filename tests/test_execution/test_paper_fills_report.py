from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_summarizer():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "paper_fills_report.py"
    spec = importlib.util.spec_from_file_location("paper_fills_report_module", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize_daily_fills


summarize_daily_fills = _load_summarizer()


def test_summarize_daily_fills_groups_by_day_and_strategy():
    fills = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-02-10 10:00",
                    "2026-02-10 10:05",
                    "2026-02-10 12:00",
                    "2026-02-11 09:20",
                ]
            ),
            "strategy_name": ["iron_condor", "iron_condor", "momentum", "iron_condor"],
            "side": ["SELL", "BUY", "BUY", "SELL"],
            "notional": [1200.0, 900.0, 500.0, 1000.0],
            "fees": [20.0, 20.0, 15.0, 20.0],
        }
    )

    daily = summarize_daily_fills(fills)
    assert len(daily) == 3

    row_ic_day1 = daily.loc[
        (daily["date"] == pd.Timestamp("2026-02-10").date())
        & (daily["strategy_name"] == "iron_condor")
    ].iloc[0]
    assert int(row_ic_day1["fill_count"]) == 2
    assert float(row_ic_day1["gross_cashflow"]) == 300.0
    assert float(row_ic_day1["fees"]) == 40.0
    assert float(row_ic_day1["net_cashflow"]) == 260.0

    row_mom_day1 = daily.loc[
        (daily["date"] == pd.Timestamp("2026-02-10").date())
        & (daily["strategy_name"] == "momentum")
    ].iloc[0]
    assert float(row_mom_day1["gross_cashflow"]) == -500.0
    assert float(row_mom_day1["net_cashflow"]) == -515.0


def test_summarize_daily_fills_empty_returns_expected_columns():
    daily = summarize_daily_fills(pd.DataFrame())
    assert list(daily.columns) == [
        "date",
        "strategy_name",
        "fill_count",
        "buy_notional",
        "sell_notional",
        "gross_cashflow",
        "fees",
        "net_cashflow",
    ]
    assert daily.empty
