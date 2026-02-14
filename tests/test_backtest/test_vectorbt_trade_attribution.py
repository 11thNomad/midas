from __future__ import annotations

import pandas as pd

from src.backtest.vectorbt_trade_attribution import build_trade_attribution


def test_build_trade_attribution_joins_schedule_context():
    trades = pd.DataFrame(
        {
            "Exit Trade Id": [1],
            "Entry Timestamp": ["2026-01-02 09:15:00"],
            "Exit Timestamp": ["2026-01-05 09:15:00"],
            "PnL": [1200.0],
            "Return": [0.05],
            "Size": [2.0],
            "Avg Entry Price": [100.0],
            "Avg Exit Price": [110.0],
            "Direction": ["Long"],
            "Status": ["Closed"],
        }
    )
    schedule = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02 09:15:00", "2026-01-03 09:15:00", "2026-01-05 09:15:00"]
            ),
            "close": [100.0, 103.0, 110.0],
            "regime": ["low_vol_trending", "low_vol_trending", "high_vol_trending"],
            "adx_14": [25.0, 26.0, 30.0],
            "vix_level": [14.0, 14.2, 17.0],
        }
    )
    out = build_trade_attribution(
        trades=trades,
        schedule=schedule,
        set_id="baseline_trend",
        fee_profile="base",
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["set_id"] == "baseline_trend"
    assert row["fee_profile"] == "base"
    assert row["entry_regime"] == "low_vol_trending"
    assert row["exit_regime"] == "high_vol_trending"
    assert row["duration_bars"] == 2
    assert row["return_pct"] == 5.0
    assert bool(row["win"]) is True
