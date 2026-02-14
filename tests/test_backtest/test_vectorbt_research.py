from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from src.backtest.simulator import FillSimulator
from src.backtest.vectorbt_research import (
    VectorBTResearchConfig,
    build_vectorbt_schedule,
    run_hybrid_from_schedule,
    run_hybrid_from_schedule_result,
    run_vectorbt_research,
    run_vectorbt_sensitivity,
    run_vectorbt_walk_forward,
)
from src.regime.classifier import RegimeThresholds


def _sample_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100 + i for i in range(30)],
        }
    )


def _sample_snapshots() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=30, freq="D")
    regimes = ["low_vol_trending" if i % 6 < 3 else "high_vol_choppy" for i in range(30)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "regime": regimes,
            "adx_14": [30.0 if r == "low_vol_trending" else 15.0 for r in regimes],
            "vix_level": [13.0 if r == "low_vol_trending" else 20.0 for r in regimes],
        }
    )


def test_build_vectorbt_schedule_has_entries_and_exits():
    schedule = build_vectorbt_schedule(
        candles=_sample_candles(),
        snapshots=_sample_snapshots(),
        config=VectorBTResearchConfig(adx_min=20.0),
    )
    assert "entry" in schedule.columns
    assert "exit" in schedule.columns
    assert bool(schedule["entry"].any())
    assert bool(schedule["exit"].any())


def test_run_vectorbt_research_returns_metrics():
    result = run_vectorbt_research(
        candles=_sample_candles(),
        snapshots=_sample_snapshots(),
        config=VectorBTResearchConfig(adx_min=20.0),
    )
    assert "total_return_pct" in result.metrics
    assert len(result.schedule) == 30
    assert not result.equity_curve.empty


def test_run_vectorbt_walk_forward_returns_fold_summary():
    folds, summary = run_vectorbt_walk_forward(
        candles=_sample_candles(),
        snapshots=_sample_snapshots(),
        config=VectorBTResearchConfig(adx_min=20.0),
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 30),
        train_months=1,
        test_months=1,
        step_months=1,
    )
    assert "folds" in summary
    assert isinstance(folds, pd.DataFrame)


def test_run_vectorbt_walk_forward_empty_has_schema_columns():
    folds, summary = run_vectorbt_walk_forward(
        candles=_sample_candles(),
        snapshots=_sample_snapshots(),
        config=VectorBTResearchConfig(adx_min=20.0),
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 30),
        train_months=12,
        test_months=3,
        step_months=3,
    )
    assert folds.empty
    assert "fold" in folds.columns
    assert "total_return_pct" in folds.columns
    assert summary["folds"] == 0.0


def test_run_hybrid_from_schedule_reuses_engine_metrics():
    schedule = build_vectorbt_schedule(
        candles=_sample_candles(),
        snapshots=_sample_snapshots(),
        config=VectorBTResearchConfig(adx_min=20.0),
    )
    metrics = run_hybrid_from_schedule(
        candles=_sample_candles(),
        schedule=schedule,
        symbol="NIFTY",
        timeframe="1d",
        initial_capital=1000.0,
        thresholds=RegimeThresholds(),
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
    )
    assert "final_equity" in metrics


def test_run_hybrid_from_schedule_result_returns_artifacts():
    schedule = build_vectorbt_schedule(
        candles=_sample_candles(),
        snapshots=_sample_snapshots(),
        config=VectorBTResearchConfig(adx_min=20.0),
    )
    result = run_hybrid_from_schedule_result(
        candles=_sample_candles(),
        schedule=schedule,
        symbol="NIFTY",
        timeframe="1d",
        initial_capital=1000.0,
        thresholds=RegimeThresholds(),
        simulator=FillSimulator(slippage_pct=0.0, commission_per_order=0.0),
    )
    assert "final_equity" in result.metrics
    assert isinstance(result.equity_curve, pd.DataFrame)
    assert isinstance(result.fills, pd.DataFrame)


def test_run_vectorbt_sensitivity_returns_variants():
    out = run_vectorbt_sensitivity(
        candles=_sample_candles(),
        snapshots=_sample_snapshots(),
        base_config=VectorBTResearchConfig(adx_min=20.0, vix_max=18.0),
        multipliers=[0.8, 1.0, 1.2],
    )
    assert not out.empty
    assert "variant" in out.columns


def test_run_vectorbt_research_handles_irregular_index_frequency():
    candles = _sample_candles().iloc[[0, 1, 4, 7, 10]].reset_index(drop=True)
    snapshots = _sample_snapshots().iloc[[0, 1, 4, 7, 10]].reset_index(drop=True)
    result = run_vectorbt_research(
        candles=candles,
        snapshots=snapshots,
        config=VectorBTResearchConfig(adx_min=20.0, freq=None),
    )
    assert "sharpe_ratio" in result.metrics


def test_run_vectorbt_research_sets_nan_sharpe_when_no_trades():
    snapshots = _sample_snapshots().copy()
    snapshots["regime"] = "high_vol_choppy"
    snapshots["adx_14"] = 10.0
    result = run_vectorbt_research(
        candles=_sample_candles(),
        snapshots=snapshots,
        config=VectorBTResearchConfig(adx_min=25.0),
    )
    assert result.metrics["trades"] == 0.0
    assert math.isnan(result.metrics["sharpe_ratio"])
