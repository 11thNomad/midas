from __future__ import annotations

import pandas as pd

from src.backtest.vectorbt_parameter_sets import (
    VectorBTParameterSet,
    apply_parameter_set,
    parse_parameter_sets,
    rank_parameter_results,
)
from src.backtest.vectorbt_research import VectorBTResearchConfig


def test_parse_parameter_sets_from_root_structure():
    sets = parse_parameter_sets(
        {
            "parameter_sets": [
                {
                    "id": "base",
                    "entry_regimes": ["low_vol_trending", "high_vol_trending"],
                    "adx_min": 25.0,
                    "vix_max": 18.0,
                }
            ]
        }
    )
    assert len(sets) == 1
    assert sets[0].set_id == "base"
    assert sets[0].entry_regimes == ("low_vol_trending", "high_vol_trending")
    assert sets[0].adx_min == 25.0
    assert sets[0].vix_max == 18.0


def test_parse_parameter_sets_from_nested_vectorbt_structure():
    sets = parse_parameter_sets(
        {
            "vectorbt": {
                "parameter_sets": [
                    {
                        "id": "alt",
                        "entry_regimes": "low_vol_trending,high_vol_trending",
                        "adx_min": 22.0,
                    }
                ]
            }
        }
    )
    assert len(sets) == 1
    assert sets[0].set_id == "alt"
    assert sets[0].entry_regimes == ("low_vol_trending", "high_vol_trending")
    assert sets[0].vix_max is None


def test_apply_parameter_set_overrides_strategy_gate_fields():
    base = VectorBTResearchConfig(
        initial_cash=1000.0,
        fees_pct=0.001,
        slippage_pct=0.002,
        freq="1D",
    )
    params = VectorBTParameterSet(
        set_id="x",
        entry_regimes=("low_vol_ranging",),
        adx_min=19.0,
        vix_max=16.0,
    )
    cfg = apply_parameter_set(base, params)
    assert cfg.initial_cash == 1000.0
    assert cfg.fees_pct == 0.001
    assert cfg.slippage_pct == 0.002
    assert cfg.entry_regimes == ("low_vol_ranging",)
    assert cfg.adx_min == 19.0
    assert cfg.vix_max == 16.0
    assert cfg.freq == "1D"


def test_rank_parameter_results_applies_filters_and_ranking():
    df = pd.DataFrame(
        [
            {
                "set_id": "a",
                "total_return_pct": 8.0,
                "sharpe_ratio": 1.0,
                "max_drawdown_pct": -10.0,
                "trades": 6.0,
            },
            {
                "set_id": "b",
                "total_return_pct": 12.0,
                "sharpe_ratio": 1.1,
                "max_drawdown_pct": -15.0,
                "trades": 2.0,
            },
            {
                "set_id": "c",
                "total_return_pct": 6.0,
                "sharpe_ratio": 0.8,
                "max_drawdown_pct": -5.0,
                "trades": 7.0,
            },
        ]
    )
    ranked = rank_parameter_results(
        df,
        rank_by="total_return_pct",
        min_trades=3.0,
        max_drawdown_pct=12.0,
    )
    assert list(ranked["set_id"])[:2] == ["a", "c"]
    assert bool(ranked.loc[ranked["set_id"] == "b", "eligible"].iloc[0]) is False


def test_rank_parameter_results_sorts_drawdown_by_absolute_value():
    df = pd.DataFrame(
        [
            {"set_id": "a", "max_drawdown_pct": -12.0, "trades": 5.0},
            {"set_id": "b", "max_drawdown_pct": -4.0, "trades": 5.0},
            {"set_id": "c", "max_drawdown_pct": -8.0, "trades": 5.0},
        ]
    )
    ranked = rank_parameter_results(df, rank_by="max_drawdown_pct", min_trades=1.0)
    assert list(ranked["set_id"]) == ["b", "c", "a"]
