from __future__ import annotations

import pandas as pd

from src.backtest.promotion_gate import (
    VectorBTPromotionGateConfig,
    evaluate_vectorbt_promotion_gate,
    parse_vectorbt_promotion_gate_config,
)


def test_parse_vectorbt_promotion_gate_config_defaults():
    cfg = parse_vectorbt_promotion_gate_config({})
    assert cfg.enabled is True
    assert cfg.min_trades_mean == 5.0
    assert cfg.max_drawdown_abs_worst == 12.0


def test_parse_vectorbt_promotion_gate_config_reads_values():
    cfg = parse_vectorbt_promotion_gate_config(
        {
            "vectorbt_promotion_gate": {
                "enabled": False,
                "min_trades_mean": 8,
                "min_metric_worst": 0.2,
                "min_metric_mean": 0.4,
                "max_drawdown_abs_worst": 9.5,
                "min_eligible_profile_share": 0.8,
            }
        }
    )
    assert cfg == VectorBTPromotionGateConfig(
        enabled=False,
        min_trades_mean=8.0,
        min_metric_worst=0.2,
        min_metric_mean=0.4,
        max_drawdown_abs_worst=9.5,
        min_eligible_profile_share=0.8,
    )


def test_evaluate_vectorbt_promotion_gate_marks_pass_and_fail_rows():
    robustness = pd.DataFrame(
        [
            {
                "set_id": "winner",
                "trades_mean": 8.0,
                "metric_worst": 0.7,
                "metric_mean": 1.2,
                "drawdown_abs_worst": 7.0,
                "eligible_profile_share": 1.0,
            },
            {
                "set_id": "loser",
                "trades_mean": 3.0,
                "metric_worst": -0.1,
                "metric_mean": 0.0,
                "drawdown_abs_worst": 20.0,
                "eligible_profile_share": 0.4,
            },
        ]
    )
    cfg = VectorBTPromotionGateConfig(
        enabled=True,
        min_trades_mean=5.0,
        min_metric_worst=0.0,
        min_metric_mean=0.1,
        max_drawdown_abs_worst=12.0,
        min_eligible_profile_share=1.0,
    )
    out = evaluate_vectorbt_promotion_gate(robustness, cfg)
    winner = out.loc[out["set_id"] == "winner"].iloc[0]
    loser = out.loc[out["set_id"] == "loser"].iloc[0]
    assert bool(winner["promotion_pass"]) is True
    assert bool(loser["promotion_pass"]) is False
    assert "trades<5" in str(loser["promotion_fail_reasons"])


def test_evaluate_vectorbt_promotion_gate_sorts_pass_rows_first():
    robustness = pd.DataFrame(
        [
            {
                "set_id": "b",
                "trades_mean": 7.0,
                "metric_worst": 0.2,
                "metric_mean": 0.5,
                "drawdown_abs_worst": 8.0,
                "eligible_profile_share": 1.0,
            },
            {
                "set_id": "a",
                "trades_mean": 7.0,
                "metric_worst": 0.3,
                "metric_mean": 0.6,
                "drawdown_abs_worst": 7.0,
                "eligible_profile_share": 1.0,
            },
            {
                "set_id": "x",
                "trades_mean": 1.0,
                "metric_worst": -1.0,
                "metric_mean": -1.0,
                "drawdown_abs_worst": 99.0,
                "eligible_profile_share": 0.0,
            },
        ]
    )
    cfg = VectorBTPromotionGateConfig()
    out = evaluate_vectorbt_promotion_gate(robustness, cfg)
    assert list(out["set_id"])[:2] == ["a", "b"]
    assert bool(out.iloc[2]["promotion_pass"]) is False
