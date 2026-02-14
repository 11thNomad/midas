"""Promotion-gate evaluation for vectorbt parameter-set candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class VectorBTPromotionGateConfig:
    enabled: bool = True
    min_trades_mean: float = 5.0
    min_metric_worst: float = 0.0
    min_metric_mean: float = 0.0
    max_drawdown_abs_worst: float = 12.0
    min_eligible_profile_share: float = 1.0


def parse_vectorbt_promotion_gate_config(
    backtest_cfg: dict[str, Any],
) -> VectorBTPromotionGateConfig:
    raw = backtest_cfg.get("vectorbt_promotion_gate", {})
    if not isinstance(raw, dict):
        raise ValueError("backtest.vectorbt_promotion_gate must be an object.")
    return VectorBTPromotionGateConfig(
        enabled=bool(raw.get("enabled", True)),
        min_trades_mean=float(raw.get("min_trades_mean", 5.0) or 5.0),
        min_metric_worst=float(raw.get("min_metric_worst", 0.0) or 0.0),
        min_metric_mean=float(raw.get("min_metric_mean", 0.0) or 0.0),
        max_drawdown_abs_worst=float(raw.get("max_drawdown_abs_worst", 12.0) or 12.0),
        min_eligible_profile_share=float(raw.get("min_eligible_profile_share", 1.0) or 1.0),
    )


def evaluate_vectorbt_promotion_gate(
    robustness: pd.DataFrame,
    config: VectorBTPromotionGateConfig,
) -> pd.DataFrame:
    if robustness.empty:
        return robustness

    out = robustness.copy()
    numeric_cols = [
        "trades_mean",
        "metric_worst",
        "metric_mean",
        "drawdown_abs_worst",
        "eligible_profile_share",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    checks = [
        ("pass_trades", out["trades_mean"] >= config.min_trades_mean, "trades_mean"),
        ("pass_metric_worst", out["metric_worst"] >= config.min_metric_worst, "metric_worst"),
        ("pass_metric_mean", out["metric_mean"] >= config.min_metric_mean, "metric_mean"),
        (
            "pass_drawdown",
            out["drawdown_abs_worst"] <= config.max_drawdown_abs_worst,
            "drawdown_abs_worst",
        ),
        (
            "pass_eligible_share",
            out["eligible_profile_share"] >= config.min_eligible_profile_share,
            "eligible_profile_share",
        ),
    ]
    for col, series, _ in checks:
        out[col] = series.fillna(False)

    out["promotion_pass"] = (
        out["pass_trades"]
        & out["pass_metric_worst"]
        & out["pass_metric_mean"]
        & out["pass_drawdown"]
        & out["pass_eligible_share"]
    )

    reasons: list[str] = []
    for _, row in out.iterrows():
        failed: list[str] = []
        if not bool(row["pass_trades"]):
            failed.append(f"trades<{config.min_trades_mean:g}")
        if not bool(row["pass_metric_worst"]):
            failed.append(f"worst<{config.min_metric_worst:g}")
        if not bool(row["pass_metric_mean"]):
            failed.append(f"mean<{config.min_metric_mean:g}")
        if not bool(row["pass_drawdown"]):
            failed.append(f"dd>{config.max_drawdown_abs_worst:g}")
        if not bool(row["pass_eligible_share"]):
            failed.append(f"eligible<{config.min_eligible_profile_share:g}")
        reasons.append("" if not failed else "; ".join(failed))
    out["promotion_fail_reasons"] = reasons

    return out.sort_values(
        by=["promotion_pass", "metric_worst", "metric_mean", "trades_mean", "set_id"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
