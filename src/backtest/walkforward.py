"""Walk-forward window generation and result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def generate_walk_forward_windows(
    *,
    start: datetime,
    end: datetime,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[WalkForwardWindow]:
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train_months, test_months, step_months must be positive.")
    if start >= end:
        return []

    windows: list[WalkForwardWindow] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_ts:
            break

        windows.append(
            WalkForwardWindow(
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
            )
        )
        cursor = cursor + pd.DateOffset(months=step_months)
        if cursor >= end_ts:
            break

    return windows


def aggregate_walk_forward_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not fold_metrics:
        return {"folds": 0.0}

    frame = pd.DataFrame(fold_metrics)
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if numeric_cols:
        frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    summary: dict[str, float] = {"folds": float(len(frame))}
    for col in numeric_cols:
        summary[f"{col}_mean"] = float(frame[col].mean())
        summary[f"{col}_median"] = float(frame[col].median())
    return summary


def build_sensitivity_variants(
    *,
    base_config: dict[str, Any],
    params: list[str],
    multipliers: list[float],
) -> list[dict[str, Any]]:
    """Build parameter-perturbed strategy config overrides."""
    variants: list[dict[str, Any]] = []
    seen: set[tuple[str, float | int]] = set()
    for param in params:
        if param not in base_config:
            continue
        base_value = base_config[param]
        if isinstance(base_value, bool):
            continue
        if not isinstance(base_value, (int, float)):
            continue
        for multiplier in multipliers:
            m = float(multiplier)
            if abs(m - 1.0) < 1e-12:
                continue
            if isinstance(base_value, int):
                candidate: float | int = max(1, int(round(base_value * m)))
            else:
                candidate = float(base_value) * m
            key = (param, candidate)
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "variant_id": f"{param}_x{m:.2f}",
                    "param": param,
                    "multiplier": m,
                    "base_value": float(base_value),
                    "new_value": float(candidate),
                    "overrides": {param: candidate},
                }
            )
    return variants


def summarize_sensitivity_results(
    *,
    variant_rows: list[dict[str, Any]],
    base_total_return_pct: float,
) -> dict[str, float]:
    """Summarize parameter sensitivity outcomes into scalar diagnostics."""
    if not variant_rows:
        return {
            "variant_count": 0.0,
            "positive_return_share": 0.0,
            "median_return_pct": 0.0,
            "worst_return_pct": 0.0,
            "robustness_score": 0.0,
        }

    frame = pd.DataFrame(variant_rows)
    returns = pd.to_numeric(
        frame.get("total_return_pct", pd.Series(dtype="float64")), errors="coerce"
    ).dropna()
    if returns.empty:
        return {
            "variant_count": float(len(frame)),
            "positive_return_share": 0.0,
            "median_return_pct": 0.0,
            "worst_return_pct": 0.0,
            "robustness_score": 0.0,
        }

    positive_share = float((returns > 0.0).mean())
    median_return = float(returns.median())
    worst_return = float(returns.min())
    if base_total_return_pct > 0:
        retention = max(0.0, min(1.0, median_return / base_total_return_pct))
        robustness = positive_share * retention
    else:
        robustness = 0.0

    return {
        "variant_count": float(len(returns)),
        "positive_return_share": positive_share,
        "median_return_pct": median_return,
        "worst_return_pct": worst_return,
        "robustness_score": float(robustness),
    }


def aggregate_cross_instrument_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-symbol backtest rows into cross-instrument summaries."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.empty or "strategy" not in frame.columns or "symbol" not in frame.columns:
        return pd.DataFrame()

    metric_cols = [
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "anti_overfit_pass",
    ]
    for c in metric_cols:
        if c not in frame.columns:
            frame[c] = pd.NA
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    grouped = frame.groupby("strategy", as_index=False).agg(
        symbols_tested=("symbol", "nunique"),
        symbols_positive_return=("total_return_pct", lambda s: int((s > 0).sum())),
        total_return_pct_mean=("total_return_pct", "mean"),
        total_return_pct_std=("total_return_pct", "std"),
        sharpe_ratio_mean=("sharpe_ratio", "mean"),
        max_drawdown_pct_mean=("max_drawdown_pct", "mean"),
        anti_overfit_pass_share=("anti_overfit_pass", "mean"),
    )
    grouped["symbols_positive_return_share"] = grouped["symbols_positive_return"] / grouped[
        "symbols_tested"
    ].replace(0, pd.NA)
    return grouped.sort_values("strategy").reset_index(drop=True)
