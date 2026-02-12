"""Walk-forward window generation and result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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


def aggregate_walk_forward_metrics(fold_metrics: list[dict]) -> dict[str, float]:
    if not fold_metrics:
        return {"folds": 0.0}

    frame = pd.DataFrame(fold_metrics)
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    summary: dict[str, float] = {"folds": float(len(frame))}
    for col in numeric_cols:
        summary[f"{col}_mean"] = float(frame[col].mean())
        summary[f"{col}_median"] = float(frame[col].median())
    return summary
