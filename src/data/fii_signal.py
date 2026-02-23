"""FII consecutive-day signal computation from cached daily equity net flows."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.calendar import nse_calendar

_ANOMALOUS_ABS_LIMIT = 20_000.0


def _resolve_thresholds(settings: dict[str, Any]) -> tuple[float, float, int]:
    regime = settings.get("regime", {}) if isinstance(settings, dict) else {}
    bearish = float(regime.get("fii_bearish_daily_threshold", -1000.0))
    bullish = float(regime.get("fii_bullish_daily_threshold", 1000.0))
    consecutive_days = int(regime.get("fii_consecutive_days", 3))
    return bearish, bullish, max(1, consecutive_days)


def _load_cache_series(cache_path: str) -> pd.Series:
    path = Path(cache_path)
    if not path.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    expected_cols = {"date", "fii_equity_net_cr"}
    if not expected_cols.issubset(df.columns):
        return pd.Series(dtype=float)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["fii_equity_net_cr"] = pd.to_numeric(out["fii_equity_net_cr"], errors="coerce")
    out = out.dropna(subset=["date", "fii_equity_net_cr"])
    # Defensive cleanup: builder already filters these, but signal computation should
    # not consume clearly anomalous records if they leak in.
    out = out[out["fii_equity_net_cr"].abs() <= _ANOMALOUS_ABS_LIMIT]
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.set_index(out["date"].dt.date)["fii_equity_net_cr"].astype(float)


def _previous_trading_days(as_of_date: date, count: int) -> list[date]:
    out: list[date] = []
    cursor = as_of_date - timedelta(days=1)
    safety = 0
    while len(out) < count and safety < 3660:
        if nse_calendar.is_trading_day(cursor):
            out.append(cursor)
        cursor -= timedelta(days=1)
        safety += 1
    return out


def get_fii_signal(
    as_of_date: date,
    cache_path: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Return consecutive-day FII signal using strictly prior trading days only."""
    bearish_threshold, bullish_threshold, consecutive_days = _resolve_thresholds(settings)
    series = _load_cache_series(cache_path)
    prior_days = _previous_trading_days(as_of_date, consecutive_days)
    values = [series.get(day) for day in prior_days]

    data_complete = len(prior_days) == consecutive_days and all(v is not None for v in values)
    if not data_complete:
        return {
            "fii_t1": values[0] if len(values) > 0 and values[0] is not None else float("nan"),
            "fii_t2": values[1] if len(values) > 1 and values[1] is not None else float("nan"),
            "fii_t3": values[2] if len(values) > 2 and values[2] is not None else float("nan"),
            "fii_consecutive_negative": False,
            "fii_consecutive_positive": False,
            "fii_signal": "neutral",
            "data_complete": False,
        }

    recent = [float(v) for v in values]
    consecutive_negative = all(v < bearish_threshold for v in recent)
    consecutive_positive = all(v > bullish_threshold for v in recent)

    if consecutive_negative:
        signal = "bearish"
    elif consecutive_positive:
        signal = "bullish"
    else:
        signal = "neutral"

    return {
        "fii_t1": recent[0],
        "fii_t2": recent[1],
        "fii_t3": recent[2],
        "fii_consecutive_negative": consecutive_negative,
        "fii_consecutive_positive": consecutive_positive,
        "fii_signal": signal,
        "data_complete": True,
    }
