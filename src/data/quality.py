"""Data quality checks for cached market datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data.calendar import NSECalendar, nse_calendar


@dataclass
class CandleQualityReport:
    rows: int
    missing_timestamps: int
    duplicate_timestamps: int
    invalid_ohlc_rows: int
    negative_or_zero_price_rows: int
    non_monotonic_timestamps: int
    missing_trading_days: int | None
    largest_gap_minutes: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "missing_timestamps": self.missing_timestamps,
            "duplicate_timestamps": self.duplicate_timestamps,
            "invalid_ohlc_rows": self.invalid_ohlc_rows,
            "negative_or_zero_price_rows": self.negative_or_zero_price_rows,
            "non_monotonic_timestamps": self.non_monotonic_timestamps,
            "missing_trading_days": self.missing_trading_days,
            "largest_gap_minutes": self.largest_gap_minutes,
        }


@dataclass
class CandleQualityThresholds:
    max_missing_timestamps: int = 0
    max_duplicate_timestamps: int = 0
    max_invalid_ohlc_rows: int = 0
    max_negative_or_zero_price_rows: int = 0
    max_non_monotonic_timestamps: int = 0
    max_missing_trading_days: int = 0
    max_largest_gap_minutes: float | None = None


@dataclass
class QualityGateResult:
    status: str
    issue_count: int
    violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issue_count": self.issue_count,
            "violations": self.violations,
        }


def _safe_minutes(delta: pd.Timedelta | None) -> float | None:
    if delta is None or pd.isna(delta):
        return None
    return round(float(delta.total_seconds()) / 60.0, 2)


def assess_candle_quality(
    df: pd.DataFrame,
    timeframe: str,
    *,
    calendar: NSECalendar | None = None,
) -> CandleQualityReport:
    if df.empty:
        return CandleQualityReport(
            rows=0,
            missing_timestamps=0,
            duplicate_timestamps=0,
            invalid_ohlc_rows=0,
            negative_or_zero_price_rows=0,
            non_monotonic_timestamps=0,
            missing_trading_days=None,
            largest_gap_minutes=None,
        )

    data = df.copy()
    rows = len(data)

    if "timestamp" not in data.columns:
        raise ValueError("Candle dataset requires 'timestamp' column for quality checks.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    missing_timestamps = int(data["timestamp"].isna().sum())
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")

    duplicate_timestamps = int(data["timestamp"].duplicated(keep=False).sum())

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required OHLC column: {col}")

    invalid_ohlc_rows = int(
        (
            (data["high"] < data[["open", "close"]].max(axis=1))
            | (data["low"] > data[["open", "close"]].min(axis=1))
        ).sum()
    )

    negative_or_zero_price_rows = int(
        (data[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    )

    # If sorting changes order materially, original data was non-monotonic.
    non_monotonic_timestamps = int(
        (pd.to_datetime(df["timestamp"], errors="coerce").diff() < pd.Timedelta(0)).sum()
    )

    cal = calendar or nse_calendar
    largest_gap_minutes = _largest_gap_minutes(
        timestamps=data["timestamp"],
        timeframe=timeframe,
        calendar=cal,
    )

    missing_trading_days: int | None = None
    if timeframe.lower() == "1d" and len(data) > 1:
        start_day = data["timestamp"].min().date()
        end_day = data["timestamp"].max().date()
        expected = set(cal.trading_days_between(start=start_day, end=end_day))
        observed = {ts.date() for ts in data["timestamp"].dt.normalize().unique()}
        missing_trading_days = int(len(expected.difference(observed)))

    return CandleQualityReport(
        rows=rows,
        missing_timestamps=missing_timestamps,
        duplicate_timestamps=duplicate_timestamps,
        invalid_ohlc_rows=invalid_ohlc_rows,
        negative_or_zero_price_rows=negative_or_zero_price_rows,
        non_monotonic_timestamps=non_monotonic_timestamps,
        missing_trading_days=missing_trading_days,
        largest_gap_minutes=largest_gap_minutes,
    )


def _largest_gap_minutes(
    *,
    timestamps: pd.Series,
    timeframe: str,
    calendar: NSECalendar,
) -> float | None:
    if timestamps.empty:
        return None
    tf = timeframe.lower().strip()

    if tf != "1d":
        deltas = timestamps.diff().dropna()
        return _safe_minutes(deltas.max() if not deltas.empty else None)

    observed_days = sorted({ts.date() for ts in timestamps.dt.normalize().dropna()})
    if len(observed_days) <= 1:
        return None

    # Trading-day-aware gap:
    # contiguous trading sessions => 1 day gap (1440m), regardless of weekends/holidays.
    largest = 0.0
    for prev_day, curr_day in zip(observed_days, observed_days[1:], strict=False):
        trading_days = calendar.trading_days_between(start=prev_day, end=curr_day)
        trading_steps = max(len(trading_days) - 1, 0)
        gap_minutes = float(trading_steps) * 1440.0
        if gap_minutes > largest:
            largest = gap_minutes

    return largest if largest > 0 else None


def summarize_issue_count(report: CandleQualityReport) -> int:
    fields = [
        report.missing_timestamps,
        report.duplicate_timestamps,
        report.invalid_ohlc_rows,
        report.negative_or_zero_price_rows,
        report.non_monotonic_timestamps,
        report.missing_trading_days or 0,
    ]
    return int(sum(fields))


def evaluate_quality_gate(
    report: CandleQualityReport,
    thresholds: CandleQualityThresholds,
) -> QualityGateResult:
    violations: list[str] = []

    checks = [
        ("missing_timestamps", report.missing_timestamps, thresholds.max_missing_timestamps),
        ("duplicate_timestamps", report.duplicate_timestamps, thresholds.max_duplicate_timestamps),
        ("invalid_ohlc_rows", report.invalid_ohlc_rows, thresholds.max_invalid_ohlc_rows),
        (
            "negative_or_zero_price_rows",
            report.negative_or_zero_price_rows,
            thresholds.max_negative_or_zero_price_rows,
        ),
        (
            "non_monotonic_timestamps",
            report.non_monotonic_timestamps,
            thresholds.max_non_monotonic_timestamps,
        ),
        (
            "missing_trading_days",
            report.missing_trading_days or 0,
            thresholds.max_missing_trading_days,
        ),
    ]

    for label, value, limit in checks:
        if value > limit:
            violations.append(f"{label}={value} > {limit}")

    if (
        thresholds.max_largest_gap_minutes is not None
        and (report.largest_gap_minutes or 0.0) > thresholds.max_largest_gap_minutes
    ):
        violations.append(
            "largest_gap_minutes="
            f"{report.largest_gap_minutes} > {thresholds.max_largest_gap_minutes}"
        )

    issue_count = summarize_issue_count(report)
    status = "ok" if not violations else "failed_thresholds"
    return QualityGateResult(status=status, issue_count=issue_count, violations=violations)


def thresholds_from_config(config: dict[str, Any]) -> CandleQualityThresholds:
    return CandleQualityThresholds(
        max_missing_timestamps=int(config.get("max_missing_timestamps", 0)),
        max_duplicate_timestamps=int(config.get("max_duplicate_timestamps", 0)),
        max_invalid_ohlc_rows=int(config.get("max_invalid_ohlc_rows", 0)),
        max_negative_or_zero_price_rows=int(config.get("max_negative_or_zero_price_rows", 0)),
        max_non_monotonic_timestamps=int(config.get("max_non_monotonic_timestamps", 0)),
        max_missing_trading_days=int(config.get("max_missing_trading_days", 0)),
        max_largest_gap_minutes=(
            float(config["max_largest_gap_minutes"])
            if config.get("max_largest_gap_minutes") is not None
            else None
        ),
    )
