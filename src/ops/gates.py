"""Operational freshness gates for paper/live readiness checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.data.store import DataStore


@dataclass(frozen=True)
class FreshnessGate:
    name: str
    dataset: str
    symbol: str | None = None
    timeframe: str | None = None
    timestamp_col: str = "timestamp"
    max_age_minutes: int = 24 * 60
    required: bool = True
    severity: str = "error"  # error | warning


@dataclass(frozen=True)
class GateResult:
    name: str
    ok: bool
    required: bool
    severity: str
    dataset: str
    symbol: str | None
    timeframe: str | None
    timestamp_col: str
    max_age_minutes: int
    latest_timestamp: str | None
    age_minutes: float | None
    message: str


def check_freshness_gate(
    store: DataStore,
    gate: FreshnessGate,
    *,
    now: datetime,
    lookback_days: int = 30,
) -> GateResult:
    start = now - timedelta(days=max(lookback_days, 1))
    frame = store.read_time_series(
        gate.dataset,
        symbol=gate.symbol,
        timeframe=gate.timeframe,
        start=start,
        end=now,
        timestamp_col=gate.timestamp_col,
    )
    if frame.empty or gate.timestamp_col not in frame.columns:
        return GateResult(
            name=gate.name,
            ok=False,
            required=gate.required,
            severity=gate.severity,
            dataset=gate.dataset,
            symbol=gate.symbol,
            timeframe=gate.timeframe,
            timestamp_col=gate.timestamp_col,
            max_age_minutes=gate.max_age_minutes,
            latest_timestamp=None,
            age_minutes=None,
            message="No rows found in lookback window",
        )

    ts = pd.to_datetime(frame[gate.timestamp_col], errors="coerce").dropna()
    if ts.empty:
        return GateResult(
            name=gate.name,
            ok=False,
            required=gate.required,
            severity=gate.severity,
            dataset=gate.dataset,
            symbol=gate.symbol,
            timeframe=gate.timeframe,
            timestamp_col=gate.timestamp_col,
            max_age_minutes=gate.max_age_minutes,
            latest_timestamp=None,
            age_minutes=None,
            message="No valid timestamps after parsing",
        )

    latest = pd.Timestamp(ts.max()).to_pydatetime().replace(tzinfo=None)
    age_minutes = (now - latest).total_seconds() / 60.0
    ok = age_minutes <= float(gate.max_age_minutes)
    return GateResult(
        name=gate.name,
        ok=bool(ok),
        required=gate.required,
        severity=gate.severity,
        dataset=gate.dataset,
        symbol=gate.symbol,
        timeframe=gate.timeframe,
        timestamp_col=gate.timestamp_col,
        max_age_minutes=gate.max_age_minutes,
        latest_timestamp=latest.isoformat(),
        age_minutes=float(age_minutes),
        message="OK" if ok else "Data stale",
    )


def evaluate_freshness_gates(
    store: DataStore,
    gates: list[FreshnessGate],
    *,
    now: datetime,
) -> list[GateResult]:
    return [check_freshness_gate(store, gate, now=now) for gate in gates]


def summarize_gate_results(results: list[GateResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.ok)
    failed = total - passed
    hard_fail = sum(1 for r in results if (not r.ok) and r.required and r.severity == "error")
    warn_fail = sum(
        1 for r in results if (not r.ok) and (not r.required or r.severity == "warning")
    )
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "hard_failures": hard_fail,
        "warning_failures": warn_fail,
        "ok": hard_fail == 0,
        "results": [asdict(r) for r in results],
    }


def build_default_open_gates(
    settings: dict[str, Any], *, symbol: str, timeframe: str
) -> list[FreshnessGate]:
    market_cfg = settings.get("market", {})
    usdinr_symbol = str(market_cfg.get("usdinr_symbol", "USDINR")).upper()
    ops_cfg = settings.get("ops", {})
    freshness = ops_cfg.get("freshness", {})

    return [
        FreshnessGate(
            name="candles_primary_daily",
            dataset="candles",
            symbol=symbol,
            timeframe="1d",
            max_age_minutes=int(freshness.get("candles_1d_max_age_minutes", 3 * 24 * 60)),
        ),
        FreshnessGate(
            name="candles_primary_runtime_tf",
            dataset="candles",
            symbol=symbol,
            timeframe=timeframe,
            max_age_minutes=int(freshness.get("candles_runtime_max_age_minutes", 24 * 60)),
        ),
        FreshnessGate(
            name="vix_daily",
            dataset="vix",
            symbol="INDIAVIX",
            timeframe="1d",
            max_age_minutes=int(freshness.get("vix_1d_max_age_minutes", 3 * 24 * 60)),
        ),
        FreshnessGate(
            name="fii_daily",
            dataset="fii_dii",
            symbol="NSE",
            timeframe="1d",
            timestamp_col="date",
            max_age_minutes=int(freshness.get("fii_1d_max_age_minutes", 5 * 24 * 60)),
        ),
        FreshnessGate(
            name="usdinr_daily",
            dataset="candles",
            symbol=usdinr_symbol,
            timeframe="1d",
            max_age_minutes=int(freshness.get("usdinr_1d_max_age_minutes", 3 * 24 * 60)),
        ),
        FreshnessGate(
            name="signal_snapshots_recent",
            dataset="signal_snapshots",
            symbol=symbol,
            timeframe=timeframe,
            max_age_minutes=int(freshness.get("signal_snapshots_max_age_minutes", 7 * 24 * 60)),
            required=False,
            severity="warning",
        ),
    ]


def build_default_intraday_gates(
    settings: dict[str, Any], *, symbol: str, timeframe: str
) -> list[FreshnessGate]:
    ops_cfg = settings.get("ops", {})
    freshness = ops_cfg.get("freshness", {})
    return [
        FreshnessGate(
            name="signal_snapshots_intraday",
            dataset="signal_snapshots",
            symbol=symbol,
            timeframe=timeframe,
            max_age_minutes=int(freshness.get("intraday_signal_max_age_minutes", 60)),
            required=False,
            severity="warning",
        ),
        FreshnessGate(
            name="regime_snapshots_intraday",
            dataset="regime_snapshots",
            symbol=symbol,
            max_age_minutes=int(freshness.get("intraday_regime_max_age_minutes", 60)),
            required=False,
            severity="warning",
        ),
    ]
