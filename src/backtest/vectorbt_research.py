"""VectorBT research layer using frozen signal snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import pandas as pd

from src.backtest.hybrid import HybridConfig, run_hybrid_schedule_backtest
from src.backtest.simulator import FillSimulator
from src.backtest.walkforward import aggregate_walk_forward_metrics, generate_walk_forward_windows
from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.signals.contracts import frame_from_signal_snapshots, signal_snapshot_from_mapping
from src.signals.pipeline import build_feature_context


@dataclass(frozen=True)
class VectorBTResearchConfig:
    initial_cash: float = 150_000.0
    fees_pct: float = 0.0002
    slippage_pct: float = 0.0005
    entry_regimes: tuple[str, ...] = ("low_vol_trending", "high_vol_trending")
    adx_min: float = 25.0
    vix_max: float | None = None
    freq: str | None = None


@dataclass
class VectorBTResearchResult:
    schedule: pd.DataFrame
    metrics: dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


def build_snapshots_from_market_data(
    *,
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    thresholds: RegimeThresholds,
    vix_df: pd.DataFrame | None = None,
    fii_df: pd.DataFrame | None = None,
    usdinr_df: pd.DataFrame | None = None,
    option_chain_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if candles.empty:
        return pd.DataFrame()

    bars = candles.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars = bars.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    vix = _prep_timeframe_df(vix_df, timestamp_col="timestamp")
    fii = _prep_timeframe_df(fii_df, timestamp_col="date")
    usdinr = _prep_timeframe_df(usdinr_df, timestamp_col="timestamp")
    chain = _prep_timeframe_df(option_chain_df, timestamp_col="timestamp")

    classifier = RegimeClassifier(thresholds=thresholds)
    previous_chain: pd.DataFrame | None = None
    rows = []
    for i in range(len(bars)):
        ts = pd.Timestamp(bars.iloc[i]["timestamp"]).to_pydatetime()
        candles_hist = bars.iloc[:i]
        vix_hist = vix.loc[vix["timestamp"] < pd.Timestamp(ts)] if not vix.empty else pd.DataFrame()
        fii_hist = fii.loc[fii["date"] < pd.Timestamp(ts)] if not fii.empty else pd.DataFrame()
        usdinr_hist = (
            usdinr.loc[usdinr["timestamp"] < pd.Timestamp(ts)]
            if not usdinr.empty
            else pd.DataFrame()
        )
        chain_asof = _latest_chain_asof(chain, ts)
        vix_series = vix_hist["close"].astype("float64") if not vix_hist.empty else None
        vix_value = (
            float(vix_series.iloc[-1])
            if vix_series is not None and not vix_series.empty
            else 0.0
        )
        snapshot, regime_signals = build_feature_context(
            timestamp=ts,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_hist,
            vix_value=vix_value,
            vix_series=vix_series,
            chain_df=chain_asof,
            previous_chain_df=previous_chain,
            fii_df=fii_hist,
            usdinr_close=(
                usdinr_hist["close"].astype("float64")
                if not usdinr_hist.empty and "close" in usdinr_hist.columns
                else None
            ),
            regime=classifier.current_regime.value,
            thresholds=thresholds,
            source="vectorbt_research",
        )
        regime = classifier.classify(regime_signals)
        rows.append({**snapshot.__dict__, "regime": regime.value})
        if chain_asof is not None and not chain_asof.empty:
            previous_chain = chain_asof

    if not rows:
        return pd.DataFrame()
    dtos = [signal_snapshot_from_mapping(row) for row in rows]
    return frame_from_signal_snapshots(dtos)


def build_vectorbt_schedule(
    *,
    candles: pd.DataFrame,
    snapshots: pd.DataFrame,
    config: VectorBTResearchConfig,
) -> pd.DataFrame:
    bars = candles[["timestamp", "close"]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    snap_cols = ["timestamp", "regime", "adx_14", "vix_level"]
    available = [c for c in snap_cols if c in snapshots.columns]
    snap = snapshots[available].copy() if available else pd.DataFrame(columns=snap_cols)
    if "timestamp" in snap.columns:
        snap["timestamp"] = pd.to_datetime(snap["timestamp"], errors="coerce")
    else:
        snap["timestamp"] = pd.NaT
    snap = (
        snap.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
    )

    merged = bars.merge(snap, on="timestamp", how="left")
    merged["regime"] = merged["regime"].fillna("unknown")
    merged["adx_14"] = pd.to_numeric(merged["adx_14"], errors="coerce").fillna(0.0)
    merged["vix_level"] = pd.to_numeric(merged["vix_level"], errors="coerce").fillna(0.0)

    entry_gate = merged["regime"].isin(config.entry_regimes) & (merged["adx_14"] >= config.adx_min)
    if config.vix_max is not None:
        entry_gate = entry_gate & (merged["vix_level"] <= float(config.vix_max))

    entries = entry_gate & (~entry_gate.shift(1, fill_value=False))
    exits = (~entry_gate) & (entry_gate.shift(1, fill_value=False))

    return pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "close": merged["close"],
            "entry": entries.astype(bool),
            "exit": exits.astype(bool),
            "regime": merged["regime"],
            "adx_14": merged["adx_14"],
            "vix_level": merged["vix_level"],
        }
    )


def run_vectorbt_research(
    *,
    candles: pd.DataFrame,
    snapshots: pd.DataFrame,
    config: VectorBTResearchConfig,
) -> VectorBTResearchResult:
    schedule = build_vectorbt_schedule(candles=candles, snapshots=snapshots, config=config)
    if schedule.empty:
        return VectorBTResearchResult(
            schedule=schedule,
            metrics={"bars": 0.0, "trades": 0.0, "total_return_pct": 0.0},
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame(),
        )

    try:
        import vectorbt as vbt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("vectorbt is required for research runs.") from exc

    indexed = schedule.set_index("timestamp")
    close = indexed["close"].astype("float64")
    entries = indexed["entry"].astype(bool)
    exits = indexed["exit"].astype(bool)
    freq = _resolve_freq(indexed.index, preferred=config.freq)
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=float(config.initial_cash),
        fees=float(config.fees_pct),
        slippage=float(config.slippage_pct),
        freq=freq,
    )

    equity = portfolio.value()
    trades = portfolio.trades.records_readable
    metrics = {
        "bars": float(len(indexed)),
        "trades": float(_as_float(portfolio.trades.count())),
        "total_return_pct": float(_as_float(portfolio.total_return()) * 100.0),
        "sharpe_ratio": float(_as_float(portfolio.sharpe_ratio())),
        "max_drawdown_pct": float(_as_float(portfolio.max_drawdown()) * 100.0),
    }
    equity_curve = (
        pd.DataFrame({"timestamp": equity.index, "equity": equity.values}).reset_index(drop=True)
    )
    return VectorBTResearchResult(
        schedule=schedule,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
    )


def run_vectorbt_walk_forward(
    *,
    candles: pd.DataFrame,
    snapshots: pd.DataFrame,
    config: VectorBTResearchConfig,
    start: datetime,
    end: datetime,
    train_months: int,
    test_months: int,
    step_months: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    windows = generate_walk_forward_windows(
        start=start,
        end=end,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )
    rows = []
    for i, window in enumerate(windows, start=1):
        candle_fold = _slice_on_timestamp(candles, start=window.test_start, end=window.test_end)
        snap_fold = _slice_on_timestamp(snapshots, start=window.test_start, end=window.test_end)
        result = run_vectorbt_research(candles=candle_fold, snapshots=snap_fold, config=config)
        rows.append(
            {
                "fold": i,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                **result.metrics,
            }
        )
    folds = pd.DataFrame(rows)
    summary = aggregate_walk_forward_metrics(rows)
    return folds, summary


def run_vectorbt_sensitivity(
    *,
    candles: pd.DataFrame,
    snapshots: pd.DataFrame,
    base_config: VectorBTResearchConfig,
    multipliers: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    seen: set[tuple[str, float]] = set()
    for m in multipliers:
        mult = float(m)
        adx_candidate = max(1.0, float(base_config.adx_min) * mult)
        key = ("adx_min", round(adx_candidate, 6))
        if key not in seen:
            seen.add(key)
            cfg = VectorBTResearchConfig(
                initial_cash=base_config.initial_cash,
                fees_pct=base_config.fees_pct,
                slippage_pct=base_config.slippage_pct,
                entry_regimes=base_config.entry_regimes,
                adx_min=adx_candidate,
                vix_max=base_config.vix_max,
                freq=base_config.freq,
            )
            out = run_vectorbt_research(candles=candles, snapshots=snapshots, config=cfg)
            rows.append(
                {
                    "variant": f"adx_min_x{mult:.2f}",
                    "adx_min": adx_candidate,
                    "vix_max": (
                        float(base_config.vix_max)
                        if base_config.vix_max is not None
                        else float("nan")
                    ),
                    "total_return_pct": float(out.metrics.get("total_return_pct", 0.0)),
                    "sharpe_ratio": float(out.metrics.get("sharpe_ratio", 0.0)),
                    "max_drawdown_pct": float(out.metrics.get("max_drawdown_pct", 0.0)),
                }
            )

        if base_config.vix_max is None:
            continue
        vix_candidate = max(1.0, float(base_config.vix_max) * mult)
        key = ("vix_max", round(vix_candidate, 6))
        if key in seen:
            continue
        seen.add(key)
        cfg = VectorBTResearchConfig(
            initial_cash=base_config.initial_cash,
            fees_pct=base_config.fees_pct,
            slippage_pct=base_config.slippage_pct,
            entry_regimes=base_config.entry_regimes,
            adx_min=base_config.adx_min,
            vix_max=vix_candidate,
            freq=base_config.freq,
        )
        out = run_vectorbt_research(candles=candles, snapshots=snapshots, config=cfg)
        rows.append(
            {
                "variant": f"vix_max_x{mult:.2f}",
                "adx_min": float(base_config.adx_min),
                "vix_max": vix_candidate,
                "total_return_pct": float(out.metrics.get("total_return_pct", 0.0)),
                "sharpe_ratio": float(out.metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown_pct": float(out.metrics.get("max_drawdown_pct", 0.0)),
            }
        )

    return pd.DataFrame(rows)


def run_hybrid_from_schedule(
    *,
    candles: pd.DataFrame,
    schedule: pd.DataFrame,
    symbol: str,
    timeframe: str,
    initial_capital: float,
    thresholds: RegimeThresholds,
    simulator: FillSimulator,
    vix_df: pd.DataFrame | None = None,
    fii_df: pd.DataFrame | None = None,
    usdinr_df: pd.DataFrame | None = None,
    option_chain_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    result = run_hybrid_schedule_backtest(
        candles=candles,
        schedule=schedule,
        config=HybridConfig(
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=initial_capital,
            max_lots=1,
        ),
        simulator=simulator,
        thresholds=thresholds,
        vix_df=vix_df,
        fii_df=fii_df,
        usdinr_df=usdinr_df,
        option_chain_df=option_chain_df,
    )
    return result.metrics


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Series):
        return float(value.iloc[-1]) if not value.empty else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _prep_timeframe_df(df: pd.DataFrame | None, *, timestamp_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if timestamp_col not in out.columns:
        return pd.DataFrame()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    return out.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)


def _latest_chain_asof(chain_df: pd.DataFrame, ts: datetime) -> pd.DataFrame | None:
    if chain_df.empty:
        return None
    eligible = chain_df.loc[chain_df["timestamp"] < pd.Timestamp(ts)]
    if eligible.empty:
        return None
    latest_ts = eligible["timestamp"].max()
    out = eligible.loc[eligible["timestamp"] == latest_ts].copy().reset_index(drop=True)
    return cast(pd.DataFrame, out)


def _slice_on_timestamp(df: pd.DataFrame, *, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "timestamp" not in out.columns:
        return pd.DataFrame()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.loc[
        (out["timestamp"] >= pd.Timestamp(start)) & (out["timestamp"] < pd.Timestamp(end))
    ].reset_index(drop=True)


def _resolve_freq(index: pd.Index, *, preferred: str | None) -> str:
    if preferred:
        return preferred
    if isinstance(index, pd.DatetimeIndex):
        inferred = pd.infer_freq(index)
        if inferred:
            return str(inferred)
        if len(index) >= 2:
            deltas = index.to_series().diff().dropna()
            if not deltas.empty:
                median_delta = deltas.median()
                total_seconds = int(
                    round(float(median_delta / pd.Timedelta(seconds=1)))
                )
                if total_seconds <= 0:
                    return "1D"
                if total_seconds % 86_400 == 0:
                    return f"{total_seconds // 86_400}D"
                if total_seconds % 3_600 == 0:
                    return f"{total_seconds // 3_600}H"
                if total_seconds % 60 == 0:
                    return f"{total_seconds // 60}T"
                return f"{total_seconds}S"
    return "1D"
