"""Dry replay of historical regimes using strict no-lookahead processing."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.store import DataStore
from src.regime import (
    RegimeClassifier,
    RegimeSnapshotStore,
    RegimeThresholds,
    StrategyTransitionStore,
)
from src.regime.replay import replay_regimes_no_lookahead


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay historical regimes from cached datasets.")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol partition")
    parser.add_argument("--timeframe", default="1d", help="Candle timeframe partition")
    parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--indicator-warmup-days",
        type=int,
        default=0,
        help="Extra days loaded before --start for indicator warmup (excluded from output).",
    )
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    parser.add_argument(
        "--persist", action="store_true", help="Persist replay snapshots/transitions into cache"
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text())


def resolve_windows(
    args: argparse.Namespace,
) -> tuple[datetime | None, datetime | None, datetime | None]:
    analysis_start = args.start
    load_start = args.start
    if args.start is not None and args.indicator_warmup_days > 0:
        load_start = args.start - timedelta(days=args.indicator_warmup_days)
    return load_start, analysis_start, args.end


def _regime_durations(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or not {"timestamp", "regime"}.issubset(snapshots.columns):
        return pd.DataFrame(columns=["regime", "avg_duration_bars", "segments"])
    out = snapshots.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return pd.DataFrame(columns=["regime", "avg_duration_bars", "segments"])
    out["segment"] = (out["regime"] != out["regime"].shift(1)).cumsum()
    seg = out.groupby(["segment", "regime"], as_index=False).size().rename(columns={"size": "bars"})
    agg = seg.groupby("regime", as_index=False).agg(
        avg_duration_bars=("bars", "mean"),
        segments=("segment", "count"),
    )
    return agg.sort_values("regime").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    load_start, analysis_start, end = resolve_windows(args)
    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")

    store = DataStore(base_dir=str(cache_dir))
    classifier = RegimeClassifier(
        thresholds=RegimeThresholds.from_config(settings.get("regime", {}))
    )

    candles = store.read_time_series(
        "candles",
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=load_start,
        end=end,
    )
    if candles.empty:
        print("No candle data available for requested window.")
        return 1

    vix = store.read_time_series(
        "vix", symbol="INDIAVIX", timeframe="1d", start=load_start, end=end
    )
    fii = store.read_time_series(
        "fii_dii",
        symbol="NSE",
        timeframe="1d",
        start=load_start,
        end=end,
        timestamp_col="date",
    )

    result = replay_regimes_no_lookahead(
        candles=candles,
        classifier=classifier,
        vix_df=vix,
        fii_df=fii,
        analysis_start=analysis_start,
    )
    snapshots = result.snapshots
    transitions = result.transitions

    print("=" * 72)
    print("Regime Dry Replay")
    print("=" * 72)
    print(f"symbol={args.symbol} timeframe={args.timeframe}")
    load_window_start = load_start.date() if load_start else "begin"
    analysis_window_start = analysis_start.date() if analysis_start else "begin"
    window_end = end.date() if end else "latest"
    print(f"load_window={load_window_start} -> {window_end}")
    print(f"analysis_window={analysis_window_start} -> {window_end}")
    print(f"indicator_warmup_days={args.indicator_warmup_days}")
    print(f"bars_loaded={len(candles)} snapshots={len(snapshots)} transitions={len(transitions)}")

    if not snapshots.empty:
        latest = snapshots.iloc[-1]
        print(f"latest_regime={latest['regime']} timestamp={latest['timestamp']}")
        dist = snapshots["regime"].value_counts().sort_index()
        print("\n[Regime Distribution]")
        print(dist.to_string())

    durations = _regime_durations(snapshots)
    print("\n[Avg Regime Duration (bars)]")
    if durations.empty:
        print("  no rows")
    else:
        print(durations.to_string(index=False))

    if args.persist:
        snapshot_store = RegimeSnapshotStore(base_dir=str(cache_dir))
        transition_store = StrategyTransitionStore(base_dir=str(cache_dir))
        persisted_snapshots = snapshot_store.persist_snapshots(
            snapshots.to_dict(orient="records"),
            symbol=args.symbol,
            source="regime_replay",
        )
        persisted_transitions = transition_store.persist_transitions(
            transitions.to_dict(orient="records"),
            symbol=args.symbol,
            source="regime_replay",
        )
        print(f"\nPersisted: snapshots={persisted_snapshots} transitions={persisted_transitions}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
