"""Report utility for regime snapshots and strategy transition activity."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.regime import (
    RegimeSnapshotStore,
    StrategyTransitionStore,
    summarize_regime_daily,
    summarize_transitions_by_strategy,
    summarize_transitions_daily,
)


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize regime snapshots and strategy transitions.")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol partition to read")
    parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    return parser.parse_args()


def load_settings(path: str) -> dict:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text())


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")

    snapshot_store = RegimeSnapshotStore(base_dir=str(cache_dir))
    transition_store = StrategyTransitionStore(base_dir=str(cache_dir))

    snapshots = snapshot_store.read_snapshots(symbol=args.symbol, start=args.start, end=args.end)
    transitions = transition_store.read_transitions(symbol=args.symbol, start=args.start, end=args.end)

    regime_daily = summarize_regime_daily(snapshots)
    transition_daily = summarize_transitions_daily(transitions)
    by_strategy = summarize_transitions_by_strategy(transitions)

    print("=" * 72)
    print("Regime / Transition Report")
    print("=" * 72)
    print(f"symbol={args.symbol}")
    print(f"window={args.start.date() if args.start else 'begin'} -> {args.end.date() if args.end else 'latest'}")
    print(f"cache_dir={cache_dir}")
    print(f"snapshot_rows={len(snapshots)} transition_rows={len(transitions)}")

    print("\n[Daily Regime Counts]")
    if regime_daily.empty:
        print("  no rows")
    else:
        print(regime_daily.to_string(index=False))

    print("\n[Daily Transition Counts]")
    if transition_daily.empty:
        print("  no rows")
    else:
        print(transition_daily.to_string(index=False))

    print("\n[Strategy Transition Counts]")
    if by_strategy.empty:
        print("  no rows")
    else:
        print(by_strategy.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
