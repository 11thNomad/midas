"""Generate a quality report for cached candle datasets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.quality import (
    CandleQualityThresholds,
    assess_candle_quality,
    evaluate_quality_gate,
    thresholds_from_config,
)
from src.data.store import DataStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data quality checks on cached candles.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    parser.add_argument("--dataset", default="candles", help="Dataset namespace to check")
    parser.add_argument("--output", default="", help="Optional explicit output JSON path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any dataset fails configured thresholds.",
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    p = REPO_ROOT / path
    if p.exists():
        return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"Settings file not found: {p}")


def parse_key(key: str) -> tuple[str | None, str | None]:
    parts = key.split(":")
    symbol = parts[1] if len(parts) >= 2 else None
    timeframe = parts[2] if len(parts) >= 3 else None
    return symbol, timeframe


def load_thresholds(settings: dict, timeframe: str) -> CandleQualityThresholds:
    cfg = settings.get("data_quality", {})
    defaults = cfg.get("thresholds", {})
    per_timeframe = cfg.get("thresholds_by_timeframe", {})
    effective = dict(defaults)
    effective.update(per_timeframe.get(timeframe.lower(), {}))
    return thresholds_from_config(effective)


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    cache_dir = settings.get("data", {}).get("cache_dir", "data/cache")
    report_dir = REPO_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    store = DataStore(base_dir=str(REPO_ROOT / cache_dir))
    metadata = store.get_metadata().get("datasets", {})

    targets = {k: v for k, v in metadata.items() if v.get("dataset") == args.dataset}
    if not targets:
        print(f"No datasets found for '{args.dataset}' in metadata.")
        return 1

    results: dict[str, dict] = {}

    for key, meta in sorted(targets.items()):
        symbol, timeframe = parse_key(key)
        timeframe = timeframe or "1d"
        df = store.read_time_series(args.dataset, symbol=symbol, timeframe=timeframe)
        q = assess_candle_quality(df, timeframe=timeframe)
        thresholds = load_thresholds(settings, timeframe)
        gate = evaluate_quality_gate(q, thresholds)

        results[key] = {
            "meta": meta,
            "quality": q.as_dict(),
            "thresholds": thresholds.__dict__,
            "gate": gate.as_dict(),
        }

        print(
            f"{key:35s} status={gate.status:16s} "
            f"rows={q.rows:<8d} issues={gate.issue_count:<4d} "
            f"largest_gap_min={q.largest_gap_minutes} violations={len(gate.violations)}"
        )
        if gate.violations:
            print(f"  - {', '.join(gate.violations)}")

    failed = [k for k, value in results.items() if value["gate"]["status"] != "ok"]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "summary": {
            "total_targets": len(results),
            "targets_failed_thresholds": len(failed),
        },
        "results": results,
    }

    output = (
        Path(args.output)
        if args.output
        else report_dir
        / f"{args.dataset}_quality_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    print(f"\nQuality report written to {output.relative_to(REPO_ROOT)}")

    if args.strict and failed:
        print(f"Threshold failures detected: {len(failed)} dataset(s)")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
