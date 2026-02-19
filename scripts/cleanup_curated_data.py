"""Build curated candle datasets from raw cache with deterministic cleanup."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.cleanup import clean_daily_candles, clean_intraday_candles
from src.data.quality import assess_candle_quality, evaluate_quality_gate, thresholds_from_config
from src.data.store import DataStore

CURATION_SCHEMA_VERSION = "candles_curated_v1"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create curated candle datasets from raw cache.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Symbol(s) to process. Defaults to all candle symbols in metadata.",
    )
    parser.add_argument(
        "--timeframe",
        action="append",
        dest="timeframes",
        help="Timeframe(s) to process. Defaults to 1d.",
    )
    parser.add_argument("--start", type=parse_date, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument(
        "--incremental-days",
        type=int,
        default=0,
        help="Shortcut mode: process only trailing N days from today.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if curated output fails quality thresholds.",
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    p = REPO_ROOT / path
    if p.exists():
        return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"Settings file not found: {p}")


def resolve_window(args: argparse.Namespace) -> tuple[datetime | None, datetime | None]:
    if args.incremental_days > 0:
        end = datetime.now(UTC).replace(tzinfo=None)
        start = end - timedelta(days=args.incremental_days)
        return start, end
    if args.start or args.end:
        return args.start, args.end
    return None, None


def load_thresholds(settings: dict, timeframe: str):
    cfg = settings.get("data_quality", {})
    defaults = cfg.get("thresholds", {})
    per_timeframe = cfg.get("thresholds_by_timeframe", {})
    effective = dict(defaults)
    effective.update(per_timeframe.get(timeframe.lower(), {}))
    return thresholds_from_config(effective)


def resolve_symbols(raw_store: DataStore, explicit_symbols: list[str] | None) -> list[str]:
    if explicit_symbols:
        return [s.upper() for s in explicit_symbols]
    metadata = raw_store.get_metadata().get("datasets", {})
    out: set[str] = set()
    for key, meta in metadata.items():
        if meta.get("dataset") != "candles":
            continue
        symbol = meta.get("symbol")
        if symbol:
            out.add(str(symbol).upper())
        else:
            parts = key.split(":")
            if len(parts) >= 2 and parts[0] == "candles":
                out.add(parts[1].upper())
    return sorted(out)


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    start, end = resolve_window(args)

    data_cfg = settings.get("data", {})
    raw_cache_dir = data_cfg.get("cache_dir", "data/cache")
    curated_cache_dir = data_cfg.get("curated_cache_dir", "data/curated_cache")
    report_dir = REPO_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_store = DataStore(base_dir=str(REPO_ROOT / raw_cache_dir))
    curated_store = DataStore(base_dir=str(REPO_ROOT / curated_cache_dir))
    symbols = resolve_symbols(raw_store, args.symbols)
    timeframes = [tf.lower() for tf in (args.timeframes or ["1d"])]

    print("=" * 72)
    print("Curated Candle Cleanup")
    print("=" * 72)
    print(f"raw_cache_dir={raw_cache_dir}")
    print(f"curated_cache_dir={curated_cache_dir}")
    print(f"symbols={symbols}")
    print(f"timeframes={timeframes}")
    if start and end:
        print(f"window={start.date()} -> {end.date()}")
    elif start:
        print(f"window_start={start.date()}")
    elif end:
        print(f"window_end={end.date()}")
    else:
        print("window=full")

    if not symbols:
        print("[FAIL] No candle symbols found to process.")
        return 1

    summary_rows: list[dict] = []
    failed_quality = 0
    processed = 0

    for symbol in symbols:
        for timeframe in timeframes:
            raw = raw_store.read_time_series(
                "candles",
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            if raw.empty:
                print(f"[SKIP] {symbol} {timeframe}: no raw rows")
                continue

            if timeframe == "1d":
                cleaned, stats = clean_daily_candles(raw)
            else:
                cleaned, stats = clean_intraday_candles(raw)

            added_rows = curated_store.write_time_series(
                "candles",
                cleaned,
                symbol=symbol,
                timeframe=timeframe,
                timestamp_col="timestamp",
                dedup_cols=["timestamp"],
                source="curated_cleanup",
            )

            quality = assess_candle_quality(cleaned, timeframe=timeframe)
            thresholds = load_thresholds(settings, timeframe)
            gate = evaluate_quality_gate(quality, thresholds)
            if gate.status != "ok":
                failed_quality += 1

            summary = {
                "symbol": symbol,
                "timeframe": timeframe,
                **stats.as_dict(),
                "curated_rows_added": int(added_rows),
                "quality_status": gate.status,
                "quality_violations": gate.violations,
                "largest_gap_minutes": quality.largest_gap_minutes,
            }
            summary_rows.append(summary)
            processed += 1

            print(
                f"[OK] {symbol} {timeframe}: input={stats.input_rows} output={stats.output_rows} "
                f"dropped={stats.dropped_rows} dup_days={stats.duplicate_trade_dates} "
                f"status={gate.status}"
            )
            if gate.violations:
                print(f"     violations={', '.join(gate.violations)}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output)
        if args.output
        else report_dir / f"curated_cleanup_{timestamp}.json"
    )
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "curation_schema_version": CURATION_SCHEMA_VERSION,
        "raw_cache_dir": raw_cache_dir,
        "curated_cache_dir": curated_cache_dir,
        "window_start": start.isoformat() if start else None,
        "window_end": end.isoformat() if end else None,
        "processed_targets": processed,
        "quality_failures": failed_quality,
        "results": summary_rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    marker_path = _write_version_marker(
        curated_store=curated_store,
        report_path=output,
        payload=payload,
    )
    print(f"\nreport={output.relative_to(REPO_ROOT)}")
    print(f"curation_marker={marker_path.relative_to(REPO_ROOT)}")

    if processed == 0:
        return 1
    if args.strict and failed_quality > 0:
        return 2
    return 0


def _write_version_marker(
    *,
    curated_store: DataStore,
    report_path: Path,
    payload: dict,
) -> Path:
    marker = curated_store.root / "curation_version.json"
    entry = {
        "schema_version": CURATION_SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "window_start": payload.get("window_start"),
        "window_end": payload.get("window_end"),
        "processed_targets": int(payload.get("processed_targets", 0) or 0),
        "quality_failures": int(payload.get("quality_failures", 0) or 0),
    }
    marker.write_text(json.dumps(entry, indent=2, sort_keys=True))
    return marker


if __name__ == "__main__":
    raise SystemExit(main())
