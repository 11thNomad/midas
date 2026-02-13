"""Download historical datasets into local parquet cache.

Examples:
  python scripts/download_historical.py --symbol NIFTY --days 365
  python scripts/download_historical.py --full
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some environments
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.fii import FiiDownloadError, fetch_fii_dii
from src.data.kite_feed import KiteFeed, KiteFeedError
from src.data.store import DataStore

DEFAULT_FULL_SYMBOLS = ["NIFTY", "BANKNIFTY"]
DEFAULT_FULL_TIMEFRAMES = ["1d", "5m"]


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical market datasets.")
    parser.add_argument(
        "--symbol", action="append", dest="symbols", help="Symbol to download. Repeatable."
    )
    parser.add_argument(
        "--timeframe", action="append", dest="timeframes", help="Timeframe (e.g., 1d, 5m)."
    )
    parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--days", type=int, default=0, help="Shortcut for end=today and start=today-days."
    )
    parser.add_argument(
        "--full", action="store_true", help="Use full preset for Phase 1 cache bootstrap."
    )
    parser.add_argument("--skip-vix", action="store_true", help="Skip India VIX series download.")
    parser.add_argument("--skip-fii", action="store_true", help="Skip FII/DII flow ingest.")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    candidate = REPO_ROOT / path
    if candidate.exists():
        return yaml.safe_load(candidate.read_text())
    raise FileNotFoundError(f"Settings file not found: {candidate}")


def resolve_window(args: argparse.Namespace, settings: dict) -> tuple[datetime, datetime]:
    if args.days and args.days > 0:
        end = datetime.now(UTC).replace(tzinfo=None)
        start = end - timedelta(days=args.days)
        return start, end

    if args.start and args.end:
        return args.start, args.end

    backtest_cfg = settings.get("backtest", {})
    start = parse_date(backtest_cfg.get("start_date", "2022-01-01"))
    end = parse_date(backtest_cfg.get("end_date", datetime.now(UTC).strftime("%Y-%m-%d")))
    return start, end


def resolve_plan(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.full:
        return DEFAULT_FULL_SYMBOLS, DEFAULT_FULL_TIMEFRAMES

    symbols = args.symbols or ["NIFTY"]
    timeframes = args.timeframes or ["1d"]
    return symbols, timeframes


def main() -> int:
    if load_dotenv:
        load_dotenv(REPO_ROOT / ".env")

    args = parse_args()
    settings = load_settings(args.settings)

    cache_dir = settings.get("data", {}).get("cache_dir", "data/cache")
    start, end = resolve_window(args, settings)
    symbols, timeframes = resolve_plan(args)

    api_key = os.getenv("KITE_API_KEY", "").strip()
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        print("[FAIL] Missing Kite credentials. Set KITE_API_KEY and KITE_ACCESS_TOKEN in .env.")
        return 1

    feed = KiteFeed(api_key=api_key, access_token=access_token)
    store = DataStore(base_dir=str(REPO_ROOT / cache_dir))

    print("=" * 64)
    print("NiftyQuant Historical Downloader")
    print("=" * 64)
    print(f"Window: {start.date()} -> {end.date()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Cache dir: {cache_dir}")

    failures = 0
    downloads = 0

    for symbol in symbols:
        for timeframe in timeframes:
            label = f"candles {symbol} {timeframe}"
            print(f"\n[RUN] {label}")
            try:
                candles = feed.get_candles(symbol=symbol, timeframe=timeframe, start=start, end=end)
                rows = store.write_time_series(
                    "candles",
                    candles,
                    symbol=symbol,
                    timeframe=timeframe,
                    source=feed.name,
                )
                print(f"  [OK] cached rows={rows}")
                downloads += 1
            except (KiteFeedError, ValueError) as exc:
                failures += 1
                print(f"  [FAIL] {exc}")

    if not args.skip_vix:
        print("\n[RUN] vix daily")
        try:
            vix = feed.get_vix(start=start, end=end)
            rows = store.write_time_series(
                "vix",
                vix,
                symbol="INDIAVIX",
                timeframe="1d",
                source=feed.name,
            )
            print(f"  [OK] cached rows={rows}")
            downloads += 1
        except (KiteFeedError, ValueError) as exc:
            failures += 1
            print(f"  [FAIL] {exc}")

    if not args.skip_fii:
        print("\n[RUN] fii_dii daily")
        try:
            fii = fetch_fii_dii(start=start, end=end)
            rows = store.write_time_series(
                "fii_dii",
                fii,
                symbol="NSE",
                timeframe="1d",
                timestamp_col="date",
                source="nse",
            )
            print(f"  [OK] cached rows={rows}")
            downloads += 1
        except (FiiDownloadError, ValueError) as exc:
            failures += 1
            print(f"  [FAIL] {exc}")

    print("\n" + "=" * 64)
    print(f"Completed: successful_tasks={downloads}, failed_tasks={failures}")
    print(f"Metadata: {Path(cache_dir) / 'metadata.json'}")

    return 0 if downloads > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
