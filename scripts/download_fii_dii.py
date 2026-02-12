"""Download FII/DII flow data from NSE and persist to local CSV/cache."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.fii import FiiDownloadError, fetch_fii_dii
from src.data.store import DataStore


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FII/DII cash market flow history.")
    parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=365, help="If no --start/--end, use trailing N days")
    parser.add_argument("--output", default="data/raw/fii_dii.csv", help="CSV output path")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    return parser.parse_args()


def load_settings(path: str) -> dict:
    p = REPO_ROOT / path
    if p.exists():
        return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"Settings file not found: {p}")


def resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.start and args.end:
        return args.start, args.end
    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=args.days)
    return start, end


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    cache_dir = settings.get("data", {}).get("cache_dir", "data/cache")
    start, end = resolve_window(args)

    print(f"Downloading FII/DII from {start.date()} to {end.date()}...")
    try:
        df = fetch_fii_dii(start=start, end=end)
    except FiiDownloadError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if df.empty:
        print("[FAIL] No FII/DII rows returned from NSE.")
        return 1

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        old = pd.DataFrame()
        try:
            old = pd.read_csv(out_path)
            if "date" in old.columns:
                old["date"] = pd.to_datetime(old["date"], errors="coerce")
        except Exception:
            old = pd.DataFrame()
        if not old.empty:
            merged = pd.concat([old, df], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
            df = merged.reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df)} rows to {out_path.relative_to(REPO_ROOT)}")

    store = DataStore(base_dir=str(REPO_ROOT / cache_dir))
    upserted = store.write_time_series(
        "fii_dii",
        df,
        symbol="NSE",
        timeframe="1d",
        timestamp_col="date",
        source="nse_api",
    )
    print(f"[OK] Cache upsert rows={upserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
