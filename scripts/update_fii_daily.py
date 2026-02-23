"""Append latest NSE FII equity net row into local daily cache (idempotent)."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_fii_cache import fetch_nse_latest_equity_net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update daily FII cache from NSE latest endpoint.")
    parser.add_argument(
        "--cache-path",
        default="data/cache/fii/fii_equity_daily.csv",
        help="Path to FII cache CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_path = REPO_ROOT / args.cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        latest = fetch_nse_latest_equity_net()
    except Exception as exc:
        print(f"[FAIL] NSE latest fetch failed: {exc}")
        return 1

    if latest.empty:
        print("[FAIL] NSE latest endpoint returned no FII/FPI row.")
        return 1

    latest = latest[["date", "fii_equity_net_cr"]].copy()
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest["fii_equity_net_cr"] = pd.to_numeric(latest["fii_equity_net_cr"], errors="coerce")
    latest = latest.dropna(subset=["date", "fii_equity_net_cr"])
    if latest.empty:
        print("[FAIL] NSE latest row could not be parsed.")
        return 1

    # Be explicit: report the most recent fetched date/value in diagnostics.
    latest = latest.sort_values("date").reset_index(drop=True)
    latest_day = latest["date"].iloc[-1].date()
    latest_value = float(latest["fii_equity_net_cr"].iloc[-1])

    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        if {"date", "fii_equity_net_cr"}.issubset(cache.columns):
            cache["date"] = pd.to_datetime(cache["date"], errors="coerce")
            cache["fii_equity_net_cr"] = pd.to_numeric(cache["fii_equity_net_cr"], errors="coerce")
            cache = cache.dropna(subset=["date", "fii_equity_net_cr"])
        else:
            cache = pd.DataFrame(columns=["date", "fii_equity_net_cr"])
    else:
        cache = pd.DataFrame(columns=["date", "fii_equity_net_cr"])

    before_rows = len(cache)
    merged = pd.concat([cache, latest], ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged.to_csv(cache_path, index=False)

    added = len(merged) - before_rows
    action = "appended" if added > 0 else "updated_existing_or_noop"
    print(
        f"[OK] {action} date={latest_day} fii_equity_net_cr={latest_value:.2f} "
        f"rows_before={before_rows} rows_after={len(merged)} utc={datetime.now(UTC).isoformat()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
