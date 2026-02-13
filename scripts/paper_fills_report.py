"""Generate daily paper-fill P&L summary by strategy from cached paper fills.

Examples:
  python scripts/paper_fills_report.py --symbol NIFTY
  python scripts/paper_fills_report.py --symbol NIFTY --start 2026-02-01 --end 2026-02-29
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.store import DataStore


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper fills into daily strategy P&L.")
    parser.add_argument(
        "--symbol", default="NIFTY", help="Underlying symbol partition used by paper fills"
    )
    parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    parser.add_argument(
        "--output-dir", default="data/reports", help="Directory for generated reports"
    )
    parser.add_argument(
        "--print-rows", type=int, default=20, help="Rows to print in terminal summary"
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    p = REPO_ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Settings file not found: {p}")
    return yaml.safe_load(p.read_text())


def summarize_daily_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paper fills into daily gross/net cashflow metrics by strategy."""
    if fills.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "strategy_name",
                "fill_count",
                "buy_notional",
                "sell_notional",
                "gross_cashflow",
                "fees",
                "net_cashflow",
            ]
        )

    out = fills.copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp"), errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return pd.DataFrame()

    out["strategy_name"] = out.get("strategy_name", "unknown").astype(str)
    out["side"] = out.get("side", "").astype(str).str.upper()
    out["notional"] = pd.to_numeric(out.get("notional", 0.0), errors="coerce").fillna(0.0)
    out["fees"] = pd.to_numeric(out.get("fees", 0.0), errors="coerce").fillna(0.0)
    out["date"] = out["timestamp"].dt.date

    out["buy_notional"] = out["notional"].where(out["side"] == "BUY", 0.0)
    out["sell_notional"] = out["notional"].where(out["side"] == "SELL", 0.0)
    out["gross_cashflow"] = out["sell_notional"] - out["buy_notional"]
    out["net_cashflow"] = out["gross_cashflow"] - out["fees"]

    agg = (
        out.groupby(["date", "strategy_name"], as_index=False)
        .agg(
            fill_count=("side", "count"),
            buy_notional=("buy_notional", "sum"),
            sell_notional=("sell_notional", "sum"),
            gross_cashflow=("gross_cashflow", "sum"),
            fees=("fees", "sum"),
            net_cashflow=("net_cashflow", "sum"),
        )
        .sort_values(["date", "strategy_name"])
        .reset_index(drop=True)
    )
    return agg


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    store = DataStore(base_dir=str(cache_dir))
    fills = store.read_time_series(
        "paper_fills",
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        timestamp_col="timestamp",
    )
    if fills.empty:
        print(f"No paper fills found for symbol={args.symbol} in requested window.")
        return 1

    daily = summarize_daily_fills(fills)
    if daily.empty:
        print("Paper fills were found, but no valid timestamps could be aggregated.")
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"paper_fills_daily_{args.symbol}_{stamp}.csv"
    json_path = output_dir / f"paper_fills_summary_{args.symbol}_{stamp}.json"
    daily.to_csv(csv_path, index=False)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol": args.symbol,
        "start": args.start.isoformat() if args.start else None,
        "end": args.end.isoformat() if args.end else None,
        "fill_rows": int(len(fills)),
        "daily_rows": int(len(daily)),
        "totals": {
            "buy_notional": float(daily["buy_notional"].sum()),
            "sell_notional": float(daily["sell_notional"].sum()),
            "gross_cashflow": float(daily["gross_cashflow"].sum()),
            "fees": float(daily["fees"].sum()),
            "net_cashflow": float(daily["net_cashflow"].sum()),
        },
        "output_csv": str(csv_path.relative_to(REPO_ROOT)),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("=" * 72)
    print("Paper Fills Daily Summary")
    print("=" * 72)
    window_start = args.start.date() if args.start else "begin"
    window_end = args.end.date() if args.end else "latest"
    print(
        f"symbol={args.symbol} rows={len(fills)} daily_rows={len(daily)} "
        f"window={window_start} -> {window_end}"
    )
    print(
        f"totals: gross_cashflow={payload['totals']['gross_cashflow']:.2f} "
        f"fees={payload['totals']['fees']:.2f} net_cashflow={payload['totals']['net_cashflow']:.2f}"
    )
    print(f"csv={csv_path.relative_to(REPO_ROOT)}")
    print(f"json={json_path.relative_to(REPO_ROOT)}")
    print("\n[Top Rows]")
    print(daily.head(max(args.print_rows, 1)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
