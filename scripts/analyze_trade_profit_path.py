"""Analyze hold-period profit path per trade from backtest artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))

from src.backtest.engine import BacktestEngine
from src.data.candle_access import build_candle_stores, read_candles
from src.data.option_symbols import parse_option_symbol, resolve_option_price
from src.data.store import DataStore


@dataclass
class TradeBlock:
    trade_id: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_rows: pd.DataFrame
    exit_rows: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-trade profit-path analysis CSV.")
    parser.add_argument("--fills-csv", required=True, help="Backtest fills CSV path")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g. NIFTY)")
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g. 1d)")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings file path")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    return parser.parse_args()


def _signed_notional(rows: pd.DataFrame, *, buy_positive: bool) -> float:
    side = rows["side"].astype(str).str.upper()
    notional = rows["notional"].astype(float)
    if buy_positive:
        return float(notional[side.eq("BUY")].sum() - notional[side.eq("SELL")].sum())
    return float(notional[side.eq("SELL")].sum() - notional[side.eq("BUY")].sum())


def _pair_trade_blocks(fills: pd.DataFrame) -> list[TradeBlock]:
    entry_ts = (
        fills.loc[fills["signal_type"].eq("entry_short"), "timestamp"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    exit_ts = (
        fills.loc[fills["signal_type"].eq("exit"), "timestamp"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    pair_count = min(len(entry_ts), len(exit_ts))
    blocks: list[TradeBlock] = []
    for idx in range(pair_count):
        ent = pd.Timestamp(entry_ts[idx])
        ex = pd.Timestamp(exit_ts[idx])
        blocks.append(
            TradeBlock(
                trade_id=idx + 1,
                entry_ts=ent,
                exit_ts=ex,
                entry_rows=fills.loc[
                    fills["signal_type"].eq("entry_short") & fills["timestamp"].eq(ent)
                ].copy(),
                exit_rows=fills.loc[
                    fills["signal_type"].eq("exit") & fills["timestamp"].eq(ex)
                ].copy(),
            )
        )
    return blocks


def _weekday_name(ts: pd.Timestamp) -> str:
    return ts.day_name()


def _build_position_legs(entry_rows: pd.DataFrame) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []
    for _, row in entry_rows.iterrows():
        symbol = str(row["instrument"])
        parts = parse_option_symbol(symbol)
        legs.append(
            {
                "symbol": symbol,
                "action": str(row["side"]).upper(),
                "quantity": float(row["quantity"]),
                "expiry": None if parts is None else parts.expiry,
                "strike": None if parts is None else float(parts.strike),
                "option_type": None if parts is None else parts.option_type,
            }
        )
    return legs


def _profit_pct(entry_credit: float, close_debit: float) -> float:
    return ((entry_credit - close_debit) / entry_credit) * 100.0


def main() -> int:
    args = parse_args()
    fills_path = REPO_ROOT / args.fills_csv
    output_path = REPO_ROOT / args.output_csv
    settings_path = REPO_ROOT / args.settings

    if not fills_path.exists():
        raise FileNotFoundError(f"fills csv not found: {fills_path}")
    if not settings_path.exists():
        raise FileNotFoundError(f"settings yaml not found: {settings_path}")

    fills = pd.read_csv(fills_path)
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], errors="coerce")
    fills = fills.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    blocks = _pair_trade_blocks(fills)
    if not blocks:
        raise RuntimeError("No entry/exit trade blocks found.")

    settings = yaml.safe_load(settings_path.read_text())
    stores = build_candle_stores(settings=settings, repo_root=REPO_ROOT)
    data_store = DataStore()

    start = blocks[0].entry_ts.to_pydatetime()
    end = blocks[-1].exit_ts.to_pydatetime()
    candles, _source = read_candles(
        stores=stores,
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        start=start,
        end=end,
    )
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], errors="coerce")
    candles = candles.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    option_chain = data_store.read_time_series(
        "option_chain",
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        start=start,
        end=end,
    )
    chain = BacktestEngine._prep_chain(option_chain)

    rows: list[dict[str, Any]] = []
    for block in blocks:
        entry_credit = _signed_notional(block.entry_rows, buy_positive=False)
        exit_debit = _signed_notional(block.exit_rows, buy_positive=True)
        if entry_credit > 0.0:
            actual_exit_pnl_pct = ((entry_credit - exit_debit) / entry_credit) * 100.0
        else:
            actual_exit_pnl_pct = float("nan")

        hold_bars = candles.loc[
            candles["timestamp"].gt(block.entry_ts) & candles["timestamp"].le(block.exit_ts),
            "timestamp",
        ]
        legs = _build_position_legs(block.entry_rows)

        max_profit_pct: float | None = None
        first_reach_50_weekday: str | None = None
        first_reach_40_weekday: str | None = None
        bars_with_missing_prices = 0
        bars_evaluated = 0

        for ts in hold_bars.tolist():
            ts_pd = pd.Timestamp(ts)
            chain_asof = BacktestEngine._latest_chain_asof(
                chain, ts_pd.to_pydatetime(), strict=True
            )
            if chain_asof is None or chain_asof.empty:
                bars_with_missing_prices += 1
                continue

            price_map = BacktestEngine._build_chain_price_map(chain_asof)
            close_debit = 0.0
            has_missing_leg = False
            for leg in legs:
                px = resolve_option_price(
                    price_lookup=price_map,
                    symbol=str(leg["symbol"]),
                    expiry=leg.get("expiry"),
                    strike=leg.get("strike"),
                    option_type=leg.get("option_type"),
                )
                if px is None or float(px) <= 0.0:
                    has_missing_leg = True
                    break
                qty = float(leg["quantity"])
                action = str(leg["action"]).upper()
                if action == "SELL":
                    close_debit += float(px) * qty
                elif action == "BUY":
                    close_debit -= float(px) * qty
            if has_missing_leg:
                bars_with_missing_prices += 1
                continue

            bars_evaluated += 1
            profit_pct = _profit_pct(entry_credit=entry_credit, close_debit=close_debit)
            if max_profit_pct is None or profit_pct > max_profit_pct:
                max_profit_pct = profit_pct
            if first_reach_40_weekday is None and profit_pct >= 40.0:
                first_reach_40_weekday = _weekday_name(ts_pd)
            if first_reach_50_weekday is None and profit_pct >= 50.0:
                first_reach_50_weekday = _weekday_name(ts_pd)

        rows.append(
            {
                "trade_id": block.trade_id,
                "entry_date": block.entry_ts.date().isoformat(),
                "entry_credit": entry_credit,
                "max_profit_pct_during_hold": max_profit_pct,
                "actual_exit_pnl_pct": actual_exit_pnl_pct,
                "exit_reason": str(block.exit_rows["reason"].iloc[0]),
                "first_reach_50_weekday": first_reach_50_weekday,
                "first_reach_40_weekday": first_reach_40_weekday,
                "bars_with_missing_prices": int(bars_with_missing_prices),
                "bars_evaluated": int(bars_evaluated),
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
