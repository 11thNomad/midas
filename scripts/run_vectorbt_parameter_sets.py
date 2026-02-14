"""Batch vectorbt research over named parameter sets with leaderboard output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtest import (
    VectorBTResearchConfig,
    apply_parameter_set,
    build_snapshots_from_market_data,
    parse_parameter_sets,
    rank_parameter_results,
    run_vectorbt_research,
    run_vectorbt_walk_forward,
)
from src.data.store import DataStore
from src.regime.classifier import RegimeThresholds
from src.regime.persistence import SignalSnapshotStore


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vectorbt parameter-set batch research.")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--from", dest="start", type=parse_date)
    parser.add_argument("--to", dest="end", type=parse_date)
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--parameter-sets", default="config/vectorbt_parameter_sets.yaml")
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument(
        "--rank-by",
        default=None,
        help=(
            "Column to rank by "
            "(default: wf_total_return_pct_mean if walk-forward else total_return_pct)."
        ),
    )
    parser.add_argument("--min-trades", type=float, default=3.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_settings(path: str) -> dict[str, Any]:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    raw = yaml.safe_load(settings_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Settings must parse into an object.")
    return raw


def load_parameter_set_config(path: str) -> list[Any]:
    config_path = REPO_ROOT / path
    if not config_path.exists():
        raise FileNotFoundError(f"Parameter set file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Parameter set file must parse into an object.")
    return parse_parameter_sets(raw)


def resolve_output_dir(*, raw_output_dir: str, run_prefix: str) -> Path:
    base = Path(raw_output_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"{run_prefix}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _timeframe_to_freq(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    mapping = {
        "1d": "1D",
        "1h": "1H",
        "60m": "1H",
        "30m": "30T",
        "15m": "15T",
        "10m": "10T",
        "5m": "5T",
        "3m": "3T",
        "1m": "1T",
    }
    return mapping.get(tf, "1D")


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = float("nan")
    return out if math.isfinite(out) else float("nan")


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    backtest_cfg = settings.get("backtest", {})
    start = args.start or parse_date(str(backtest_cfg.get("start_date", "2022-01-01")))
    end = args.end or parse_date(str(backtest_cfg.get("end_date", "2025-12-31")))

    cache_dir = REPO_ROOT / str(settings.get("data", {}).get("cache_dir", "data/cache"))
    store = DataStore(base_dir=str(cache_dir))
    snapshot_store = SignalSnapshotStore(base_dir=str(cache_dir))
    out_dir = resolve_output_dir(raw_output_dir=args.output_dir, run_prefix="vectorbt_paramsets")

    candles = store.read_time_series(
        "candles",
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
    )
    if candles.empty:
        print("No candles found for requested window.")
        return 1

    snapshots = snapshot_store.read_snapshots(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
    )
    if snapshots.empty:
        print("No signal snapshots in cache for window; building from raw datasets.")
        vix = store.read_time_series("vix", symbol="INDIAVIX", timeframe="1d", start=start, end=end)
        fii = store.read_time_series(
            "fii_dii",
            symbol="NSE",
            timeframe="1d",
            start=start,
            end=end,
            timestamp_col="date",
        )
        usdinr_symbol = str(settings.get("market", {}).get("usdinr_symbol", "USDINR")).upper()
        usdinr = store.read_time_series(
            "candles",
            symbol=usdinr_symbol,
            timeframe="1d",
            start=start,
            end=end,
        )
        option_chain = store.read_time_series(
            "option_chain",
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start,
            end=end,
        )
        snapshots = build_snapshots_from_market_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            candles=candles,
            thresholds=RegimeThresholds.from_config(settings.get("regime", {})),
            vix_df=vix,
            fii_df=fii,
            usdinr_df=usdinr,
            option_chain_df=option_chain,
        )

    param_sets = load_parameter_set_config(args.parameter_sets)
    if not param_sets:
        print("No parameter sets found.")
        return 1

    base_config = VectorBTResearchConfig(
        initial_cash=float(settings.get("risk", {}).get("initial_capital", 150_000.0) or 150_000.0),
        fees_pct=float(backtest_cfg.get("slippage_pct", 0.05) or 0.05) / 100.0,
        slippage_pct=float(backtest_cfg.get("slippage_pct", 0.05) or 0.05) / 100.0,
        entry_regimes=("low_vol_trending", "high_vol_trending"),
        adx_min=25.0,
        vix_max=None,
        freq=_timeframe_to_freq(args.timeframe),
    )

    rows: list[dict[str, Any]] = []
    for params in param_sets:
        cfg = apply_parameter_set(base_config, params)
        run = run_vectorbt_research(candles=candles, snapshots=snapshots, config=cfg)
        row: dict[str, Any] = {
            "set_id": params.set_id,
            "entry_regimes": ",".join(params.entry_regimes),
            "adx_min": float(params.adx_min),
            "vix_max": float(params.vix_max) if params.vix_max is not None else float("nan"),
            "notes": params.notes,
        }
        row.update({key: _as_float(value) for key, value in run.metrics.items()})

        if args.walk_forward:
            folds, wf_summary = run_vectorbt_walk_forward(
                candles=candles,
                snapshots=snapshots,
                config=cfg,
                start=start,
                end=end,
                train_months=int(backtest_cfg.get("train_months", 12)),
                test_months=int(backtest_cfg.get("test_months", 3)),
                step_months=int(backtest_cfg.get("step_months", 3)),
            )
            folds_path = out_dir / f"{params.set_id}_walkforward_folds.csv"
            folds.to_csv(folds_path, index=False)
            for key, value in wf_summary.items():
                row[f"wf_{key}"] = _as_float(value)

        rows.append(row)
        print(
            f"set={params.set_id} trades={row.get('trades')} "
            f"return={row.get('total_return_pct')}"
        )

    results_df = pd.DataFrame(rows)
    rank_by = args.rank_by or (
        "wf_total_return_pct_mean" if args.walk_forward else "total_return_pct"
    )
    leaderboard = rank_parameter_results(
        results_df,
        rank_by=rank_by,
        min_trades=float(args.min_trades),
        max_drawdown_pct=args.max_drawdown_pct,
    )
    top_k = max(1, int(args.top_k))
    top_df = leaderboard.head(top_k).reset_index(drop=True)
    eligible = leaderboard.loc[leaderboard["eligible"]].reset_index(drop=True)
    best = eligible.iloc[0].to_dict() if not eligible.empty else None

    results_path = out_dir / "vectorbt_parameter_set_results.csv"
    leaderboard_path = out_dir / "vectorbt_parameter_set_leaderboard.csv"
    top_path = out_dir / "vectorbt_parameter_set_top.csv"
    summary_path = out_dir / "vectorbt_parameter_set_summary.json"
    results_df.to_csv(results_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    top_df.to_csv(top_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "walk_forward": bool(args.walk_forward),
                "rank_by": rank_by,
                "min_trades": float(args.min_trades),
                "max_drawdown_pct": (
                    float(args.max_drawdown_pct) if args.max_drawdown_pct is not None else None
                ),
                "parameter_set_count": int(len(param_sets)),
                "eligible_count": int(len(eligible)),
                "best_eligible_set": best,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print(f"results={results_path}")
    print(f"leaderboard={leaderboard_path}")
    print(f"top={top_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
