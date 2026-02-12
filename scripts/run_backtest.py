"""Run a Phase 4 scaffold backtest from cached parquet datasets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtest import (
    BacktestEngine,
    FillSimulator,
    aggregate_walk_forward_metrics,
    generate_walk_forward_windows,
    write_backtest_report,
)
from src.data.store import DataStore
from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.strategies.regime_probe import RegimeProbeStrategy


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest scaffold from cached data.")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol partition")
    parser.add_argument("--timeframe", default="1d", help="Candle timeframe partition")
    parser.add_argument("--from", dest="start", type=parse_date, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end", type=parse_date, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        help="Strategy id; repeatable or comma-separated (currently: regime_probe)",
    )
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward windows instead of single run")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings file path")
    parser.add_argument("--output-dir", default="data/reports", help="Report output directory")
    return parser.parse_args()


def load_settings(path: str) -> dict:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text())


def resolve_strategies(values: list[str] | None) -> list[str]:
    if not values:
        return ["regime_probe"]
    out: list[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out or ["regime_probe"]


def build_strategy(strategy_id: str, *, settings: dict, symbol: str):
    if strategy_id != "regime_probe":
        raise ValueError(f"Unsupported strategy '{strategy_id}'. Available: regime_probe")
    strategy_cfg = settings.get("strategies", {}).get("momentum", {})
    strategy_cfg = {**strategy_cfg, "instrument": symbol, "lots": strategy_cfg.get("max_lots", 1)}
    strategy = RegimeProbeStrategy(name=strategy_id, config=strategy_cfg)
    capital = float(strategy_cfg.get("capital_per_trade", 200000) or 200000)
    return strategy, capital


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    strategy_ids = resolve_strategies(args.strategies)

    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")
    store = DataStore(base_dir=str(cache_dir))

    backtest_cfg = settings.get("backtest", {})
    start = args.start or parse_date(backtest_cfg.get("start_date", "2022-01-01"))
    end = args.end or parse_date(backtest_cfg.get("end_date", "2025-12-31"))

    candles = store.read_time_series("candles", symbol=args.symbol, timeframe=args.timeframe, start=start, end=end)
    if candles.empty:
        print("No candle data in cache for the requested window.")
        print("Run: python scripts/download_historical.py --symbol NIFTY --timeframe 1d --days 365")
        return 1
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], errors="coerce")
    candles = candles.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    vix = store.read_time_series("vix", symbol="INDIAVIX", timeframe="1d", start=start, end=end)
    fii = store.read_time_series(
        "fii_dii",
        symbol="NSE",
        timeframe="1d",
        start=start,
        end=end,
        timestamp_col="date",
    )

    simulator = FillSimulator(
        slippage_pct=float(backtest_cfg.get("slippage_pct", 0.05) or 0.05),
        commission_per_order=float(backtest_cfg.get("commission_per_order", 20.0) or 20.0),
    )
    output_dir = str(REPO_ROOT / args.output_dir)
    periods_per_year = 252 if args.timeframe == "1d" else 252 * 75

    print("=" * 72)
    print("Backtest Run")
    print("=" * 72)
    print(f"symbol={args.symbol} timeframe={args.timeframe} start={start.date()} end={end.date()}")
    print(f"strategies={strategy_ids}")
    print(f"walk_forward={args.walk_forward}")

    for strategy_id in strategy_ids:
        strategy, initial_capital = build_strategy(strategy_id, settings=settings, symbol=args.symbol)

        if args.walk_forward:
            windows = generate_walk_forward_windows(
                start=start,
                end=end,
                train_months=int(backtest_cfg.get("train_months", 12)),
                test_months=int(backtest_cfg.get("test_months", 3)),
                step_months=int(backtest_cfg.get("step_months", 3)),
            )
            if not windows:
                raise ValueError("No walk-forward windows were generated for this date range/config.")

            fold_rows: list[dict] = []
            for i, w in enumerate(windows, start=1):
                classifier = RegimeClassifier(thresholds=RegimeThresholds.from_config(settings.get("regime", {})))
                strategy_fold, _ = build_strategy(strategy_id, settings=settings, symbol=args.symbol)
                engine = BacktestEngine(
                    classifier=classifier,
                    strategy=strategy_fold,
                    simulator=simulator,
                    initial_capital=initial_capital,
                    periods_per_year=periods_per_year,
                )
                test_candles = candles.loc[
                    (candles["timestamp"] >= pd.Timestamp(w.test_start))
                    & (candles["timestamp"] < pd.Timestamp(w.test_end))
                ]
                result = engine.run(candles=test_candles, vix_df=vix, fii_df=fii)
                row = {
                    "fold": i,
                    "train_start": w.train_start.isoformat(),
                    "train_end": w.train_end.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    **result.metrics,
                }
                fold_rows.append(row)

            fold_df = pd.DataFrame(fold_rows)
            summary = aggregate_walk_forward_metrics(fold_rows)
            run_name = f"{strategy_id}_{args.symbol.lower()}_{args.timeframe}_{start.date()}_{end.date()}_walkforward"
            csv_path = Path(output_dir) / f"{run_name}_folds.csv"
            json_path = Path(output_dir) / f"{run_name}_summary.json"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fold_df.to_csv(csv_path, index=False)

            json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

            print(f"\n[{strategy_id}] Walk-forward complete")
            print(f"  folds={len(fold_df)} return_mean={summary.get('total_return_pct_mean', 0.0):.2f}")
            print(f"  folds_csv={csv_path}")
            print(f"  summary_json={json_path}")
        else:
            classifier = RegimeClassifier(thresholds=RegimeThresholds.from_config(settings.get("regime", {})))
            engine = BacktestEngine(
                classifier=classifier,
                strategy=strategy,
                simulator=simulator,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
            )
            result = engine.run(candles=candles, vix_df=vix, fii_df=fii)

            run_name = f"{strategy_id}_{args.symbol.lower()}_{args.timeframe}_{start.date()}_{end.date()}"
            paths = write_backtest_report(result, output_dir=output_dir, run_name=run_name)

            print(f"\n[{strategy_id}] Backtest complete")
            print(f"  fills={int(result.metrics['fill_count'])} final_equity={result.metrics['final_equity']:.2f}")
            print(
                f"  total_return_pct={result.metrics['total_return_pct']:.2f} "
                f"max_drawdown_pct={result.metrics['max_drawdown_pct']:.2f}"
            )
            for k, p in paths.items():
                print(f"  {k}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
