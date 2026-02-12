"""Run a Phase 4 scaffold backtest from cached parquet datasets."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtest import BacktestEngine, FillSimulator, write_backtest_report
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
    parser.add_argument("--strategy", default="regime_probe", help="Strategy id (currently: regime_probe)")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings file path")
    parser.add_argument("--output-dir", default="data/reports", help="Report output directory")
    return parser.parse_args()


def load_settings(path: str) -> dict:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text())


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)

    if args.strategy != "regime_probe":
        raise ValueError("Only 'regime_probe' is available in this scaffold.")

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

    vix = store.read_time_series("vix", symbol="INDIAVIX", timeframe="1d", start=start, end=end)
    fii = store.read_time_series(
        "fii_dii",
        symbol="NSE",
        timeframe="1d",
        start=start,
        end=end,
        timestamp_col="date",
    )

    classifier = RegimeClassifier(thresholds=RegimeThresholds.from_config(settings.get("regime", {})))
    strategy_cfg = settings.get("strategies", {}).get("momentum", {})
    strategy_cfg = {**strategy_cfg, "instrument": args.symbol, "lots": strategy_cfg.get("max_lots", 1)}
    strategy = RegimeProbeStrategy(name="regime_probe", config=strategy_cfg)

    simulator = FillSimulator(
        slippage_pct=float(backtest_cfg.get("slippage_pct", 0.05) or 0.05),
        commission_per_order=float(backtest_cfg.get("commission_per_order", 20.0) or 20.0),
    )
    engine = BacktestEngine(
        classifier=classifier,
        strategy=strategy,
        simulator=simulator,
        initial_capital=float(strategy_cfg.get("capital_per_trade", 200000) or 200000),
        periods_per_year=252 if args.timeframe == "1d" else 252 * 75,
    )
    result = engine.run(candles=candles, vix_df=vix, fii_df=fii)

    run_name = f"{args.strategy}_{args.symbol.lower()}_{args.timeframe}_{start.date()}_{end.date()}"
    paths = write_backtest_report(
        result,
        output_dir=str(REPO_ROOT / args.output_dir),
        run_name=run_name,
    )

    print("=" * 72)
    print("Backtest Complete")
    print("=" * 72)
    print(f"symbol={args.symbol} timeframe={args.timeframe} start={start.date()} end={end.date()}")
    print(f"fills={int(result.metrics['fill_count'])} final_equity={result.metrics['final_equity']:.2f}")
    print(f"total_return_pct={result.metrics['total_return_pct']:.2f} max_drawdown_pct={result.metrics['max_drawdown_pct']:.2f}")
    print("\nReport files:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
