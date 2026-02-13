"""Run a Phase 4 scaffold backtest from cached parquet datasets."""

from __future__ import annotations

import argparse
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
    regime_segmented_returns,
    write_backtest_report,
    write_walkforward_report,
)
from src.data.store import DataStore
from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.risk.circuit_breaker import CircuitBreaker
from src.strategies.iron_condor import IronCondorStrategy
from src.strategies.momentum import MomentumStrategy
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
        help="Strategy id; repeatable or comma-separated (available: regime_probe, momentum, iron_condor)",
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
    if strategy_id == "regime_probe":
        base_cfg = settings.get("strategies", {}).get("momentum", {})
        probe_cfg = {**base_cfg, "instrument": symbol, "lots": base_cfg.get("max_lots", 1)}
        strategy = RegimeProbeStrategy(name=strategy_id, config=probe_cfg)
        capital = float(probe_cfg.get("capital_per_trade", 200000) or 200000)
        return strategy, capital
    if strategy_id == "momentum":
        base_cfg = settings.get("strategies", {}).get("momentum", {})
        momentum_cfg = {**base_cfg, "instrument": symbol}
        strategy = MomentumStrategy(name=strategy_id, config=momentum_cfg)
        capital = float(momentum_cfg.get("capital_per_trade", 200000) or 200000)
        return strategy, capital
    if strategy_id == "iron_condor":
        base_cfg = settings.get("strategies", {}).get("iron_condor", {})
        ic_cfg = {**base_cfg, "instrument": symbol}
        strategy = IronCondorStrategy(name=strategy_id, config=ic_cfg)
        capital = float(ic_cfg.get("capital_per_trade", 100000) or 100000)
        return strategy, capital
    raise ValueError(f"Unsupported strategy '{strategy_id}'. Available: regime_probe, momentum, iron_condor")


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
    option_chain = store.read_time_series(
        "option_chain",
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
    )

    simulator = FillSimulator(
        slippage_pct=float(backtest_cfg.get("slippage_pct", 0.05) or 0.05),
        commission_per_order=float(backtest_cfg.get("commission_per_order", 20.0) or 20.0),
        stt_pct=float(backtest_cfg.get("stt_pct", 0.0125) or 0.0125),
        exchange_txn_charges_pct=float(backtest_cfg.get("exchange_txn_charges_pct", 0.053) or 0.053),
        gst_pct=float(backtest_cfg.get("gst_pct", 18.0) or 18.0),
        sebi_fee_pct=float(backtest_cfg.get("sebi_fee_pct", 0.0001) or 0.0001),
        stamp_duty_pct=float(backtest_cfg.get("stamp_duty_pct", 0.003) or 0.003),
    )
    output_dir = str(REPO_ROOT / args.output_dir)
    periods_per_year = 252 if args.timeframe == "1d" else 252 * 75
    risk_free_rate_annual = float(backtest_cfg.get("risk_free_rate_annual", 0.07) or 0.07)
    monte_carlo_permutations = int(backtest_cfg.get("monte_carlo_permutations", 200) or 200)
    minimum_trade_count = int(backtest_cfg.get("minimum_trade_count", 50) or 50)

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
            fold_regime_rows: list[pd.DataFrame] = []
            for i, w in enumerate(windows, start=1):
                classifier = RegimeClassifier(thresholds=RegimeThresholds.from_config(settings.get("regime", {})))
                strategy_fold, _ = build_strategy(strategy_id, settings=settings, symbol=args.symbol)
                engine = BacktestEngine(
                    classifier=classifier,
                    strategy=strategy_fold,
                    simulator=simulator,
                    initial_capital=initial_capital,
                    periods_per_year=periods_per_year,
                    risk_free_rate_annual=risk_free_rate_annual,
                    monte_carlo_permutations=monte_carlo_permutations,
                    minimum_trade_count=minimum_trade_count,
                    circuit_breaker=CircuitBreaker(
                        initial_capital=initial_capital,
                        max_daily_loss_pct=float(settings.get("risk", {}).get("max_daily_loss_pct", 3.0) or 3.0),
                        max_drawdown_pct=float(settings.get("risk", {}).get("max_drawdown_pct", 15.0) or 15.0),
                        max_open_positions=int(settings.get("risk", {}).get("max_open_positions", 4) or 4),
                    ),
                )
                test_candles = candles.loc[
                    (candles["timestamp"] >= pd.Timestamp(w.test_start))
                    & (candles["timestamp"] < pd.Timestamp(w.test_end))
                ]
                result = engine.run(candles=test_candles, vix_df=vix, fii_df=fii, option_chain_df=option_chain)
                row = {
                    "fold": i,
                    "train_start": w.train_start.isoformat(),
                    "train_end": w.train_end.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    **result.metrics,
                }
                if not result.regimes.empty:
                    regime_counts = result.regimes["regime"].value_counts(normalize=True)
                    row["dominant_regime"] = str(regime_counts.index[0])
                    row["dominant_regime_share"] = float(regime_counts.iloc[0])
                fold_rows.append(row)
                seg = regime_segmented_returns(result.equity_curve, result.regimes)
                if not seg.empty:
                    seg = seg.copy()
                    seg["fold"] = i
                    fold_regime_rows.append(seg)

            fold_df = pd.DataFrame(fold_rows)
            summary = aggregate_walk_forward_metrics(fold_rows)
            regime_table = pd.DataFrame()
            if fold_regime_rows:
                all_seg = pd.concat(fold_regime_rows, ignore_index=True)
                regime_table = (
                    all_seg.groupby("regime", as_index=False)
                    .agg(
                        folds=("fold", "nunique"),
                        bars=("bars", "sum"),
                        mean_bar_return_pct=("mean_bar_return_pct", "mean"),
                        cumulative_return_pct=("cumulative_return_pct", "mean"),
                    )
                    .sort_values("regime")
                    .reset_index(drop=True)
                )
            run_name = f"{strategy_id}_{args.symbol.lower()}_{args.timeframe}_{start.date()}_{end.date()}_walkforward"
            paths = write_walkforward_report(
                folds=fold_df,
                summary=summary,
                regime_table=regime_table,
                output_dir=output_dir,
                run_name=run_name,
            )

            print(f"\n[{strategy_id}] Walk-forward complete")
            print(f"  folds={len(fold_df)} return_mean={summary.get('total_return_pct_mean', 0.0):.2f}")
            for k, p in paths.items():
                print(f"  {k}: {p}")
        else:
            classifier = RegimeClassifier(thresholds=RegimeThresholds.from_config(settings.get("regime", {})))
            engine = BacktestEngine(
                classifier=classifier,
                strategy=strategy,
                simulator=simulator,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
                risk_free_rate_annual=risk_free_rate_annual,
                monte_carlo_permutations=monte_carlo_permutations,
                minimum_trade_count=minimum_trade_count,
                circuit_breaker=CircuitBreaker(
                    initial_capital=initial_capital,
                    max_daily_loss_pct=float(settings.get("risk", {}).get("max_daily_loss_pct", 3.0) or 3.0),
                    max_drawdown_pct=float(settings.get("risk", {}).get("max_drawdown_pct", 15.0) or 15.0),
                    max_open_positions=int(settings.get("risk", {}).get("max_open_positions", 4) or 4),
                ),
            )
            result = engine.run(candles=candles, vix_df=vix, fii_df=fii, option_chain_df=option_chain)

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
