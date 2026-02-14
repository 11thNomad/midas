"""Run VectorBT research over frozen signal snapshots."""

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
    FillSimulator,
    VectorBTResearchConfig,
    build_snapshots_from_market_data,
    parse_vectorbt_fee_profiles,
    resolve_vectorbt_costs,
    run_hybrid_from_schedule_result,
    run_vectorbt_research,
    run_vectorbt_sensitivity,
    run_vectorbt_walk_forward,
    select_vectorbt_fee_profiles,
)
from src.data.store import DataStore
from src.regime.classifier import RegimeThresholds
from src.regime.persistence import SignalSnapshotStore


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VectorBT research runner.")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--from", dest="start", type=parse_date)
    parser.add_argument("--to", dest="end", type=parse_date)
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--entry-regimes", default="low_vol_trending,high_vol_trending")
    parser.add_argument("--adx-min", type=float, default=25.0)
    parser.add_argument("--vix-max", type=float, default=None)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument(
        "--fee-profile",
        default=None,
        help=(
            "Fee profile name from settings backtest.vectorbt_fee_profiles "
            "(default: settings default)."
        ),
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text())


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


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")
    store = DataStore(base_dir=str(cache_dir))
    snapshot_store = SignalSnapshotStore(base_dir=str(cache_dir))

    backtest_cfg = settings.get("backtest", {})
    start = args.start or parse_date(backtest_cfg.get("start_date", "2022-01-01"))
    end = args.end or parse_date(backtest_cfg.get("end_date", "2025-12-31"))
    default_fee_profile, fee_profiles = parse_vectorbt_fee_profiles(backtest_cfg)
    selected_name = args.fee_profile or default_fee_profile
    selected = select_vectorbt_fee_profiles(fee_profiles, selected_name)
    if len(selected) != 1:
        raise ValueError("run_vectorbt_research accepts exactly one fee profile.")
    fee_profile = selected[0]
    fees_pct, slippage_pct = resolve_vectorbt_costs(backtest_cfg=backtest_cfg, profile=fee_profile)

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
        vix = store.read_time_series(
            "vix",
            symbol="INDIAVIX",
            timeframe="1d",
            start=start,
            end=end,
        )
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

    entry_regimes = tuple(r.strip() for r in args.entry_regimes.split(",") if r.strip())
    cfg = VectorBTResearchConfig(
        initial_cash=float(settings.get("risk", {}).get("initial_capital", 150_000.0) or 150_000.0),
        fees_pct=fees_pct,
        slippage_pct=slippage_pct,
        entry_regimes=entry_regimes,
        adx_min=float(args.adx_min),
        vix_max=args.vix_max,
        freq=_timeframe_to_freq(args.timeframe),
    )
    print(
        f"fee_profile={fee_profile.name} fees_pct={cfg.fees_pct:.6f} "
        f"slippage_pct={cfg.slippage_pct:.6f}"
    )

    out_dir = resolve_output_dir(raw_output_dir=args.output_dir, run_prefix="vectorbt")
    if args.walk_forward:
        folds, summary = run_vectorbt_walk_forward(
            candles=candles,
            snapshots=snapshots,
            config=cfg,
            start=start,
            end=end,
            train_months=int(backtest_cfg.get("train_months", 12)),
            test_months=int(backtest_cfg.get("test_months", 3)),
            step_months=int(backtest_cfg.get("step_months", 3)),
        )
        folds_path = out_dir / "vectorbt_walkforward_folds.csv"
        summary_path = out_dir / "vectorbt_walkforward_summary.json"
        folds.to_csv(folds_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"walkforward_folds={folds_path}")
        print(f"walkforward_summary={summary_path}")
        return 0

    result = run_vectorbt_research(candles=candles, snapshots=snapshots, config=cfg)
    metrics_path = out_dir / "vectorbt_metrics.json"
    schedule_path = out_dir / "vectorbt_schedule.csv"
    equity_path = out_dir / "vectorbt_equity.csv"
    trades_path = out_dir / "vectorbt_trades.csv"
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True))
    result.schedule.to_csv(schedule_path, index=False)
    result.equity_curve.to_csv(equity_path, index=False)
    if isinstance(result.trades, pd.DataFrame):
        result.trades.to_csv(trades_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)
    print(f"vectorbt_metrics={metrics_path}")
    print(f"vectorbt_schedule={schedule_path}")
    print(f"vectorbt_equity={equity_path}")
    multipliers = [
        float(v)
        for v in backtest_cfg.get("sensitivity", {}).get("multipliers", [0.8, 1.0, 1.2])
    ]
    sensitivity_df = run_vectorbt_sensitivity(
        candles=candles,
        snapshots=snapshots,
        base_config=cfg,
        multipliers=multipliers,
    )
    sensitivity_path = out_dir / "vectorbt_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"vectorbt_sensitivity={sensitivity_path}")

    if args.hybrid:
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
        simulator = FillSimulator(
            slippage_pct=float(backtest_cfg.get("slippage_pct", 0.05) or 0.05),
            commission_per_order=float(backtest_cfg.get("commission_per_order", 20.0) or 20.0),
            stt_pct=float(backtest_cfg.get("stt_pct", 0.0125) or 0.0125),
            exchange_txn_charges_pct=float(
                backtest_cfg.get("exchange_txn_charges_pct", 0.053) or 0.053
            ),
            gst_pct=float(backtest_cfg.get("gst_pct", 18.0) or 18.0),
            sebi_fee_pct=float(backtest_cfg.get("sebi_fee_pct", 0.0001) or 0.0001),
            stamp_duty_pct=float(backtest_cfg.get("stamp_duty_pct", 0.003) or 0.003),
        )
        hybrid_result = run_hybrid_from_schedule_result(
            candles=candles,
            schedule=result.schedule,
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_capital=float(
                settings.get("risk", {}).get("initial_capital", 150_000.0) or 150_000.0
            ),
            thresholds=RegimeThresholds.from_config(settings.get("regime", {})),
            simulator=simulator,
            vix_df=vix,
            fii_df=fii,
            usdinr_df=usdinr,
            option_chain_df=option_chain,
        )
        hybrid_metrics = hybrid_result.metrics
        hybrid_path = out_dir / "hybrid_metrics.json"
        hybrid_equity_path = out_dir / "hybrid_equity.csv"
        hybrid_fills_path = out_dir / "hybrid_fills.csv"
        hybrid_regimes_path = out_dir / "hybrid_regimes.csv"
        hybrid_snapshots_path = out_dir / "hybrid_signal_snapshots.csv"
        hybrid_path.write_text(json.dumps(hybrid_metrics, indent=2, sort_keys=True))
        hybrid_result.equity_curve.to_csv(hybrid_equity_path, index=False)
        hybrid_result.fills.to_csv(hybrid_fills_path, index=False)
        hybrid_result.regimes.to_csv(hybrid_regimes_path, index=False)
        hybrid_result.signal_snapshots.to_csv(hybrid_snapshots_path, index=False)
        print(f"hybrid_metrics={hybrid_path}")
        print(f"hybrid_equity={hybrid_equity_path}")
        print(f"hybrid_fills={hybrid_fills_path}")
        print(f"hybrid_regimes={hybrid_regimes_path}")
        print(f"hybrid_signal_snapshots={hybrid_snapshots_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
