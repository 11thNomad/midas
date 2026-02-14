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
    build_trade_attribution,
    evaluate_vectorbt_promotion_gate,
    parse_parameter_sets,
    parse_vectorbt_fee_profiles,
    parse_vectorbt_promotion_gate_config,
    rank_parameter_results,
    resolve_vectorbt_costs,
    run_vectorbt_research,
    run_vectorbt_walk_forward,
    select_vectorbt_fee_profiles,
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
        "--fee-profiles",
        default=None,
        help=(
            "Comma-separated fee profiles from settings "
            "backtest.vectorbt_fee_profiles (default: run all configured profiles)."
        ),
    )
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


def summarize_set_robustness(leaderboard: pd.DataFrame, *, walk_forward: bool) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard
    metric = "wf_total_return_pct_mean" if walk_forward else "total_return_pct"
    if metric not in leaderboard.columns:
        metric = "total_return_pct"

    frame = leaderboard.copy()
    frame[metric] = pd.to_numeric(frame.get(metric), errors="coerce")
    if "max_drawdown_pct" in frame.columns:
        frame["max_drawdown_pct"] = pd.to_numeric(frame["max_drawdown_pct"], errors="coerce")
    frame["eligible"] = frame["eligible"].astype(bool)

    grouped = frame.groupby("set_id", as_index=False).agg(
        profiles_tested=("fee_profile", "nunique"),
        eligible_profiles=("eligible", "sum"),
        metric_mean=(metric, "mean"),
        metric_median=(metric, "median"),
        metric_worst=(metric, "min"),
        metric_best=(metric, "max"),
        trades_mean=("trades", "mean"),
    )
    grouped["eligible_profile_share"] = (
        grouped["eligible_profiles"] / grouped["profiles_tested"].replace(0, pd.NA)
    )

    if "max_drawdown_pct" in frame.columns:
        drawdown_abs = frame.assign(max_drawdown_abs=frame["max_drawdown_pct"].abs())
        dd_summary = drawdown_abs.groupby("set_id", as_index=False).agg(
            drawdown_abs_mean=("max_drawdown_abs", "mean"),
            drawdown_abs_worst=("max_drawdown_abs", "max"),
        )
        grouped = grouped.merge(dd_summary, on="set_id", how="left")

    return grouped.sort_values(
        by=["eligible_profile_share", "metric_worst", "metric_mean", "set_id"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    backtest_cfg = settings.get("backtest", {})
    default_fee_profile, all_fee_profiles = parse_vectorbt_fee_profiles(backtest_cfg)
    selected_profiles = select_vectorbt_fee_profiles(
        all_fee_profiles,
        args.fee_profiles if args.fee_profiles is not None else None,
    )
    if args.fee_profiles is None:
        selected_profiles = all_fee_profiles
    if not selected_profiles:
        raise ValueError(f"No fee profiles selected (default={default_fee_profile}).")

    start = args.start or parse_date(str(backtest_cfg.get("start_date", "2022-01-01")))
    end = args.end or parse_date(str(backtest_cfg.get("end_date", "2025-12-31")))

    cache_dir = REPO_ROOT / str(settings.get("data", {}).get("cache_dir", "data/cache"))
    store = DataStore(base_dir=str(cache_dir))
    snapshot_store = SignalSnapshotStore(base_dir=str(cache_dir))
    out_dir = resolve_output_dir(raw_output_dir=args.output_dir, run_prefix="vectorbt_paramsets")
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

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
        fees_pct=0.0,
        slippage_pct=0.0,
        entry_regimes=("low_vol_trending", "high_vol_trending"),
        adx_min=25.0,
        vix_max=None,
        freq=_timeframe_to_freq(args.timeframe),
    )

    rows: list[dict[str, Any]] = []
    for fee_profile in selected_profiles:
        fees_pct, slippage_pct = resolve_vectorbt_costs(
            backtest_cfg=backtest_cfg,
            profile=fee_profile,
        )
        print(
            f"fee_profile={fee_profile.name} fees_pct={fees_pct:.6f} "
            f"slippage_pct={slippage_pct:.6f}"
        )
        for params in param_sets:
            cfg = apply_parameter_set(base_config, params)
            cfg = VectorBTResearchConfig(
                initial_cash=cfg.initial_cash,
                fees_pct=fees_pct,
                slippage_pct=slippage_pct,
                entry_regimes=cfg.entry_regimes,
                adx_min=cfg.adx_min,
                vix_max=cfg.vix_max,
                freq=cfg.freq,
            )
            run = run_vectorbt_research(candles=candles, snapshots=snapshots, config=cfg)
            detail_prefix = f"{params.set_id}_{fee_profile.name}"
            schedule_path = details_dir / f"{detail_prefix}_schedule.csv"
            equity_path = details_dir / f"{detail_prefix}_equity.csv"
            trades_path = details_dir / f"{detail_prefix}_trades.csv"
            attribution_path = details_dir / f"{detail_prefix}_trade_attribution.csv"
            run.schedule.to_csv(schedule_path, index=False)
            run.equity_curve.to_csv(equity_path, index=False)
            if isinstance(run.trades, pd.DataFrame):
                run.trades.to_csv(trades_path, index=False)
                attribution = build_trade_attribution(
                    trades=run.trades,
                    schedule=run.schedule,
                    set_id=params.set_id,
                    fee_profile=fee_profile.name,
                )
            else:
                pd.DataFrame().to_csv(trades_path, index=False)
                attribution = pd.DataFrame()
            attribution.to_csv(attribution_path, index=False)

            row: dict[str, Any] = {
                "set_id": params.set_id,
                "fee_profile": fee_profile.name,
                "fee_multiplier": float(fee_profile.fee_multiplier),
                "slippage_multiplier": float(fee_profile.slippage_multiplier),
                "fees_pct_applied": float(fees_pct),
                "slippage_pct_applied": float(slippage_pct),
                "entry_regimes": ",".join(params.entry_regimes),
                "adx_min": float(params.adx_min),
                "vix_max": float(params.vix_max) if params.vix_max is not None else float("nan"),
                "vix_max_label": (
                    f"{float(params.vix_max):.2f}" if params.vix_max is not None else "none"
                ),
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
                folds_path = out_dir / f"{params.set_id}_{fee_profile.name}_walkforward_folds.csv"
                folds.to_csv(folds_path, index=False)
                for key, value in wf_summary.items():
                    row[f"wf_{key}"] = _as_float(value)

            rows.append(row)
            print(
                f"set={params.set_id} fee_profile={fee_profile.name} "
                f"trades={row.get('trades')} return={row.get('total_return_pct')}"
            )

    results_df = pd.DataFrame(rows)
    rank_by = args.rank_by or (
        "wf_total_return_pct_mean" if args.walk_forward else "total_return_pct"
    )
    if rank_by not in results_df.columns:
        fallback = "total_return_pct"
        if fallback not in results_df.columns:
            raise ValueError(
                f"rank_by column not found: {rank_by} and fallback column missing: {fallback}"
            )
        print(f"rank_by={rank_by} not found; falling back to {fallback}")
        rank_by = fallback
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
    robustness_df = summarize_set_robustness(leaderboard, walk_forward=args.walk_forward)
    top_sets_df = robustness_df.head(top_k).reset_index(drop=True)
    best_set = top_sets_df.iloc[0].to_dict() if not top_sets_df.empty else None
    gate_cfg = parse_vectorbt_promotion_gate_config(backtest_cfg)
    gate_df = (
        evaluate_vectorbt_promotion_gate(robustness_df, gate_cfg)
        if gate_cfg.enabled and not robustness_df.empty
        else pd.DataFrame()
    )
    approved_sets = (
        gate_df.loc[gate_df["promotion_pass"]].copy()
        if not gate_df.empty and "promotion_pass" in gate_df.columns
        else pd.DataFrame()
    )

    results_path = out_dir / "vectorbt_parameter_set_results.csv"
    leaderboard_path = out_dir / "vectorbt_parameter_set_leaderboard.csv"
    top_path = out_dir / "vectorbt_parameter_set_top.csv"
    robustness_path = out_dir / "vectorbt_parameter_set_robustness.csv"
    top_sets_path = out_dir / "vectorbt_parameter_set_top_sets.csv"
    gate_path = out_dir / "vectorbt_parameter_set_promotion_gate.csv"
    summary_path = out_dir / "vectorbt_parameter_set_summary.json"
    results_df.to_csv(results_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    top_df.to_csv(top_path, index=False)
    robustness_df.to_csv(robustness_path, index=False)
    top_sets_df.to_csv(top_sets_path, index=False)
    gate_df.to_csv(gate_path, index=False)
    try:
        details_dir_label = str(details_dir.relative_to(REPO_ROOT))
    except ValueError:
        details_dir_label = str(details_dir)
    summary_path.write_text(
        json.dumps(
            {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "walk_forward": bool(args.walk_forward),
                "fee_profiles": [profile.name for profile in selected_profiles],
                "rank_by": rank_by,
                "min_trades": float(args.min_trades),
                "max_drawdown_pct": (
                    float(args.max_drawdown_pct) if args.max_drawdown_pct is not None else None
                ),
                "parameter_set_count": int(len(param_sets)),
                "run_count": int(len(results_df)),
                "details_dir": details_dir_label,
                "eligible_count": int(len(eligible)),
                "best_eligible_set": best,
                "best_robust_set": best_set,
                "promotion_gate": {
                    "enabled": gate_cfg.enabled,
                    "min_trades_mean": gate_cfg.min_trades_mean,
                    "min_metric_worst": gate_cfg.min_metric_worst,
                    "min_metric_mean": gate_cfg.min_metric_mean,
                    "max_drawdown_abs_worst": gate_cfg.max_drawdown_abs_worst,
                    "min_eligible_profile_share": gate_cfg.min_eligible_profile_share,
                    "approved_set_count": int(len(approved_sets)),
                    "approved_sets": (
                        approved_sets["set_id"].astype(str).unique().tolist()
                        if not approved_sets.empty and "set_id" in approved_sets.columns
                        else []
                    ),
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print(f"results={results_path}")
    print(f"leaderboard={leaderboard_path}")
    print(f"top={top_path}")
    print(f"robustness={robustness_path}")
    print(f"top_sets={top_sets_path}")
    print(f"promotion_gate={gate_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
