"""Run backtest and walk-forward analysis from cached parquet datasets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtest import (
    BacktestEngine,
    FillSimulator,
    aggregate_cross_instrument_results,
    aggregate_walk_forward_metrics,
    build_sensitivity_variants,
    generate_walk_forward_windows,
    regime_segmented_returns,
    summarize_sensitivity_results,
    write_backtest_report,
    write_walkforward_report,
)
from src.data.candle_access import CandleStores, build_candle_stores, read_candles
from src.data.option_chain_quality import OptionChainQualityThresholds
from src.data.store import DataStore
from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.risk.circuit_breaker import CircuitBreaker
from src.strategies.baseline_trend import BaselineTrendStrategy
from src.strategies.iron_condor import IronCondorStrategy
from src.strategies.jade_lizard import JadeLizardStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime_probe import RegimeProbeStrategy

DEFAULT_SENSITIVITY_PARAMS = {
    "iron_condor": ["call_delta", "put_delta", "wing_width", "profit_target_pct", "stop_loss_pct"],
    "jade_lizard": [
        "call_delta",
        "put_delta",
        "spread_width",
        "profit_target_pct",
        "stop_loss_pct",
    ],
    "momentum": ["fast_ema", "slow_ema", "adx_filter", "atr_multiplier"],
    "baseline_trend": ["adx_min", "vix_max"],
    "regime_probe": ["lots"],
}


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest scaffold from cached data.")
    parser.add_argument("--symbol", default="NIFTY", help="Primary symbol partition")
    parser.add_argument(
        "--symbols",
        action="append",
        help=(
            "Optional cross-instrument symbols; repeatable or comma-separated "
            "(e.g. NIFTY,BANKNIFTY)"
        ),
    )
    parser.add_argument("--timeframe", default="1d", help="Candle timeframe partition")
    parser.add_argument("--from", dest="start", type=parse_date, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end", type=parse_date, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--indicator-warmup-days",
        type=int,
        default=0,
        help=(
            "Extra days loaded before --from to warm indicators "
            "(excluded from backtest metrics/trades)."
        ),
    )
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        help=(
            "Strategy id; repeatable or comma-separated "
            "(available: regime_probe, baseline_trend, momentum, iron_condor, jade_lizard)"
        ),
    )
    parser.add_argument(
        "--walk-forward", action="store_true", help="Run walk-forward windows instead of single run"
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run parameter sensitivity variants (default: from settings)",
    )
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings file path")
    parser.add_argument("--output-dir", default="data/reports", help="Report output directory")
    parser.add_argument(
        "--no-timestamp-subdir",
        action="store_true",
        help="Write artifacts directly in --output-dir instead of run timestamp subfolder.",
    )
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
    for value in values:
        out.extend([part.strip() for part in value.split(",") if part.strip()])
    return out or ["regime_probe"]


def resolve_symbols(primary_symbol: str, values: list[str] | None) -> list[str]:
    if not values:
        return [primary_symbol]
    out: list[str] = []
    for value in values:
        out.extend([part.strip().upper() for part in value.split(",") if part.strip()])
    if primary_symbol.upper() not in out:
        out.insert(0, primary_symbol.upper())

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in out:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def build_strategy(
    strategy_id: str,
    *,
    settings: dict,
    symbol: str,
    timeframe: str = "1d",
    config_overrides: dict | None = None,
):
    config_overrides = config_overrides or {}
    if strategy_id == "regime_probe":
        base_cfg = settings.get("strategies", {}).get("momentum", {})
        probe_cfg = {
            **base_cfg,
            "instrument": symbol,
            "timeframe": timeframe,
            "lots": base_cfg.get("max_lots", 1),
            **config_overrides,
        }
        strategy = RegimeProbeStrategy(name=strategy_id, config=probe_cfg)
        capital = float(probe_cfg.get("capital_per_trade", 200000) or 200000)
        return strategy, capital, probe_cfg
    if strategy_id == "momentum":
        base_cfg = settings.get("strategies", {}).get("momentum", {})
        momentum_cfg = {
            **base_cfg,
            "instrument": symbol,
            "timeframe": timeframe,
            **config_overrides,
        }
        strategy = MomentumStrategy(name=strategy_id, config=momentum_cfg)
        capital = float(momentum_cfg.get("capital_per_trade", 200000) or 200000)
        return strategy, capital, momentum_cfg
    if strategy_id == "baseline_trend":
        base_cfg = settings.get("strategies", {}).get("baseline_trend", {})
        baseline_cfg = {
            **base_cfg,
            "instrument": symbol,
            "timeframe": timeframe,
            **config_overrides,
        }
        strategy = BaselineTrendStrategy(name=strategy_id, config=baseline_cfg)
        capital = float(baseline_cfg.get("capital_per_trade", 150000) or 150000)
        return strategy, capital, baseline_cfg
    if strategy_id == "iron_condor":
        base_cfg = settings.get("strategies", {}).get("iron_condor", {})
        ic_cfg = {**base_cfg, "instrument": symbol, "timeframe": timeframe, **config_overrides}
        strategy = IronCondorStrategy(name=strategy_id, config=ic_cfg)
        capital = float(ic_cfg.get("capital_per_trade", 100000) or 100000)
        return strategy, capital, ic_cfg
    if strategy_id == "jade_lizard":
        base_cfg = settings.get("strategies", {}).get("jade_lizard", {})
        jl_cfg = {**base_cfg, "instrument": symbol, "timeframe": timeframe, **config_overrides}
        strategy = JadeLizardStrategy(name=strategy_id, config=jl_cfg)
        capital = float(jl_cfg.get("capital_per_trade", 100000) or 100000)
        return strategy, capital, jl_cfg
    raise ValueError(
        "Unsupported strategy "
        f"'{strategy_id}'. Available: regime_probe, baseline_trend, momentum, "
        "iron_condor, jade_lizard"
    )


def _load_symbol_data(
    *,
    candle_stores: CandleStores,
    raw_store: DataStore,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candles, candle_source = read_candles(
        stores=candle_stores,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    if not candles.empty:
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], errors="coerce")
        candles = (
            candles.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        )
    print(f"  candles_source={candle_source}")

    option_chain = raw_store.read_time_series(
        "option_chain",
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    return candles, option_chain


def _build_engine(
    *,
    settings: dict,
    simulator: FillSimulator,
    initial_capital: float,
    strategy,
    periods_per_year: int,
    risk_free_rate_annual: float,
    monte_carlo_permutations: int,
    minimum_trade_count: int,
) -> BacktestEngine:
    risk_cfg = settings.get("risk", {})
    chain_quality_thresholds = OptionChainQualityThresholds.from_config(
        settings.get("data_quality", {}).get("option_chain", {})
    )
    return BacktestEngine(
        classifier=RegimeClassifier(
            thresholds=RegimeThresholds.from_config(settings.get("regime", {}))
        ),
        strategy=strategy,
        simulator=simulator,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=risk_free_rate_annual,
        monte_carlo_permutations=monte_carlo_permutations,
        minimum_trade_count=minimum_trade_count,
        circuit_breaker=CircuitBreaker(
            initial_capital=initial_capital,
            max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 3.0) or 3.0),
            max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 15.0) or 15.0),
            max_open_positions=int(risk_cfg.get("max_open_positions", 4) or 4),
        ),
        chain_quality_thresholds=chain_quality_thresholds,
    )


def _run_single_backtest(
    *,
    settings: dict,
    simulator: FillSimulator,
    strategy,
    initial_capital: float,
    candles: pd.DataFrame,
    vix: pd.DataFrame,
    fii: pd.DataFrame,
    usdinr: pd.DataFrame,
    option_chain: pd.DataFrame,
    analysis_start: datetime,
    periods_per_year: int,
    risk_free_rate_annual: float,
    monte_carlo_permutations: int,
    minimum_trade_count: int,
    output_dir: str,
    run_name: str,
    write_report: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    engine = _build_engine(
        settings=settings,
        simulator=simulator,
        initial_capital=initial_capital,
        strategy=strategy,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=risk_free_rate_annual,
        monte_carlo_permutations=monte_carlo_permutations,
        minimum_trade_count=minimum_trade_count,
    )
    result = engine.run(
        candles=candles,
        vix_df=vix,
        fii_df=fii,
        usdinr_df=usdinr,
        option_chain_df=option_chain,
        analysis_start=analysis_start,
    )
    paths: dict[str, str] = {}
    if write_report:
        paths = write_backtest_report(result, output_dir=output_dir, run_name=run_name)
    return result.metrics, paths


def _run_walk_forward_backtest(
    *,
    settings: dict,
    simulator: FillSimulator,
    strategy_id: str,
    symbol: str,
    candles: pd.DataFrame,
    vix: pd.DataFrame,
    fii: pd.DataFrame,
    usdinr: pd.DataFrame,
    option_chain: pd.DataFrame,
    start: datetime,
    end: datetime,
    backtest_cfg: dict,
    periods_per_year: int,
    risk_free_rate_annual: float,
    monte_carlo_permutations: int,
    minimum_trade_count: int,
    output_dir: str,
    run_name: str,
    write_report: bool,
    timeframe: str,
    indicator_warmup_days: int = 0,
) -> tuple[dict[str, float], dict[str, str]]:
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
    for i, window in enumerate(windows, start=1):
        strategy_fold, capital_fold, _ = build_strategy(
            strategy_id, settings=settings, symbol=symbol, timeframe=timeframe
        )
        engine = _build_engine(
            settings=settings,
            simulator=simulator,
            initial_capital=capital_fold,
            strategy=strategy_fold,
            periods_per_year=periods_per_year,
            risk_free_rate_annual=risk_free_rate_annual,
            monte_carlo_permutations=monte_carlo_permutations,
            minimum_trade_count=minimum_trade_count,
        )
        test_candles = candles.loc[
            (
                candles["timestamp"]
                >= pd.Timestamp(window.test_start - timedelta(days=indicator_warmup_days))
            )
            & (candles["timestamp"] < pd.Timestamp(window.test_end))
        ]
        if test_candles.empty:
            continue

        result = engine.run(
            candles=test_candles,
            vix_df=vix,
            fii_df=fii,
            usdinr_df=usdinr,
            option_chain_df=option_chain,
            analysis_start=window.test_start,
        )
        row = {
            "fold": i,
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "test_start": window.test_start.isoformat(),
            "test_end": window.test_end.isoformat(),
            **result.metrics,
        }
        if not result.regimes.empty:
            regime_counts = result.regimes["regime"].value_counts(normalize=True)
            row["dominant_regime"] = str(regime_counts.index[0])
            row["dominant_regime_share"] = float(regime_counts.iloc[0])
        fold_rows.append(row)

        segmented = regime_segmented_returns(result.equity_curve, result.regimes)
        if not segmented.empty:
            segmented = segmented.copy()
            segmented["fold"] = i
            fold_regime_rows.append(segmented)

    if not fold_rows:
        return {"folds": 0.0}, {}

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

    paths: dict[str, str] = {}
    if write_report:
        paths = write_walkforward_report(
            folds=fold_df,
            summary=summary,
            regime_table=regime_table,
            output_dir=output_dir,
            run_name=run_name,
        )
    return summary, paths


def _normalized_metric(metrics: dict[str, float], *, walk_forward: bool, metric: str) -> float:
    if walk_forward:
        key = f"{metric}_mean"
        if key in metrics:
            return float(metrics[key])
    if metric in metrics:
        return float(metrics[metric])
    return 0.0


def _resolve_sensitivity_params(
    *,
    strategy_id: str,
    strategy_cfg: dict,
    sensitivity_cfg: dict,
) -> list[str]:
    params_by_strategy = sensitivity_cfg.get("params_by_strategy", {})
    configured = params_by_strategy.get(strategy_id)
    candidates = configured if configured else DEFAULT_SENSITIVITY_PARAMS.get(strategy_id, [])
    params: list[str] = []
    for param in candidates:
        if param not in strategy_cfg:
            continue
        value = strategy_cfg[param]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            params.append(param)

    max_params = int(sensitivity_cfg.get("max_params_per_strategy", 3) or 3)
    if max_params > 0:
        params = params[:max_params]
    return params


def _run_strategy(
    *,
    walk_forward: bool,
    settings: dict,
    simulator: FillSimulator,
    strategy_id: str,
    symbol: str,
    strategy,
    initial_capital: float,
    candles: pd.DataFrame,
    vix: pd.DataFrame,
    fii: pd.DataFrame,
    usdinr: pd.DataFrame,
    option_chain: pd.DataFrame,
    analysis_start: datetime,
    start: datetime,
    end: datetime,
    backtest_cfg: dict,
    periods_per_year: int,
    risk_free_rate_annual: float,
    monte_carlo_permutations: int,
    minimum_trade_count: int,
    output_dir: str,
    run_name: str,
    write_report: bool,
    timeframe: str,
    indicator_warmup_days: int = 0,
) -> tuple[dict[str, float], dict[str, str]]:
    if walk_forward:
        return _run_walk_forward_backtest(
            settings=settings,
            simulator=simulator,
            strategy_id=strategy_id,
            symbol=symbol,
            candles=candles,
            vix=vix,
            fii=fii,
            usdinr=usdinr,
            option_chain=option_chain,
            start=start,
            end=end,
            backtest_cfg=backtest_cfg,
            periods_per_year=periods_per_year,
            risk_free_rate_annual=risk_free_rate_annual,
            monte_carlo_permutations=monte_carlo_permutations,
            minimum_trade_count=minimum_trade_count,
            output_dir=output_dir,
            run_name=run_name,
            write_report=write_report,
            timeframe=timeframe,
            indicator_warmup_days=indicator_warmup_days,
        )
    return _run_single_backtest(
        settings=settings,
        simulator=simulator,
        strategy=strategy,
        initial_capital=initial_capital,
        candles=candles,
        vix=vix,
        fii=fii,
        usdinr=usdinr,
        option_chain=option_chain,
        analysis_start=analysis_start,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=risk_free_rate_annual,
        monte_carlo_permutations=monte_carlo_permutations,
        minimum_trade_count=minimum_trade_count,
        output_dir=output_dir,
        run_name=run_name,
        write_report=write_report,
    )


def resolve_output_dir(*, raw_output_dir: str, run_prefix: str, no_timestamp_subdir: bool) -> Path:
    base = Path(raw_output_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    if no_timestamp_subdir:
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"{run_prefix}_{stamp}"


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    strategy_ids = resolve_strategies(args.strategies)
    symbols = resolve_symbols(args.symbol.upper(), args.symbols)

    candle_stores = build_candle_stores(settings=settings, repo_root=REPO_ROOT)
    raw_store = candle_stores.raw

    backtest_cfg = settings.get("backtest", {})
    start = args.start or parse_date(backtest_cfg.get("start_date", "2022-01-01"))
    end = args.end or parse_date(backtest_cfg.get("end_date", "2025-12-31"))
    load_start = start - timedelta(days=max(int(args.indicator_warmup_days), 0))

    vix = raw_store.read_time_series(
        "vix", symbol="INDIAVIX", timeframe="1d", start=load_start, end=end
    )
    fii = raw_store.read_time_series(
        "fii_dii",
        symbol="NSE",
        timeframe="1d",
        start=load_start,
        end=end,
        timestamp_col="date",
    )
    usdinr_symbol = str(settings.get("market", {}).get("usdinr_symbol", "USDINR")).upper()
    usdinr = raw_store.read_time_series(
        "candles", symbol=usdinr_symbol, timeframe="1d", start=load_start, end=end
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
    run_prefix = "walkforward" if args.walk_forward else "backtest"
    output_dir_path = resolve_output_dir(
        raw_output_dir=args.output_dir,
        run_prefix=run_prefix,
        no_timestamp_subdir=args.no_timestamp_subdir,
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir_path)

    periods_per_year = 252 if args.timeframe == "1d" else 252 * 75
    risk_free_rate_annual = float(backtest_cfg.get("risk_free_rate_annual", 0.07) or 0.07)
    monte_carlo_permutations = int(backtest_cfg.get("monte_carlo_permutations", 200) or 200)
    minimum_trade_count = int(backtest_cfg.get("minimum_trade_count", 50) or 50)

    sensitivity_cfg = backtest_cfg.get("sensitivity", {})
    sensitivity_enabled = bool(args.sensitivity or sensitivity_cfg.get("enabled", False))
    sensitivity_multipliers = [
        float(v) for v in sensitivity_cfg.get("multipliers", [0.8, 1.0, 1.2])
    ]

    print("=" * 72)
    print("Backtest Run")
    print("=" * 72)
    print(f"symbols={symbols} timeframe={args.timeframe} start={start.date()} end={end.date()}")
    print(f"load_start={load_start.date()} indicator_warmup_days={args.indicator_warmup_days}")
    print(f"output_dir={output_dir_path}")
    print(f"strategies={strategy_ids}")
    print(f"walk_forward={args.walk_forward} sensitivity={sensitivity_enabled}")

    runs_completed = 0
    cross_rows: list[dict] = []
    sensitivity_overview_rows: list[dict] = []

    for symbol in symbols:
        candles, option_chain = _load_symbol_data(
            candle_stores=candle_stores,
            raw_store=raw_store,
            symbol=symbol,
            timeframe=args.timeframe,
            start=load_start,
            end=end,
        )
        if candles.empty:
            print(f"\n[{symbol}] No candle data in cache for requested window; skipping symbol")
            continue

        for strategy_id in strategy_ids:
            strategy, initial_capital, strategy_cfg = build_strategy(
                strategy_id, settings=settings, symbol=symbol, timeframe=args.timeframe
            )
            run_suffix = "walkforward" if args.walk_forward else "backtest"
            run_name = (
                f"{strategy_id}_{symbol.lower()}_{args.timeframe}_"
                f"{start.date()}_{end.date()}_{run_suffix}"
            )

            metrics, paths = _run_strategy(
                walk_forward=args.walk_forward,
                settings=settings,
                simulator=simulator,
                strategy_id=strategy_id,
                symbol=symbol,
                strategy=strategy,
                initial_capital=initial_capital,
                candles=candles,
                vix=vix,
                fii=fii,
                usdinr=usdinr,
                option_chain=option_chain,
                analysis_start=start,
                start=start,
                end=end,
                backtest_cfg=backtest_cfg,
                periods_per_year=periods_per_year,
                risk_free_rate_annual=risk_free_rate_annual,
                monte_carlo_permutations=monte_carlo_permutations,
                minimum_trade_count=minimum_trade_count,
                output_dir=output_dir,
                run_name=run_name,
                write_report=True,
                timeframe=args.timeframe,
                indicator_warmup_days=max(int(args.indicator_warmup_days), 0),
            )
            runs_completed += 1

            total_return = _normalized_metric(
                metrics, walk_forward=args.walk_forward, metric="total_return_pct"
            )
            sharpe = _normalized_metric(
                metrics, walk_forward=args.walk_forward, metric="sharpe_ratio"
            )
            max_dd = _normalized_metric(
                metrics, walk_forward=args.walk_forward, metric="max_drawdown_pct"
            )
            anti_overfit = _normalized_metric(
                metrics, walk_forward=args.walk_forward, metric="anti_overfit_pass"
            )

            print(f"\n[{strategy_id}] symbol={symbol}")
            print(
                "  "
                f"total_return_pct={total_return:.2f} "
                f"sharpe={sharpe:.3f} "
                f"max_drawdown_pct={max_dd:.2f}"
            )
            for key, path in paths.items():
                print(f"  {key}: {path}")

            cross_rows.append(
                {
                    "strategy": strategy_id,
                    "symbol": symbol,
                    "timeframe": args.timeframe,
                    "walk_forward": float(args.walk_forward),
                    "total_return_pct": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown_pct": max_dd,
                    "anti_overfit_pass": anti_overfit,
                }
            )

            if not sensitivity_enabled:
                continue

            params = _resolve_sensitivity_params(
                strategy_id=strategy_id,
                strategy_cfg=strategy_cfg,
                sensitivity_cfg=sensitivity_cfg,
            )
            variants = build_sensitivity_variants(
                base_config=strategy_cfg,
                params=params,
                multipliers=sensitivity_multipliers,
            )
            if not variants:
                print("  sensitivity: no eligible numeric params configured")
                continue

            variant_rows: list[dict] = []
            base_total_return = total_return
            for variant in variants:
                variant_strategy, variant_capital, _ = build_strategy(
                    strategy_id,
                    settings=settings,
                    symbol=symbol,
                    timeframe=args.timeframe,
                    config_overrides=variant["overrides"],
                )
                variant_metrics, _ = _run_strategy(
                    walk_forward=args.walk_forward,
                    settings=settings,
                    simulator=simulator,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    strategy=variant_strategy,
                    initial_capital=variant_capital,
                    candles=candles,
                    vix=vix,
                    fii=fii,
                    usdinr=usdinr,
                    option_chain=option_chain,
                    analysis_start=start,
                    start=start,
                    end=end,
                    backtest_cfg=backtest_cfg,
                    periods_per_year=periods_per_year,
                    risk_free_rate_annual=risk_free_rate_annual,
                    monte_carlo_permutations=monte_carlo_permutations,
                    minimum_trade_count=minimum_trade_count,
                    output_dir=output_dir,
                    run_name=f"{run_name}_sens_{variant['variant_id']}",
                    write_report=False,
                    timeframe=args.timeframe,
                    indicator_warmup_days=max(int(args.indicator_warmup_days), 0),
                )
                variant_rows.append(
                    {
                        "strategy": strategy_id,
                        "symbol": symbol,
                        "variant_id": variant["variant_id"],
                        "param": variant["param"],
                        "multiplier": variant["multiplier"],
                        "base_value": variant["base_value"],
                        "new_value": variant["new_value"],
                        "total_return_pct": _normalized_metric(
                            variant_metrics,
                            walk_forward=args.walk_forward,
                            metric="total_return_pct",
                        ),
                        "sharpe_ratio": _normalized_metric(
                            variant_metrics,
                            walk_forward=args.walk_forward,
                            metric="sharpe_ratio",
                        ),
                        "max_drawdown_pct": _normalized_metric(
                            variant_metrics,
                            walk_forward=args.walk_forward,
                            metric="max_drawdown_pct",
                        ),
                    }
                )

            variant_df = pd.DataFrame(variant_rows)
            sensitivity_base_name = (
                f"{strategy_id}_{symbol.lower()}_{args.timeframe}_"
                f"{start.date()}_{end.date()}_sensitivity"
            )
            sensitivity_csv = Path(output_dir) / f"{sensitivity_base_name}.csv"
            variant_df.to_csv(sensitivity_csv, index=False)

            sensitivity_summary = summarize_sensitivity_results(
                variant_rows=variant_rows,
                base_total_return_pct=base_total_return,
            )
            sensitivity_summary.update(
                {
                    "strategy": strategy_id,
                    "symbol": symbol,
                    "timeframe": args.timeframe,
                    "walk_forward": float(args.walk_forward),
                    "base_total_return_pct": float(base_total_return),
                }
            )
            sensitivity_json = Path(output_dir) / f"{sensitivity_base_name}_summary.json"
            sensitivity_json.write_text(json.dumps(sensitivity_summary, indent=2, sort_keys=True))
            print(f"  sensitivity_csv: {sensitivity_csv}")
            print(f"  sensitivity_summary_json: {sensitivity_json}")
            sensitivity_overview_rows.append(sensitivity_summary)

    if runs_completed == 0:
        print("No backtests executed (missing candle data for selected symbols/date range).")
        return 1

    if len(strategy_ids) > 1 and cross_rows:
        comparison_df = (
            pd.DataFrame(cross_rows)
            .sort_values(
                ["symbol", "total_return_pct", "sharpe_ratio"],
                ascending=[True, False, False],
            )
            .reset_index(drop=True)
        )
        comparison_stamp = f"{args.timeframe}_{start.date()}_{end.date()}"
        comparison_csv = Path(output_dir) / f"strategy_comparison_{comparison_stamp}.csv"
        comparison_json = Path(output_dir) / f"strategy_comparison_{comparison_stamp}.json"
        comparison_df.to_csv(comparison_csv, index=False)
        comparison_json.write_text(
            json.dumps(comparison_df.to_dict(orient="records"), indent=2, sort_keys=True)
        )
        print("\nStrategy comparison")
        print(f"  comparison_csv: {comparison_csv}")
        print(f"  comparison_json: {comparison_json}")

    if len(symbols) > 1 and cross_rows:
        cross_detail = (
            pd.DataFrame(cross_rows).sort_values(["strategy", "symbol"]).reset_index(drop=True)
        )
        cross_summary = aggregate_cross_instrument_results(cross_rows)
        stamp = f"{args.timeframe}_{start.date()}_{end.date()}"

        detail_csv = Path(output_dir) / f"cross_instrument_detail_{stamp}.csv"
        summary_csv = Path(output_dir) / f"cross_instrument_summary_{stamp}.csv"
        summary_json = Path(output_dir) / f"cross_instrument_summary_{stamp}.json"

        cross_detail.to_csv(detail_csv, index=False)
        cross_summary.to_csv(summary_csv, index=False)
        summary_json.write_text(
            json.dumps(cross_summary.to_dict(orient="records"), indent=2, sort_keys=True)
        )

        print("\nCross-instrument validation")
        print(f"  detail_csv: {detail_csv}")
        print(f"  summary_csv: {summary_csv}")
        print(f"  summary_json: {summary_json}")

    if sensitivity_overview_rows:
        overview_df = (
            pd.DataFrame(sensitivity_overview_rows)
            .sort_values(["strategy", "symbol"])
            .reset_index(drop=True)
        )
        overview_path = (
            Path(output_dir)
            / f"sensitivity_overview_{args.timeframe}_{start.date()}_{end.date()}.csv"
        )
        overview_df.to_csv(overview_path, index=False)
        print("\nSensitivity overview")
        print(f"  overview_csv: {overview_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
