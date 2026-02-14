"""Paper loop scaffold that exercises regime/runtime/strategy routing with paper execution.

Examples:
  python scripts/run_paper.py --symbol NIFTY --timeframe 5m --iterations 1
  python scripts/run_paper.py --symbol NIFTY --timeframe 5m --iterations 20 --sleep-seconds 30
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some environments
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.contracts import option_dtos_from_chain
from src.data.fii import FiiDownloadError, fetch_fii_dii
from src.data.kite_feed import KiteFeed, KiteFeedError
from src.data.store import DataStore
from src.execution import PaperExecutionEngine
from src.regime import (
    RegimeClassifier,
    RegimeRuntime,
    RegimeSnapshotStore,
    RegimeThresholds,
    SignalSnapshotStore,
    StrategyTransitionStore,
)
from src.risk.circuit_breaker import CircuitBreaker
from src.signals.regime import build_regime_signals
from src.strategies.base import BaseStrategy
from src.strategies.iron_condor import IronCondorStrategy
from src.strategies.jade_lizard import JadeLizardStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime_probe import RegimeProbeStrategy
from src.strategies.router import StrategyRouter

OPTION_CHAIN_DEDUP_COLS = ["timestamp", "expiry", "strike", "option_type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-loop scaffold (paper execution mode).")
    parser.add_argument("--symbol", default="NIFTY", help="Underlying symbol")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (e.g., 5m, 1d)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of loop iterations")
    parser.add_argument("--sleep-seconds", type=int, default=15, help="Sleep between iterations")
    parser.add_argument(
        "--lookback-days", type=int, default=180, help="History window for signal computation"
    )
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    return parser.parse_args()


def load_settings(path: str) -> dict:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text())


def build_strategies(settings: dict) -> list[BaseStrategy]:
    strategies_cfg = settings.get("strategies", {})
    strategies: list[BaseStrategy] = []
    unknown_enabled: list[str] = []
    for name, cfg in strategies_cfg.items():
        if not cfg.get("enabled", False):
            continue
        if name == "momentum":
            strategies.append(MomentumStrategy(name=name, config=cfg))
        elif name == "iron_condor":
            strategies.append(IronCondorStrategy(name=name, config=cfg))
        elif name == "jade_lizard":
            strategies.append(JadeLizardStrategy(name=name, config=cfg))
        elif name == "regime_probe":
            strategies.append(RegimeProbeStrategy(name=name, config=cfg))
        else:
            unknown_enabled.append(name)

    if unknown_enabled:
        known = ["regime_probe", "momentum", "iron_condor", "jade_lizard"]
        raise ValueError(
            "Unknown enabled strategy id(s): "
            f"{', '.join(sorted(unknown_enabled))}. "
            f"Known strategy ids: {', '.join(known)}"
        )
    return strategies


def _load_or_fetch_candles(
    *,
    store: DataStore,
    feed: KiteFeed,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    candles = store.read_time_series(
        "candles",
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    if not candles.empty:
        return candles

    fetched = feed.get_candles(symbol=symbol, timeframe=timeframe, start=start, end=end)
    if fetched.empty:
        return fetched
    store.write_time_series(
        "candles", fetched, symbol=symbol, timeframe=timeframe, source=feed.name
    )
    return fetched


def _load_or_fetch_vix(
    *, store: DataStore, feed: KiteFeed, start: datetime, end: datetime
) -> pd.DataFrame:
    vix = store.read_time_series("vix", symbol="INDIAVIX", timeframe="1d", start=start, end=end)
    if not vix.empty:
        return vix
    fetched = feed.get_vix(start=start, end=end)
    if fetched.empty:
        return fetched
    store.write_time_series("vix", fetched, symbol="INDIAVIX", timeframe="1d", source=feed.name)
    return fetched


def _load_or_fetch_fii(*, store: DataStore, start: datetime, end: datetime) -> pd.DataFrame:
    fii = store.read_time_series(
        "fii_dii",
        symbol="NSE",
        timeframe="1d",
        start=start,
        end=end,
        timestamp_col="date",
    )
    if not fii.empty:
        return fii
    fetched = fetch_fii_dii(start=start, end=end)
    if fetched.empty:
        return fetched
    store.write_time_series(
        "fii_dii",
        fetched,
        symbol="NSE",
        timeframe="1d",
        timestamp_col="date",
        source="nse",
    )
    return fetched


def _option_chain_dte_bounds(settings: dict) -> tuple[int, int | None]:
    strategies_cfg = settings.get("strategies", {})
    option_cfgs = [
        strategies_cfg.get("iron_condor", {}),
        strategies_cfg.get("jade_lizard", {}),
    ]
    enabled_cfgs = [cfg for cfg in option_cfgs if cfg and cfg.get("enabled", False)]
    if not enabled_cfgs:
        return 0, None

    mins: list[int] = []
    maxs: list[int] = []
    for cfg in enabled_cfgs:
        dte_min = int(cfg.get("dte_min", 5) or 5)
        dte_max = int(cfg.get("dte_max", 14) or 14)
        if dte_min > dte_max:
            dte_min, dte_max = dte_max, dte_min
        mins.append(dte_min)
        maxs.append(dte_max)

    return min(mins), max(maxs)


def _select_target_expiry(
    *,
    feed: KiteFeed,
    symbol: str,
    asof: datetime,
    dte_min: int,
    dte_max: int | None,
) -> datetime | None:
    expiries = feed.get_option_expiries(symbol=symbol, start=asof)
    if not expiries:
        return None

    in_window: list[datetime] = []
    beyond_min: list[datetime] = []
    for expiry in expiries:
        dte = (expiry.date() - asof.date()).days
        if dte < dte_min:
            continue
        beyond_min.append(expiry)
        if dte_max is None or dte <= dte_max:
            in_window.append(expiry)

    if in_window:
        return in_window[0]
    if beyond_min:
        return beyond_min[0]
    return expiries[0]


def _frame_from_option_chain(*, source: str, chain) -> pd.DataFrame:
    dtos = option_dtos_from_chain(chain, source=source)
    if not dtos:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "underlying",
                "underlying_price",
                "expiry",
                "strike",
                "option_type",
                "symbol",
                "ltp",
                "bid",
                "ask",
                "volume",
                "oi",
                "change_in_oi",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
                "source",
            ]
        )

    rows: list[dict] = []
    for dto, contract in zip(dtos, chain.contracts, strict=False):
        rows.append(
            {
                "timestamp": dto.timestamp,
                "underlying": dto.underlying,
                "underlying_price": chain.underlying_price,
                "expiry": dto.expiry,
                "strike": dto.strike,
                "option_type": dto.option_type,
                "symbol": contract.symbol,
                "ltp": dto.ltp,
                "bid": dto.bid,
                "ask": dto.ask,
                "volume": dto.volume,
                "oi": dto.oi,
                "change_in_oi": dto.change_in_oi,
                "iv": dto.iv,
                "delta": dto.delta,
                "gamma": dto.gamma,
                "theta": dto.theta,
                "vega": dto.vega,
                "rho": dto.rho,
                "source": dto.source,
            }
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["expiry"] = pd.to_datetime(frame["expiry"], errors="coerce")
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    frame["option_type"] = frame["option_type"].astype(str).str.upper()
    frame = frame.dropna(subset=["timestamp", "expiry", "strike"])
    return frame.sort_values(["timestamp", "expiry", "strike", "option_type"]).reset_index(
        drop=True
    )


def _load_latest_option_chain_snapshot(
    *,
    store: DataStore,
    symbol: str,
    timeframe: str,
    before: datetime,
    lookback_days: int = 30,
) -> pd.DataFrame:
    window_start = before - timedelta(days=lookback_days)
    cutoff = before - timedelta(microseconds=1)
    prior = store.read_time_series(
        "option_chain",
        symbol=symbol,
        timeframe=timeframe,
        start=window_start,
        end=cutoff,
        timestamp_col="timestamp",
    )
    if prior.empty or "timestamp" not in prior.columns:
        return pd.DataFrame()

    prior["timestamp"] = pd.to_datetime(prior["timestamp"], errors="coerce")
    prior = prior.dropna(subset=["timestamp"])
    if prior.empty:
        return pd.DataFrame()
    latest_ts = prior["timestamp"].max()
    return prior.loc[prior["timestamp"] == latest_ts].reset_index(drop=True)


def _fetch_and_persist_option_chain(
    *,
    store: DataStore,
    feed: KiteFeed,
    symbol: str,
    timeframe: str,
    asof: datetime,
    dte_min: int,
    dte_max: int | None,
) -> tuple[pd.DataFrame, datetime | None]:
    expiry = _select_target_expiry(
        feed=feed,
        symbol=symbol,
        asof=asof,
        dte_min=dte_min,
        dte_max=dte_max,
    )
    if expiry is None:
        return pd.DataFrame(), None

    chain = feed.get_option_chain(symbol=symbol, expiry=expiry, timestamp=asof)
    chain_df = _frame_from_option_chain(source=feed.name, chain=chain)
    if chain_df.empty:
        return chain_df, expiry

    store.write_time_series(
        "option_chain",
        chain_df,
        symbol=symbol,
        timeframe=timeframe,
        timestamp_col="timestamp",
        dedup_cols=OPTION_CHAIN_DEDUP_COLS,
        source=feed.name,
    )
    return chain_df, expiry


def main() -> int:
    if load_dotenv:
        load_dotenv(REPO_ROOT / ".env")

    args = parse_args()
    settings = load_settings(args.settings)

    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")
    now = datetime.now(UTC).replace(tzinfo=None)
    start = now - timedelta(days=args.lookback_days)

    api_key = os.getenv("KITE_API_KEY", "").strip()
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        print("[FAIL] Missing Kite credentials. Set KITE_API_KEY and KITE_ACCESS_TOKEN in .env.")
        return 1

    feed = KiteFeed(api_key=api_key, access_token=access_token)
    store = DataStore(base_dir=str(cache_dir))
    chain_dte_min, chain_dte_max = _option_chain_dte_bounds(settings)
    previous_chain_df = _load_latest_option_chain_snapshot(
        store=store,
        symbol=args.symbol,
        timeframe=args.timeframe,
        before=now,
    )
    classifier = RegimeClassifier(
        thresholds=RegimeThresholds.from_config(settings.get("regime", {}))
    )
    router = StrategyRouter(strategies=build_strategies(settings))
    runtime = RegimeRuntime(
        classifier=classifier,
        router=router,
        snapshot_store=RegimeSnapshotStore(base_dir=str(cache_dir)),
        signal_snapshot_store=SignalSnapshotStore(base_dir=str(cache_dir)),
        transition_store=StrategyTransitionStore(base_dir=str(cache_dir)),
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    backtest_cfg = settings.get("backtest", {})
    risk_cfg = settings.get("risk", {})
    initial_capital = float(risk_cfg.get("initial_capital", 150_000.0) or 150_000.0)
    slippage_pct = float(backtest_cfg.get("slippage_pct", 0.05) or 0.05)
    breaker = CircuitBreaker(
        initial_capital=initial_capital,
        max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 3.0) or 3.0),
        max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 15.0) or 15.0),
        max_open_positions=int(risk_cfg.get("max_open_positions", 4) or 4),
    )
    executor = PaperExecutionEngine(
        base_dir=str(cache_dir),
        slippage_bps=slippage_pct * 100.0,
        commission_per_order=float(backtest_cfg.get("commission_per_order", 20.0) or 20.0),
        initial_capital=initial_capital,
        circuit_breaker=breaker,
    )
    stop_requested = False

    def _request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\nReceived signal {signum}; shutting down gracefully after current iteration.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("=" * 72)
    print("NiftyQuant Paper Loop (Paper Execution)")
    print("=" * 72)
    print(f"symbol={args.symbol} timeframe={args.timeframe} iterations={args.iterations}")
    print(f"initial_capital={initial_capital:.2f}")
    print(f"cache_dir={cache_dir}")
    print(f"enabled_strategies={[s.name for s in router.strategies]}")
    chain_dte_max_label = chain_dte_max if chain_dte_max is not None else "none"
    print(f"option_chain_dte_bounds=min={chain_dte_min} max={chain_dte_max_label}")
    print(f"bootstrapped_previous_chain_rows={len(previous_chain_df)}")

    for i in range(args.iterations):
        if stop_requested:
            break
        loop_ts = datetime.now(UTC).replace(tzinfo=None)
        try:
            candles = _load_or_fetch_candles(
                store=store,
                feed=feed,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start=start,
                end=loop_ts,
            )
            if candles.empty:
                print(f"[{i + 1}/{args.iterations}] no candle data available")
                if i < args.iterations - 1:
                    time.sleep(args.sleep_seconds)
                continue

            vix = _load_or_fetch_vix(store=store, feed=feed, start=start, end=loop_ts)
            fii = _load_or_fetch_fii(store=store, start=start, end=loop_ts)
            chain_df, chain_expiry = _fetch_and_persist_option_chain(
                store=store,
                feed=feed,
                symbol=args.symbol,
                timeframe=args.timeframe,
                asof=loop_ts,
                dte_min=chain_dte_min,
                dte_max=chain_dte_max,
            )

            vix_series = (
                vix["close"].astype("float64") if not vix.empty and "close" in vix.columns else None
            )
            vix_value = (
                float(vix_series.iloc[-1])
                if vix_series is not None and not vix_series.empty
                else 0.0
            )
            fii_net_3d = (
                float(fii["fii_net"].tail(3).sum())
                if not fii.empty and "fii_net" in fii.columns
                else 0.0
            )

            regime_signals = build_regime_signals(
                timestamp=loop_ts,
                candles=candles,
                vix_value=vix_value,
                vix_series=vix_series,
                chain_df=chain_df if not chain_df.empty else None,
                previous_chain_df=previous_chain_df if not previous_chain_df.empty else None,
                fii_net_3d=fii_net_3d,
            )

            underlying_price = (
                float(chain_df["underlying_price"].iloc[-1])
                if not chain_df.empty and "underlying_price" in chain_df.columns
                else float(candles["close"].iloc[-1])
            )
            regime, transition_signals = runtime.process(regime_signals)
            strategy_signals = router.generate_signals(
                market_data={
                    "timestamp": loop_ts,
                    "candles": candles,
                    "vix": vix_value,
                    "option_chain": chain_df,
                    "underlying_price": underlying_price,
                },
                regime=regime,
            )
            all_signals = transition_signals + strategy_signals
            fills = executor.execute_signals(
                all_signals,
                market_data={
                    "timestamp": loop_ts,
                    "vix": vix_value,
                    "symbol": args.symbol,
                    "close_price": underlying_price,
                },
            )

            chain_expiry_label = chain_expiry.date().isoformat() if chain_expiry else "none"
            print(
                f"[{i + 1}/{args.iterations}] regime={regime.value} "
                f"vix={vix_value:.2f} adx={regime_signals.adx_14:.2f} "
                f"chain_rows={len(chain_df)} chain_expiry={chain_expiry_label} "
                f"transition_signals={len(transition_signals)} "
                f"strategy_signals={len(strategy_signals)} "
                f"fills={len(fills)}"
            )
            if not chain_df.empty:
                previous_chain_df = chain_df
        except (KiteFeedError, FiiDownloadError) as exc:
            print(f"[{i + 1}/{args.iterations}] data unavailable: {exc}")
        except Exception as exc:
            print(f"[{i + 1}/{args.iterations}] error: {exc}")
            traceback.print_exc()
            break

        if i < args.iterations - 1:
            time.sleep(args.sleep_seconds)

    print("Done. Regime snapshots, signal snapshots, and strategy transitions persisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
