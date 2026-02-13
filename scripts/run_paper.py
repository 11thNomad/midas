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
from dataclasses import dataclass
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

from src.data.fii import FiiDownloadError, fetch_fii_dii
from src.data.kite_feed import KiteFeed, KiteFeedError
from src.data.store import DataStore
from src.execution import PaperExecutionEngine
from src.risk.circuit_breaker import CircuitBreaker
from src.regime import (
    RegimeClassifier,
    RegimeRuntime,
    RegimeSnapshotStore,
    RegimeThresholds,
    StrategyTransitionStore,
)
from src.signals.regime import build_regime_signals
from src.strategies.iron_condor import IronCondorStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType
from src.strategies.regime_probe import RegimeProbeStrategy
from src.strategies.router import StrategyRouter


@dataclass
class NoOpStrategy(BaseStrategy):
    """Placeholder strategy for activation-routing testing."""

    def __init__(self, name: str, config: dict):
        super().__init__(name=name, config=config)

    def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=self.config.get("instrument", "NIFTY"),
            timestamp=market_data.get("timestamp", datetime.now(UTC).replace(tzinfo=None)),
            regime=regime,
            reason="No-op paper scaffold strategy",
        )

    def get_exit_conditions(self, market_data: dict) -> Signal | None:
        return None

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-loop scaffold (paper execution mode).")
    parser.add_argument("--symbol", default="NIFTY", help="Underlying symbol")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (e.g., 5m, 1d)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of loop iterations")
    parser.add_argument("--sleep-seconds", type=int, default=15, help="Sleep between iterations")
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for signal computation")
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
    for name, cfg in strategies_cfg.items():
        if not cfg.get("enabled", False):
            continue
        if name == "momentum":
            strategies.append(MomentumStrategy(name=name, config=cfg))
        elif name == "iron_condor":
            strategies.append(IronCondorStrategy(name=name, config=cfg))
        elif name == "regime_probe":
            strategies.append(RegimeProbeStrategy(name=name, config=cfg))
        else:
            strategies.append(NoOpStrategy(name=name, config=cfg))
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
    store.write_time_series("candles", fetched, symbol=symbol, timeframe=timeframe, source=feed.name)
    return fetched


def _load_or_fetch_vix(*, store: DataStore, feed: KiteFeed, start: datetime, end: datetime) -> pd.DataFrame:
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
    classifier = RegimeClassifier(thresholds=RegimeThresholds.from_config(settings.get("regime", {})))
    router = StrategyRouter(strategies=build_strategies(settings))
    runtime = RegimeRuntime(
        classifier=classifier,
        router=router,
        snapshot_store=RegimeSnapshotStore(base_dir=str(cache_dir)),
        transition_store=StrategyTransitionStore(base_dir=str(cache_dir)),
        symbol=args.symbol,
    )
    backtest_cfg = settings.get("backtest", {})
    risk_cfg = settings.get("risk", {})
    slippage_pct = float(backtest_cfg.get("slippage_pct", 0.05) or 0.05)
    breaker = CircuitBreaker(
        initial_capital=500_000.0,
        max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 3.0) or 3.0),
        max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 15.0) or 15.0),
        max_open_positions=int(risk_cfg.get("max_open_positions", 4) or 4),
    )
    executor = PaperExecutionEngine(
        base_dir=str(cache_dir),
        slippage_bps=slippage_pct * 100.0,
        commission_per_order=float(backtest_cfg.get("commission_per_order", 20.0) or 20.0),
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
    print(f"cache_dir={cache_dir}")
    print(f"enabled_strategies={[s.name for s in router.strategies]}")

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

            vix_series = vix["close"].astype("float64") if not vix.empty and "close" in vix.columns else None
            vix_value = float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else 0.0
            fii_net_3d = float(fii["fii_net"].tail(3).sum()) if not fii.empty and "fii_net" in fii.columns else 0.0

            # TODO: Replace None chain placeholders with true option-chain snapshots once feed integration is ready.
            regime_signals = build_regime_signals(
                timestamp=loop_ts,
                candles=candles,
                vix_value=vix_value,
                vix_series=vix_series,
                chain_df=None,
                previous_chain_df=None,
                fii_net_3d=fii_net_3d,
            )

            regime, transition_signals = runtime.process(regime_signals)
            strategy_signals = router.generate_signals(
                market_data={"timestamp": loop_ts, "candles": candles, "vix": vix_value},
                regime=regime,
            )
            all_signals = transition_signals + strategy_signals
            fills = executor.execute_signals(
                all_signals,
                market_data={
                    "timestamp": loop_ts,
                    "vix": vix_value,
                    "symbol": args.symbol,
                    "close_price": float(candles["close"].iloc[-1]),
                },
            )

            print(
                f"[{i + 1}/{args.iterations}] regime={regime.value} "
                f"vix={vix_value:.2f} adx={regime_signals.adx_14:.2f} "
                f"transition_signals={len(transition_signals)} strategy_signals={len(strategy_signals)} "
                f"fills={len(fills)}"
            )
        except (KiteFeedError, FiiDownloadError) as exc:
            print(f"[{i + 1}/{args.iterations}] data unavailable: {exc}")
        except Exception as exc:
            print(f"[{i + 1}/{args.iterations}] error: {exc}")
            traceback.print_exc()
            break

        if i < args.iterations - 1:
            time.sleep(args.sleep_seconds)

    print("Done. Regime snapshots and strategy transitions persisted to parquet cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
