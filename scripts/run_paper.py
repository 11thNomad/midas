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
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some environments
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.candle_access import CandleStores, build_candle_stores, read_candles
from src.data.contracts import option_dtos_from_chain
from src.data.fii import FiiDownloadError, fetch_fii_dii
from src.data.kite_feed import KiteFeed, KiteFeedError
from src.data.option_chain_quality import OptionChainQualityThresholds
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
from src.signals.pipeline import build_feature_context
from src.strategies.base import BaseStrategy, Signal, SignalType
from src.strategies.baseline_trend import BaselineTrendStrategy
from src.strategies.iron_condor import IronCondorStrategy
from src.strategies.jade_lizard import JadeLizardStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime_probe import RegimeProbeStrategy
from src.strategies.router import StrategyRouter

OPTION_CHAIN_DEDUP_COLS = ["timestamp", "expiry", "strike", "option_type"]
MARKET_TZ = ZoneInfo("Asia/Kolkata")


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
    market_cfg = settings.get("market", {})
    strategies: list[BaseStrategy] = []
    unknown_enabled: list[str] = []
    for name, cfg in strategies_cfg.items():
        if not cfg.get("enabled", False):
            continue
        if name == "momentum":
            strategies.append(MomentumStrategy(name=name, config=cfg))
        elif name == "baseline_trend":
            strategies.append(BaselineTrendStrategy(name=name, config=cfg))
        elif name == "iron_condor":
            strategy_cfg = {
                **cfg,
                "time_exit_window_start": market_cfg.get("exit_window_start", "15:10"),
                "time_exit_window_end": market_cfg.get("exit_window_end", "15:20"),
            }
            strategies.append(IronCondorStrategy(name=name, config=strategy_cfg))
        elif name == "jade_lizard":
            strategies.append(JadeLizardStrategy(name=name, config=cfg))
        elif name == "regime_probe":
            strategies.append(RegimeProbeStrategy(name=name, config=cfg))
        else:
            unknown_enabled.append(name)

    if unknown_enabled:
        known = ["regime_probe", "baseline_trend", "momentum", "iron_condor", "jade_lizard"]
        raise ValueError(
            "Unknown enabled strategy id(s): "
            f"{', '.join(sorted(unknown_enabled))}. "
            f"Known strategy ids: {', '.join(known)}"
        )
    return strategies


def _parse_hhmm(value: object, default: str) -> dt_time:
    raw = str(value or default).strip()
    try:
        return datetime.strptime(raw, "%H:%M").time()
    except ValueError:
        return datetime.strptime(default, "%H:%M").time()


def _resolve_entry_window(settings: dict) -> tuple[dt_time, dt_time]:
    market_cfg = settings.get("market", {})
    open_time = _parse_hhmm(market_cfg.get("open_time"), "09:15")
    close_time = _parse_hhmm(market_cfg.get("close_time"), "15:30")
    entry_buffer_minutes = int(market_cfg.get("entry_buffer_minutes", 15) or 15)
    exit_buffer_minutes = int(market_cfg.get("exit_buffer_minutes", 10) or 10)
    last_new_entry_time = market_cfg.get("last_new_entry_time")

    start_dt = datetime.combine(datetime.now().date(), open_time) + timedelta(
        minutes=entry_buffer_minutes
    )
    if last_new_entry_time:
        end_time = _parse_hhmm(last_new_entry_time, "15:15")
    else:
        end_dt = datetime.combine(datetime.now().date(), close_time) - timedelta(
            minutes=exit_buffer_minutes + 5
        )
        end_time = end_dt.time()
    return start_dt.time(), end_time


def _to_ist_naive(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(MARKET_TZ).replace(tzinfo=None)


def _filter_entry_signals_by_market_window(
    *,
    signals: list[Signal],
    entry_window_start: dt_time,
    entry_window_end: dt_time,
) -> tuple[list[Signal], int]:
    filtered: list[Signal] = []
    blocked = 0
    for strategy_signal in signals:
        if strategy_signal.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
            signal_ts_ist = _to_ist_naive(strategy_signal.timestamp)
            signal_time = signal_ts_ist.time()
            if signal_time < entry_window_start or signal_time > entry_window_end:
                blocked += 1
                entry_window_label = (
                    f"{entry_window_start.strftime('%H:%M')}-"
                    f"{entry_window_end.strftime('%H:%M')}"
                )
                print(
                    "ABORT_ENTRY market-hours buffer: "
                    f"timestamp_ist={signal_ts_ist.isoformat(timespec='seconds')} "
                    f"window={entry_window_label} "
                    f"strategy={strategy_signal.strategy_name} "
                    f"instrument={strategy_signal.instrument}"
                )
                continue
        filtered.append(strategy_signal)
    return filtered, blocked


def _kite_available_cash(feed: KiteFeed) -> float | None:
    client = getattr(feed, "_kite", None)
    if client is None:
        return None
    try:
        margins = client.margins()
    except Exception:
        return None
    if not isinstance(margins, dict):
        return None

    def _as_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            parsed = float(value)
            if parsed <= 0.0:
                return None
            return parsed
        except (TypeError, ValueError):
            return None

    equity = margins.get("equity") if isinstance(margins.get("equity"), dict) else {}
    available = equity.get("available") if isinstance(equity.get("available"), dict) else {}
    candidates = (
        available.get("live_balance"),
        available.get("cash"),
        available.get("opening_balance"),
        available.get("net"),
        margins.get("available_cash"),
    )
    for raw in candidates:
        resolved = _as_float(raw)
        if resolved is not None:
            return resolved
    return None


def _load_or_fetch_candles(
    *,
    store: DataStore,
    candle_stores: CandleStores,
    feed: KiteFeed,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    candles, _ = read_candles(
        stores=candle_stores,
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


def _load_or_fetch_usdinr(
    *,
    store: DataStore,
    feed: KiteFeed,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    usdinr = store.read_time_series("candles", symbol=symbol, timeframe="1d", start=start, end=end)
    if not usdinr.empty:
        return usdinr
    try:
        fetched = feed.get_candles(symbol=symbol, timeframe="1d", start=start, end=end)
    except (KiteFeedError, ValueError):
        return pd.DataFrame()
    if fetched.empty:
        return fetched
    store.write_time_series("candles", fetched, symbol=symbol, timeframe="1d", source=feed.name)
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


def _instrument_price_map(
    *, chain_df: pd.DataFrame, underlying_symbol: str, underlying_price: float
) -> dict[str, float]:
    prices: dict[str, float] = {}
    if not chain_df.empty and {"symbol", "ltp"}.issubset(chain_df.columns):
        priced = chain_df[["symbol", "ltp"]].dropna(subset=["symbol", "ltp"])
        for _, row in priced.iterrows():
            symbol = str(row["symbol"]).strip()
            if not symbol:
                continue
            try:
                ltp = float(row["ltp"])
            except (TypeError, ValueError):
                continue
            if ltp <= 0.0:
                continue
            prices[symbol] = ltp
    if underlying_price > 0.0:
        prices[str(underlying_symbol).strip()] = float(underlying_price)
    return prices


def _option_quote_map(chain_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    quotes: dict[str, dict[str, float]] = {}
    if chain_df.empty or "symbol" not in chain_df.columns:
        return quotes
    for _, row in chain_df.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue
        bid = pd.to_numeric(row.get("bid"), errors="coerce")
        ask = pd.to_numeric(row.get("ask"), errors="coerce")
        ltp = pd.to_numeric(row.get("ltp"), errors="coerce")
        quote: dict[str, float] = {}
        if not pd.isna(bid) and float(bid) > 0.0:
            quote["bid"] = float(bid)
        if not pd.isna(ask) and float(ask) > 0.0:
            quote["ask"] = float(ask)
        if "bid" in quote and "ask" in quote:
            quote["mid"] = (quote["bid"] + quote["ask"]) / 2.0
        if not pd.isna(ltp) and float(ltp) > 0.0:
            quote["ltp"] = float(ltp)
        if quote:
            quotes[symbol] = quote
    return quotes


def _lot_size_by_underlying(settings: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    strategies_cfg = settings.get("strategies", {})
    for name in ("iron_condor", "jade_lizard"):
        cfg = strategies_cfg.get(name, {})
        if not cfg or not cfg.get("enabled", False):
            continue
        instrument = str(cfg.get("instrument", "NIFTY")).upper()
        lot_raw = cfg.get("lot_size")
        try:
            lot_size = int(lot_raw) if lot_raw is not None else 0
        except (TypeError, ValueError):
            lot_size = 0
        if lot_size > 0:
            out[instrument] = lot_size
    return out


def _is_wednesday_watchdog_trigger(timestamp: datetime) -> bool:
    ts_ist = _to_ist_naive(timestamp)
    if ts_ist.weekday() != 2:
        return False
    return ts_ist.time() >= dt_time(15, 25)


def _freshness_issues(
    *,
    candles: pd.DataFrame,
    vix: pd.DataFrame,
    now: datetime,
    settings: dict,
) -> list[str]:
    def _as_market_ist_naive(ts: pd.Timestamp) -> datetime:
        if ts.tzinfo is None:
            return ts.to_pydatetime().replace(tzinfo=None)
        return ts.tz_convert(MARKET_TZ).to_pydatetime().replace(tzinfo=None)

    issues: list[str] = []
    now_ist = _to_ist_naive(now)
    freshness_cfg = settings.get("ops", {}).get("freshness", {})
    max_candle_age = int(freshness_cfg.get("candles_runtime_max_age_minutes", 24 * 60) or 24 * 60)
    max_vix_age = int(freshness_cfg.get("vix_1d_max_age_minutes", 3 * 24 * 60) or 3 * 24 * 60)

    if candles.empty:
        issues.append("candles_missing")
    else:
        if "timestamp" in candles.columns:
            candle_ts = pd.to_datetime(candles["timestamp"], errors="coerce").dropna()
        else:
            candle_ts = pd.to_datetime(candles.index, errors="coerce")
            candle_ts = pd.Series(candle_ts).dropna()
        if candle_ts.empty:
            issues.append("candles_timestamp_invalid")
        else:
            latest_ist = _as_market_ist_naive(pd.Timestamp(candle_ts.max()))
            age = max(0.0, (now_ist - latest_ist).total_seconds() / 60.0)
            if age > float(max_candle_age):
                issues.append(f"candles_stale_{age:.1f}m>{max_candle_age}m")

    if vix.empty:
        issues.append("vix_missing")
    else:
        if "timestamp" in vix.columns:
            vix_ts = pd.to_datetime(vix["timestamp"], errors="coerce").dropna()
        else:
            vix_ts = pd.to_datetime(vix.index, errors="coerce")
            vix_ts = pd.Series(vix_ts).dropna()
        if vix_ts.empty:
            issues.append("vix_timestamp_invalid")
        else:
            latest_ist = _as_market_ist_naive(pd.Timestamp(vix_ts.max()))
            age = max(0.0, (now_ist - latest_ist).total_seconds() / 60.0)
            if age > float(max_vix_age):
                issues.append(f"vix_stale_{age:.1f}m>{max_vix_age}m")
    return issues


def _sync_strategy_state_from_fills(router: StrategyRouter, fills: list[dict]) -> None:
    by_name = {strategy.name: strategy for strategy in router.strategies}
    for fill in fills:
        strategy_name = str(fill.get("strategy_name", "")).strip()
        strategy = by_name.get(strategy_name)
        if strategy is None:
            continue
        raw_signal_type = str(fill.get("signal_type", "")).strip().lower()
        signal_type = next(
            (signal for signal in SignalType if signal.value == raw_signal_type),
            None,
        )
        timestamp = pd.to_datetime(fill.get("timestamp"), errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.Timestamp(datetime.now())
        strategy.on_fill(
            str(fill.get("fill_id", "")),
            float(fill.get("price", 0.0) or 0.0),
            int(fill.get("quantity", 0) or 0),
            timestamp.to_pydatetime(),
            signal_type=signal_type,
        )


def _rollback_unfilled_entry_state(
    *,
    router: StrategyRouter,
    signals: list[Signal],
    fills: list[dict],
) -> None:
    by_name = {strategy.name: strategy for strategy in router.strategies}
    filled_keys: set[tuple[str, str, str]] = set()
    for fill in fills:
        strategy_name = str(fill.get("strategy_name", "")).strip()
        signal_type = str(fill.get("signal_type", "")).strip().lower()
        ts = pd.to_datetime(fill.get("timestamp"), errors="coerce")
        if not strategy_name or not signal_type or pd.isna(ts):
            continue
        filled_keys.add((strategy_name, signal_type, ts.isoformat()))

    for strategy_signal in signals:
        if strategy_signal.signal_type not in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
            continue
        key = (
            strategy_signal.strategy_name,
            strategy_signal.signal_type.value,
            pd.Timestamp(strategy_signal.timestamp).isoformat(),
        )
        if key in filled_keys:
            continue
        strategy = by_name.get(strategy_signal.strategy_name)
        if strategy is None:
            continue
        strategy.state.current_position = None


def _clear_strategy_positions_for_system_fills(router: StrategyRouter, fills: list[dict]) -> None:
    has_system_close = any(
        str(fill.get("strategy_name", "")).strip() in {"watchdog", "expiry_settlement"}
        for fill in fills
    )
    if not has_system_close:
        return
    for strategy in router.strategies:
        strategy.state.current_position = None

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
    candle_stores = build_candle_stores(settings=settings, repo_root=REPO_ROOT)
    store = candle_stores.raw
    chain_dte_min, chain_dte_max = _option_chain_dte_bounds(settings)
    chain_quality_thresholds = OptionChainQualityThresholds.from_config(
        settings.get("data_quality", {}).get("option_chain", {})
    )
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
    paper_cfg = settings.get("paper_trading", {})
    paper_capital = float(paper_cfg.get("paper_capital", initial_capital) or initial_capital)
    margin_buffer_pct = float(paper_cfg.get("margin_buffer_pct", 15.0) or 15.0)
    slippage_multiplier = float(paper_cfg.get("slippage_multiplier", 1.5) or 1.5)
    if paper_capital > 0:
        print(
            "WARNING: paper trading is using mock margin — not connected to live account"
        )
        print(
            "MOCK_MARGIN: using "
            f"paper_capital={paper_capital:.2f} (bypassing Kite /margins; "
            "runtime balance persisted in SQLite)"
        )
        available_cash_resolver = None
    else:
        def _resolve_kite_cash() -> float | None:
            return _kite_available_cash(feed)

        available_cash_resolver = _resolve_kite_cash
    executor = PaperExecutionEngine(
        base_dir=str(cache_dir),
        slippage_bps=slippage_pct * 100.0,
        slippage_multiplier=slippage_multiplier,
        commission_per_order=float(backtest_cfg.get("commission_per_order", 20.0) or 20.0),
        initial_capital=initial_capital,
        paper_capital=paper_capital,
        margin_buffer_pct=margin_buffer_pct,
        available_cash_resolver=available_cash_resolver,
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
    print(
        "paper_capital="
        f"{paper_capital:.2f} "
        f"margin_buffer_pct={margin_buffer_pct:.2f} "
        f"slippage_multiplier={slippage_multiplier:.2f}"
    )
    print(f"cache_dir={cache_dir}")
    print(f"enabled_strategies={[s.name for s in router.strategies]}")
    chain_dte_max_label = chain_dte_max if chain_dte_max is not None else "none"
    print(f"option_chain_dte_bounds=min={chain_dte_min} max={chain_dte_max_label}")
    print(f"bootstrapped_previous_chain_rows={len(previous_chain_df)}")
    entry_window_start, entry_window_end = _resolve_entry_window(settings)
    lot_size_by_underlying = _lot_size_by_underlying(settings)
    print(
        "entry_window_ist="
        f"{entry_window_start.strftime('%H:%M')}-{entry_window_end.strftime('%H:%M')}"
    )

    for i in range(args.iterations):
        if stop_requested:
            break
        loop_ts = datetime.now(UTC).replace(tzinfo=None)
        try:
            candles = _load_or_fetch_candles(
                store=store,
                candle_stores=candle_stores,
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
            usdinr_symbol = str(settings.get("market", {}).get("usdinr_symbol", "USDINR")).upper()
            usdinr = _load_or_fetch_usdinr(
                store=store,
                feed=feed,
                symbol=usdinr_symbol,
                start=start,
                end=loop_ts,
            )
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
            underlying_price = (
                float(chain_df["underlying_price"].iloc[-1])
                if not chain_df.empty and "underlying_price" in chain_df.columns
                else float(candles["close"].iloc[-1])
            )
            option_quotes = _option_quote_map(chain_df)
            instrument_prices = _instrument_price_map(
                chain_df=chain_df,
                underlying_symbol=args.symbol,
                underlying_price=underlying_price,
            )
            watchdog_force_exit = _is_wednesday_watchdog_trigger(loop_ts)
            freshness_issues = _freshness_issues(
                candles=candles,
                vix=vix,
                now=loop_ts,
                settings=settings,
            )
            if freshness_issues:
                fills = executor.execute_signals(
                    [],
                    market_data={
                        "timestamp": loop_ts,
                        "vix": vix_value,
                        "symbol": args.symbol,
                        "close_price": underlying_price,
                        "instrument_prices": instrument_prices,
                        "option_quotes": option_quotes,
                        "underlying_prices": {str(args.symbol).upper(): underlying_price},
                        "lot_size_by_underlying": lot_size_by_underlying,
                        "force_exit_all": watchdog_force_exit,
                        "force_exit_reason": "watchdog_wed_1525_force_exit",
                    },
                )
                _sync_strategy_state_from_fills(router, fills)
                _clear_strategy_positions_for_system_fills(router, fills)
                print(
                    f"[{i + 1}/{args.iterations}] stale_data="
                    f"{'|'.join(freshness_issues)} watchdog={int(watchdog_force_exit)} "
                    f"fills={len(fills)}"
                )
                if not chain_df.empty:
                    previous_chain_df = chain_df
                if i < args.iterations - 1:
                    time.sleep(args.sleep_seconds)
                continue

            signal_snapshot, regime_signals = build_feature_context(
                timestamp=loop_ts,
                symbol=args.symbol,
                timeframe=args.timeframe,
                candles=candles,
                vix_value=vix_value,
                vix_series=vix_series,
                chain_df=chain_df if not chain_df.empty else None,
                previous_chain_df=previous_chain_df if not previous_chain_df.empty else None,
                fii_df=fii,
                usdinr_close=(
                    usdinr["close"].astype("float64")
                    if not usdinr.empty and "close" in usdinr.columns
                    else None
                ),
                regime=runtime.classifier.current_regime.value,
                thresholds=runtime.classifier.thresholds,
                chain_quality_thresholds=chain_quality_thresholds,
                source="paper_loop",
            )
            regime, transition_signals = runtime.process(
                regime_signals,
                signal_snapshot=signal_snapshot,
            )
            strategy_market_data = {
                "timestamp": loop_ts,
                "candles": candles,
                "vix": vix_value,
                "option_chain": chain_df,
                "underlying_price": underlying_price,
                "regime": regime,
            }
            strategy_exit_signals = router.generate_exit_signals(market_data=strategy_market_data)
            transition_non_exit_signals = [
                s for s in transition_signals if s.signal_type != SignalType.EXIT
            ]
            transition_exit_signals = [
                s for s in transition_signals if s.signal_type == SignalType.EXIT
            ]
            dedup_exit_by_strategy: dict[str, Signal] = {}
            for sig in transition_exit_signals + strategy_exit_signals:
                dedup_exit_by_strategy.setdefault(sig.strategy_name, sig)
            exit_signals = list(dedup_exit_by_strategy.values())

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
            blocked_entry_strategies = {sig.strategy_name for sig in exit_signals}
            strategy_signals = [
                sig
                for sig in strategy_signals
                if sig.strategy_name not in blocked_entry_strategies
            ]
            all_signals = transition_non_exit_signals + exit_signals + strategy_signals
            filtered_signals, blocked_entries = _filter_entry_signals_by_market_window(
                signals=all_signals,
                entry_window_start=entry_window_start,
                entry_window_end=entry_window_end,
            )
            fills = executor.execute_signals(
                filtered_signals,
                market_data={
                    "timestamp": loop_ts,
                    "vix": vix_value,
                    "adx": float(regime_signals.adx_14),
                    "symbol": args.symbol,
                    "close_price": underlying_price,
                    "instrument_prices": instrument_prices,
                    "option_quotes": option_quotes,
                    "underlying_prices": {str(args.symbol).upper(): underlying_price},
                    "lot_size_by_underlying": lot_size_by_underlying,
                    "force_exit_all": watchdog_force_exit,
                    "force_exit_reason": "watchdog_wed_1525_force_exit",
                },
            )
            _sync_strategy_state_from_fills(router, fills)
            _rollback_unfilled_entry_state(router=router, signals=filtered_signals, fills=fills)
            _clear_strategy_positions_for_system_fills(router, fills)

            chain_expiry_label = chain_expiry.date().isoformat() if chain_expiry else "none"
            print(
                f"[{i + 1}/{args.iterations}] regime={regime.value} "
                f"vix={vix_value:.2f} adx={regime_signals.adx_14:.2f} "
                f"chain_rows={len(chain_df)} chain_expiry={chain_expiry_label} "
                f"transition_signals={len(transition_signals)} "
                f"exit_signals={len(exit_signals)} "
                f"strategy_signals={len(strategy_signals)} "
                f"blocked_entries={blocked_entries} "
                f"watchdog={int(watchdog_force_exit)} "
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
