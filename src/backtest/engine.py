"""Event-driven backtest engine scaffold (Phase 4)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import pandas as pd

from src.backtest.metrics import summarize_backtest
from src.backtest.simulator import FillSimulator
from src.data.option_chain_quality import OptionChainQualityThresholds
from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.risk.circuit_breaker import CircuitBreaker
from src.signals.contracts import SignalSnapshotDTO, frame_from_signal_snapshots
from src.signals.pipeline import build_feature_context
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    regimes: pd.DataFrame
    metrics: dict[str, float]
    signal_snapshots: pd.DataFrame = field(default_factory=pd.DataFrame)
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class PrecomputedBarContext:
    timestamp: pd.Timestamp
    regime: RegimeState
    snapshot: SignalSnapshotDTO
    chain_timestamp: pd.Timestamp | None


@dataclass
class BacktestPrecomputedData:
    """Reusable per-symbol backtest context shared across strategies."""

    vix: pd.DataFrame
    fii: pd.DataFrame
    usdinr: pd.DataFrame
    chain: pd.DataFrame
    chain_ranges: dict[pd.Timestamp, tuple[int, int]]
    chain_timestamps: pd.DatetimeIndex
    chain_price_maps: dict[pd.Timestamp, dict[str, float]]
    bar_contexts: dict[pd.Timestamp, PrecomputedBarContext]

    def chain_snapshot(self, chain_ts: pd.Timestamp | None) -> pd.DataFrame | None:
        if chain_ts is None:
            return None
        bounds = self.chain_ranges.get(chain_ts)
        if bounds is None:
            return None
        start, end = bounds
        return self.chain.iloc[start:end].reset_index(drop=True)


@dataclass
class BacktestEngine:
    """Run a single strategy over historical bars with regime awareness."""

    classifier: RegimeClassifier
    strategy: BaseStrategy
    simulator: FillSimulator
    initial_capital: float = 1_000_000.0
    periods_per_year: int = 252
    risk_free_rate_annual: float = 0.07
    monte_carlo_permutations: int = 200
    minimum_trade_count: int = 50
    circuit_breaker: CircuitBreaker | None = None
    fill_on: str = "open"
    chain_quality_thresholds: OptionChainQualityThresholds = field(
        default_factory=OptionChainQualityThresholds
    )

    def run(
        self,
        *,
        candles: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        fii_df: pd.DataFrame | None = None,
        usdinr_df: pd.DataFrame | None = None,
        option_chain_df: pd.DataFrame | None = None,
        analysis_start: datetime | None = None,
        precomputed_data: BacktestPrecomputedData | None = None,
    ) -> BacktestResult:
        if candles.empty:
            empty = pd.DataFrame()
            return BacktestResult(
                equity_curve=empty,
                fills=empty,
                regimes=empty,
                signal_snapshots=empty,
                decisions=empty,
                metrics=summarize_backtest(
                    equity_curve=empty,
                    fills=empty,
                    initial_capital=self.initial_capital,
                    risk_free_rate_annual=self.risk_free_rate_annual,
                    monte_carlo_permutations=self.monte_carlo_permutations,
                    minimum_trade_count=self.minimum_trade_count,
                ),
            )

        bars = candles.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
        bars = bars.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if precomputed_data is None:
            vix = self._prep_vix(vix_df)
            fii = self._prep_fii(fii_df)
            usdinr = self._prep_usdinr(usdinr_df)
            chain = self._prep_chain(option_chain_df)
        else:
            vix = precomputed_data.vix
            fii = precomputed_data.fii
            usdinr = precomputed_data.usdinr
            chain = precomputed_data.chain

        cash = float(self.initial_capital)
        positions: dict[str, int] = {}
        avg_cost_by_instrument: dict[str, float] = {}
        last_price_by_instrument: dict[str, float] = {}
        realized_pnl_today = 0.0
        fill_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        regime_rows: list[dict[str, Any]] = []
        signal_snapshot_rows: list[SignalSnapshotDTO] = []
        decision_rows: list[dict[str, Any]] = []
        previous_regime = self.classifier.current_regime
        previous_chain_asof: pd.DataFrame | None = None
        analysis_cutoff = pd.Timestamp(analysis_start) if analysis_start is not None else None

        for i in range(len(bars)):
            row = bars.iloc[i]
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            ts_key = pd.Timestamp(ts)
            open_price = float(row["open"])
            close_price = float(row["close"])
            candles_hist = bars.iloc[:i]
            pre_ctx = (
                precomputed_data.bar_contexts.get(ts_key)
                if precomputed_data is not None
                else None
            )
            if pre_ctx is not None:
                precomputed = precomputed_data
                if precomputed is None:
                    raise RuntimeError("precomputed context unexpectedly unavailable")
                chain_asof = precomputed.chain_snapshot(pre_ctx.chain_timestamp)
                chain_prices = (
                    precomputed.chain_price_maps.get(pre_ctx.chain_timestamp, {})
                    if pre_ctx.chain_timestamp is not None
                    else {}
                )
                mark_prices = self._compose_mark_price_map(
                    chain_prices=chain_prices,
                    default_underlying_price=close_price,
                    underlying_symbol=str(self.strategy.config.get("instrument", "UNDERLYING")),
                )
                signal_snapshot = pre_ctx.snapshot
                regime = pre_ctx.regime
                vix_value = float(signal_snapshot.vix_level)
            else:
                vix_hist = (
                    vix.loc[vix["timestamp"] < ts_key] if not vix.empty else pd.DataFrame()
                )
                fii_hist = fii.loc[fii["date"] < ts_key] if not fii.empty else pd.DataFrame()
                usdinr_hist = (
                    usdinr.loc[usdinr["timestamp"] < ts_key]
                    if not usdinr.empty
                    else pd.DataFrame()
                )
                chain_asof = self._latest_chain_asof(chain, ts, strict=True)
                mark_prices = self._build_mark_price_map(
                    chain_asof=chain_asof,
                    default_underlying_price=close_price,
                    underlying_symbol=str(self.strategy.config.get("instrument", "UNDERLYING")),
                )

                vix_series = vix_hist["close"].astype("float64") if not vix_hist.empty else None
                vix_value = (
                    float(vix_series.iloc[-1])
                    if vix_series is not None and not vix_series.empty
                    else 0.0
                )

                signal_snapshot, regime_signals = build_feature_context(
                    timestamp=ts,
                    symbol=str(self.strategy.config.get("instrument", "NIFTY")),
                    timeframe=str(self.strategy.config.get("timeframe", "1d")),
                    candles=candles_hist,
                    vix_value=vix_value,
                    vix_series=vix_series,
                    chain_df=chain_asof,
                    previous_chain_df=previous_chain_asof,
                    fii_df=fii_hist,
                    usdinr_close=(
                        usdinr_hist["close"].astype("float64")
                        if not usdinr_hist.empty and "close" in usdinr_hist.columns
                        else None
                    ),
                    regime=self.classifier.current_regime.value,
                    thresholds=self.classifier.thresholds,
                    chain_quality_thresholds=self.chain_quality_thresholds,
                    source="backtest_engine",
                )
                regime = self.classifier.classify(regime_signals)
                if chain_asof is not None and not chain_asof.empty:
                    previous_chain_asof = chain_asof
            if analysis_cutoff is not None and pd.Timestamp(ts) < analysis_cutoff:
                previous_regime = regime
                continue

            regime_rows.append(
                {
                    "timestamp": ts,
                    "regime": regime.value,
                    "vix": float(signal_snapshot.vix_level),
                    "adx": float(signal_snapshot.adx_14),
                }
            )
            signal_snapshot_rows.append(
                SignalSnapshotDTO(
                    **{
                        **signal_snapshot.__dict__,
                        "regime": regime.value,
                    }
                )
            )

            can_trade = (
                self.circuit_breaker.can_trade() if self.circuit_breaker is not None else True
            )
            position_before = copy.deepcopy(self.strategy.state.current_position)
            signal = self._next_signal(
                timestamp=ts,
                regime=regime,
                previous_regime=previous_regime,
                candles_hist=candles_hist,
                vix_value=vix_value,
                option_chain=chain_asof,
                underlying_price=close_price,
                can_trade=can_trade,
            )
            previous_regime = regime

            if signal is not None:
                decision_rows.append(
                    {
                        "timestamp": ts,
                        "strategy_name": signal.strategy_name,
                        "signal_type": signal.signal_type.value,
                        "instrument": signal.instrument,
                        "regime": signal.regime.value,
                        "is_actionable": float(signal.is_actionable),
                        "can_trade": float(can_trade),
                        "orders_count": len(signal.orders or []),
                        "reason": signal.reason,
                        "indicators_json": json.dumps(
                            signal.indicators, sort_keys=True, default=str
                        ),
                        "greeks_snapshot_json": json.dumps(
                            signal.greeks_snapshot, sort_keys=True, default=str
                        ),
                    }
                )

            fill_reference_price = open_price if self.fill_on == "open" else close_price
            if signal is not None and signal.is_actionable:
                is_exit = signal.signal_type == SignalType.EXIT
                if not can_trade and not is_exit:
                    # Circuit-breaker halt blocks new entries/adjustments.
                    # Exits must still be allowed.
                    continue
                if signal.signal_type == SignalType.EXIT and not signal.orders:
                    net_qty = int(positions.get(signal.instrument, 0))
                    signal.orders = [
                        {
                            "symbol": signal.instrument,
                            "action": "BUY" if net_qty < 0 else "SELL",
                            "quantity": abs(net_qty) if net_qty != 0 else 1,
                        }
                    ]
                fills = self.simulator.simulate(
                    signal,
                    close_price=fill_reference_price,
                    timestamp=ts,
                    price_lookup=mark_prices,
                )
                if not fills:
                    # Keep strategy state aligned to realized execution.
                    self.strategy.state.current_position = position_before
                    continue
                for fill in fills:
                    instrument = str(fill["instrument"])
                    qty = int(fill["quantity"])
                    side = str(fill["side"]).upper()
                    fill_price = float(fill["price"])
                    fees = float(fill["fees"])
                    realized_delta = self._update_position_and_realized_pnl(
                        positions=positions,
                        avg_cost_by_instrument=avg_cost_by_instrument,
                        instrument=instrument,
                        side=side,
                        quantity=qty,
                        fill_price=fill_price,
                    )
                    realized_pnl_today += realized_delta - fees

                    notional = float(fill.get("notional", fill["price"] * qty))
                    if side == "BUY":
                        cash -= notional + fees
                    else:
                        cash += notional - fees

                    last_price_by_instrument[instrument] = fill_price
                    fill_rows.append(fill)
                if signal.signal_type == SignalType.EXIT:
                    self.strategy.state.current_position = None

            equity = self._mark_to_market(
                cash=cash,
                positions=positions,
                mark_prices=mark_prices,
                fallback_prices=last_price_by_instrument,
                default_underlying_price=close_price,
            )
            open_positions = sum(1 for _, qty in positions.items() if qty != 0)
            equity_rows.append(
                {"timestamp": ts, "cash": cash, "open_positions": open_positions, "equity": equity}
            )

            if self.circuit_breaker is not None:
                unrealized_pnl = self._compute_unrealized_pnl(
                    positions=positions,
                    avg_cost_by_instrument=avg_cost_by_instrument,
                    mark_prices=mark_prices,
                    fallback_prices=last_price_by_instrument,
                    default_underlying_price=close_price,
                )
                self.circuit_breaker.update(
                    current_equity=equity,
                    realized_pnl_today=realized_pnl_today,
                    unrealized_pnl=unrealized_pnl,
                    open_positions=open_positions,
                    timestamp=ts,
                )

        fills_df = pd.DataFrame(fill_rows)
        equity_df = pd.DataFrame(equity_rows)
        regimes_df = pd.DataFrame(regime_rows)
        signal_snapshots_df = frame_from_signal_snapshots(signal_snapshot_rows)
        decisions_df = pd.DataFrame(decision_rows)
        metrics = summarize_backtest(
            equity_curve=equity_df,
            fills=fills_df,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year,
            risk_free_rate_annual=self.risk_free_rate_annual,
            monte_carlo_permutations=self.monte_carlo_permutations,
            minimum_trade_count=self.minimum_trade_count,
        )
        return BacktestResult(
            equity_curve=equity_df,
            fills=fills_df,
            regimes=regimes_df,
            signal_snapshots=signal_snapshots_df,
            decisions=decisions_df,
            metrics=metrics,
        )

    @classmethod
    def prepare_precomputed_data(
        cls,
        *,
        candles: pd.DataFrame,
        thresholds: RegimeThresholds,
        symbol: str,
        timeframe: str,
        vix_df: pd.DataFrame | None = None,
        fii_df: pd.DataFrame | None = None,
        usdinr_df: pd.DataFrame | None = None,
        option_chain_df: pd.DataFrame | None = None,
        chain_quality_thresholds: OptionChainQualityThresholds | None = None,
    ) -> BacktestPrecomputedData:
        bars = candles.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
        bars = bars.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        vix = cls._prep_vix(vix_df)
        fii = cls._prep_fii(fii_df)
        usdinr = cls._prep_usdinr(usdinr_df)
        chain = cls._prep_chain(option_chain_df)
        chain_ranges = cls._build_chain_ranges(chain)
        chain_timestamps = pd.DatetimeIndex(sorted(chain_ranges.keys()))

        chain_price_maps: dict[pd.Timestamp, dict[str, float]] = {}
        for chain_ts in chain_timestamps:
            snap = cls._slice_chain_by_timestamp(chain, chain_ranges, chain_ts)
            chain_price_maps[chain_ts] = cls._build_chain_price_map(snap)

        classifier = RegimeClassifier(thresholds=thresholds)
        quality_thresholds = chain_quality_thresholds or OptionChainQualityThresholds()
        previous_chain_ts: pd.Timestamp | None = None
        contexts: dict[pd.Timestamp, PrecomputedBarContext] = {}

        for i in range(len(bars)):
            row = bars.iloc[i]
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            ts_key = pd.Timestamp(ts)
            candles_hist = bars.iloc[:i]

            vix_hist = vix.loc[vix["timestamp"] < ts_key] if not vix.empty else pd.DataFrame()
            fii_hist = fii.loc[fii["date"] < ts_key] if not fii.empty else pd.DataFrame()
            usdinr_hist = (
                usdinr.loc[usdinr["timestamp"] < ts_key] if not usdinr.empty else pd.DataFrame()
            )

            chain_ts: pd.Timestamp | None = cls._latest_chain_timestamp(
                chain_timestamps=chain_timestamps,
                ts=ts,
                strict=True,
            )
            chain_asof = cls._slice_chain_by_timestamp(chain, chain_ranges, chain_ts)
            previous_chain_asof = cls._slice_chain_by_timestamp(
                chain,
                chain_ranges,
                previous_chain_ts,
            )

            vix_series = vix_hist["close"].astype("float64") if not vix_hist.empty else None
            vix_value = (
                float(vix_series.iloc[-1])
                if vix_series is not None and not vix_series.empty
                else 0.0
            )
            snapshot, regime_signals = build_feature_context(
                timestamp=ts,
                symbol=symbol,
                timeframe=timeframe,
                candles=candles_hist,
                vix_value=vix_value,
                vix_series=vix_series,
                chain_df=chain_asof,
                previous_chain_df=previous_chain_asof,
                fii_df=fii_hist,
                usdinr_close=(
                    usdinr_hist["close"].astype("float64")
                    if not usdinr_hist.empty and "close" in usdinr_hist.columns
                    else None
                ),
                regime=classifier.current_regime.value,
                thresholds=thresholds,
                chain_quality_thresholds=quality_thresholds,
                source="backtest_engine",
            )
            regime = classifier.classify(regime_signals)
            contexts[ts_key] = PrecomputedBarContext(
                timestamp=ts_key,
                regime=regime,
                snapshot=snapshot,
                chain_timestamp=chain_ts,
            )
            if chain_ts is not None:
                previous_chain_ts = chain_ts

        return BacktestPrecomputedData(
            vix=vix,
            fii=fii,
            usdinr=usdinr,
            chain=chain,
            chain_ranges=chain_ranges,
            chain_timestamps=chain_timestamps,
            chain_price_maps=chain_price_maps,
            bar_contexts=contexts,
        )

    def _next_signal(
        self,
        *,
        timestamp: datetime,
        regime: RegimeState,
        previous_regime: RegimeState,
        candles_hist: pd.DataFrame,
        vix_value: float,
        option_chain: pd.DataFrame | None,
        underlying_price: float,
        can_trade: bool,
    ) -> Signal | None:
        market_data = {
            "timestamp": timestamp,
            "candles": candles_hist,
            "vix": vix_value,
            "option_chain": option_chain,
            "underlying_price": underlying_price,
        }
        if self.strategy.state.current_position is None and not can_trade:
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                strategy_name=self.strategy.name,
                instrument=str(self.strategy.config.get("instrument", "NIFTY")),
                timestamp=timestamp,
                regime=regime,
                reason="Circuit breaker halted new entries",
            )
        if self.strategy.state.current_position is not None:
            exit_signal = self.strategy.get_exit_conditions(market_data)
            if exit_signal is not None and exit_signal.is_actionable:
                return exit_signal

        if self.strategy.should_be_active(regime):
            return self.strategy.generate_signal(market_data=market_data, regime=regime)
        return self.strategy.on_regime_change(previous_regime, regime)

    @staticmethod
    def _prep_vix(vix_df: pd.DataFrame | None) -> pd.DataFrame:
        if vix_df is None or vix_df.empty:
            return pd.DataFrame(columns=["timestamp", "close"])
        out = vix_df.copy()
        if "timestamp" not in out.columns or "close" not in out.columns:
            return pd.DataFrame(columns=["timestamp", "close"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return (
            out.dropna(subset=["timestamp", "close"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    @staticmethod
    def _prep_fii(fii_df: pd.DataFrame | None) -> pd.DataFrame:
        if fii_df is None or fii_df.empty:
            return pd.DataFrame(columns=["date", "fii_net"])
        out = fii_df.copy()
        if "date" not in out.columns or "fii_net" not in out.columns:
            return pd.DataFrame(columns=["date", "fii_net"])
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["fii_net"] = pd.to_numeric(out["fii_net"], errors="coerce")
        return out.dropna(subset=["date", "fii_net"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _prep_usdinr(usdinr_df: pd.DataFrame | None) -> pd.DataFrame:
        if usdinr_df is None or usdinr_df.empty:
            return pd.DataFrame(columns=["timestamp", "close"])
        out = usdinr_df.copy()
        if "timestamp" not in out.columns or "close" not in out.columns:
            return pd.DataFrame(columns=["timestamp", "close"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return (
            out.dropna(subset=["timestamp", "close"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    @staticmethod
    def _prep_chain(chain_df: pd.DataFrame | None) -> pd.DataFrame:
        if chain_df is None or chain_df.empty:
            return pd.DataFrame(columns=["timestamp", "option_type", "strike"])
        out = chain_df.copy()
        if "timestamp" not in out.columns:
            return pd.DataFrame(columns=["timestamp", "option_type", "strike"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        if "option_type" in out.columns:
            out["option_type"] = out["option_type"].astype(str).str.upper()
        if "strike" in out.columns:
            out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return out

    @staticmethod
    def _build_chain_ranges(chain_df: pd.DataFrame) -> dict[pd.Timestamp, tuple[int, int]]:
        if chain_df.empty:
            return {}
        out: dict[pd.Timestamp, tuple[int, int]] = {}
        for ts, idx in chain_df.groupby("timestamp", sort=True).indices.items():
            ts_key = pd.to_datetime(ts, errors="coerce")
            if pd.isna(ts_key):
                continue
            lo = int(min(idx))
            hi = int(max(idx)) + 1
            out[pd.Timestamp(ts_key)] = (lo, hi)
        return out

    @staticmethod
    def _latest_chain_timestamp(
        *, chain_timestamps: pd.DatetimeIndex, ts: datetime, strict: bool = False
    ) -> pd.Timestamp | None:
        if chain_timestamps.empty:
            return None
        cutoff = pd.Timestamp(ts)
        if strict:
            pos = int(chain_timestamps.searchsorted(cutoff, side="left")) - 1
        else:
            pos = int(chain_timestamps.searchsorted(cutoff, side="right")) - 1
        if pos < 0:
            return None
        return pd.Timestamp(chain_timestamps[pos])

    @staticmethod
    def _slice_chain_by_timestamp(
        chain_df: pd.DataFrame,
        chain_ranges: dict[pd.Timestamp, tuple[int, int]],
        chain_ts: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        if chain_ts is None:
            return None
        bounds = chain_ranges.get(chain_ts)
        if bounds is None:
            return None
        lo, hi = bounds
        return chain_df.iloc[lo:hi].reset_index(drop=True)

    @staticmethod
    def _latest_chain_asof(
        chain_df: pd.DataFrame, ts: datetime, *, strict: bool = False
    ) -> pd.DataFrame | None:
        if chain_df.empty:
            return None
        cutoff = pd.Timestamp(ts)
        if strict:
            eligible = chain_df.loc[chain_df["timestamp"] < cutoff]
        else:
            eligible = chain_df.loc[chain_df["timestamp"] <= cutoff]
        if eligible.empty:
            return None
        latest_ts = eligible["timestamp"].max()
        snap = eligible.loc[eligible["timestamp"] == latest_ts].copy()
        return cast(pd.DataFrame, snap.reset_index(drop=True))

    @staticmethod
    def _build_chain_price_map(chain_asof: pd.DataFrame | None) -> dict[str, float]:
        if chain_asof is None or chain_asof.empty:
            return {}

        symbol_col = (
            "symbol"
            if "symbol" in chain_asof.columns
            else "tradingsymbol"
            if "tradingsymbol" in chain_asof.columns
            else None
        )
        if symbol_col is None:
            return {}

        price_col = None
        for candidate in ("ltp", "last_price", "close", "price"):
            if candidate in chain_asof.columns:
                price_col = candidate
                break
        if price_col is None:
            return {}

        prices: dict[str, float] = {}
        for _, row in chain_asof.iterrows():
            symbol = str(row.get(symbol_col, "")).strip()
            if not symbol:
                continue
            raw_price = row.get(price_col)
            if raw_price is None:
                continue
            price = pd.to_numeric(raw_price, errors="coerce")
            if pd.isna(price):
                continue
            prices[symbol] = float(price)
        return prices

    @staticmethod
    def _compose_mark_price_map(
        *,
        chain_prices: dict[str, float],
        default_underlying_price: float,
        underlying_symbol: str,
    ) -> dict[str, float]:
        prices = dict(chain_prices)
        prices["UNDERLYING"] = float(default_underlying_price)
        prices[str(underlying_symbol)] = float(default_underlying_price)
        return prices

    @staticmethod
    def _build_mark_price_map(
        *,
        chain_asof: pd.DataFrame | None,
        default_underlying_price: float,
        underlying_symbol: str,
    ) -> dict[str, float]:
        chain_prices = BacktestEngine._build_chain_price_map(chain_asof)
        return BacktestEngine._compose_mark_price_map(
            chain_prices=chain_prices,
            default_underlying_price=default_underlying_price,
            underlying_symbol=underlying_symbol,
        )

    @staticmethod
    def _mark_to_market(
        *,
        cash: float,
        positions: dict[str, int],
        mark_prices: dict[str, float],
        fallback_prices: dict[str, float],
        default_underlying_price: float,
    ) -> float:
        total = float(cash)
        for instrument, qty in positions.items():
            if qty == 0:
                continue
            price = mark_prices.get(instrument)
            if price is None:
                price = fallback_prices.get(instrument, default_underlying_price)
            total += float(qty) * float(price)
        return total

    @staticmethod
    def _resolve_mark_price(
        *,
        instrument: str,
        mark_prices: dict[str, float],
        fallback_prices: dict[str, float],
        default_underlying_price: float,
    ) -> float:
        price = mark_prices.get(instrument)
        if price is None:
            price = fallback_prices.get(instrument, default_underlying_price)
        return float(price)

    @classmethod
    def _compute_unrealized_pnl(
        cls,
        *,
        positions: dict[str, int],
        avg_cost_by_instrument: dict[str, float],
        mark_prices: dict[str, float],
        fallback_prices: dict[str, float],
        default_underlying_price: float,
    ) -> float:
        unrealized = 0.0
        for instrument, qty in positions.items():
            if qty == 0:
                continue
            avg_cost = float(avg_cost_by_instrument.get(instrument, 0.0))
            mark = cls._resolve_mark_price(
                instrument=instrument,
                mark_prices=mark_prices,
                fallback_prices=fallback_prices,
                default_underlying_price=default_underlying_price,
            )
            if qty > 0:
                unrealized += (mark - avg_cost) * qty
            else:
                unrealized += (avg_cost - mark) * abs(qty)
        return float(unrealized)

    @staticmethod
    def _update_position_and_realized_pnl(
        *,
        positions: dict[str, int],
        avg_cost_by_instrument: dict[str, float],
        instrument: str,
        side: str,
        quantity: int,
        fill_price: float,
    ) -> float:
        realized = 0.0
        qty_change = quantity if side == "BUY" else -quantity
        current_qty = int(positions.get(instrument, 0))
        current_avg = float(avg_cost_by_instrument.get(instrument, 0.0))

        if (
            current_qty == 0
            or (current_qty > 0 and qty_change > 0)
            or (current_qty < 0 and qty_change < 0)
        ):
            new_qty = current_qty + qty_change
            if new_qty == 0:
                positions[instrument] = 0
                avg_cost_by_instrument.pop(instrument, None)
                return 0.0

            total_qty = abs(current_qty) + abs(qty_change)
            if total_qty > 0:
                new_avg = (
                    (abs(current_qty) * current_avg) + (abs(qty_change) * fill_price)
                ) / total_qty
                positions[instrument] = new_qty
                avg_cost_by_instrument[instrument] = float(new_avg)
            return 0.0

        if current_qty > 0 and qty_change < 0:
            closing_qty = min(current_qty, abs(qty_change))
            realized += (fill_price - current_avg) * closing_qty
            remaining = current_qty - closing_qty
            new_short = abs(qty_change) - closing_qty
            if remaining > 0:
                positions[instrument] = remaining
                avg_cost_by_instrument[instrument] = current_avg
            elif new_short > 0:
                positions[instrument] = -new_short
                avg_cost_by_instrument[instrument] = fill_price
            else:
                positions[instrument] = 0
                avg_cost_by_instrument.pop(instrument, None)
            return float(realized)

        if current_qty < 0 and qty_change > 0:
            closing_qty = min(abs(current_qty), qty_change)
            realized += (current_avg - fill_price) * closing_qty
            remaining_short = abs(current_qty) - closing_qty
            new_long = qty_change - closing_qty
            if remaining_short > 0:
                positions[instrument] = -remaining_short
                avg_cost_by_instrument[instrument] = current_avg
            elif new_long > 0:
                positions[instrument] = new_long
                avg_cost_by_instrument[instrument] = fill_price
            else:
                positions[instrument] = 0
                avg_cost_by_instrument.pop(instrument, None)
            return float(realized)

        return 0.0
