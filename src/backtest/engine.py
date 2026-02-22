"""Event-driven backtest engine scaffold (Phase 4)."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import pandas as pd

from src.backtest.metrics import summarize_backtest
from src.backtest.simulator import FillSimulator
from src.data.option_chain_quality import OptionChainQualityThresholds
from src.data.option_symbols import option_lookup_keys, resolve_option_price
from src.regime.classifier import RegimeClassifier
from src.risk.circuit_breaker import CircuitBreaker
from src.signals.contracts import SignalSnapshotDTO, frame_from_signal_snapshots
from src.signals.pipeline import build_feature_context
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    regimes: pd.DataFrame
    metrics: dict[str, Any]
    signal_snapshots: pd.DataFrame = field(default_factory=pd.DataFrame)
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    number_of_symbols_in_run: int = 1
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
        vix = self._prep_vix(vix_df)
        fii = self._prep_fii(fii_df)
        usdinr = self._prep_usdinr(usdinr_df)
        chain = self._prep_chain(option_chain_df)

        cash = float(self.initial_capital)
        positions: dict[str, int] = {}
        avg_cost_by_instrument: dict[str, float] = {}
        last_price_by_instrument: dict[str, float] = {}
        mark_prices: dict[str, float] = {}
        if self.strategy.state.current_position is not None:
            LOGGER.warning(
                "Backtest run started with non-empty strategy state; bootstrapping positions",
                extra={"strategy_name": self.strategy.name},
            )
        self._bootstrap_positions_from_strategy_state(
            positions=positions,
            avg_cost_by_instrument=avg_cost_by_instrument,
        )
        realized_pnl_today = 0.0
        fill_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        regime_rows: list[dict[str, Any]] = []
        signal_snapshot_rows: list[SignalSnapshotDTO] = []
        decision_rows: list[dict[str, Any]] = []
        # Exit counters are event-level (one per exit signal), not per leg/fill.
        exit_attempted_count = 0
        exit_filled_count = 0
        exit_unfilled_count = 0
        exit_unfilled_reason_counts: dict[str, int] = {}
        forced_liquidation_count = 0
        forced_liquidation_symbols: set[str] = set()
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
            vix_hist = vix.loc[vix["timestamp"] < ts_key] if not vix.empty else pd.DataFrame()
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
            if vix_series is not None and not vix_series.empty:
                vix_value = float(vix_series.iloc[-1])
            else:
                vix_value = 0.0

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
                if is_exit:
                    exit_attempted_count += 1
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
                    if is_exit:
                        exit_unfilled_count += 1
                        reason = str(signal.reason or "unknown")
                        exit_unfilled_reason_counts[reason] = (
                            exit_unfilled_reason_counts.get(reason, 0) + 1
                        )
                    # Keep strategy state aligned to realized execution.
                    self.strategy.state.current_position = position_before
                    continue
                if is_exit:
                    exit_filled_count += 1
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

        if not bars.empty:
            final_ts = pd.Timestamp(bars.iloc[-1]["timestamp"]).to_pydatetime()
            final_close = float(bars.iloc[-1]["close"])
            forced_fills, impacted_symbols = self._force_liquidation_at_window_end(
                timestamp=final_ts,
                close_price=final_close,
                positions=positions,
                strategy_position=self.strategy.state.current_position,
                mark_prices=mark_prices,
                regime=previous_regime,
            )
            if forced_fills:
                forced_liquidation_count = max(int(len(impacted_symbols)), 1)
                forced_liquidation_symbols.update(impacted_symbols)
                exit_attempted_count += 1
                exit_filled_count += 1
                for fill in forced_fills:
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
                self.strategy.state.current_position = None
                decision_rows.append(
                    {
                        "timestamp": final_ts,
                        "strategy_name": self.strategy.name,
                        "signal_type": SignalType.EXIT.value,
                        "instrument": str(self.strategy.config.get("instrument", "NIFTY")),
                        "regime": previous_regime.value,
                        "is_actionable": 1.0,
                        "can_trade": 1.0,
                        "orders_count": len(forced_fills),
                        "reason": "backtest_window_end",
                        "indicators_json": json.dumps({}, sort_keys=True, default=str),
                        "greeks_snapshot_json": json.dumps({}, sort_keys=True, default=str),
                    }
                )
                equity = self._mark_to_market(
                    cash=cash,
                    positions=positions,
                    mark_prices=mark_prices,
                    fallback_prices=last_price_by_instrument,
                    default_underlying_price=final_close,
                )
                open_positions = sum(1 for _, qty in positions.items() if qty != 0)
                replacement = {
                    "timestamp": final_ts,
                    "cash": cash,
                    "open_positions": open_positions,
                    "equity": equity,
                }
                if equity_rows:
                    equity_rows[-1] = replacement
                else:
                    equity_rows.append(replacement)
            elif self.strategy.state.current_position is not None:
                fallback_symbol = str(self.strategy.config.get("instrument", "NIFTY"))
                forced_liquidation_symbols.add(fallback_symbol)
                forced_liquidation_count = max(int(len(forced_liquidation_symbols)), 1)
                exit_attempted_count += 1
                exit_unfilled_count += 1
                exit_unfilled_reason_counts["backtest_window_end"] = (
                    exit_unfilled_reason_counts.get("backtest_window_end", 0) + 1
                )
                decision_rows.append(
                    {
                        "timestamp": final_ts,
                        "strategy_name": self.strategy.name,
                        "signal_type": SignalType.EXIT.value,
                        "instrument": str(self.strategy.config.get("instrument", "NIFTY")),
                        "regime": previous_regime.value,
                        "is_actionable": 1.0,
                        "can_trade": 1.0,
                        "orders_count": 0,
                        "reason": "backtest_window_end",
                        "indicators_json": json.dumps({"fill_failed": True}, sort_keys=True),
                        "greeks_snapshot_json": json.dumps({}, sort_keys=True, default=str),
                    }
                )

        fills_df = pd.DataFrame(fill_rows)
        equity_df = pd.DataFrame(equity_rows)
        regimes_df = pd.DataFrame(regime_rows)
        signal_snapshots_df = frame_from_signal_snapshots(signal_snapshot_rows)
        decisions_df = pd.DataFrame(decision_rows)
        decisions_df = self._annotate_early_exit_opportunities(
            decisions_df=decisions_df,
            fills=fills_df,
        )
        run_integrity = {
            "forced_liquidations": {
                "count": int(forced_liquidation_count),
                "symbols": sorted(forced_liquidation_symbols),
                "threshold": int(max(self.number_of_symbols_in_run, 1)),
                "flag": bool(forced_liquidation_count > max(self.number_of_symbols_in_run, 1)),
            },
            "unfilled_exits": {
                "attempted": int(exit_attempted_count),
                "filled": int(exit_filled_count),
                "unfilled": int(exit_unfilled_count),
                "failure_reasons": exit_unfilled_reason_counts,
            },
        }
        metrics = summarize_backtest(
            equity_curve=equity_df,
            fills=fills_df,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year,
            risk_free_rate_annual=self.risk_free_rate_annual,
            monte_carlo_permutations=self.monte_carlo_permutations,
            minimum_trade_count=self.minimum_trade_count,
            run_integrity=run_integrity,
        )
        return BacktestResult(
            equity_curve=equity_df,
            fills=fills_df,
            regimes=regimes_df,
            signal_snapshots=signal_snapshots_df,
            decisions=decisions_df,
            metrics=metrics,
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
        if previous_regime != regime:
            return self.strategy.on_regime_change(previous_regime, regime)
        return None

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
            resolved_price = float(price)
            for lookup_key in option_lookup_keys(
                symbol=symbol,
                expiry=row.get("expiry"),
                strike=row.get("strike"),
                option_type=row.get("option_type"),
            ):
                prices[lookup_key] = resolved_price
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

    def _force_liquidation_at_window_end(
        self,
        *,
        timestamp: datetime,
        close_price: float,
        positions: dict[str, int],
        strategy_position: dict[str, Any] | None,
        mark_prices: dict[str, float],
        regime: RegimeState,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        non_zero_positions = {k: int(v) for k, v in positions.items() if int(v) != 0}
        if not non_zero_positions:
            return [], set()

        orders: list[dict[str, Any]] = []
        impacted_symbols: set[str] = set()
        leg_lookup: dict[str, dict[str, Any]] = {}
        if strategy_position is not None:
            for leg in strategy_position.get("legs", []):
                leg_symbol = str(leg.get("symbol", "")).strip()
                if leg_symbol:
                    leg_lookup[leg_symbol] = leg

        for instrument, qty in non_zero_positions.items():
            action = "BUY" if qty < 0 else "SELL"
            order: dict[str, Any] = {
                "symbol": instrument,
                "action": action,
                "quantity": abs(qty),
            }
            leg = leg_lookup.get(instrument)
            if leg is not None:
                for key in ("expiry", "strike", "option_type"):
                    if key in leg and leg.get(key) is not None:
                        order[key] = leg.get(key)
            orders.append(order)
            impacted_symbols.add(self._root_symbol_from_instrument(instrument))

        signal = Signal(
            signal_type=SignalType.EXIT,
            strategy_name=self.strategy.name,
            instrument=str(self.strategy.config.get("instrument", "NIFTY")),
            timestamp=timestamp,
            orders=orders,
            regime=regime,
            reason="backtest_window_end",
        )
        fills = self.simulator.simulate(
            signal,
            close_price=close_price,
            timestamp=timestamp,
            price_lookup=mark_prices,
        )
        return fills, impacted_symbols

    @staticmethod
    def _root_symbol_from_instrument(instrument: str) -> str:
        raw = str(instrument).strip().upper()
        if not raw:
            return "UNKNOWN"
        if "_" in raw:
            return raw.split("_", 1)[0]
        root_chars: list[str] = []
        for ch in raw:
            if ch.isalpha():
                root_chars.append(ch)
                continue
            break
        return "".join(root_chars) or raw

    def _bootstrap_positions_from_strategy_state(
        self,
        *,
        positions: dict[str, int],
        avg_cost_by_instrument: dict[str, float],
    ) -> None:
        current = self.strategy.state.current_position
        if not isinstance(current, dict):
            return

        legs = current.get("legs")
        if isinstance(legs, list) and legs:
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                symbol = str(leg.get("symbol", "")).strip()
                action = str(leg.get("action", "")).strip().upper()
                quantity = int(leg.get("quantity", 0) or 0)
                if not symbol or quantity <= 0 or action not in {"BUY", "SELL"}:
                    continue
                qty_delta = quantity if action == "BUY" else -quantity
                positions[symbol] = int(positions.get(symbol, 0)) + qty_delta
                avg_cost_by_instrument[symbol] = float(leg.get("price", 0.0) or 0.0)
            return

        symbol = str(current.get("symbol", self.strategy.config.get("instrument", "NIFTY"))).strip()
        quantity = int(current.get("quantity", 0) or 0)
        if not symbol or quantity == 0:
            return
        positions[symbol] = quantity
        avg_cost_by_instrument[symbol] = float(current.get("entry_price", 0.0) or 0.0)

    @staticmethod
    def _annotate_early_exit_opportunities(
        *,
        decisions_df: pd.DataFrame,
        fills: pd.DataFrame,
    ) -> pd.DataFrame:
        if decisions_df.empty:
            out = decisions_df.copy()
            out["early_exit_opportunity"] = pd.Series(dtype="bool")
            out["earliest_exit_day"] = pd.Series(dtype="object")
            out["earliest_exit_pnl"] = pd.Series(dtype="float64")
            out["actual_exit_pnl"] = pd.Series(dtype="float64")
            out["pnl_delta_vs_earliest_exit"] = pd.Series(dtype="float64")
            return out

        out = decisions_df.copy()
        out["early_exit_opportunity"] = False
        out["earliest_exit_day"] = pd.NA
        out["earliest_exit_pnl"] = pd.NA
        out["actual_exit_pnl"] = pd.NA
        out["pnl_delta_vs_earliest_exit"] = pd.NA

        if fills.empty:
            return out

        fill_frame = fills.copy()
        fill_frame["timestamp"] = pd.to_datetime(fill_frame["timestamp"], errors="coerce")
        fill_frame = fill_frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        if fill_frame.empty:
            return out

        signed_cash = []
        for _, row in fill_frame.iterrows():
            side = str(row.get("side", "")).upper()
            raw_notional = pd.to_numeric(row.get("notional", 0.0), errors="coerce")
            raw_fees = pd.to_numeric(row.get("fees", 0.0), errors="coerce")
            notional = 0.0 if pd.isna(raw_notional) else float(raw_notional)
            fees = 0.0 if pd.isna(raw_fees) else float(raw_fees)
            signed = notional if side == "SELL" else -notional
            signed_cash.append(float(signed - fees))
        fill_frame["_net_cash"] = signed_cash

        entry_blocks = (
            fill_frame.loc[fill_frame["signal_type"].astype(str).str.contains("entry", case=False)]
            .groupby("timestamp", as_index=False)
            .agg(entry_net=("_net_cash", "sum"))
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        exit_blocks = (
            fill_frame.loc[fill_frame["signal_type"].astype(str).str.contains("exit", case=False)]
            .groupby("timestamp", as_index=False)
            .agg(exit_net=("_net_cash", "sum"), exit_fees=("fees", "sum"))
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if entry_blocks.empty or exit_blocks.empty:
            return out

        if len(entry_blocks) != len(exit_blocks):
            LOGGER.warning(
                "Early-exit opportunity pairing mismatch; pairing chronologically by index",
                extra={
                    "entry_block_count": int(len(entry_blocks)),
                    "exit_block_count": int(len(exit_blocks)),
                },
            )

        pair_count = min(len(entry_blocks), len(exit_blocks))
        if pair_count <= 0:
            return out

        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        for i in range(pair_count):
            entry_ts = pd.Timestamp(entry_blocks.iloc[i]["timestamp"])
            exit_ts = pd.Timestamp(exit_blocks.iloc[i]["timestamp"])
            entry_net = float(entry_blocks.iloc[i]["entry_net"])
            actual_exit_net = float(exit_blocks.iloc[i]["exit_net"])
            proxy_exit_fees = float(exit_blocks.iloc[i]["exit_fees"])
            actual_trade_pnl = entry_net + actual_exit_net

            pre_exit = out.loc[
                (out["timestamp"] > entry_ts)
                & (out["timestamp"] < exit_ts)
                & (out["signal_type"].astype(str).str.lower() == "exit")
            ].copy()
            if pre_exit.empty:
                continue

            pre_exit["reason_lower"] = pre_exit["reason"].astype(str).str.lower()
            candidates = pre_exit.loc[
                pre_exit["reason_lower"].str.startswith("profit target hit")
            ].copy()
            if candidates.empty:
                continue

            candidates = candidates.sort_values("timestamp").reset_index(drop=True)
            earliest = candidates.iloc[0]
            indicators = {}
            raw_indicators = earliest.get("indicators_json")
            if isinstance(raw_indicators, str) and raw_indicators.strip():
                try:
                    indicators = json.loads(raw_indicators)
                except json.JSONDecodeError:
                    indicators = {}

            close_debit = pd.to_numeric(indicators.get("close_debit"), errors="coerce")
            if pd.isna(close_debit):
                continue
            earliest_exit_pnl = entry_net - float(close_debit) - proxy_exit_fees
            delta_vs_earliest = actual_trade_pnl - earliest_exit_pnl

            mask_actual_exit = (
                (out["timestamp"] == exit_ts)
                & (out["signal_type"].astype(str).str.lower() == "exit")
            )
            if not mask_actual_exit.any():
                continue
            actual_exit_reason = out.loc[mask_actual_exit, "reason"].astype(str).str.lower()
            if actual_exit_reason.str.startswith("profit target hit").any():
                continue

            out.loc[mask_actual_exit, "early_exit_opportunity"] = True
            out.loc[mask_actual_exit, "earliest_exit_day"] = pd.Timestamp(
                earliest["timestamp"]
            ).day_name()
            out.loc[mask_actual_exit, "earliest_exit_pnl"] = float(earliest_exit_pnl)
            out.loc[mask_actual_exit, "actual_exit_pnl"] = float(actual_trade_pnl)
            out.loc[mask_actual_exit, "pnl_delta_vs_earliest_exit"] = float(delta_vs_earliest)

        return out

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
            price = resolve_option_price(
                price_lookup=mark_prices,
                symbol=instrument,
            )
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
        price = resolve_option_price(
            price_lookup=mark_prices,
            symbol=instrument,
        )
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
