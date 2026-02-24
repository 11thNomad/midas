"""Minimal paper execution simulator for actionable strategy signals."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.store import DataStore
from src.risk.circuit_breaker import CircuitBreaker
from src.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class PaperExecutionEngine:
    """Immediate-fill paper executor with basic cost and persistence hooks."""

    base_dir: str = "data/cache"
    dataset: str = "paper_fills"
    slippage_bps: float = 5.0
    slippage_multiplier: float = 1.0
    commission_per_order: float = 20.0
    initial_capital: float = 150_000.0
    paper_capital: float | None = None
    margin_buffer_pct: float = 15.0
    available_cash_resolver: Callable[[], float | None] | None = None
    paper_log_dir: str = "data/paper_trading"
    circuit_breaker: CircuitBreaker | None = None
    _store: DataStore = field(init=False, repr=False)
    _fill_seq: int = field(default=0, init=False, repr=False)
    _cash: float = field(default=0.0, init=False, repr=False)
    _positions: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _avg_cost_by_instrument: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_price_by_instrument: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _realized_pnl_today: float = field(default=0.0, init=False, repr=False)
    _gross_realized_pnl_today: float = field(default=0.0, init=False, repr=False)
    _fees_paid_today: float = field(default=0.0, init=False, repr=False)
    _entries_today: int = field(default=0, init=False, repr=False)
    _exits_today: int = field(default=0, init=False, repr=False)
    _paper_log_path: Path = field(init=False, repr=False)
    _summary_day_key: str | None = field(default=None, init=False, repr=False)

    _FILL_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "trade_id",
        "signal_type",
        "instrument",
        "leg",
        "action",
        "strike",
        "expiry",
        "quantity",
        "fill_price",
        "mid_price",
        "slippage_applied",
        "fees_estimated",
        "regime",
        "vix",
        "adx",
    )
    _SUMMARY_COLUMNS: tuple[str, ...] = (
        "date",
        "entries",
        "exits",
        "open_positions",
        "gross_pnl",
        "fees",
        "net_pnl",
        "cash_balance",
        "margin_utilisation_pct",
    )

    def __post_init__(self) -> None:
        self._store = DataStore(base_dir=self.base_dir)
        self._cash = float(self.initial_capital)
        if self.paper_capital is None:
            self.paper_capital = float(self.initial_capital)
        self._paper_log_path = Path(self.paper_log_dir)
        self._paper_log_path.mkdir(parents=True, exist_ok=True)

    def execute_signals(
        self, signals: list[Signal], *, market_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute actionable signals as immediate paper fills and persist events."""
        market_data = market_data or {}
        self._roll_daily_counters_if_needed(market_data=market_data)
        fills: list[dict[str, Any]] = []
        executed_signals: list[Signal] = []

        for signal in signals:
            if not signal.is_actionable:
                continue
            if self.circuit_breaker is not None and not self.circuit_breaker.can_trade():
                continue
            if signal.signal_type == SignalType.ENTRY_SHORT and not self._can_enter_short(
                signal, market_data=market_data
            ):
                continue
            signal_fills = self._fills_for_signal(signal, market_data=market_data)
            if signal_fills:
                executed_signals.append(signal)
                if signal.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
                    self._entries_today += 1
                elif signal.signal_type == SignalType.EXIT:
                    self._exits_today += 1
            fills.extend(signal_fills)

        if fills:
            frame = pd.DataFrame(fills)
            self._store.write_time_series(
                self.dataset,
                frame,
                symbol=str(market_data.get("symbol", "NIFTY")),
                timestamp_col="timestamp",
                source="paper_executor",
            )

        if self.circuit_breaker is not None:
            default_price = float(
                market_data.get("last_price", market_data.get("close_price", 0.0)) or 0.0
            )
            equity = self._mark_to_market(default_price=default_price)
            unrealized_pnl = self._compute_unrealized_pnl(default_price=default_price)
            open_positions = sum(1 for _, qty in self._positions.items() if qty != 0)
            self.circuit_breaker.update(
                current_equity=equity,
                realized_pnl_today=self._realized_pnl_today,
                unrealized_pnl=unrealized_pnl,
                open_positions=open_positions,
            )

        self._write_csv_logs(fills=fills, signals=executed_signals, market_data=market_data)
        return fills

    def _can_enter_short(self, signal: Signal, *, market_data: dict[str, Any]) -> bool:
        required_margin = self._estimate_required_margin(signal, market_data=market_data)
        required_with_buffer = required_margin * (1.0 + (self.margin_buffer_pct / 100.0))
        available_cash, cash_source = self._resolve_available_cash()
        if available_cash >= required_with_buffer:
            return True
        logger.warning(
            "ABORT_ENTRY - insufficient margin (available: %.2f, required: %.2f, source: %s)",
            available_cash,
            required_with_buffer,
            cash_source,
        )
        return False

    def _estimate_required_margin(self, signal: Signal, *, market_data: dict[str, Any]) -> float:
        indicators = signal.indicators if isinstance(signal.indicators, dict) else {}
        wing_width = float(
            indicators.get("call_wing")
            or indicators.get("put_wing")
            or indicators.get("wing_width")
            or 100.0
        )
        orders = signal.orders or []
        quantity_units = max((int(order.get("quantity", 1) or 1) for order in orders), default=1)
        spot = self._resolve_spot(signal=signal, market_data=market_data)
        wing_based = wing_width * float(quantity_units) * 1.5
        notional_based = float(quantity_units) * spot * 0.06 if spot > 0.0 else 0.0
        minimum_floor = 100_000.0
        return max(wing_based, notional_based, minimum_floor)

    @staticmethod
    def _resolve_spot(*, signal: Signal, market_data: dict[str, Any]) -> float:
        indicators = signal.indicators if isinstance(signal.indicators, dict) else {}
        candidates = (
            indicators.get("spot"),
            indicators.get("underlying_price"),
            market_data.get("close_price"),
            market_data.get("last_price"),
        )
        for value in candidates:
            try:
                if value is None:
                    continue
                parsed = float(value)
                if parsed > 0.0:
                    return parsed
            except (TypeError, ValueError):
                continue
        return 0.0

    def _resolve_available_cash(self) -> tuple[float, str]:
        if self.available_cash_resolver is not None:
            try:
                resolved = self.available_cash_resolver()
            except Exception:  # pragma: no cover - defensive runtime guard
                resolved = None
            if resolved is not None and float(resolved) > 0.0:
                return float(resolved), "kite"
        return float(self.paper_capital or self.initial_capital), "paper_capital"

    def read_fills(
        self,
        *,
        symbol: str = "NIFTY",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return self._store.read_time_series(
            self.dataset,
            symbol=symbol,
            start=start,
            end=end,
            timestamp_col="timestamp",
        )

    def _fills_for_signal(
        self, signal: Signal, *, market_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        orders = signal.orders or [self._default_order(signal)]
        ts = signal.timestamp
        trade_id = self._resolve_trade_id(signal)

        out: list[dict[str, Any]] = []
        for order in orders:
            self._fill_seq += 1
            instrument = str(order.get("symbol", signal.instrument))
            raw_price = self._resolve_order_price(order=order, market_data=market_data)
            action = str(
                order.get("action", self._default_action(signal, symbol=instrument))
            ).upper()
            filled_price = self._apply_slippage(raw_price, action=action)
            option_type = self._resolve_option_type(order=order, instrument=instrument)
            leg = self._resolve_leg_label(
                signal_type=signal.signal_type.value,
                action=action,
                option_type=option_type,
            )
            quantity = int(order.get("quantity", 1) or 1)

            notional = abs(filled_price * quantity)
            fees = float(self.commission_per_order)
            slippage_applied = float(filled_price) - float(raw_price)
            out.append(
                {
                    "timestamp": ts,
                    "fill_id": f"PAPER-{self._fill_seq:08d}",
                    "trade_id": trade_id,
                    "strategy_name": signal.strategy_name,
                    "signal_type": signal.signal_type.value,
                    "instrument": instrument,
                    "side": action,
                    "quantity": quantity,
                    "price": float(filled_price),
                    "notional": float(notional),
                    "fees": fees,
                    "mid_price": float(raw_price),
                    "slippage_applied": float(slippage_applied),
                    "leg": leg,
                    "strike": order.get("strike"),
                    "expiry": order.get("expiry"),
                    "option_type": option_type,
                    "reason": signal.reason,
                    "regime": signal.regime.value,
                    "confidence": float(signal.confidence),
                }
            )
            realized_delta = self._update_position_and_realized_pnl(
                instrument=instrument,
                side=action,
                quantity=quantity,
                fill_price=filled_price,
            )
            self._gross_realized_pnl_today += realized_delta
            self._fees_paid_today += fees
            self._realized_pnl_today += realized_delta - fees
            if action == "BUY":
                self._cash -= notional + fees
            else:
                self._cash += notional - fees
            self._last_price_by_instrument[instrument] = float(filled_price)
            # TODO: Wire strategy state updates (on_fill)
            # once runtime tracks strategy instances per signal.
        return out

    def _write_csv_logs(
        self,
        *,
        fills: list[dict[str, Any]],
        signals: list[Signal],
        market_data: dict[str, Any],
    ) -> None:
        self._write_fills_csv(fills=fills, signals=signals, market_data=market_data)
        self._write_daily_summary_csv(market_data=market_data)

    def _write_fills_csv(
        self,
        *,
        fills: list[dict[str, Any]],
        signals: list[Signal],
        market_data: dict[str, Any],
    ) -> None:
        if not fills:
            return
        _ = signals  # consumed at source via per-fill metadata
        rows: list[dict[str, Any]] = []
        for fill in fills:
            ts = pd.Timestamp(fill.get("timestamp"))
            signal_type = str(fill.get("signal_type", "")).strip().lower()
            instrument = str(fill.get("instrument", "")).strip()
            trade_id = str(fill.get("trade_id", "")).strip()
            if not trade_id:
                trade_id = self._default_trade_id_from_fill(fill)
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "trade_id": trade_id,
                    "signal_type": signal_type,
                    "instrument": instrument,
                    "leg": str(fill.get("leg", "unknown")),
                    "action": str(fill.get("side", "")).upper(),
                    "strike": fill.get("strike"),
                    "expiry": self._serialize_expiry(fill.get("expiry")),
                    "quantity": int(fill.get("quantity", 0) or 0),
                    "fill_price": float(fill.get("price", 0.0) or 0.0),
                    "mid_price": float(fill.get("mid_price", 0.0) or 0.0),
                    "slippage_applied": float(fill.get("slippage_applied", 0.0) or 0.0),
                    "fees_estimated": float(fill.get("fees", 0.0) or 0.0),
                    "regime": str(fill.get("regime", "")),
                    "vix": self._as_float(market_data.get("vix")),
                    "adx": self._as_float(market_data.get("adx")),
                }
            )
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=list(self._FILL_COLUMNS))
        by_day = frame.groupby(pd.to_datetime(frame["timestamp"]).dt.strftime("%Y%m%d"), sort=True)
        for day_key, subset in by_day:
            path = self._paper_log_path / f"fills_{day_key}.csv"
            # Append-only audit trail by design: restarts can create duplicate logical fills
            # for the same day. Downstream reconciliation can deduplicate by fill_id/trade_id.
            subset.to_csv(path, index=False, mode="a", header=not path.exists())

    def _write_daily_summary_csv(self, *, market_data: dict[str, Any]) -> None:
        ts = pd.to_datetime(
            market_data.get("timestamp", datetime.now()),
            errors="coerce",
        )
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        day_key = ts.strftime("%Y%m%d")
        open_positions = sum(1 for _, qty in self._positions.items() if qty != 0)
        gross_pnl = float(self._gross_realized_pnl_today)
        fees = float(self._fees_paid_today)
        net_pnl = float(self._realized_pnl_today)
        summary_row = {
            "date": ts.strftime("%Y-%m-%d"),
            "entries": int(self._entries_today),
            "exits": int(self._exits_today),
            "open_positions": int(open_positions),
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "cash_balance": float(self._cash),
            "margin_utilisation_pct": self._estimate_margin_utilisation_pct(),
        }
        summary_path = self._paper_log_path / f"daily_summary_{day_key}.csv"
        pd.DataFrame([summary_row], columns=list(self._SUMMARY_COLUMNS)).to_csv(
            summary_path,
            index=False,
        )

    def _roll_daily_counters_if_needed(self, *, market_data: dict[str, Any]) -> None:
        ts = pd.to_datetime(
            market_data.get("timestamp", datetime.now()),
            errors="coerce",
        )
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        day_key = ts.strftime("%Y%m%d")
        if self._summary_day_key is None:
            self._summary_day_key = day_key
            return
        if day_key == self._summary_day_key:
            return
        self._summary_day_key = day_key
        self._entries_today = 0
        self._exits_today = 0
        self._gross_realized_pnl_today = 0.0
        self._fees_paid_today = 0.0
        self._realized_pnl_today = 0.0

    @staticmethod
    def _resolve_option_type(*, order: dict[str, Any], instrument: str) -> str | None:
        raw = order.get("option_type")
        if raw is not None:
            opt = str(raw).strip().upper()
            if opt in {"CE", "PE"}:
                return opt
        symbol = instrument.strip().upper()
        if symbol.endswith("CE"):
            return "CE"
        if symbol.endswith("PE"):
            return "PE"
        return None

    @staticmethod
    def _resolve_leg_label(*, signal_type: str, action: str, option_type: str | None) -> str:
        # Label indicates the original strategy leg being opened/unwound, not the trade action verb.
        opt = (option_type or "").upper()
        side = action.upper()
        st = signal_type.lower()
        if st == SignalType.ENTRY_SHORT.value:
            if opt == "CE" and side == "SELL":
                return "call_short"
            if opt == "PE" and side == "SELL":
                return "put_short"
            if opt == "CE" and side == "BUY":
                return "call_hedge"
            if opt == "PE" and side == "BUY":
                return "put_hedge"
        if st == SignalType.EXIT.value:
            if opt == "CE" and side == "BUY":
                return "call_short"
            if opt == "PE" and side == "BUY":
                return "put_short"
            if opt == "CE" and side == "SELL":
                return "call_hedge"
            if opt == "PE" and side == "SELL":
                return "put_hedge"
        return "unknown"

    @staticmethod
    def _serialize_expiry(value: Any) -> str | None:
        if value is None:
            return None
        expiry = pd.to_datetime(value, errors="coerce")
        if pd.isna(expiry):
            return str(value)
        return pd.Timestamp(expiry).strftime("%Y-%m-%d")

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_trade_id(self, signal: Signal) -> str:
        indicators = signal.indicators if isinstance(signal.indicators, dict) else {}
        for key in ("trade_id", "position_id"):
            value = indicators.get(key)
            if value is not None and str(value).strip():
                return str(value)
        ts = pd.Timestamp(signal.timestamp).strftime("%Y%m%dT%H%M%S")
        return f"{signal.strategy_name}:{signal.instrument}:{ts}"

    @staticmethod
    def _default_trade_id_from_fill(fill: dict[str, Any]) -> str:
        ts = pd.Timestamp(fill.get("timestamp")).strftime("%Y%m%dT%H%M%S")
        strategy = str(fill.get("strategy_name", "strategy")).strip() or "strategy"
        instrument = str(fill.get("instrument", "instrument")).strip() or "instrument"
        return f"{strategy}:{instrument}:{ts}"

    def _estimate_margin_utilisation_pct(self) -> float:
        capital = float(self.paper_capital or self.initial_capital)
        if capital <= 0.0:
            return 0.0
        gross_notional = 0.0
        for instrument, qty in self._positions.items():
            if qty == 0:
                continue
            price = float(self._last_price_by_instrument.get(instrument, 0.0) or 0.0)
            gross_notional += abs(float(qty) * price)
        # Approximation only: uses notional*6% and can overstate utilisation for hedged spreads.
        # Broker-reported margin should be treated as source of truth for operational decisions.
        margin_estimate = gross_notional * 0.06
        return (margin_estimate / capital) * 100.0

    def _default_order(self, signal: Signal) -> dict[str, Any]:
        symbol = signal.instrument
        action = self._default_action(signal, symbol=symbol)
        quantity = abs(int(self._positions.get(symbol, 0)))
        if quantity <= 0:
            quantity = 1
        return {"symbol": symbol, "action": action, "quantity": quantity}

    def _default_action(self, signal: Signal, *, symbol: str) -> str:
        if signal.signal_type == SignalType.ENTRY_SHORT:
            return "SELL"
        if signal.signal_type == SignalType.EXIT:
            net_qty = int(self._positions.get(symbol, 0))
            if net_qty < 0:
                return "BUY"
            if net_qty > 0:
                return "SELL"
            side_hint = str(signal.indicators.get("position_side", "")).lower()
            if side_hint == "short":
                return "BUY"
            if side_hint == "long":
                return "SELL"
            return "SELL"
        return "BUY"

    @staticmethod
    def _resolve_order_price(order: dict[str, Any], market_data: dict[str, Any]) -> float:
        if "price" in order and order["price"] is not None:
            return float(order["price"])
        if "ltp" in order and order["ltp"] is not None:
            return float(order["ltp"])
        if "last_price" in market_data and market_data["last_price"] is not None:
            return float(market_data["last_price"])
        if "close_price" in market_data and market_data["close_price"] is not None:
            return float(market_data["close_price"])
        return 1.0

    def _apply_slippage(self, price: float, *, action: str) -> float:
        effective_bps = self.slippage_bps * max(float(self.slippage_multiplier), 0.0)
        slip = (effective_bps / 10_000.0) * price
        if action == "BUY":
            return price + slip
        return max(0.0, price - slip)

    def _mark_to_market(self, *, default_price: float) -> float:
        equity = float(self._cash)
        for instrument, qty in self._positions.items():
            if qty == 0:
                continue
            price = self._last_price_by_instrument.get(instrument, default_price)
            equity += float(qty) * float(price)
        return equity

    def _compute_unrealized_pnl(self, *, default_price: float) -> float:
        unrealized = 0.0
        for instrument, qty in self._positions.items():
            if qty == 0:
                continue
            mark = float(self._last_price_by_instrument.get(instrument, default_price))
            avg_cost = float(self._avg_cost_by_instrument.get(instrument, mark))
            if qty > 0:
                unrealized += (mark - avg_cost) * qty
            else:
                unrealized += (avg_cost - mark) * abs(qty)
        return float(unrealized)

    def _update_position_and_realized_pnl(
        self,
        *,
        instrument: str,
        side: str,
        quantity: int,
        fill_price: float,
    ) -> float:
        realized = 0.0
        qty_change = quantity if side == "BUY" else -quantity
        current_qty = int(self._positions.get(instrument, 0))
        current_avg = float(self._avg_cost_by_instrument.get(instrument, 0.0))

        if (
            current_qty == 0
            or (current_qty > 0 and qty_change > 0)
            or (current_qty < 0 and qty_change < 0)
        ):
            new_qty = current_qty + qty_change
            if new_qty == 0:
                self._positions[instrument] = 0
                self._avg_cost_by_instrument.pop(instrument, None)
                return 0.0
            total_qty = abs(current_qty) + abs(qty_change)
            if total_qty > 0:
                new_avg = (
                    (abs(current_qty) * current_avg) + (abs(qty_change) * fill_price)
                ) / total_qty
                self._positions[instrument] = new_qty
                self._avg_cost_by_instrument[instrument] = float(new_avg)
            return 0.0

        if current_qty > 0 and qty_change < 0:
            closing_qty = min(current_qty, abs(qty_change))
            realized += (fill_price - current_avg) * closing_qty
            remaining = current_qty - closing_qty
            new_short = abs(qty_change) - closing_qty
            if remaining > 0:
                self._positions[instrument] = remaining
                self._avg_cost_by_instrument[instrument] = current_avg
            elif new_short > 0:
                self._positions[instrument] = -new_short
                self._avg_cost_by_instrument[instrument] = fill_price
            else:
                self._positions[instrument] = 0
                self._avg_cost_by_instrument.pop(instrument, None)
            return float(realized)

        if current_qty < 0 and qty_change > 0:
            closing_qty = min(abs(current_qty), qty_change)
            realized += (current_avg - fill_price) * closing_qty
            remaining_short = abs(current_qty) - closing_qty
            new_long = qty_change - closing_qty
            if remaining_short > 0:
                self._positions[instrument] = -remaining_short
                self._avg_cost_by_instrument[instrument] = current_avg
            elif new_long > 0:
                self._positions[instrument] = new_long
                self._avg_cost_by_instrument[instrument] = fill_price
            else:
                self._positions[instrument] = 0
                self._avg_cost_by_instrument.pop(instrument, None)
            return float(realized)

        return 0.0
