"""Minimal paper execution simulator for actionable strategy signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.store import DataStore
from src.risk.circuit_breaker import CircuitBreaker
from src.strategies.base import Signal, SignalType


@dataclass
class PaperExecutionEngine:
    """Immediate-fill paper executor with basic cost and persistence hooks."""

    base_dir: str = "data/cache"
    dataset: str = "paper_fills"
    slippage_bps: float = 5.0
    commission_per_order: float = 20.0
    initial_capital: float = 1_000_000.0
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

    def __post_init__(self):
        self._store = DataStore(base_dir=self.base_dir)
        self._cash = float(self.initial_capital)

    def execute_signals(
        self, signals: list[Signal], *, market_data: dict[str, Any] | None = None
    ) -> list[dict]:
        """Execute actionable signals as immediate paper fills and persist events."""
        market_data = market_data or {}
        fills: list[dict] = []

        for signal in signals:
            if not signal.is_actionable:
                continue
            if self.circuit_breaker is not None and not self.circuit_breaker.can_trade():
                continue
            fills.extend(self._fills_for_signal(signal, market_data=market_data))

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

        return fills

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

    def _fills_for_signal(self, signal: Signal, *, market_data: dict[str, Any]) -> list[dict]:
        orders = signal.orders or [self._default_order(signal)]
        ts = signal.timestamp

        out: list[dict] = []
        for order in orders:
            self._fill_seq += 1
            instrument = str(order.get("symbol", signal.instrument))
            raw_price = self._resolve_order_price(order=order, market_data=market_data)
            action = str(
                order.get("action", self._default_action(signal, symbol=instrument))
            ).upper()
            filled_price = self._apply_slippage(raw_price, action=action)
            quantity = int(order.get("quantity", 1) or 1)

            notional = abs(filled_price * quantity)
            fees = float(self.commission_per_order)
            out.append(
                {
                    "timestamp": ts,
                    "fill_id": f"PAPER-{self._fill_seq:08d}",
                    "strategy_name": signal.strategy_name,
                    "signal_type": signal.signal_type.value,
                    "instrument": instrument,
                    "side": action,
                    "quantity": quantity,
                    "price": float(filled_price),
                    "notional": float(notional),
                    "fees": fees,
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
            self._realized_pnl_today += realized_delta - fees
            if action == "BUY":
                self._cash -= notional + fees
            else:
                self._cash += notional - fees
            self._last_price_by_instrument[instrument] = float(filled_price)
            # TODO: Wire strategy state updates (on_fill)
            # once runtime tracks strategy instances per signal.
        return out

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
    def _resolve_order_price(order: dict, market_data: dict[str, Any]) -> float:
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
        slip = (self.slippage_bps / 10_000.0) * price
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
