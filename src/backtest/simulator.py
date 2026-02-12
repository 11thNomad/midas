"""Backtest fill simulator with configurable slippage and transaction costs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.strategies.base import Signal, SignalType


@dataclass
class FillSimulator:
    """Simple bar-close fill model used by the backtest engine."""

    slippage_pct: float = 0.05  # percent per side (0.05 means 0.05%)
    commission_per_order: float = 20.0

    def simulate(self, signal: Signal, *, close_price: float, timestamp: datetime) -> list[dict[str, Any]]:
        if not signal.is_actionable:
            return []

        orders = signal.orders or [{"symbol": signal.instrument, "action": self._default_action(signal), "quantity": 1}]
        fills: list[dict[str, Any]] = []
        for order in orders:
            side = str(order.get("action", self._default_action(signal))).upper()
            qty = int(order.get("quantity", 1) or 1)
            raw_price = float(order.get("price", close_price) or close_price)
            slip = raw_price * (self.slippage_pct / 100.0)
            price = raw_price + slip if side == "BUY" else max(0.0, raw_price - slip)

            fills.append(
                {
                    "timestamp": timestamp,
                    "strategy_name": signal.strategy_name,
                    "signal_type": signal.signal_type.value,
                    "instrument": str(order.get("symbol", signal.instrument)),
                    "side": side,
                    "quantity": qty,
                    "price": float(price),
                    "fees": float(self.commission_per_order),
                    "regime": signal.regime.value,
                    "reason": signal.reason,
                }
            )
        return fills

    @staticmethod
    def _default_action(signal: Signal) -> str:
        if signal.signal_type == SignalType.ENTRY_SHORT:
            return "SELL"
        if signal.signal_type == SignalType.EXIT:
            return "SELL"
        return "BUY"
