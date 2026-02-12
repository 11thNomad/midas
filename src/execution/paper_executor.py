"""Minimal paper execution simulator for actionable strategy signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.store import DataStore
from src.strategies.base import Signal, SignalType


@dataclass
class PaperExecutionEngine:
    """Immediate-fill paper executor with basic cost and persistence hooks."""

    base_dir: str = "data/cache"
    dataset: str = "paper_fills"
    slippage_bps: float = 5.0
    commission_per_order: float = 20.0
    _store: DataStore = field(init=False, repr=False)
    _fill_seq: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self._store = DataStore(base_dir=self.base_dir)

    def execute_signals(self, signals: list[Signal], *, market_data: dict[str, Any] | None = None) -> list[dict]:
        """Execute actionable signals as immediate paper fills and persist events."""
        market_data = market_data or {}
        fills: list[dict] = []

        for signal in signals:
            if not signal.is_actionable:
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
        orders = signal.orders or [{"symbol": signal.instrument, "action": self._default_action(signal)}]
        ts = signal.timestamp

        out: list[dict] = []
        for order in orders:
            self._fill_seq += 1
            raw_price = self._resolve_order_price(order=order, market_data=market_data)
            action = str(order.get("action", self._default_action(signal))).upper()
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
                    "instrument": str(order.get("symbol", signal.instrument)),
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
            # TODO: Wire strategy state updates (on_fill) once runtime tracks strategy instances per signal.
        return out

    @staticmethod
    def _default_action(signal: Signal) -> str:
        if signal.signal_type == SignalType.ENTRY_SHORT:
            return "SELL"
        if signal.signal_type == SignalType.EXIT:
            return "SELL"
        return "BUY"

    @staticmethod
    def _resolve_order_price(order: dict, market_data: dict[str, Any]) -> float:
        if "price" in order and order["price"] is not None:
            return float(order["price"])
        if "ltp" in order and order["ltp"] is not None:
            return float(order["ltp"])
        vix_hint = float(market_data.get("vix", 0.0) or 0.0)
        return max(1.0, vix_hint)

    def _apply_slippage(self, price: float, *, action: str) -> float:
        slip = (self.slippage_bps / 10_000.0) * price
        if action == "BUY":
            return price + slip
        return max(0.0, price - slip)
