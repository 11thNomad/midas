"""Backtest fill simulator with configurable slippage and transaction costs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.strategies.base import Signal, SignalType


@dataclass
class FillSimulator:
    """Simple bar-close fill model used by the backtest engine."""

    slippage_pct: float = 0.05  # percent per side (0.05 means 0.05%)
    commission_per_order: float = 20.0
    stt_pct: float = 0.1
    exchange_txn_charges_pct: float = 0.03503
    gst_pct: float = 18.0
    sebi_fee_pct: float = 0.0001
    stamp_duty_pct: float = 0.003

    def simulate(
        self,
        signal: Signal,
        *,
        close_price: float,
        timestamp: datetime,
        price_lookup: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        if not signal.is_actionable:
            return []

        orders = signal.orders or [
            {"symbol": signal.instrument, "action": self._default_action(signal), "quantity": 1}
        ]
        resolved_prices: list[tuple[dict[str, Any], float]] = []
        for order in orders:
            raw_price = self._resolve_raw_price(
                order=order,
                default_price=close_price,
                price_lookup=price_lookup,
            )
            if raw_price is None:
                # For option legs, we require an explicit/lookup price.
                # Falling back to underlying close creates invalid fills.
                return []
            resolved_prices.append((order, raw_price))

        fills: list[dict[str, Any]] = []
        for order, raw_price in resolved_prices:
            side = str(order.get("action", self._default_action(signal))).upper()
            qty = int(order.get("quantity", 1) or 1)
            slip = raw_price * (self.slippage_pct / 100.0)
            price = raw_price + slip if side == "BUY" else max(0.0, raw_price - slip)
            notional = abs(price * qty)
            fee_parts = self._compute_fees(side=side, notional=notional)
            total_fees = float(sum(fee_parts.values()))

            fills.append(
                {
                    "timestamp": timestamp,
                    "strategy_name": signal.strategy_name,
                    "signal_type": signal.signal_type.value,
                    "instrument": str(order.get("symbol", signal.instrument)),
                    "side": side,
                    "quantity": qty,
                    "price": float(price),
                    "notional": float(notional),
                    "fees": total_fees,
                    **fee_parts,
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

    def _compute_fees(self, *, side: str, notional: float) -> dict[str, float]:
        brokerage = float(self.commission_per_order)
        exchange = notional * (self.exchange_txn_charges_pct / 100.0)
        stt = notional * (self.stt_pct / 100.0) if side == "SELL" else 0.0
        sebi = notional * (self.sebi_fee_pct / 100.0)
        stamp = notional * (self.stamp_duty_pct / 100.0) if side == "BUY" else 0.0
        gst = (brokerage + exchange + sebi) * (self.gst_pct / 100.0)
        return {
            "brokerage": float(brokerage),
            "exchange_charges": float(exchange),
            "stt": float(stt),
            "sebi_fee": float(sebi),
            "stamp_duty": float(stamp),
            "gst": float(gst),
        }

    @staticmethod
    def _resolve_raw_price(
        *,
        order: dict[str, Any],
        default_price: float,
        price_lookup: dict[str, float] | None,
    ) -> float | None:
        if "price" in order and order["price"] is not None:
            return float(order["price"])
        if "ltp" in order and order["ltp"] is not None:
            return float(order["ltp"])
        symbol = str(order.get("symbol", "")).strip()
        if symbol and price_lookup and symbol in price_lookup:
            return float(price_lookup[symbol])
        if price_lookup:
            option_key = FillSimulator._canonical_option_key_from_order(order)
            if option_key is None and symbol:
                option_key = FillSimulator._canonical_option_key_from_symbol(symbol)
            if option_key is not None and option_key in price_lookup:
                return float(price_lookup[option_key])
        if symbol and FillSimulator._is_option_symbol(symbol):
            return None
        return float(default_price)

    @staticmethod
    def _is_option_symbol(symbol: str) -> bool:
        upper = symbol.upper()
        return upper.endswith("CE") or upper.endswith("PE")

    @staticmethod
    def _canonical_option_key_from_order(order: dict[str, Any]) -> str | None:
        expiry = order.get("expiry")
        strike = order.get("strike")
        option_type = order.get("option_type")
        if expiry is None or strike is None or option_type is None:
            return None
        return FillSimulator._canonical_option_key(
            expiry=expiry,
            strike=strike,
            option_type=option_type,
        )

    @staticmethod
    def _canonical_option_key_from_symbol(symbol: str) -> str | None:
        sym = str(symbol).strip().upper()
        if not sym:
            return None

        # Format: NIFTY_20240718_24700CE
        m = re.match(r"^[A-Z]+_(\d{8})_(\d+)(CE|PE)$", sym)
        if m:
            return f"OPT::{m.group(1)}_{int(m.group(2))}_{m.group(3)}"

        # Format: NIFTY2471824700CE -> YYMMDD + strike + option type
        m = re.match(r"^[A-Z]+(\d{6})(\d+)(CE|PE)$", sym)
        if m:
            yymmdd = m.group(1)
            yyyy = 2000 + int(yymmdd[:2])
            mm = int(yymmdd[2:4])
            dd = int(yymmdd[4:6])
            try:
                expiry = pd.Timestamp(year=yyyy, month=mm, day=dd)
            except ValueError:
                return None
            return f"OPT::{expiry.strftime('%Y%m%d')}_{int(m.group(2))}_{m.group(3)}"

        return None

    @staticmethod
    def _canonical_option_key(
        *,
        expiry: Any,
        strike: Any,
        option_type: Any,
    ) -> str | None:
        expiry_ts = pd.to_datetime(expiry, errors="coerce")
        strike_value = pd.to_numeric(strike, errors="coerce")
        if pd.isna(expiry_ts) or pd.isna(strike_value):
            return None
        opt = str(option_type).strip().upper()
        if opt not in {"CE", "PE"}:
            return None
        return f"OPT::{expiry_ts.strftime('%Y%m%d')}_{int(float(strike_value))}_{opt}"
