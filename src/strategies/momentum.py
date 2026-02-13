"""EMA crossover momentum strategy (Phase 5 starter, backtest-ready)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.signals import trend
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class MomentumStrategy(BaseStrategy):
    """Long/short momentum using EMA crossover with ADX trend filter."""

    def generate_signal(self, market_data: dict[str, Any], regime: RegimeState) -> Signal:
        raw_ts = market_data.get("timestamp")
        ts = raw_ts if isinstance(raw_ts, datetime) else datetime.now()
        candles = market_data.get("candles")
        instrument = self.config.get("instrument", "NIFTY")

        if candles is None or len(candles) < 60:
            return self._no_signal(ts, regime, instrument, reason="Insufficient candles")

        close = candles["close"].astype("float64")
        high = candles["high"].astype("float64")
        low = candles["low"].astype("float64")

        fast = int(self.config.get("fast_ema", 20))
        slow = int(self.config.get("slow_ema", 50))
        adx_filter = float(self.config.get("adx_filter", 25.0))
        lots = self.compute_position_size(capital=0, risk_per_trade=0)

        ema_signal = trend.ema_crossover(close, fast=fast, slow=slow)
        latest_cross = int(ema_signal.iloc[-1])
        adx_value = float(trend.adx(high=high, low=low, close=close, period=14).fillna(0).iloc[-1])

        if adx_value < adx_filter:
            return self._no_signal(ts, regime, instrument, reason="ADX filter not met")

        # Position management via signal direction.
        if self.state.current_position is None:
            if latest_cross > 0:
                self.state.current_position = {"side": "LONG", "quantity": lots}
                return Signal(
                    signal_type=SignalType.ENTRY_LONG,
                    strategy_name=self.name,
                    instrument=instrument,
                    timestamp=ts,
                    orders=[{"symbol": instrument, "action": "BUY", "quantity": lots}],
                    regime=regime,
                    indicators={"adx": adx_value},
                    reason="EMA bullish crossover with ADX filter",
                )
            self.state.current_position = {"side": "SHORT", "quantity": lots}
            return Signal(
                signal_type=SignalType.ENTRY_SHORT,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[{"symbol": instrument, "action": "SELL", "quantity": lots}],
                regime=regime,
                indicators={"adx": adx_value},
                reason="EMA bearish crossover with ADX filter",
            )

        current_side = str(self.state.current_position.get("side", ""))
        qty = int(self.state.current_position.get("quantity", lots))

        if current_side == "LONG" and latest_cross < 0:
            self.state.current_position = {"side": "SHORT", "quantity": qty}
            return Signal(
                signal_type=SignalType.ADJUST,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[
                    {"symbol": instrument, "action": "SELL", "quantity": qty},  # exit long
                    {"symbol": instrument, "action": "SELL", "quantity": qty},  # enter short
                ],
                regime=regime,
                indicators={"adx": adx_value},
                reason="Crossed from bullish to bearish",
            )

        if current_side == "SHORT" and latest_cross > 0:
            self.state.current_position = {"side": "LONG", "quantity": qty}
            return Signal(
                signal_type=SignalType.ADJUST,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[
                    {"symbol": instrument, "action": "BUY", "quantity": qty},  # exit short
                    {"symbol": instrument, "action": "BUY", "quantity": qty},  # enter long
                ],
                regime=regime,
                indicators={"adx": adx_value},
                reason="Crossed from bearish to bullish",
            )

        return self._no_signal(ts, regime, instrument, reason="No crossover change")

    def get_exit_conditions(self, market_data: dict[str, Any]) -> Signal | None:
        return None

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return int(self.config.get("max_lots", 1) or 1)

    def _no_signal(
        self, ts: datetime, regime: RegimeState, instrument: str, *, reason: str
    ) -> Signal:
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason=reason,
        )
