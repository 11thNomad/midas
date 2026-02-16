"""Baseline trend strategy aligned with vectorbt baseline gate logic."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.signals import trend
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class BaselineTrendStrategy(BaseStrategy):
    """Long-only strategy gated by regime + ADX (+ optional VIX cap)."""

    def generate_signal(self, market_data: dict[str, Any], regime: RegimeState) -> Signal:
        raw_ts = market_data.get("timestamp")
        ts = raw_ts if isinstance(raw_ts, datetime) else datetime.now()
        instrument = self.config.get("instrument", "NIFTY")
        candles = market_data.get("candles")
        adx_period = int(self.config.get("adx_period", 14) or 14)

        if candles is None or len(candles) < max(20, adx_period + 1):
            return self._no_signal(ts, regime, instrument, reason="Insufficient candles")

        high = candles["high"].astype("float64")
        low = candles["low"].astype("float64")
        close = candles["close"].astype("float64")
        adx_value = float(trend.adx(high=high, low=low, close=close, period=adx_period).iloc[-1])
        if adx_value != adx_value:  # NaN guard
            adx_value = 0.0

        adx_min = float(self.config.get("adx_min", 25.0) or 25.0)
        vix_raw = self.config.get("vix_max")
        vix_max = float(vix_raw) if vix_raw is not None else None
        vix_value = float(market_data.get("vix", 0.0) or 0.0)

        gate = (regime in self.active_regimes) and (adx_value >= adx_min)
        if vix_max is not None:
            gate = gate and (vix_value <= vix_max)

        if self.state.current_position is None and gate:
            qty = self.compute_position_size(capital=0.0, risk_per_trade=0.0)
            self.state.current_position = {"side": "LONG", "quantity": qty}
            return Signal(
                signal_type=SignalType.ENTRY_LONG,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[{"symbol": instrument, "action": "BUY", "quantity": qty}],
                regime=regime,
                indicators={"adx": adx_value, "vix": vix_value},
                reason="Baseline trend entry gate open",
            )

        if self.state.current_position is not None and not gate:
            qty = int(self.state.current_position.get("quantity", 1) or 1)
            self.state.current_position = None
            return Signal(
                signal_type=SignalType.EXIT,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[{"symbol": instrument, "action": "SELL", "quantity": qty}],
                regime=regime,
                indicators={"adx": adx_value, "vix": vix_value},
                reason="Baseline trend exit gate closed",
            )

        return self._no_signal(ts, regime, instrument, reason="No baseline trend action")

    def get_exit_conditions(self, market_data: dict[str, Any]) -> Signal | None:
        return None

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return int(self.config.get("max_lots", 1) or 1)

    def _no_signal(
        self,
        ts: datetime,
        regime: RegimeState,
        instrument: str,
        *,
        reason: str,
    ) -> Signal:
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason=reason,
        )
