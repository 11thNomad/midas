"""Minimal strategy used to exercise Phase 4 backtest plumbing."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class RegimeProbeStrategy(BaseStrategy):
    """Enter long in trending regimes, exit otherwise."""

    def generate_signal(self, market_data: dict[str, Any], regime: RegimeState) -> Signal:
        raw_ts = market_data.get("timestamp")
        ts = raw_ts if isinstance(raw_ts, datetime) else datetime.now()
        instrument = self.config.get("instrument", "NIFTY")

        if self.state.current_position is None and regime in (
            RegimeState.LOW_VOL_TRENDING,
            RegimeState.HIGH_VOL_TRENDING,
        ):
            qty = self.compute_position_size(capital=0, risk_per_trade=0)
            self.state.current_position = {"symbol": instrument, "quantity": qty}
            return Signal(
                signal_type=SignalType.ENTRY_LONG,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[{"symbol": instrument, "action": "BUY", "quantity": qty}],
                regime=regime,
                reason="Probe entry in trending regime",
            )

        if self.state.current_position is not None and regime not in (
            RegimeState.LOW_VOL_TRENDING,
            RegimeState.HIGH_VOL_TRENDING,
        ):
            qty = int(self.state.current_position.get("quantity", 1))
            self.state.current_position = None
            return Signal(
                signal_type=SignalType.EXIT,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=[{"symbol": instrument, "action": "SELL", "quantity": qty}],
                regime=regime,
                reason="Probe exit outside trending regime",
            )

        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason="No probe action",
        )

    def get_exit_conditions(self, market_data: dict[str, Any]) -> Signal | None:
        return None

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return int(self.config.get("lots", 1) or 1)
