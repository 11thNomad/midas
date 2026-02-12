"""Iron condor scaffold strategy for regime-aware backtest plumbing."""

from __future__ import annotations

from datetime import datetime

from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class IronCondorStrategy(BaseStrategy):
    """Simplified iron condor lifecycle for early backtest integration."""

    def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
        ts = market_data.get("timestamp", datetime.now())
        instrument = self.config.get("instrument", "NIFTY")
        lots = self.compute_position_size(capital=0, risk_per_trade=0)

        if self.state.current_position is None:
            self.state.current_position = {
                "structure": "iron_condor",
                "quantity": lots,
                "entry_time": ts,
            }
            return Signal(
                signal_type=SignalType.ENTRY_SHORT,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                # Placeholder legs; replace with live chain strike selection later.
                orders=[
                    {"symbol": f"{instrument}_CALL_SHORT", "action": "SELL", "quantity": lots},
                    {"symbol": f"{instrument}_PUT_SHORT", "action": "SELL", "quantity": lots},
                    {"symbol": f"{instrument}_CALL_HEDGE", "action": "BUY", "quantity": lots},
                    {"symbol": f"{instrument}_PUT_HEDGE", "action": "BUY", "quantity": lots},
                ],
                regime=regime,
                reason="Iron condor entry in allowed low-vol regime",
            )

        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason="Position already open",
        )

    def get_exit_conditions(self, market_data: dict) -> Signal | None:
        return None

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return int(self.config.get("max_lots", 1) or 1)

    def on_regime_change(self, old_regime: RegimeState, new_regime: RegimeState) -> Signal | None:
        if self.state.current_position is None:
            return None
        if self.should_be_active(new_regime):
            return None

        instrument = self.config.get("instrument", "NIFTY")
        qty = int(self.state.current_position.get("quantity", 1))
        self.state.current_position = None
        return Signal(
            signal_type=SignalType.EXIT,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=datetime.now(),
            orders=[
                {"symbol": f"{instrument}_CALL_SHORT", "action": "BUY", "quantity": qty},
                {"symbol": f"{instrument}_PUT_SHORT", "action": "BUY", "quantity": qty},
                {"symbol": f"{instrument}_CALL_HEDGE", "action": "SELL", "quantity": qty},
                {"symbol": f"{instrument}_PUT_HEDGE", "action": "SELL", "quantity": qty},
            ],
            regime=new_regime,
            reason=f"Regime changed from {old_regime.value} to {new_regime.value}",
        )
