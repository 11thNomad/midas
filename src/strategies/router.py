"""Strategy routing based on current regime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.strategies.base import BaseStrategy, RegimeState, Signal


@dataclass
class StrategyRouter:
    """Controls strategy activation/deactivation as regimes change."""

    strategies: list[BaseStrategy]
    current_regime: RegimeState = RegimeState.UNKNOWN
    transition_log: list[dict] = field(default_factory=list)

    def on_regime_change(
        self,
        new_regime: RegimeState,
        *,
        timestamp: datetime | None = None,
    ) -> list[Signal]:
        """Update strategy active flags and return any generated transition signals."""
        ts = timestamp or datetime.now()
        out: list[Signal] = []

        old_regime = self.current_regime
        self.current_regime = new_regime

        for strategy in self.strategies:
            should_be_active = strategy.should_be_active(new_regime)
            was_active = strategy.state.is_active

            if should_be_active and not was_active:
                strategy.state.is_active = True

            if not should_be_active and was_active:
                exit_signal = strategy.on_regime_change(old_regime=old_regime, new_regime=new_regime)
                strategy.state.is_active = False
                if exit_signal is not None:
                    out.append(exit_signal)

            self.transition_log.append(
                {
                    "timestamp": ts.isoformat(),
                    "strategy": strategy.name,
                    "from_active": was_active,
                    "to_active": strategy.state.is_active,
                    "regime": new_regime.value,
                }
            )

        return out

    def generate_signals(self, market_data: dict, regime: RegimeState) -> list[Signal]:
        """Call active strategies and return actionable signals only."""
        out: list[Signal] = []
        for strategy in self.strategies:
            if not strategy.state.is_active:
                continue
            signal = strategy.generate_signal(market_data=market_data, regime=regime)
            if signal.is_actionable:
                out.append(signal)
        return out
