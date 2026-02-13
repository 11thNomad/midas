from __future__ import annotations

from datetime import datetime

from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType
from src.strategies.router import StrategyRouter


class DummyStrategy(BaseStrategy):
    def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=self.config.get("instrument", "NIFTY"),
            timestamp=market_data.get("timestamp", datetime(2026, 1, 1)),
            regime=regime,
        )

    def get_exit_conditions(self, market_data: dict) -> Signal | None:
        return None

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return 1


def test_router_deactivates_strategy_outside_regime_and_emits_exit():
    strategy = DummyStrategy(
        name="dummy",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_TRENDING.value]},
    )
    strategy.state.current_position = {"symbol": "NIFTY"}
    strategy.state.is_active = True

    router = StrategyRouter(strategies=[strategy], current_regime=RegimeState.LOW_VOL_TRENDING)
    transition_signals = router.on_regime_change(
        RegimeState.HIGH_VOL_CHOPPY, timestamp=datetime(2026, 1, 1)
    )

    assert strategy.state.is_active is False
    assert len(transition_signals) == 1
    assert transition_signals[0].signal_type == SignalType.EXIT


def test_router_activates_strategy_when_regime_matches():
    strategy = DummyStrategy(
        name="dummy",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_TRENDING.value]},
    )
    strategy.state.is_active = False

    router = StrategyRouter(strategies=[strategy], current_regime=RegimeState.HIGH_VOL_CHOPPY)
    transition_signals = router.on_regime_change(
        RegimeState.LOW_VOL_TRENDING, timestamp=datetime(2026, 1, 1)
    )

    assert transition_signals == []
    assert strategy.state.is_active is True
