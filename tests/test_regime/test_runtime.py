from __future__ import annotations

from datetime import datetime

from src.regime.classifier import RegimeClassifier, RegimeSignals, RegimeThresholds
from src.regime.persistence import RegimeSnapshotStore, SignalSnapshotStore, StrategyTransitionStore
from src.regime.runtime import RegimeRuntime
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


def test_runtime_process_routes_and_persists_snapshot(tmp_path):
    classifier = RegimeClassifier(thresholds=RegimeThresholds())
    strategy = DummyStrategy(
        name="dummy",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_TRENDING.value]},
    )
    strategy.state.is_active = False
    router = StrategyRouter(strategies=[strategy])
    store = RegimeSnapshotStore(base_dir=str(tmp_path / "cache"))
    signal_store = SignalSnapshotStore(base_dir=str(tmp_path / "cache"))
    transition_store = StrategyTransitionStore(base_dir=str(tmp_path / "cache"))
    runtime = RegimeRuntime(
        classifier=classifier,
        router=router,
        snapshot_store=store,
        signal_snapshot_store=signal_store,
        transition_store=transition_store,
        symbol="NIFTY",
        timeframe="15m",
    )

    signals = RegimeSignals(
        timestamp=datetime(2026, 1, 2, 9, 15),
        india_vix=12.0,
        adx_14=30.0,
    )
    regime, transition_signals = runtime.process(signals)

    assert regime == RegimeState.LOW_VOL_TRENDING
    assert transition_signals == []
    assert strategy.state.is_active is True

    persisted = store.read_snapshots(symbol="NIFTY")
    assert len(persisted) == 1
    assert persisted.loc[0, "regime"] == RegimeState.LOW_VOL_TRENDING.value
    signal_persisted = signal_store.read_snapshots(symbol="NIFTY", timeframe="15m")
    assert len(signal_persisted) == 1
    assert signal_persisted.loc[0, "symbol"] == "NIFTY"
    assert signal_persisted.loc[0, "timeframe"] == "15m"
    assert signal_persisted.loc[0, "regime"] == RegimeState.LOW_VOL_TRENDING.value

    transitions = transition_store.read_transitions(symbol="NIFTY")
    assert len(transitions) == 1
    assert transitions.loc[0, "strategy"] == "dummy"
