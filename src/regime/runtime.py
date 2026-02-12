"""Runtime glue for regime classification, strategy routing, and persistence."""

from __future__ import annotations

from dataclasses import dataclass

from src.regime.classifier import RegimeClassifier, RegimeSignals
from src.regime.persistence import RegimeSnapshotStore, StrategyTransitionStore
from src.strategies.base import RegimeState, Signal
from src.strategies.router import StrategyRouter


@dataclass
class RegimeRuntime:
    """Single-step coordinator for live/paper loops."""

    classifier: RegimeClassifier
    router: StrategyRouter
    snapshot_store: RegimeSnapshotStore | None = None
    transition_store: StrategyTransitionStore | None = None
    symbol: str = "NIFTY"

    def process(self, signals: RegimeSignals) -> tuple[RegimeState, list[Signal]]:
        """Classify regime, route strategy activations, and persist snapshot."""
        regime = self.classifier.classify(signals)
        start_log_idx = len(self.router.transition_log)
        transition_signals = self.router.on_regime_change(regime, timestamp=signals.timestamp)
        new_transitions = self.router.transition_log[start_log_idx:]

        if self.snapshot_store is not None:
            snapshot = self.classifier.snapshot(signals, regime=regime)
            self.snapshot_store.persist_snapshot(snapshot, symbol=self.symbol)
        if self.transition_store is not None and new_transitions:
            self.transition_store.persist_transitions(new_transitions, symbol=self.symbol)

        return regime, transition_signals
