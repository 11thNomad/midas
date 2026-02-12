"""Runtime glue for regime classification, strategy routing, and persistence."""

from __future__ import annotations

from dataclasses import dataclass

from src.regime.classifier import RegimeClassifier, RegimeSignals
from src.regime.persistence import RegimeSnapshotStore
from src.strategies.base import RegimeState, Signal
from src.strategies.router import StrategyRouter


@dataclass
class RegimeRuntime:
    """Single-step coordinator for live/paper loops."""

    classifier: RegimeClassifier
    router: StrategyRouter
    snapshot_store: RegimeSnapshotStore | None = None
    symbol: str = "NIFTY"

    def process(self, signals: RegimeSignals) -> tuple[RegimeState, list[Signal]]:
        """Classify regime, route strategy activations, and persist snapshot."""
        regime = self.classifier.classify(signals)
        transition_signals = self.router.on_regime_change(regime, timestamp=signals.timestamp)

        if self.snapshot_store is not None:
            snapshot = self.classifier.snapshot(signals, regime=regime)
            self.snapshot_store.persist_snapshot(snapshot, symbol=self.symbol)

        return regime, transition_signals
