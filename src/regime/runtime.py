"""Runtime glue for regime classification, strategy routing, and persistence."""

from __future__ import annotations

from dataclasses import dataclass

from src.regime.classifier import RegimeClassifier, RegimeSignals
from src.regime.persistence import RegimeSnapshotStore, SignalSnapshotStore, StrategyTransitionStore
from src.signals.contracts import SignalSnapshotDTO, signal_snapshot_from_mapping
from src.strategies.base import RegimeState, Signal
from src.strategies.router import StrategyRouter


@dataclass
class RegimeRuntime:
    """Single-step coordinator for live/paper loops."""

    classifier: RegimeClassifier
    router: StrategyRouter
    snapshot_store: RegimeSnapshotStore | None = None
    signal_snapshot_store: SignalSnapshotStore | None = None
    transition_store: StrategyTransitionStore | None = None
    symbol: str = "NIFTY"
    timeframe: str = "1d"

    def process(
        self,
        signals: RegimeSignals,
        *,
        signal_snapshot: SignalSnapshotDTO | None = None,
    ) -> tuple[RegimeState, list[Signal]]:
        """Classify regime, route strategy activations, and persist snapshot."""
        regime = self.classifier.classify(signals)
        start_log_idx = len(self.router.transition_log)
        transition_signals = self.router.on_regime_change(regime, timestamp=signals.timestamp)
        new_transitions = self.router.transition_log[start_log_idx:]

        if self.snapshot_store is not None:
            snapshot = self.classifier.snapshot(signals, regime=regime)
            self.snapshot_store.persist_snapshot(snapshot, symbol=self.symbol)
        if self.signal_snapshot_store is not None:
            if signal_snapshot is None:
                signal_snapshot = signal_snapshot_from_mapping(
                    {
                        "timestamp": signals.timestamp,
                        "symbol": self.symbol,
                        "timeframe": self.timeframe,
                        "vix_level": signals.india_vix,
                        "vix_roc_5d": signals.vix_change_5d,
                        "adx_14": signals.adx_14,
                        "pcr_oi": signals.pcr,
                        "fii_net_3d": signals.fii_net_3d,
                        "regime": regime.value,
                        "regime_confidence": 0.0,
                        "source": "regime_runtime",
                    }
                )
            if signal_snapshot.regime != regime.value:
                signal_snapshot = signal_snapshot_from_mapping(
                    {**signal_snapshot.__dict__, "regime": regime.value}
                )
            self.signal_snapshot_store.persist_snapshot(
                signal_snapshot,
                symbol=self.symbol,
                timeframe=self.timeframe,
                source="regime_runtime",
            )
        if self.transition_store is not None and new_transitions:
            self.transition_store.persist_transitions(new_transitions, symbol=self.symbol)

        return regime, transition_signals
