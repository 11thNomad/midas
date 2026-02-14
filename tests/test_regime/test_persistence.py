from __future__ import annotations

from datetime import datetime

from src.regime.persistence import RegimeSnapshotStore, SignalSnapshotStore, StrategyTransitionStore
from src.signals.contracts import SignalSnapshotDTO


def test_strategy_transition_store_roundtrip(tmp_path):
    store = StrategyTransitionStore(base_dir=str(tmp_path / "cache"))
    rows = store.persist_transitions(
        [
            {
                "timestamp": datetime(2026, 1, 2, 9, 15).isoformat(),
                "strategy": "dummy",
                "from_active": True,
                "to_active": False,
                "regime": "high_vol_choppy",
            }
        ],
        symbol="NIFTY",
    )
    assert rows == 1

    loaded = store.read_transitions(symbol="NIFTY")
    assert len(loaded) == 1
    assert loaded.loc[0, "strategy"] == "dummy"


def test_regime_snapshot_store_roundtrip(tmp_path):
    store = RegimeSnapshotStore(base_dir=str(tmp_path / "cache"))
    rows = store.persist_snapshot(
        {
            "timestamp": datetime(2026, 1, 2, 9, 15).isoformat(),
            "regime": "low_vol_trending",
            "india_vix": 12.0,
            "adx_14": 30.0,
        },
        symbol="NIFTY",
    )
    assert rows == 1

    loaded = store.read_snapshots(symbol="NIFTY")
    assert len(loaded) == 1
    assert loaded.loc[0, "regime"] == "low_vol_trending"


def test_signal_snapshot_store_roundtrip(tmp_path):
    store = SignalSnapshotStore(base_dir=str(tmp_path / "cache"))
    rows = store.persist_snapshot(
        SignalSnapshotDTO(
            timestamp=datetime(2026, 1, 2, 9, 15),
            symbol="NIFTY",
            timeframe="15m",
            vix_level=12.0,
            adx_14=30.0,
            regime="low_vol_trending",
            regime_confidence=0.6,
            source="test",
        ),
        symbol="NIFTY",
        timeframe="15m",
        source="test",
    )
    assert rows == 1

    loaded = store.read_snapshots(symbol="NIFTY", timeframe="15m")
    assert len(loaded) == 1
    assert loaded.loc[0, "symbol"] == "NIFTY"
    assert loaded.loc[0, "timeframe"] == "15m"
    assert loaded.loc[0, "regime"] == "low_vol_trending"
