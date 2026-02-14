from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.signals.contracts import (
    SignalContractError,
    frame_from_signal_snapshots,
    signal_snapshot_from_mapping,
    signal_snapshots_from_frame,
)


def test_signal_snapshot_contract_roundtrip():
    payload = {
        "timestamp": datetime(2026, 1, 1, 9, 15),
        "symbol": "NIFTY",
        "timeframe": "15m",
        "vix_level": 13.5,
        "adx_14": 27.1,
        "regime": "low_vol_trending",
        "regime_confidence": 0.82,
    }
    dto = signal_snapshot_from_mapping(payload)
    out = frame_from_signal_snapshots([dto])
    assert len(out) == 1
    assert out.loc[0, "symbol"] == "NIFTY"
    assert out.loc[0, "timeframe"] == "15m"
    assert out.loc[0, "regime"] == "low_vol_trending"


def test_signal_snapshot_contract_requires_core_keys():
    with pytest.raises(SignalContractError):
        signal_snapshot_from_mapping({"timestamp": datetime(2026, 1, 1, 9, 15), "symbol": "NIFTY"})


def test_signal_snapshot_contract_rejects_invalid_confidence():
    with pytest.raises(SignalContractError):
        signal_snapshot_from_mapping(
            {
                "timestamp": datetime(2026, 1, 1, 9, 15),
                "symbol": "NIFTY",
                "timeframe": "15m",
                "regime_confidence": 1.5,
            }
        )


def test_signal_snapshot_contract_from_frame():
    frame = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 1, 1, 9, 15),
                "symbol": "NIFTY",
                "timeframe": "15m",
                "regime_confidence": 0.5,
            },
            {
                "timestamp": datetime(2026, 1, 1, 9, 30),
                "symbol": "BANKNIFTY",
                "timeframe": "15m",
                "regime_confidence": 0.7,
            },
        ]
    )
    dtos = signal_snapshots_from_frame(frame)
    assert len(dtos) == 2
    assert dtos[0].symbol == "NIFTY"
    assert dtos[1].symbol == "BANKNIFTY"
