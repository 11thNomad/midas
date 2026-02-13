from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.contracts import (
    DTOValidationError,
    candle_dtos_from_frame,
    fii_dtos_from_frame,
    frame_from_candle_dtos,
    frame_from_fii_dtos,
    option_dtos_from_chain,
)
from src.data.schemas import InstrumentType, OptionChain, OptionContract


def test_candle_dto_roundtrip_keeps_shape():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:15:00", "2026-01-01 09:20:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1200],
            "oi": [200, 210],
        }
    )
    dtos = candle_dtos_from_frame(frame, source="test", symbol="NIFTY", timeframe="5m")
    out = frame_from_candle_dtos(dtos)
    assert len(out) == 2
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume", "oi"]


def test_candle_dto_raises_for_missing_required_columns():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"]), "open": [100.0]})
    with pytest.raises(DTOValidationError):
        candle_dtos_from_frame(frame, source="test", symbol="NIFTY", timeframe="1d")


def test_option_chain_maps_to_normalized_option_dtos():
    chain = OptionChain(
        underlying="NIFTY",
        underlying_price=22000.0,
        timestamp=datetime(2026, 1, 1, 15, 30),
        expiry=datetime(2026, 1, 8),
        contracts=[
            OptionContract(
                symbol="NIFTY26JAN22000CE",
                instrument_type=InstrumentType.CALL,
                strike=22000.0,
                expiry=datetime(2026, 1, 8),
                ltp=180.0,
                oi=1000,
            ),
            OptionContract(
                symbol="NIFTY26JAN22000PE",
                instrument_type=InstrumentType.PUT,
                strike=22000.0,
                expiry=datetime(2026, 1, 8),
                ltp=150.0,
                oi=1200,
            ),
        ],
    )
    rows = option_dtos_from_chain(chain, source="test")
    assert len(rows) == 2
    assert {row.option_type for row in rows} == {"CE", "PE"}


def test_fii_dto_roundtrip_has_expected_columns():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "fii_buy": [100.0, 120.0],
            "fii_sell": [90.0, 100.0],
            "fii_net": [10.0, 20.0],
            "dii_buy": [80.0, 60.0],
            "dii_sell": [70.0, 65.0],
            "dii_net": [10.0, -5.0],
        }
    )
    dtos = fii_dtos_from_frame(frame, source="test")
    out = frame_from_fii_dtos(dtos)
    assert list(out.columns) == [
        "date",
        "fii_buy",
        "fii_sell",
        "fii_net",
        "dii_buy",
        "dii_sell",
        "dii_net",
    ]
    assert len(out) == 2
