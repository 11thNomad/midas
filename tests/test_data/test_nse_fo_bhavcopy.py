from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.nse_fo_bhavcopy import normalize_fo_bhavcopy_to_option_chain


def test_normalize_legacy_fo_bhavcopy_to_option_chain():
    raw = pd.DataFrame(
        {
            "INSTRUMENT": ["OPTIDX", "OPTIDX"],
            "SYMBOL": ["NIFTY", "BANKNIFTY"],
            "EXPIRY_DT": ["11-Jul-2024", "11-Jul-2024"],
            "STRIKE_PR": [22000.0, 50000.0],
            "OPTION_TYP": ["CE", "PE"],
            "CLOSE": [120.5, 250.0],
            "CONTRACTS": [100, 50],
            "OPEN_INT": [1000, 2000],
            "CHG_IN_OI": [120, -50],
            "TIMESTAMP": ["05-JUL-2024", "05-JUL-2024"],
        }
    )
    out = normalize_fo_bhavcopy_to_option_chain(raw, symbol="NIFTY", trade_date=date(2024, 7, 5))

    assert len(out) == 1
    row = out.iloc[0]
    assert row["underlying"] == "NIFTY"
    assert row["option_type"] == "CE"
    assert float(row["strike"]) == 22000.0
    assert float(row["ltp"]) == 120.5
    assert float(row["oi"]) == 1000.0
    assert float(row["change_in_oi"]) == 120.0
    assert pd.Timestamp(row["timestamp"]) == pd.Timestamp("2024-07-05 18:30:00")


def test_normalize_udiff_fo_bhavcopy_to_option_chain():
    raw = pd.DataFrame(
        {
            "FinInstrmTp": ["IDO", "IDO", "IDO"],
            "TckrSymb": ["NIFTY", "NIFTYNXT50", "NIFTY"],
            "XpryDt": ["2026-02-26", "2026-02-26", "2026-02-26"],
            "StrkPric": [23000.0, 73000.0, 22800.0],
            "OptnTp": ["CE", "PE", "PE"],
            "LastPric": [0.0, 100.0, 85.0],
            "ClsPric": [122.4, 100.0, 85.0],
            "TtlTradgVol": [2500, 5, 1400],
            "OpnIntrst": [50000, 100, 42000],
            "ChngInOpnIntrst": [1800, 0, -1200],
            "UndrlygPric": [22807.2, 73508.8, 22807.2],
            "FinInstrmNm": ["NIFTY26FEB23000CE", "NIFTYNXT5026FEB73000PE", "NIFTY26FEB22800PE"],
            "TradDt": ["2026-02-12", "2026-02-12", "2026-02-12"],
        }
    )
    out = normalize_fo_bhavcopy_to_option_chain(raw, symbol="NIFTY", trade_date=date(2026, 2, 12))

    assert len(out) == 2
    assert set(out["option_type"].tolist()) == {"CE", "PE"}
    assert set(out["symbol"].tolist()) == {"NIFTY26FEB23000CE", "NIFTY26FEB22800PE"}
    ce = out.loc[out["option_type"] == "CE"].iloc[0]
    # Last price is 0.0 for CE row, so normalizer should fall back to close.
    assert float(ce["ltp"]) == 122.4
    assert float(ce["underlying_price"]) == 22807.2
    assert pd.Timestamp(ce["timestamp"]) == pd.Timestamp("2026-02-12 18:30:00")

