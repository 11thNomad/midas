from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.signals.option_chain_features import (
    OPTION_FEATURE_ARTIFACT_COLUMNS,
    build_option_chain_feature_row,
    option_feature_artifact_from_snapshots,
)


def _chain(iv_shift: float = 0.0, put_extra: float = 0.0) -> pd.DataFrame:
    near = pd.Timestamp("2026-02-26")
    far = pd.Timestamp("2026-03-26")
    strikes = [21800, 21900, 22000, 22100, 22200]
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for expiry in [near, far]:
        for strike in strikes:
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-02-10 09:30:00"),
                    "expiry": expiry,
                    "underlying_price": 22020.0,
                    "strike": strike,
                    "option_type": "PE",
                    "oi": 1000 + (22000 - strike),
                    "iv": 18.0 + iv_shift + (put_extra if strike < 22000 else 0.0),
                }
            )
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-02-10 09:30:00"),
                    "expiry": expiry,
                    "underlying_price": 22020.0,
                    "strike": strike,
                    "option_type": "CE",
                    "oi": 900 + (strike - 22000),
                    "iv": 16.0 + iv_shift,
                }
            )
    return pd.DataFrame(rows)


def test_option_chain_features_extracts_band_term_and_wall_metrics():
    row = build_option_chain_feature_row(
        chain_df=_chain(),
        asof=datetime(2026, 2, 10),
        fallback_spot=22000.0,
    )
    assert row.rows > 0
    assert row.atm_strike == 22000.0
    assert row.strike_step == 100.0
    assert row.pcr_oi_total > 1.0
    assert row.pcr_oi_atm_band > 1.0
    assert row.oi_support < row.atm_strike
    assert row.oi_resistance > row.atm_strike
    assert row.near_term_pcr_oi > 1.0
    assert row.next_term_pcr_oi > 1.0
    assert row.atm_iv_near > 0.0
    assert row.atm_iv_next > 0.0


def test_option_chain_features_detects_parallel_shift_and_tilt_change():
    prev = _chain(iv_shift=0.0, put_extra=0.0)
    curr = _chain(iv_shift=2.0, put_extra=1.0)
    row = build_option_chain_feature_row(
        chain_df=curr,
        previous_chain_df=prev,
        asof=datetime(2026, 2, 10),
        fallback_spot=22000.0,
    )
    assert row.iv_surface_parallel_shift > 2.0
    assert row.iv_surface_tilt_change > 0.0
    assert row.iv_skew_otm > 0.0


def test_option_chain_features_empty_chain_returns_zero_row():
    row = build_option_chain_feature_row(chain_df=pd.DataFrame())
    assert row.rows == 0
    assert row.pcr_oi_total == 0.0
    assert row.oi_support == 0.0
    assert row.iv_surface_parallel_shift == 0.0


def test_option_feature_artifact_extracts_stable_columns():
    snapshots = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 2, 10, 9, 15),
                "symbol": "NIFTY",
                "timeframe": "1d",
                "regime": "low_vol_ranging",
                "pcr_oi": 1.02,
                "pcr_oi_total": 1.1,
                "option_spot": 22010.0,
                "option_atm_strike": 22000.0,
                "iv_surface_parallel_shift": 1.3,
            }
        ]
    )
    artifact = option_feature_artifact_from_snapshots(snapshots)
    assert len(artifact) == 1
    assert set(artifact.columns).issubset(set(OPTION_FEATURE_ARTIFACT_COLUMNS))
    assert float(artifact.loc[0, "option_spot"]) == 22010.0
