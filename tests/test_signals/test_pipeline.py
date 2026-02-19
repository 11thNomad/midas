from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.regime.classifier import RegimeThresholds
from src.signals.pipeline import build_feature_context


def test_build_feature_context_emits_snapshot_and_regime_signals():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="D"),
            "open": [100 + i for i in range(40)],
            "high": [101 + i for i in range(40)],
            "low": [99 + i for i in range(40)],
            "close": [100 + i for i in range(40)],
            "volume": [1000] * 40,
        }
    )
    vix_series = pd.Series([12.0 + (i * 0.1) for i in range(40)])
    fii = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "fii_net": [100.0] * 40,
        }
    )
    usdinr = pd.Series([82.0 + (i * 0.01) for i in range(40)])
    chain = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-02-10")] * 6,
            "underlying_price": [22000.0] * 6,
            "expiry": [pd.Timestamp("2026-02-26")] * 6,
            "symbol": ["CE1", "CE2", "CE3", "PE1", "PE2", "PE3"],
            "option_type": ["CE", "CE", "CE", "PE", "PE", "PE"],
            "strike": [22000, 22100, 22200, 22000, 21900, 21800],
            "iv": [18.0, 19.0, 19.5, 20.0, 21.0, 21.5],
            "oi": [1000, 900, 850, 1200, 800, 700],
        }
    )
    snapshot, regime = build_feature_context(
        timestamp=datetime(2026, 2, 10),
        symbol="NIFTY",
        timeframe="1d",
        candles=candles,
        vix_value=13.2,
        vix_series=vix_series,
        chain_df=chain,
        previous_chain_df=chain,
        fii_df=fii,
        usdinr_close=usdinr,
        thresholds=RegimeThresholds(),
    )

    assert snapshot.schema_version == "1.0.0"
    assert snapshot.symbol == "NIFTY"
    assert snapshot.timeframe == "1d"
    assert snapshot.vix_level == 13.2
    assert snapshot.fii_net_5d != 0.0
    assert snapshot.usdinr_roc_1d != 0.0
    assert snapshot.chain_quality_status == "ok"
    assert snapshot.option_spot > 0.0
    assert snapshot.pcr_oi_total > 0.0
    assert snapshot.pcr_oi_atm_band > 0.0
    assert regime.india_vix == 13.2
    assert regime.pcr > 0.0
