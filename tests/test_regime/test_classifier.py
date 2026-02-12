from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.regime.classifier import RegimeClassifier, RegimeSignals, RegimeThresholds
from src.signals.regime import build_regime_signals
from src.strategies.base import RegimeState


def test_regime_classifier_base_low_vol_trending():
    classifier = RegimeClassifier(thresholds=RegimeThresholds())
    signals = RegimeSignals(timestamp=datetime(2026, 1, 1), india_vix=12.0, adx_14=30.0)
    regime = classifier.classify(signals)
    assert regime == RegimeState.LOW_VOL_TRENDING


def test_iv_surface_stress_overrides_to_high_vol_choppy():
    classifier = RegimeClassifier(thresholds=RegimeThresholds(iv_surface_shift_alert=1.0))
    signals = RegimeSignals(
        timestamp=datetime(2026, 1, 1),
        india_vix=13.0,
        adx_14=28.0,
        iv_surface_parallel_shift=2.5,
    )
    regime = classifier.classify(signals)
    assert regime == RegimeState.HIGH_VOL_CHOPPY


def test_vix_spike_override_to_high_vol_choppy():
    classifier = RegimeClassifier(thresholds=RegimeThresholds(vix_spike_5d_alert=2.0))
    signals = RegimeSignals(
        timestamp=datetime(2026, 1, 1),
        india_vix=13.5,
        adx_14=28.0,
        vix_change_5d=3.5,
    )
    regime = classifier.classify(signals)
    assert regime == RegimeState.HIGH_VOL_CHOPPY


def test_build_regime_signals_populates_iv_and_vix_change_fields():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=80, freq="D"),
            "open": [100 + i for i in range(80)],
            "high": [101 + i for i in range(80)],
            "low": [99 + i for i in range(80)],
            "close": [100 + i for i in range(80)],
        }
    )
    vix_series = pd.Series([12, 12.2, 12.4, 12.6, 12.8, 13.5, 13.8], dtype="float64")

    previous_chain = pd.DataFrame(
        {
            "strike": [21900, 22000, 22100, 21900, 22000, 22100],
            "option_type": ["PE", "PE", "PE", "CE", "CE", "CE"],
            "iv": [18, 17, 16, 15, 14, 13],
            "oi": [100, 110, 120, 100, 110, 120],
        }
    )
    current_chain = previous_chain.copy()
    current_chain["iv"] = current_chain["iv"] + 2

    signals = build_regime_signals(
        timestamp=datetime(2026, 1, 1),
        candles=candles,
        vix_value=13.8,
        vix_series=vix_series,
        chain_df=current_chain,
        previous_chain_df=previous_chain,
    )

    assert signals.vix_change_5d > 0
    assert signals.iv_surface_parallel_shift == 2.0
