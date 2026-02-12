from __future__ import annotations

import pandas as pd

from src.signals import composite, mean_reversion, options_signals, regime_filters, trend, volatility


def test_ema_crossover_returns_binary_signal():
    close = pd.Series([100, 101, 102, 103, 104, 103, 102, 101, 100])
    sig = trend.ema_crossover(close, fast=2, slow=4)
    assert set(sig.dropna().unique()).issubset({-1, 1})


def test_bollinger_percent_b_range_is_reasonable():
    close = pd.Series([100 + i * 0.5 for i in range(60)])
    pb = mean_reversion.bollinger_percent_b(close, period=20)
    assert pb.dropna().between(-0.5, 1.5).all()


def test_iv_rank_within_0_100_for_stable_series():
    iv = pd.Series([10 + (i % 20) for i in range(300)], dtype="float64")
    rank = volatility.iv_rank(iv, lookback=252)
    valid = rank.dropna()
    assert valid.between(0, 100).all()


def test_put_call_ratio_and_max_pain():
    chain = pd.DataFrame(
        {
            "strike": [100, 100, 110, 110, 120, 120],
            "option_type": ["CE", "PE", "CE", "PE", "CE", "PE"],
            "oi": [100, 120, 90, 110, 80, 140],
            "change_in_oi": [10, 15, -5, 20, 8, -3],
        }
    )
    pcr = options_signals.put_call_ratio(chain)
    pain = options_signals.max_pain(chain)
    assert pcr > 1
    assert pain in {100, 110, 120}


def test_composite_boring_alpha_setup():
    adx = pd.Series([18, 22, 15])
    vix = pd.Series([13, 13, 19])
    pcr = pd.Series([1.0, 1.0, 1.0])
    setup = composite.boring_alpha_setup(adx, vix, pcr)
    assert list(setup) == [1, 0, 0]


def test_vix_band_labels():
    vix = pd.Series([12.0, 16.0, 20.0])
    labels = regime_filters.vix_regime_band(vix)
    assert list(labels) == ["low", "transition", "high"]
