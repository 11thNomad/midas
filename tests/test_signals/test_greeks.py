from __future__ import annotations

from src.signals.greeks import mibian_greeks


def test_mibian_greeks_returns_non_zero_for_valid_inputs():
    out = mibian_greeks(
        spot=22000.0,
        strike=22000.0,
        rate_pct=8.0,
        days_to_expiry=14,
        iv_pct=18.0,
        option_type="CE",
    )
    assert out["gamma"] >= 0.0
    assert out["vega"] >= 0.0
    assert out["delta"] > 0.0


def test_mibian_greeks_returns_zero_for_invalid_inputs():
    out = mibian_greeks(
        spot=0.0,
        strike=22000.0,
        rate_pct=8.0,
        days_to_expiry=14,
        iv_pct=18.0,
        option_type="PE",
    )
    assert out == {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
