"""Greeks utilities computed locally for contract parity across modes."""

from __future__ import annotations

from typing import Literal


def mibian_implied_iv(
    *,
    spot: float,
    strike: float,
    rate_pct: float,
    days_to_expiry: int,
    option_price: float,
    option_type: Literal["CE", "PE"],
) -> float | None:
    """Solve implied volatility (%) from option price using mibian."""
    if spot <= 0 or strike <= 0 or option_price <= 0 or days_to_expiry <= 0:
        return None

    try:
        import mibian
    except ImportError:
        return None

    try:
        if option_type.upper() == "PE":
            bs = mibian.BS(
                [float(spot), float(strike), float(rate_pct), int(days_to_expiry)],
                putPrice=float(option_price),
            )
        else:
            bs = mibian.BS(
                [float(spot), float(strike), float(rate_pct), int(days_to_expiry)],
                callPrice=float(option_price),
            )
    except Exception:
        return None

    iv = float(getattr(bs, "impliedVolatility", 0.0) or 0.0)
    if not (0.01 <= iv <= 300.0):
        return None
    return iv


def mibian_greeks(
    *,
    spot: float,
    strike: float,
    rate_pct: float,
    days_to_expiry: int,
    iv_pct: float,
    option_type: Literal["CE", "PE"],
) -> dict[str, float]:
    """Return Greeks from mibian Black-Scholes implementation.

    All percent-style inputs are expected as whole numbers (e.g., 8.0, 18.5).
    """
    if spot <= 0 or strike <= 0 or iv_pct <= 0 or days_to_expiry <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    try:
        import mibian
    except ImportError:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    bs = mibian.BS(
        [float(spot), float(strike), float(rate_pct), int(days_to_expiry)],
        volatility=float(iv_pct),
    )
    opt = option_type.upper()
    if opt == "PE":
        delta = float(getattr(bs, "putDelta", 0.0) or 0.0)
        theta = float(getattr(bs, "putTheta", 0.0) or 0.0)
        rho = float(getattr(bs, "putRho", 0.0) or 0.0)
    else:
        delta = float(getattr(bs, "callDelta", 0.0) or 0.0)
        theta = float(getattr(bs, "callTheta", 0.0) or 0.0)
        rho = float(getattr(bs, "callRho", 0.0) or 0.0)

    return {
        "delta": delta,
        "gamma": float(getattr(bs, "gamma", 0.0) or 0.0),
        "theta": theta,
        "vega": float(getattr(bs, "vega", 0.0) or 0.0),
        "rho": rho,
    }
