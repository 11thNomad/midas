"""Options-specific signals from chain snapshots."""

from __future__ import annotations

import pandas as pd


def put_call_ratio(chain_df: pd.DataFrame) -> float:
    """Chain must have columns: option_type (CE/PE), oi."""
    if chain_df.empty:
        return 0.0
    calls = chain_df.loc[chain_df["option_type"].str.upper() == "CE", "oi"].sum()
    puts = chain_df.loc[chain_df["option_type"].str.upper() == "PE", "oi"].sum()
    if calls == 0:
        return 0.0
    return float(puts / calls)


def max_pain(chain_df: pd.DataFrame) -> float | None:
    """Approximate max pain using OI-weighted payoff across strikes."""
    required = {"strike", "option_type", "oi"}
    if chain_df.empty or not required.issubset(chain_df.columns):
        return None

    strikes = sorted(chain_df["strike"].dropna().unique())
    if not strikes:
        return None

    def payout_at_expiry(settlement: float) -> float:
        calls = chain_df[chain_df["option_type"].str.upper() == "CE"]
        puts = chain_df[chain_df["option_type"].str.upper() == "PE"]
        call_pain = ((settlement - calls["strike"]).clip(lower=0) * calls["oi"]).sum()
        put_pain = ((puts["strike"] - settlement).clip(lower=0) * puts["oi"]).sum()
        return float(call_pain + put_pain)

    pains = {strike: payout_at_expiry(strike) for strike in strikes}
    return min(pains, key=pains.get)


def oi_change_by_strike(chain_df: pd.DataFrame) -> pd.Series:
    """Return strike-wise OI change; requires strike + change_in_oi columns."""
    if chain_df.empty:
        return pd.Series(dtype="float64")
    if not {"strike", "change_in_oi"}.issubset(chain_df.columns):
        return pd.Series(dtype="float64")
    return chain_df.groupby("strike")["change_in_oi"].sum().sort_index()


def total_call_put_oi(chain_df: pd.DataFrame) -> pd.Series:
    if chain_df.empty:
        return pd.Series({"call_oi": 0.0, "put_oi": 0.0})
    call_oi = chain_df.loc[chain_df["option_type"].str.upper() == "CE", "oi"].sum()
    put_oi = chain_df.loc[chain_df["option_type"].str.upper() == "PE", "oi"].sum()
    return pd.Series({"call_oi": float(call_oi), "put_oi": float(put_oi)})
