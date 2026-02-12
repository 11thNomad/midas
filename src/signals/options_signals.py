"""Options-specific signals from chain snapshots."""

from __future__ import annotations

import pandas as pd


def _normalize_iv_chain(chain_df: pd.DataFrame) -> pd.DataFrame:
    required = {"strike", "option_type", "iv"}
    if chain_df.empty or not required.issubset(chain_df.columns):
        return pd.DataFrame(columns=["strike", "option_type", "iv"])
    out = chain_df[["strike", "option_type", "iv"]].copy()
    out["option_type"] = out["option_type"].astype(str).str.upper()
    out = out[out["option_type"].isin(["CE", "PE"])]
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["iv"] = pd.to_numeric(out["iv"], errors="coerce")
    out = out.dropna(subset=["strike", "iv"])
    return out


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


def iv_surface_change(previous_chain_df: pd.DataFrame, current_chain_df: pd.DataFrame) -> pd.DataFrame:
    """Return strike/type aligned IV change table between two snapshots."""
    prev = _normalize_iv_chain(previous_chain_df).rename(columns={"iv": "iv_prev"})
    curr = _normalize_iv_chain(current_chain_df).rename(columns={"iv": "iv_curr"})
    if prev.empty or curr.empty:
        return pd.DataFrame(columns=["strike", "option_type", "iv_prev", "iv_curr", "iv_change"])

    merged = prev.merge(curr, on=["strike", "option_type"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["strike", "option_type", "iv_prev", "iv_curr", "iv_change"])
    merged["iv_change"] = merged["iv_curr"] - merged["iv_prev"]
    return merged.sort_values(["strike", "option_type"]).reset_index(drop=True)


def iv_surface_parallel_shift(previous_chain_df: pd.DataFrame, current_chain_df: pd.DataFrame) -> float:
    """Average IV move across overlapping strikes/contracts."""
    delta = iv_surface_change(previous_chain_df, current_chain_df)
    if delta.empty:
        return 0.0
    return float(delta["iv_change"].mean())


def iv_surface_tilt(chain_df: pd.DataFrame, underlying_price: float | None = None) -> float:
    """Simple smile tilt proxy: OTM put IV minus OTM call IV."""
    normalized = _normalize_iv_chain(chain_df)
    if normalized.empty:
        return 0.0

    if underlying_price is None:
        underlying_price = float(normalized["strike"].median())

    puts = normalized[(normalized["option_type"] == "PE") & (normalized["strike"] < underlying_price)]["iv"]
    calls = normalized[(normalized["option_type"] == "CE") & (normalized["strike"] > underlying_price)]["iv"]
    if puts.empty or calls.empty:
        return 0.0
    return float(puts.mean() - calls.mean())


def iv_surface_tilt_change(
    previous_chain_df: pd.DataFrame,
    current_chain_df: pd.DataFrame,
    *,
    previous_underlying: float | None = None,
    current_underlying: float | None = None,
) -> float:
    prev_tilt = iv_surface_tilt(previous_chain_df, underlying_price=previous_underlying)
    curr_tilt = iv_surface_tilt(current_chain_df, underlying_price=current_underlying)
    return float(curr_tilt - prev_tilt)
