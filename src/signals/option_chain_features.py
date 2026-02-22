"""Canonical option-chain feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.signals import options_signals

OPTION_FEATURE_ARTIFACT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "timeframe",
    "regime",
    "regime_confidence",
    "chain_rows",
    "chain_quality_status",
    "chain_quality_issue_count",
    "option_spot",
    "option_atm_strike",
    "option_strike_step",
    "pcr_oi",
    "pcr_oi_total",
    "pcr_oi_atm_band",
    "pcr_oi_otm_band",
    "near_term_pcr_oi",
    "next_term_pcr_oi",
    "oi_support",
    "oi_resistance",
    "atm_iv_near",
    "atm_iv_next",
    "iv_term_structure",
    "iv_skew_otm",
    "iv_surface_parallel_shift",
    "iv_surface_tilt_change",
    "atm_call_delta",
    "atm_put_delta",
    "atm_gamma",
    "atm_theta",
    "atm_vega",
    "atm_rho",
)


@dataclass(frozen=True)
class OptionChainFeatureRow:
    rows: int = 0
    spot: float = 0.0
    atm_strike: float = 0.0
    strike_step: float = 0.0
    pcr_oi_total: float = 0.0
    pcr_oi_atm_band: float = 0.0
    pcr_oi_otm_band: float = 0.0
    oi_support: float = 0.0
    oi_resistance: float = 0.0
    near_term_pcr_oi: float = 0.0
    next_term_pcr_oi: float = 0.0
    atm_iv_near: float = 0.0
    atm_iv_next: float = 0.0
    iv_term_structure: float = 0.0
    iv_skew_otm: float = 0.0
    iv_surface_parallel_shift: float = 0.0
    iv_surface_tilt_change: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "spot": self.spot,
            "atm_strike": self.atm_strike,
            "strike_step": self.strike_step,
            "pcr_oi_total": self.pcr_oi_total,
            "pcr_oi_atm_band": self.pcr_oi_atm_band,
            "pcr_oi_otm_band": self.pcr_oi_otm_band,
            "oi_support": self.oi_support,
            "oi_resistance": self.oi_resistance,
            "near_term_pcr_oi": self.near_term_pcr_oi,
            "next_term_pcr_oi": self.next_term_pcr_oi,
            "atm_iv_near": self.atm_iv_near,
            "atm_iv_next": self.atm_iv_next,
            "iv_term_structure": self.iv_term_structure,
            "iv_skew_otm": self.iv_skew_otm,
            "iv_surface_parallel_shift": self.iv_surface_parallel_shift,
            "iv_surface_tilt_change": self.iv_surface_tilt_change,
        }


def build_option_chain_feature_row(
    *,
    chain_df: pd.DataFrame | None,
    previous_chain_df: pd.DataFrame | None = None,
    asof: datetime | None = None,
    fallback_spot: float = 0.0,
    band_steps: int = 2,
) -> OptionChainFeatureRow:
    normalized = _normalize_chain(chain_df)
    if normalized.empty:
        return OptionChainFeatureRow()

    spot = _resolve_spot(normalized, fallback=fallback_spot)
    strikes = sorted(float(v) for v in normalized["strike"].dropna().unique())
    if not strikes:
        return OptionChainFeatureRow(rows=int(len(normalized)), spot=spot)

    atm_strike = min(strikes, key=lambda strike: abs(strike - spot))
    strike_step = _infer_strike_step(strikes)
    band_width = strike_step * max(int(band_steps), 1)

    total_pcr = _pcr_oi(normalized)
    atm_band = normalized.loc[(normalized["strike"] - atm_strike).abs() <= band_width]
    atm_band_pcr = _pcr_oi(atm_band)
    otm_band = normalized.loc[
        ((normalized["option_type"] == "PE") & (normalized["strike"] < atm_strike))
        | ((normalized["option_type"] == "CE") & (normalized["strike"] > atm_strike))
    ]
    otm_band = otm_band.loc[(otm_band["strike"] - atm_strike).abs() <= band_width]
    otm_band_pcr = _pcr_oi(otm_band)

    expiries = _sorted_expiries(normalized, asof=asof)
    near_term = (
        normalized.loc[normalized["expiry"] == expiries[0]].copy()
        if expiries
        else normalized.copy()
    )
    next_term = (
        normalized.loc[normalized["expiry"] == expiries[1]].copy() if len(expiries) > 1 else None
    )

    near_term_pcr = _pcr_oi(near_term)
    next_term_pcr = _pcr_oi(next_term) if next_term is not None else 0.0
    oi_support, oi_resistance = options_signals.oi_support_resistance(near_term)

    atm_iv_near = _atm_iv_for_expiry(near_term, atm_strike=atm_strike)
    atm_iv_next = (
        _atm_iv_for_expiry(next_term, atm_strike=atm_strike)
        if next_term is not None
        else 0.0
    )
    iv_term_structure = float(atm_iv_next - atm_iv_near) if atm_iv_next > 0.0 else 0.0
    iv_skew_otm = _otm_iv_skew(near_term, atm_strike=atm_strike, band_width=band_width)

    iv_shift = 0.0
    iv_tilt_change = 0.0
    previous_normalized = _normalize_chain(previous_chain_df)
    if not previous_normalized.empty:
        iv_shift = options_signals.iv_surface_parallel_shift(previous_normalized, normalized)
        iv_tilt_change = options_signals.iv_surface_tilt_change(
            previous_normalized,
            normalized,
            previous_underlying=_resolve_spot(previous_normalized, fallback=spot),
            current_underlying=spot,
        )

    return OptionChainFeatureRow(
        rows=int(len(normalized)),
        spot=float(spot),
        atm_strike=float(atm_strike),
        strike_step=float(strike_step),
        pcr_oi_total=float(total_pcr),
        pcr_oi_atm_band=float(atm_band_pcr),
        pcr_oi_otm_band=float(otm_band_pcr),
        oi_support=float(oi_support),
        oi_resistance=float(oi_resistance),
        near_term_pcr_oi=float(near_term_pcr),
        next_term_pcr_oi=float(next_term_pcr),
        atm_iv_near=float(atm_iv_near),
        atm_iv_next=float(atm_iv_next),
        iv_term_structure=float(iv_term_structure),
        iv_skew_otm=float(iv_skew_otm),
        iv_surface_parallel_shift=float(iv_shift),
        iv_surface_tilt_change=float(iv_tilt_change),
    )


def frame_from_option_chain_feature_rows(rows: list[OptionChainFeatureRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(OptionChainFeatureRow().as_dict().keys()))
    return pd.DataFrame([row.as_dict() for row in rows])


def option_feature_artifact_from_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Extract a stable option-feature artifact table from signal snapshots."""
    if snapshots.empty:
        return pd.DataFrame(columns=list(OPTION_FEATURE_ARTIFACT_COLUMNS))
    available = [col for col in OPTION_FEATURE_ARTIFACT_COLUMNS if col in snapshots.columns]
    if not available:
        return pd.DataFrame(columns=list(OPTION_FEATURE_ARTIFACT_COLUMNS))
    out = snapshots[available].copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def _series_from_frame(
    frame: pd.DataFrame, column: str, *, default: Any
) -> pd.Series[Any]:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _normalize_chain(chain_df: pd.DataFrame | None) -> pd.DataFrame:
    if chain_df is None or chain_df.empty:
        return pd.DataFrame()

    required = {"option_type", "strike"}
    if not required.issubset(chain_df.columns):
        return pd.DataFrame()

    out = chain_df.copy()
    out["option_type"] = out["option_type"].astype(str).str.upper()
    out = out[out["option_type"].isin(["CE", "PE"])]
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["oi"] = pd.to_numeric(_series_from_frame(out, "oi", default=0.0), errors="coerce")
    out["iv"] = pd.to_numeric(_series_from_frame(out, "iv", default=0.0), errors="coerce")
    out["underlying_price"] = pd.to_numeric(
        _series_from_frame(out, "underlying_price", default=0.0),
        errors="coerce",
    )
    out["expiry"] = pd.to_datetime(
        _series_from_frame(out, "expiry", default=pd.NaT),
        errors="coerce",
    )
    out = out.dropna(subset=["strike"]).copy()
    if out.empty:
        return out

    dedup_cols = [c for c in ["expiry", "strike", "option_type"] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="last")
    return out.reset_index(drop=True)


def _resolve_spot(chain_df: pd.DataFrame, *, fallback: float) -> float:
    price = pd.to_numeric(
        _series_from_frame(chain_df, "underlying_price", default=0.0),
        errors="coerce",
    ).dropna()
    if not price.empty and float(price.iloc[-1]) > 0.0:
        return float(price.iloc[-1])
    if fallback > 0.0:
        return float(fallback)
    strike = pd.to_numeric(
        _series_from_frame(chain_df, "strike", default=0.0),
        errors="coerce",
    ).dropna()
    if strike.empty:
        return 0.0
    return float(strike.median())


def _infer_strike_step(strikes: list[float]) -> float:
    if len(strikes) < 2:
        return 50.0
    diffs = pd.Series(strikes).sort_values().diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 50.0
    return float(diffs.median())


def _pcr_oi(chain_df: pd.DataFrame | None) -> float:
    if chain_df is None or chain_df.empty:
        return 0.0
    calls = pd.to_numeric(
        chain_df.loc[chain_df["option_type"] == "CE", "oi"], errors="coerce"
    ).sum()
    puts = pd.to_numeric(chain_df.loc[chain_df["option_type"] == "PE", "oi"], errors="coerce").sum()
    if calls <= 0:
        return 0.0
    return float(puts / calls)


def _sorted_expiries(chain_df: pd.DataFrame, *, asof: datetime | None) -> list[pd.Timestamp]:
    expiry = pd.to_datetime(_series_from_frame(chain_df, "expiry", default=pd.NaT), errors="coerce")
    expiry = expiry.dropna().drop_duplicates().sort_values()
    if expiry.empty:
        return []
    if asof is None:
        return [pd.Timestamp(v) for v in expiry]
    anchor = pd.Timestamp(asof).normalize()
    filtered = expiry.loc[expiry.dt.normalize() >= anchor]
    chosen = filtered if not filtered.empty else expiry
    return [pd.Timestamp(v) for v in chosen]


def _atm_iv_for_expiry(chain_df: pd.DataFrame | None, *, atm_strike: float) -> float:
    if chain_df is None or chain_df.empty or "iv" not in chain_df.columns:
        return 0.0

    out = chain_df.copy()
    out["strike"] = pd.to_numeric(
        _series_from_frame(out, "strike", default=float("nan")),
        errors="coerce",
    )
    out["iv"] = pd.to_numeric(
        _series_from_frame(out, "iv", default=float("nan")),
        errors="coerce",
    )
    out = out.dropna(subset=["strike", "iv"])
    if out.empty:
        return 0.0
    out["_dist"] = (out["strike"] - float(atm_strike)).abs()
    nearest = out.loc[out["_dist"] == out["_dist"].min()]
    if nearest.empty:
        return 0.0
    return float(nearest["iv"].mean())


def _otm_iv_skew(chain_df: pd.DataFrame, *, atm_strike: float, band_width: float) -> float:
    puts = chain_df.loc[
        (chain_df["option_type"] == "PE")
        & (chain_df["strike"] < atm_strike)
        & ((chain_df["strike"] - atm_strike).abs() <= band_width)
    ]
    calls = chain_df.loc[
        (chain_df["option_type"] == "CE")
        & (chain_df["strike"] > atm_strike)
        & ((chain_df["strike"] - atm_strike).abs() <= band_width)
    ]
    if puts.empty or calls.empty:
        return 0.0
    puts_iv = pd.to_numeric(puts["iv"], errors="coerce").dropna()
    calls_iv = pd.to_numeric(calls["iv"], errors="coerce").dropna()
    if puts_iv.empty or calls_iv.empty:
        return 0.0
    return float(puts_iv.mean() - calls_iv.mean())
