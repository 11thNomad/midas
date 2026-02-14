"""Fee/slippage profile helpers for vectorbt experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VectorBTFeeProfile:
    name: str
    slippage_multiplier: float = 1.0
    fee_multiplier: float = 1.0


def parse_vectorbt_fee_profiles(
    backtest_cfg: dict[str, Any],
) -> tuple[str, list[VectorBTFeeProfile]]:
    raw = backtest_cfg.get("vectorbt_fee_profiles", {})
    if not isinstance(raw, dict):
        raise ValueError("backtest.vectorbt_fee_profiles must be an object.")

    default_profile = str(raw.get("default", "base")).strip() or "base"
    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        raise ValueError("backtest.vectorbt_fee_profiles.profiles must be an object.")

    profiles: list[VectorBTFeeProfile] = []
    for name, cfg in profiles_raw.items():
        profile_name = str(name).strip()
        if not profile_name:
            continue
        if not isinstance(cfg, dict):
            raise ValueError(f"Profile '{profile_name}' must be an object.")
        profiles.append(
            VectorBTFeeProfile(
                name=profile_name,
                slippage_multiplier=float(cfg.get("slippage_multiplier", 1.0) or 1.0),
                fee_multiplier=float(cfg.get("fee_multiplier", 1.0) or 1.0),
            )
        )

    if not profiles:
        profiles = [VectorBTFeeProfile(name="base")]
        default_profile = "base"

    names = {profile.name for profile in profiles}
    if default_profile not in names:
        raise ValueError(
            "Default vectorbt fee profile "
            f"'{default_profile}' not found in profiles: {sorted(names)}"
        )
    return default_profile, profiles


def select_vectorbt_fee_profiles(
    all_profiles: list[VectorBTFeeProfile],
    selected: str | None,
) -> list[VectorBTFeeProfile]:
    if selected is None or not selected.strip():
        return all_profiles
    wanted = {part.strip() for part in selected.split(",") if part.strip()}
    if not wanted:
        return all_profiles
    out = [profile for profile in all_profiles if profile.name in wanted]
    found = {profile.name for profile in out}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown vectorbt fee profile(s): {missing}")
    return out


def resolve_vectorbt_costs(
    *,
    backtest_cfg: dict[str, Any],
    profile: VectorBTFeeProfile,
) -> tuple[float, float]:
    base_slippage_pct = float(backtest_cfg.get("slippage_pct", 0.05) or 0.05)
    base_fees_pct = float(
        backtest_cfg.get("vectorbt_fees_pct", base_slippage_pct) or base_slippage_pct
    )
    fees_pct = (base_fees_pct * float(profile.fee_multiplier)) / 100.0
    slippage_pct = (base_slippage_pct * float(profile.slippage_multiplier)) / 100.0
    return fees_pct, slippage_pct
