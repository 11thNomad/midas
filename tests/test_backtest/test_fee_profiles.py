from __future__ import annotations

from src.backtest.fee_profiles import (
    VectorBTFeeProfile,
    parse_vectorbt_fee_profiles,
    resolve_vectorbt_costs,
    select_vectorbt_fee_profiles,
)


def test_parse_vectorbt_fee_profiles_reads_default_and_profiles():
    default_name, profiles = parse_vectorbt_fee_profiles(
        {
            "vectorbt_fee_profiles": {
                "default": "base",
                "profiles": {
                    "base": {"slippage_multiplier": 1.0, "fee_multiplier": 1.0},
                    "stress": {"slippage_multiplier": 2.0, "fee_multiplier": 1.5},
                },
            }
        }
    )
    assert default_name == "base"
    assert [p.name for p in profiles] == ["base", "stress"]


def test_parse_vectorbt_fee_profiles_falls_back_to_base_when_missing():
    default_name, profiles = parse_vectorbt_fee_profiles({})
    assert default_name == "base"
    assert len(profiles) == 1
    assert profiles[0].name == "base"


def test_select_vectorbt_fee_profiles_filters_requested_names():
    profiles = [
        VectorBTFeeProfile(name="base"),
        VectorBTFeeProfile(name="penalized"),
        VectorBTFeeProfile(name="stress"),
    ]
    selected = select_vectorbt_fee_profiles(profiles, "base,stress")
    assert [p.name for p in selected] == ["base", "stress"]


def test_resolve_vectorbt_costs_applies_multipliers():
    fees_pct, slippage_pct = resolve_vectorbt_costs(
        backtest_cfg={"slippage_pct": 0.05, "vectorbt_fees_pct": 0.04},
        profile=VectorBTFeeProfile(name="stress", slippage_multiplier=2.0, fee_multiplier=1.5),
    )
    assert fees_pct == (0.04 * 1.5) / 100.0
    assert slippage_pct == (0.05 * 2.0) / 100.0
