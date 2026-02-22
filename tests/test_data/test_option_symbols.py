from __future__ import annotations

import pandas as pd

from src.data.option_symbols import option_lookup_keys, parse_option_symbol, resolve_option_price


def test_parse_option_symbol_handles_underscore_and_compact_formats():
    underscore = parse_option_symbol("NIFTY_20240718_24700CE")
    compact = parse_option_symbol("NIFTY2471824700CE")

    assert underscore is not None
    assert compact is not None
    assert underscore.root == compact.root == "NIFTY"
    assert underscore.expiry == compact.expiry == pd.Timestamp("2024-07-18")
    assert underscore.strike == compact.strike == 24700
    assert underscore.option_type == compact.option_type == "CE"


def test_option_lookup_keys_include_both_symbol_variants_and_canonical_key():
    keys = set(option_lookup_keys(symbol="NIFTY_20240718_24700CE"))
    assert "NIFTY_20240718_24700CE" in keys
    assert "NIFTY2471824700CE" in keys
    assert "OPT::20240718_24700_CE" in keys


def test_resolve_option_price_matches_cross_format_lookup():
    lookup = {"NIFTY2471824700CE": 44.55}
    resolved = resolve_option_price(
        price_lookup=lookup,
        symbol="NIFTY_20240718_24700CE",
    )
    assert resolved == 44.55
