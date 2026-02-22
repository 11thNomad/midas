"""Option symbol parsing and lookup helpers for mixed NSE symbol formats."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

_UNDERSCORE_RE = re.compile(r"^(?P<root>[A-Z]+)_(?P<yyyymmdd>\d{8})_(?P<strike>\d+)(?P<opt>CE|PE)$")
_COMPACT_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<body>\d+)(?P<opt>CE|PE)$")


@dataclass(frozen=True)
class OptionSymbolParts:
    root: str
    expiry: pd.Timestamp
    strike: int
    option_type: str


def parse_option_symbol(symbol: str) -> OptionSymbolParts | None:
    raw = str(symbol).strip().upper()
    if not raw:
        return None

    underscore = _UNDERSCORE_RE.match(raw)
    if underscore:
        expiry = pd.to_datetime(underscore.group("yyyymmdd"), format="%Y%m%d", errors="coerce")
        if pd.isna(expiry):
            return None
        return OptionSymbolParts(
            root=underscore.group("root"),
            expiry=pd.Timestamp(expiry),
            strike=int(underscore.group("strike")),
            option_type=underscore.group("opt"),
        )

    compact = _COMPACT_RE.match(raw)
    if compact:
        root = compact.group("root")
        body = compact.group("body")
        opt = compact.group("opt")
        # Two compact date encodings appear in cache:
        # 1) YYMMDD + strike (e.g., 24071824700)
        # 2) YYMDD  + strike (e.g., 2471824700 where month=7)
        for date_len in (6, 5):
            if len(body) <= date_len:
                continue
            date_part = body[:date_len]
            strike_part = body[date_len:]
            if not strike_part.isdigit():
                continue
            yy = int(date_part[:2])
            if date_len == 6:
                mm = int(date_part[2:4])
                dd = int(date_part[4:6])
            else:
                mm = int(date_part[2:3])
                dd = int(date_part[3:5])
            try:
                expiry = pd.Timestamp(year=2000 + yy, month=mm, day=dd)
            except ValueError:
                continue
            return OptionSymbolParts(
                root=root,
                expiry=expiry,
                strike=int(strike_part),
                option_type=opt,
            )

    return None


def option_symbol_underscore(parts: OptionSymbolParts) -> str:
    return (
        f"{parts.root}_{parts.expiry.strftime('%Y%m%d')}_{int(parts.strike)}{parts.option_type.upper()}"
    )


def option_symbol_compact(parts: OptionSymbolParts) -> str:
    yy = parts.expiry.strftime("%y")
    month = str(parts.expiry.month)
    dd = parts.expiry.strftime("%d")
    strike = int(parts.strike)
    option_type = parts.option_type.upper()
    return f"{parts.root}{yy}{month}{dd}{strike}{option_type}"


def option_symbol_compact_padded(parts: OptionSymbolParts) -> str:
    return (
        f"{parts.root}{parts.expiry.strftime('%y%m%d')}"
        f"{int(parts.strike)}{parts.option_type.upper()}"
    )


def option_canonical_key(parts: OptionSymbolParts) -> str:
    return (
        f"OPT::{parts.expiry.strftime('%Y%m%d')}_{int(parts.strike)}_{parts.option_type.upper()}"
    )


def option_canonical_key_from_contract(
    *,
    expiry: Any,
    strike: Any,
    option_type: Any,
) -> str | None:
    expiry_ts = pd.to_datetime(expiry, errors="coerce")
    strike_value = pd.to_numeric(strike, errors="coerce")
    if pd.isna(expiry_ts) or pd.isna(strike_value):
        return None
    opt = str(option_type).strip().upper()
    if opt not in {"CE", "PE"}:
        return None
    return f"OPT::{pd.Timestamp(expiry_ts).strftime('%Y%m%d')}_{int(float(strike_value))}_{opt}"


def option_lookup_keys(
    *,
    symbol: str,
    expiry: Any | None = None,
    strike: Any | None = None,
    option_type: Any | None = None,
) -> tuple[str, ...]:
    """Return candidate lookup keys across symbol formats and canonical key."""

    candidates: list[str] = []
    raw = str(symbol).strip()
    if raw:
        candidates.append(raw)

    parts = parse_option_symbol(raw)
    if parts is not None:
        candidates.append(option_symbol_underscore(parts))
        candidates.append(option_symbol_compact(parts))
        candidates.append(option_symbol_compact_padded(parts))
        candidates.append(option_canonical_key(parts))

    contract_key = option_canonical_key_from_contract(
        expiry=expiry,
        strike=strike,
        option_type=option_type,
    )
    if contract_key is not None:
        candidates.append(contract_key)

    # Deduplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        key_norm = str(key).strip()
        if not key_norm or key_norm in seen:
            continue
        seen.add(key_norm)
        out.append(key_norm)
    return tuple(out)


def resolve_option_price(
    *,
    price_lookup: Mapping[str, float],
    symbol: str,
    expiry: Any | None = None,
    strike: Any | None = None,
    option_type: Any | None = None,
) -> float | None:
    for key in option_lookup_keys(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
    ):
        if key in price_lookup:
            return float(price_lookup[key])
    return None
