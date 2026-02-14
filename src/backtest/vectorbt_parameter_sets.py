"""Parameter-set utilities for vectorbt research sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.vectorbt_research import VectorBTResearchConfig


@dataclass(frozen=True)
class VectorBTParameterSet:
    set_id: str
    entry_regimes: tuple[str, ...]
    adx_min: float
    vix_max: float | None = None
    notes: str = ""


def parse_parameter_sets(raw: dict[str, Any]) -> list[VectorBTParameterSet]:
    entries = _extract_entries(raw)
    out: list[VectorBTParameterSet] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"parameter_sets[{i - 1}] must be an object.")
        set_id = str(entry.get("id", "")).strip()
        if not set_id:
            raise ValueError(f"parameter_sets[{i - 1}].id is required.")
        if set_id in seen:
            raise ValueError(f"Duplicate parameter set id: {set_id}")
        seen.add(set_id)

        regimes_raw = entry.get("entry_regimes", [])
        entry_regimes = _parse_regimes(regimes_raw)
        if not entry_regimes:
            raise ValueError(f"parameter_sets[{i - 1}].entry_regimes cannot be empty.")

        if "adx_min" not in entry:
            raise ValueError(f"parameter_sets[{i - 1}].adx_min is required.")
        adx_min = float(entry["adx_min"])

        vix_raw = entry.get("vix_max")
        vix_max = None if vix_raw is None else float(vix_raw)
        notes = str(entry.get("notes", "")).strip()
        out.append(
            VectorBTParameterSet(
                set_id=set_id,
                entry_regimes=entry_regimes,
                adx_min=adx_min,
                vix_max=vix_max,
                notes=notes,
            )
        )
    return out


def apply_parameter_set(
    base_config: VectorBTResearchConfig,
    params: VectorBTParameterSet,
) -> VectorBTResearchConfig:
    return VectorBTResearchConfig(
        initial_cash=base_config.initial_cash,
        fees_pct=base_config.fees_pct,
        slippage_pct=base_config.slippage_pct,
        entry_regimes=params.entry_regimes,
        adx_min=float(params.adx_min),
        vix_max=params.vix_max,
        freq=base_config.freq,
    )


def rank_parameter_results(
    results: pd.DataFrame,
    *,
    rank_by: str,
    min_trades: float,
    max_drawdown_pct: float | None = None,
) -> pd.DataFrame:
    if results.empty:
        return results
    if rank_by not in results.columns:
        raise ValueError(f"rank_by column not found: {rank_by}")

    out = results.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    trade_input = out["trades"] if "trades" in out.columns else pd.Series(0.0, index=out.index)
    out["trades"] = pd.to_numeric(trade_input, errors="coerce").fillna(0.0)

    eligible = out["trades"] >= float(min_trades)
    if max_drawdown_pct is not None and "max_drawdown_pct" in out.columns:
        dd = pd.to_numeric(out["max_drawdown_pct"], errors="coerce").abs()
        eligible = eligible & (dd <= float(max_drawdown_pct))
    out["eligible"] = eligible

    rank_metric = pd.to_numeric(out[rank_by], errors="coerce")
    ascending = "drawdown" in rank_by.lower()
    if ascending:
        rank_metric = rank_metric.abs()
    out["_rank_metric"] = rank_metric

    out = out.sort_values(
        by=["eligible", "_rank_metric", "trades", "set_id"],
        ascending=[False, ascending, False, True],
        na_position="last",
    ).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out.drop(columns=["_rank_metric"])


def _extract_entries(raw: dict[str, Any]) -> list[Any]:
    if "parameter_sets" in raw:
        entries = raw.get("parameter_sets")
    else:
        vectorbt = raw.get("vectorbt", {})
        entries = vectorbt.get("parameter_sets", []) if isinstance(vectorbt, dict) else []
    if not isinstance(entries, list):
        raise ValueError("parameter_sets must be a list.")
    return entries


def _parse_regimes(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(part).strip() for part in raw]
    else:
        raise ValueError("entry_regimes must be a list or comma-separated string.")
    out = tuple(item for item in items if item)
    return out
