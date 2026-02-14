"""Trade attribution helpers for vectorbt parameter-set experiments."""

from __future__ import annotations

import pandas as pd


def build_trade_attribution(
    *,
    trades: pd.DataFrame,
    schedule: pd.DataFrame,
    set_id: str,
    fee_profile: str,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    out = trades.copy()
    entry_col = _pick_column(out, ["Entry Timestamp", "entry_timestamp"])
    exit_col = _pick_column(out, ["Exit Timestamp", "exit_timestamp"])
    pnl_col = _pick_column(out, ["PnL", "pnl"])
    ret_col = _pick_column(out, ["Return", "return"])
    size_col = _pick_column(out, ["Size", "size"])
    entry_price_col = _pick_column(out, ["Avg Entry Price", "entry_price"])
    exit_price_col = _pick_column(out, ["Avg Exit Price", "exit_price"])
    status_col = _pick_column(out, ["Status", "status"])
    direction_col = _pick_column(out, ["Direction", "direction"])
    trade_id_col = _pick_column(out, ["Exit Trade Id", "trade_id", "id"])

    out["entry_timestamp"] = pd.to_datetime(_series_or_empty(out, entry_col), errors="coerce")
    out["exit_timestamp"] = pd.to_datetime(_series_or_empty(out, exit_col), errors="coerce")
    out["pnl"] = pd.to_numeric(out[pnl_col], errors="coerce") if pnl_col else pd.NA
    out["return_raw"] = pd.to_numeric(out[ret_col], errors="coerce") if ret_col else pd.NA
    out["return_pct"] = (
        pd.to_numeric(out["return_raw"], errors="coerce") * 100.0
        if "return_raw" in out.columns
        else pd.NA
    )
    out["size"] = pd.to_numeric(out[size_col], errors="coerce") if size_col else pd.NA
    out["entry_price"] = (
        pd.to_numeric(out[entry_price_col], errors="coerce") if entry_price_col else pd.NA
    )
    out["exit_price"] = (
        pd.to_numeric(out[exit_price_col], errors="coerce") if exit_price_col else pd.NA
    )
    out["status"] = out[status_col].astype(str) if status_col else pd.NA
    out["direction"] = out[direction_col].astype(str) if direction_col else pd.NA
    if trade_id_col:
        out["trade_id"] = out[trade_id_col]
    else:
        out["trade_id"] = pd.Series(range(len(out)), index=out.index)

    schedule_ctx = _prepare_schedule_context(schedule)
    out = out.merge(
        schedule_ctx.rename(
            columns={
                "timestamp": "entry_timestamp",
                "close": "entry_close",
                "regime": "entry_regime",
                "adx_14": "entry_adx_14",
                "vix_level": "entry_vix_level",
                "_bar_index": "entry_bar_index",
            }
        ),
        on="entry_timestamp",
        how="left",
    )
    out = out.merge(
        schedule_ctx.rename(
            columns={
                "timestamp": "exit_timestamp",
                "close": "exit_close",
                "regime": "exit_regime",
                "adx_14": "exit_adx_14",
                "vix_level": "exit_vix_level",
                "_bar_index": "exit_bar_index",
            }
        ),
        on="exit_timestamp",
        how="left",
    )
    out["duration_bars"] = (
        pd.to_numeric(out["exit_bar_index"], errors="coerce")
        - pd.to_numeric(out["entry_bar_index"], errors="coerce")
    )
    out["win"] = pd.to_numeric(out["pnl"], errors="coerce") > 0.0
    out["set_id"] = set_id
    out["fee_profile"] = fee_profile

    keep_cols = [
        "set_id",
        "fee_profile",
        "trade_id",
        "entry_timestamp",
        "exit_timestamp",
        "duration_bars",
        "direction",
        "status",
        "size",
        "entry_price",
        "exit_price",
        "pnl",
        "return_raw",
        "return_pct",
        "win",
        "entry_regime",
        "exit_regime",
        "entry_adx_14",
        "exit_adx_14",
        "entry_vix_level",
        "exit_vix_level",
        "entry_close",
        "exit_close",
    ]
    existing = [col for col in keep_cols if col in out.columns]
    return out[existing].sort_values("entry_timestamp").reset_index(drop=True)


def _prepare_schedule_context(schedule: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "close", "regime", "adx_14", "vix_level"]
    existing = [col for col in cols if col in schedule.columns]
    if "timestamp" not in existing:
        return pd.DataFrame(columns=cols + ["_bar_index"])
    out = schedule[existing].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["_bar_index"] = out.index.astype("int64")
    return out


def _pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    normalized = {
        _normalize_column_name(str(col)): str(col)
        for col in frame.columns
    }
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _series_or_empty(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column and column in frame.columns:
        return frame[column]
    return pd.Series(index=frame.index, dtype="object")


def _normalize_column_name(name: str) -> str:
    out = name.strip().lower()
    for token in (" ", "_", "-", "%", "(", ")", "[", "]"):
        out = out.replace(token, "")
    return out
