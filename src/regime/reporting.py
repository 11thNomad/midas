"""Summary builders for regime snapshots and strategy transition logs."""

from __future__ import annotations

import pandas as pd


def summarize_regime_daily(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Daily count of snapshots by regime."""
    if snapshots.empty or "timestamp" not in snapshots.columns or "regime" not in snapshots.columns:
        return pd.DataFrame(columns=["date", "regime", "snapshots"])

    out = snapshots.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return pd.DataFrame(columns=["date", "regime", "snapshots"])

    out["date"] = out["timestamp"].dt.date
    grouped = out.groupby(["date", "regime"], as_index=False).size().rename(columns={"size": "snapshots"})
    return grouped.sort_values(["date", "regime"]).reset_index(drop=True)


def summarize_transitions_daily(transitions: pd.DataFrame) -> pd.DataFrame:
    """Daily count of activation/deactivation events."""
    expected = {"timestamp", "from_active", "to_active"}
    if transitions.empty or not expected.issubset(transitions.columns):
        return pd.DataFrame(columns=["date", "activations", "deactivations", "events"])

    out = transitions.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return pd.DataFrame(columns=["date", "activations", "deactivations", "events"])

    out["date"] = out["timestamp"].dt.date
    out["activation"] = (~out["from_active"].astype(bool) & out["to_active"].astype(bool)).astype(int)
    out["deactivation"] = (out["from_active"].astype(bool) & ~out["to_active"].astype(bool)).astype(int)

    grouped = out.groupby("date", as_index=False).agg(
        activations=("activation", "sum"),
        deactivations=("deactivation", "sum"),
        events=("date", "count"),
    )
    return grouped.sort_values("date").reset_index(drop=True)


def summarize_transitions_by_strategy(transitions: pd.DataFrame) -> pd.DataFrame:
    """Transition counts by strategy."""
    expected = {"strategy", "from_active", "to_active"}
    if transitions.empty or not expected.issubset(transitions.columns):
        return pd.DataFrame(columns=["strategy", "activations", "deactivations", "events"])

    out = transitions.copy()
    out["activation"] = (~out["from_active"].astype(bool) & out["to_active"].astype(bool)).astype(int)
    out["deactivation"] = (out["from_active"].astype(bool) & ~out["to_active"].astype(bool)).astype(int)

    grouped = out.groupby("strategy", as_index=False).agg(
        activations=("activation", "sum"),
        deactivations=("deactivation", "sum"),
        events=("strategy", "count"),
    )
    return grouped.sort_values("strategy").reset_index(drop=True)
