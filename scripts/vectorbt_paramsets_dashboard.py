"""Streamlit dashboard for vectorbt parameter-set reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.backtest import (
    list_vectorbt_paramset_runs,
    load_detail_artifact,
    load_run_summary,
    load_run_table,
    load_walkforward_folds,
)

DEFAULT_REPORTS_ROOT = Path("data/reports")


def _format_percent(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):.2f}%"


@st.cache_data(show_spinner=False)
def _cached_run_data(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    summary = load_run_summary(run_dir)
    results = load_run_table(run_dir, "vectorbt_parameter_set_results.csv")
    leaderboard = load_run_table(run_dir, "vectorbt_parameter_set_leaderboard.csv")
    robustness = load_run_table(run_dir, "vectorbt_parameter_set_robustness.csv")
    top_sets = load_run_table(run_dir, "vectorbt_parameter_set_top_sets.csv")
    promotion_gate = load_run_table(run_dir, "vectorbt_parameter_set_promotion_gate.csv")
    set_ids = (
        sorted(results["set_id"].dropna().astype(str).unique().tolist())
        if "set_id" in results
        else []
    )
    fee_profiles = (
        sorted(results["fee_profile"].dropna().astype(str).unique().tolist())
        if "fee_profile" in results
        else []
    )
    folds = load_walkforward_folds(run_dir, set_ids=set_ids, fee_profiles=fee_profiles)
    trade_attribution = load_detail_artifact(
        run_dir,
        set_ids=set_ids,
        fee_profiles=fee_profiles,
        suffix="trade_attribution",
    )
    equity = load_detail_artifact(
        run_dir,
        set_ids=set_ids,
        fee_profiles=fee_profiles,
        suffix="equity",
    )
    return {
        "summary": summary,
        "results": results,
        "leaderboard": leaderboard,
        "robustness": robustness,
        "top_sets": top_sets,
        "promotion_gate": promotion_gate,
        "folds": folds,
        "trade_attribution": trade_attribution,
        "equity": equity,
        "set_ids": set_ids,
        "fee_profiles": fee_profiles,
    }


def _filter_leaderboard(
    leaderboard: pd.DataFrame,
    *,
    set_ids: list[str],
    fee_profiles: list[str],
    min_trades: float,
    eligible_only: bool,
) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard
    out = leaderboard.copy()
    if "set_id" in out.columns:
        out = out.loc[out["set_id"].astype(str).isin(set_ids)]
    if "fee_profile" in out.columns:
        out = out.loc[out["fee_profile"].astype(str).isin(fee_profiles)]
    if "trades" in out.columns:
        out["trades"] = pd.to_numeric(out["trades"], errors="coerce")
        out = out.loc[out["trades"] >= float(min_trades)]
    if eligible_only and "eligible" in out.columns:
        out = out.loc[out["eligible"].astype(bool)]
    if "rank" in out.columns:
        out = out.sort_values("rank")
    return out.reset_index(drop=True)


def _metric_options(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "wf_total_return_pct_mean",
        "total_return_pct",
        "wf_sharpe_ratio_mean",
        "sharpe_ratio",
        "wf_max_drawdown_pct_mean",
        "max_drawdown_pct",
        "trades",
    ]
    cols = set(frame.columns)
    out = [c for c in preferred if c in cols]
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    for col in numeric:
        if col not in out:
            out.append(col)
    return out


def _render_overview(summary: dict[str, Any]) -> None:
    st.subheader("Run Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", str(summary.get("symbol", "n/a")))
    c2.metric("Timeframe", str(summary.get("timeframe", "n/a")))
    c3.metric("Runs", str(summary.get("run_count", "n/a")))
    c4.metric("Set Count", str(summary.get("parameter_set_count", "n/a")))

    best = summary.get("best_robust_set", {})
    if isinstance(best, dict) and best:
        st.caption(
            "Best robust set: "
            f"{best.get('set_id', 'n/a')} "
            f"(worst metric={best.get('metric_worst', 'n/a')})"
        )
    st.json(summary, expanded=False)


def _render_recommendation(promotion_gate: pd.DataFrame, *, set_ids: list[str]) -> None:
    st.subheader("Promotion Recommendation")
    if promotion_gate.empty:
        st.info("No promotion gate artifact found for this run.")
        return
    gate = promotion_gate.copy()
    if "set_id" in gate.columns:
        gate = gate.loc[gate["set_id"].astype(str).isin(set_ids)]
    if gate.empty:
        st.info("No promotion-gate rows for current filters.")
        return
    if "promotion_pass" not in gate.columns:
        st.info("Promotion gate rows are missing `promotion_pass`.")
        return

    passed = gate.loc[gate["promotion_pass"].astype(bool)].copy()
    if not passed.empty:
        st.success(
            "Recommended for paper-candidate shortlist: "
            + ", ".join(passed["set_id"].astype(str).tolist())
        )
        show_cols = [
            col
            for col in [
                "set_id",
                "metric_worst",
                "metric_mean",
                "trades_mean",
                "drawdown_abs_worst",
                "eligible_profile_share",
            ]
            if col in passed.columns
        ]
        st.dataframe(passed[show_cols], use_container_width=True)
        return

    pass_cols = [col for col in gate.columns if col.startswith("pass_")]
    if pass_cols:
        gate["pass_score"] = gate[pass_cols].astype(bool).sum(axis=1)
    else:
        gate["pass_score"] = 0
    near = gate.sort_values(
        by=["pass_score", "metric_worst", "metric_mean", "set_id"],
        ascending=[False, False, False, True],
        na_position="last",
    ).head(3)
    st.warning("No set passes the promotion gate yet. Nearest candidates:")
    show_cols = [
        col
        for col in [
            "set_id",
            "pass_score",
            "promotion_fail_reasons",
            "metric_worst",
            "metric_mean",
            "trades_mean",
            "drawdown_abs_worst",
        ]
        if col in near.columns
    ]
    st.dataframe(near[show_cols], use_container_width=True)


def _render_leaderboard(
    leaderboard: pd.DataFrame,
    *,
    metric: str,
) -> None:
    st.subheader("Leaderboard")
    if leaderboard.empty:
        st.info("No rows after filters.")
        return

    st.dataframe(leaderboard, use_container_width=True)
    if metric in leaderboard.columns and "set_id" in leaderboard.columns:
        pivot = leaderboard.pivot_table(
            index="set_id",
            columns="fee_profile" if "fee_profile" in leaderboard.columns else None,
            values=metric,
            aggfunc="mean",
        )
        if isinstance(pivot, pd.DataFrame) and not pivot.empty:
            st.caption(f"{metric} by set and fee profile")
            st.bar_chart(pivot)

    scatter_cols = {"total_return_pct", "max_drawdown_pct", "set_id"}
    if scatter_cols.issubset(set(leaderboard.columns)):
        scatter = leaderboard[list(scatter_cols)].copy()
        scatter["label"] = scatter["set_id"].astype(str)
        st.caption("Risk/return scatter")
        st.scatter_chart(
            scatter,
            x="max_drawdown_pct",
            y="total_return_pct",
            color="label",
        )


def _render_robustness(robustness: pd.DataFrame) -> None:
    st.subheader("Set Robustness")
    if robustness.empty:
        st.info("No robustness table in this run.")
        return
    st.dataframe(robustness, use_container_width=True)
    if {"set_id", "metric_worst"}.issubset(set(robustness.columns)):
        chart = robustness.set_index("set_id")[["metric_worst", "metric_mean"]]
        st.caption("Worst vs mean metric across fee profiles")
        st.bar_chart(chart)


def _render_walkforward(
    folds: pd.DataFrame,
    *,
    selected_set: str,
    selected_fee_profile: str,
) -> None:
    st.subheader("Walk-Forward")
    if folds.empty:
        st.info("No walk-forward fold files found in this run.")
        return
    subset = folds.loc[
        (folds["set_id"].astype(str) == selected_set)
        & (folds["fee_profile"].astype(str) == selected_fee_profile)
    ].copy()
    if subset.empty:
        st.info("No fold rows for current set/profile filter.")
        return

    subset["fold"] = pd.to_numeric(subset.get("fold"), errors="coerce")
    subset = subset.sort_values("fold")
    if "total_return_pct" in subset.columns:
        returns = subset[["fold", "total_return_pct"]].set_index("fold")
        st.caption("Fold return %")
        st.line_chart(returns)
        subset["cum_return_pct"] = (
            pd.to_numeric(subset["total_return_pct"], errors="coerce").cumsum()
        )
        st.caption("Cumulative fold return %")
        st.line_chart(subset.set_index("fold")[["cum_return_pct"]])
    if "max_drawdown_pct" in subset.columns:
        dd = subset[["fold", "max_drawdown_pct"]].set_index("fold")
        st.caption("Fold max drawdown %")
        st.line_chart(dd)
    st.dataframe(subset, use_container_width=True)


def _render_trades(
    trade_attribution: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    selected_set: str,
    selected_fee_profile: str,
) -> None:
    st.subheader("Trades & Signal Context")
    if trade_attribution.empty:
        st.info("No trade attribution files found for this run.")
        return

    subset = trade_attribution.loc[
        (trade_attribution["set_id"].astype(str) == selected_set)
        & (trade_attribution["fee_profile"].astype(str) == selected_fee_profile)
    ].copy()
    if subset.empty:
        st.info("No trades for current set/profile filter.")
        return

    subset["pnl"] = pd.to_numeric(subset.get("pnl"), errors="coerce")
    subset["return_pct"] = pd.to_numeric(subset.get("return_pct"), errors="coerce")
    subset["duration_bars"] = pd.to_numeric(subset.get("duration_bars"), errors="coerce")
    subset["entry_timestamp"] = pd.to_datetime(subset.get("entry_timestamp"), errors="coerce")
    subset = subset.sort_values("entry_timestamp")

    wins = int((subset["pnl"] > 0.0).sum())
    total = int(len(subset))
    win_rate = (wins / total) * 100.0 if total > 0 else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", str(total))
    c2.metric("Total PnL", f"{subset['pnl'].sum():.2f}")
    c3.metric("Win rate", f"{win_rate:.1f}%")
    c4.metric("Avg return/trade", f"{subset['return_pct'].mean():.2f}%")

    if {"entry_regime", "pnl"}.issubset(set(subset.columns)):
        by_regime = (
            subset.groupby("entry_regime", dropna=False)["pnl"]
            .mean()
            .sort_values(ascending=False)
        )
        st.caption("Average PnL by entry regime")
        st.bar_chart(by_regime)

    if {"entry_vix_level", "pnl"}.issubset(set(subset.columns)):
        scatter = subset[["entry_vix_level", "pnl"]].copy()
        st.caption("PnL vs entry VIX")
        st.scatter_chart(scatter, x="entry_vix_level", y="pnl")

    eq_subset = equity.loc[
        (equity["set_id"].astype(str) == selected_set)
        & (equity["fee_profile"].astype(str) == selected_fee_profile)
    ].copy() if not equity.empty else pd.DataFrame()
    if not eq_subset.empty and {"timestamp", "equity"}.issubset(set(eq_subset.columns)):
        eq_subset["timestamp"] = pd.to_datetime(eq_subset["timestamp"], errors="coerce")
        eq_subset = eq_subset.dropna(subset=["timestamp"]).sort_values("timestamp")
        st.caption("Equity curve")
        st.line_chart(eq_subset.set_index("timestamp")[["equity"]])

    st.dataframe(subset, use_container_width=True)


def _render_cohorts_and_gate(
    trade_attribution: pd.DataFrame,
    promotion_gate: pd.DataFrame,
    *,
    set_ids: list[str],
    fee_profiles: list[str],
) -> None:
    st.subheader("Cohorts & Promotion Gate")
    cohort_tab, gate_tab = st.tabs(["Cohort Analysis", "Promotion Gate"])

    with cohort_tab:
        if trade_attribution.empty:
            st.info("No trade attribution files found for this run.")
        else:
            data = trade_attribution.copy()
            data = data.loc[
                data["set_id"].astype(str).isin(set_ids)
                & data["fee_profile"].astype(str).isin(fee_profiles)
            ].copy()
            if data.empty:
                st.info("No trade rows for selected filters.")
            else:
                data["entry_timestamp"] = pd.to_datetime(
                    data.get("entry_timestamp"),
                    errors="coerce",
                )
                data["pnl"] = pd.to_numeric(data.get("pnl"), errors="coerce")
                data["return_pct"] = pd.to_numeric(data.get("return_pct"), errors="coerce")
                data["duration_bars"] = pd.to_numeric(data.get("duration_bars"), errors="coerce")
                data["win"] = pd.to_numeric(data.get("pnl"), errors="coerce") > 0.0
                data["entry_weekday"] = data["entry_timestamp"].dt.day_name()
                data["entry_month"] = data["entry_timestamp"].dt.strftime("%Y-%m")

                cohort_options = [
                    column
                    for column in [
                        "set_id",
                        "fee_profile",
                        "entry_regime",
                        "exit_regime",
                        "entry_weekday",
                        "entry_month",
                    ]
                    if column in data.columns
                ]
                cohort_dim = st.selectbox("Cohort dimension", options=cohort_options, index=0)
                grouped = (
                    data.groupby(cohort_dim, dropna=False)
                    .agg(
                        trades=("pnl", "count"),
                        win_rate_pct=("win", lambda s: float(s.mean()) * 100.0),
                        avg_pnl=("pnl", "mean"),
                        total_pnl=("pnl", "sum"),
                        avg_return_pct=("return_pct", "mean"),
                        median_return_pct=("return_pct", "median"),
                        avg_duration_bars=("duration_bars", "mean"),
                    )
                    .reset_index()
                    .sort_values("avg_pnl", ascending=False)
                )
                st.dataframe(grouped, use_container_width=True)

                metric = st.selectbox(
                    "Cohort metric",
                    options=[
                        "trades",
                        "win_rate_pct",
                        "avg_pnl",
                        "total_pnl",
                        "avg_return_pct",
                        "median_return_pct",
                        "avg_duration_bars",
                    ],
                    index=2,
                )
                st.bar_chart(grouped.set_index(cohort_dim)[[metric]])

    with gate_tab:
        if promotion_gate.empty:
            st.info("No promotion gate artifact found for this run.")
        else:
            gate = promotion_gate.copy()
            if "set_id" in gate.columns:
                gate = gate.loc[gate["set_id"].astype(str).isin(set_ids)]
            st.dataframe(gate, use_container_width=True)
            if "promotion_pass" in gate.columns:
                pass_count = int(gate["promotion_pass"].astype(bool).sum())
                total = int(len(gate))
                st.caption(f"Promotion pass: {pass_count}/{total}")


def main() -> None:
    st.set_page_config(
        page_title="VectorBT Parameter Set Dashboard",
        layout="wide",
    )
    st.title("VectorBT Parameter Set Dashboard")

    reports_root_raw = st.sidebar.text_input("Reports root", value=str(DEFAULT_REPORTS_ROOT))
    reports_root = Path(reports_root_raw)
    run_dirs = list_vectorbt_paramset_runs(reports_root)
    if not run_dirs:
        st.error(f"No vectorbt paramset runs found under: {reports_root}")
        return

    run_names = [path.name for path in run_dirs]
    selected_run_name = st.sidebar.selectbox("Run directory", run_names, index=0)
    selected_run_dir = reports_root / selected_run_name
    payload = _cached_run_data(str(selected_run_dir))
    summary = payload["summary"]
    results = payload["results"]
    leaderboard = payload["leaderboard"]
    robustness = payload["robustness"]
    promotion_gate = payload["promotion_gate"]
    folds = payload["folds"]
    trade_attribution = payload["trade_attribution"]
    equity = payload["equity"]
    available_set_ids = payload["set_ids"]
    available_fee_profiles = payload["fee_profiles"]

    selected_set_ids = st.sidebar.multiselect(
        "Set IDs",
        options=available_set_ids,
        default=available_set_ids,
    )
    selected_fee_profiles = st.sidebar.multiselect(
        "Fee profiles",
        options=available_fee_profiles,
        default=available_fee_profiles,
    )
    min_trades = float(
        st.sidebar.number_input(
            "Min trades",
            min_value=0.0,
            max_value=10_000.0,
            value=3.0,
            step=1.0,
        )
    )
    eligible_only = st.sidebar.checkbox("Eligible only", value=False)
    metric_options = _metric_options(leaderboard if not leaderboard.empty else results)
    default_metric = "wf_total_return_pct_mean"
    if default_metric not in metric_options:
        default_metric = metric_options[0]
    metric = st.sidebar.selectbox(
        "Metric",
        options=metric_options,
        index=metric_options.index(default_metric),
    )

    if not selected_set_ids or not selected_fee_profiles:
        st.warning("Pick at least one set_id and one fee profile.")
        return

    filtered = _filter_leaderboard(
        leaderboard if not leaderboard.empty else results,
        set_ids=selected_set_ids,
        fee_profiles=selected_fee_profiles,
        min_trades=min_trades,
        eligible_only=eligible_only,
    )

    tabs = st.tabs(
        ["Overview", "Leaderboard", "Robustness", "Walk-Forward", "Trades", "Cohorts", "Raw"]
    )
    with tabs[0]:
        _render_overview(summary)
        _render_recommendation(promotion_gate, set_ids=selected_set_ids)
        if not filtered.empty and metric in filtered.columns:
            best = filtered.iloc[0]
            st.caption(
                "Current top row: "
                f"{best.get('set_id')} / {best.get('fee_profile')} "
                f"metric={best.get(metric)} "
                f"return={_format_percent(best.get('total_return_pct'))} "
                f"dd={_format_percent(best.get('max_drawdown_pct'))}"
            )
    with tabs[1]:
        _render_leaderboard(filtered, metric=metric)
    with tabs[2]:
        _render_robustness(robustness)
    with tabs[3]:
        wf_set = st.selectbox("Walk-forward set_id", options=selected_set_ids, index=0)
        wf_profile = st.selectbox(
            "Walk-forward fee profile",
            options=selected_fee_profiles,
            index=0,
        )
        _render_walkforward(folds, selected_set=wf_set, selected_fee_profile=wf_profile)
    with tabs[4]:
        trade_set = st.selectbox("Trade set_id", options=selected_set_ids, index=0)
        trade_profile = st.selectbox(
            "Trade fee profile",
            options=selected_fee_profiles,
            index=0,
        )
        _render_trades(
            trade_attribution,
            equity,
            selected_set=trade_set,
            selected_fee_profile=trade_profile,
        )
    with tabs[5]:
        _render_cohorts_and_gate(
            trade_attribution,
            promotion_gate,
            set_ids=selected_set_ids,
            fee_profiles=selected_fee_profiles,
        )
    with tabs[6]:
        st.subheader("Raw Artifacts")
        st.write(f"Run dir: `{selected_run_dir}`")
        st.caption("Results")
        st.dataframe(results, use_container_width=True)
        st.caption("Top sets")
        st.dataframe(payload["top_sets"], use_container_width=True)


if __name__ == "__main__":
    main()
