"""Streamlit dashboard for backtest/walk-forward/hybrid strategy result artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DEFAULT_REPORTS_ROOT = Path("data/reports")
KNOWN_STRATEGY_IDS = (
    "regime_probe",
    "baseline_trend",
    "momentum",
    "iron_condor",
    "jade_lizard",
)


@dataclass(frozen=True)
class HybridRunRef:
    label: str
    run_dir: Path


def _list_runs(root: Path, prefix: str) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _list_hybrid_runs(root: Path) -> list[HybridRunRef]:
    if not root.exists() or not root.is_dir():
        return []
    refs: list[HybridRunRef] = []
    for parent in sorted(root.iterdir(), key=lambda p: p.name):
        if not parent.is_dir() or not parent.name.startswith("hybrid_"):
            continue
        for child in sorted(parent.iterdir(), key=lambda p: p.name, reverse=True):
            if not child.is_dir() or not child.name.startswith("vectorbt_"):
                continue
            refs.append(HybridRunRef(label=f"{parent.name}/{child.name}", run_dir=child))
    return refs


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _format_pct(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):.2f}%"


def _extract_strategy_ids(comparison: pd.DataFrame) -> list[str]:
    if comparison.empty or "strategy" not in comparison.columns:
        return []
    return sorted(comparison["strategy"].dropna().astype(str).unique().tolist())


def _find_backtest_artifact(run_dir: Path, strategy_id: str, suffix: str) -> Path | None:
    matches = sorted(run_dir.glob(f"{strategy_id}_*_{suffix}"))
    return matches[0] if matches else None


def _render_comparison_block(title: str, frame: pd.DataFrame) -> None:
    st.subheader(title)
    if frame.empty:
        st.info("No comparison file in this run.")
        return
    st.dataframe(frame, use_container_width=True)
    if {"strategy", "total_return_pct"}.issubset(frame.columns):
        chart = (
            frame[["strategy", "total_return_pct"]]
            .copy()
            .set_index("strategy")
            .sort_values("total_return_pct", ascending=False)
        )
        st.caption("Total return by strategy")
        st.bar_chart(chart)
    if {"strategy", "max_drawdown_pct"}.issubset(frame.columns):
        dd = (
            frame[["strategy", "max_drawdown_pct"]]
            .copy()
            .set_index("strategy")
            .sort_values("max_drawdown_pct", ascending=True)
        )
        st.caption("Max drawdown by strategy")
        st.bar_chart(dd)


@st.cache_data(show_spinner=False)
def _load_backtest_payload(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    comparison_path = next(iter(sorted(run_dir.glob("strategy_comparison_*.csv"))), None)
    comparison = _read_csv(comparison_path) if comparison_path is not None else pd.DataFrame()
    return {"comparison": comparison}


@st.cache_data(show_spinner=False)
def _load_walkforward_payload(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    comparison_path = next(iter(sorted(run_dir.glob("strategy_comparison_*.csv"))), None)
    comparison = _read_csv(comparison_path) if comparison_path is not None else pd.DataFrame()
    return {"comparison": comparison}


def _strategy_ids_from_artifacts(run_dir: Path) -> list[str]:
    ids: set[str] = set()
    for strategy_id in KNOWN_STRATEGY_IDS:
        has_backtest = any(run_dir.glob(f"{strategy_id}_*_backtest_metrics.json"))
        has_walkforward = any(run_dir.glob(f"{strategy_id}_*_walkforward_summary.json"))
        if has_backtest or has_walkforward:
            ids.add(strategy_id)
    return sorted(ids)


@st.cache_data(show_spinner=False)
def _load_hybrid_payload(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    return {
        "hybrid_metrics": _read_json(run_dir / "hybrid_metrics.json"),
        "hybrid_equity": _read_csv(run_dir / "hybrid_equity.csv"),
        "hybrid_fills": _read_csv(run_dir / "hybrid_fills.csv"),
        "vectorbt_metrics": _read_json(run_dir / "vectorbt_metrics.json"),
    }


def _render_backtest_strategy_details(run_dir: Path, strategy_id: str) -> None:
    st.subheader(f"Backtest Details: {strategy_id}")
    metrics_path = _find_backtest_artifact(run_dir, strategy_id, "backtest_metrics.json")
    equity_path = _find_backtest_artifact(run_dir, strategy_id, "backtest_equity.csv")
    fills_path = _find_backtest_artifact(run_dir, strategy_id, "backtest_fills.csv")
    report_path = _find_backtest_artifact(run_dir, strategy_id, "backtest_report.html")

    metrics = _read_json(metrics_path) if metrics_path is not None else {}
    equity = _read_csv(equity_path) if equity_path is not None else pd.DataFrame()
    fills = _read_csv(fills_path) if fills_path is not None else pd.DataFrame()

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", _format_pct(metrics.get("total_return_pct")))
        c2.metric("Sharpe", f"{float(metrics.get('sharpe_ratio', 0.0)):.3f}")
        c3.metric("Max DD", _format_pct(metrics.get("max_drawdown_pct")))
        c4.metric("Fills", str(int(float(metrics.get("fill_count", 0.0) or 0.0))))
        st.json(metrics, expanded=False)
    else:
        st.info("No metrics file for selected strategy in this run.")

    if not equity.empty and {"timestamp", "equity"}.issubset(equity.columns):
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce")
        equity = equity.dropna(subset=["timestamp"]).sort_values("timestamp")
        st.caption("Equity curve")
        st.line_chart(equity.set_index("timestamp")[["equity"]])

    if not fills.empty and {"timestamp", "fees"}.issubset(fills.columns):
        fills["timestamp"] = pd.to_datetime(fills["timestamp"], errors="coerce")
        fills["fees"] = pd.to_numeric(fills["fees"], errors="coerce")
        fee_summary = fills["fees"].agg(["count", "sum", "mean"]).to_dict()
        st.caption(
            f"Fill count={int(fee_summary.get('count', 0))} "
            f"total_fees={fee_summary.get('sum', 0.0):.2f} "
            f"avg_fee={fee_summary.get('mean', 0.0):.2f}"
        )
        st.dataframe(fills, use_container_width=True)

    if report_path is not None:
        st.code(str(report_path))


def _render_walkforward_strategy_details(run_dir: Path, strategy_id: str) -> None:
    st.subheader(f"Walk-Forward Details: {strategy_id}")
    folds_path = _find_backtest_artifact(run_dir, strategy_id, "walkforward_folds.csv")
    summary_path = _find_backtest_artifact(run_dir, strategy_id, "walkforward_summary.json")
    report_path = _find_backtest_artifact(run_dir, strategy_id, "walkforward_report.html")

    folds = _read_csv(folds_path) if folds_path is not None else pd.DataFrame()
    summary = _read_json(summary_path) if summary_path is not None else {}

    if summary:
        st.json(summary, expanded=False)
    if folds.empty:
        st.info("No fold file for selected strategy.")
    else:
        folds["fold"] = pd.to_numeric(folds.get("fold"), errors="coerce")
        folds = folds.sort_values("fold")
        if "total_return_pct" in folds.columns:
            st.caption("Fold return %")
            st.line_chart(folds.set_index("fold")[["total_return_pct"]])
        if "max_drawdown_pct" in folds.columns:
            st.caption("Fold max drawdown %")
            st.line_chart(folds.set_index("fold")[["max_drawdown_pct"]])
        st.dataframe(folds, use_container_width=True)
    if report_path is not None:
        st.code(str(report_path))


def _render_hybrid_block(hybrid_refs: list[HybridRunRef], selected_labels: list[str]) -> None:
    st.subheader("Hybrid Runs")
    if not selected_labels:
        st.info("Pick one or more hybrid runs from sidebar.")
        return

    rows: list[dict[str, Any]] = []
    for label in selected_labels:
        ref = next((r for r in hybrid_refs if r.label == label), None)
        if ref is None:
            continue
        payload = _load_hybrid_payload(str(ref.run_dir))
        metrics = payload["hybrid_metrics"]
        rows.append(
            {
                "hybrid_run": label,
                "total_return_pct": metrics.get("total_return_pct"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "fill_count": metrics.get("fill_count"),
                "fees_paid": metrics.get("fees_paid"),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        st.info("No hybrid metric rows found.")
        return
    st.dataframe(summary, use_container_width=True)
    if {"hybrid_run", "total_return_pct"}.issubset(summary.columns):
        st.caption("Hybrid total return %")
        st.bar_chart(summary.set_index("hybrid_run")[["total_return_pct"]])

    selected_label = st.selectbox("Hybrid run detail", options=selected_labels, index=0)
    selected_ref = next((r for r in hybrid_refs if r.label == selected_label), None)
    if selected_ref is None:
        return
    payload = _load_hybrid_payload(str(selected_ref.run_dir))
    equity = payload["hybrid_equity"]
    fills = payload["hybrid_fills"]
    st.json(payload["hybrid_metrics"], expanded=False)
    if not equity.empty and {"timestamp", "equity"}.issubset(equity.columns):
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce")
        equity = equity.dropna(subset=["timestamp"]).sort_values("timestamp")
        st.caption("Hybrid equity curve")
        st.line_chart(equity.set_index("timestamp")[["equity"]])
    if not fills.empty:
        st.caption("Hybrid fills")
        st.dataframe(fills, use_container_width=True)
    st.code(str(selected_ref.run_dir))


def main() -> None:
    st.set_page_config(page_title="Strategy Results Dashboard", layout="wide")
    st.title("Strategy Results Dashboard")

    reports_root_input = st.sidebar.text_input(
        "Reports root",
        value=str(DEFAULT_REPORTS_ROOT),
    ).strip()
    reports_root = Path(reports_root_input or "data/reports")
    backtest_runs = _list_runs(reports_root, "backtest_")
    walkforward_runs = _list_runs(reports_root, "walkforward_")
    hybrid_runs = _list_hybrid_runs(reports_root)

    if not backtest_runs and not walkforward_runs and not hybrid_runs:
        st.error(f"No supported runs found under: {reports_root}")
        return

    selected_backtest = None
    if backtest_runs:
        selected_backtest_name = st.sidebar.selectbox(
            "Backtest run",
            options=[p.name for p in backtest_runs],
            index=0,
        )
        selected_backtest = reports_root / selected_backtest_name

    selected_walkforward = None
    if walkforward_runs:
        selected_walkforward_name = st.sidebar.selectbox(
            "Walk-forward run",
            options=[p.name for p in walkforward_runs],
            index=0,
        )
        selected_walkforward = reports_root / selected_walkforward_name

    selected_hybrid = st.sidebar.multiselect(
        "Hybrid runs",
        options=[r.label for r in hybrid_runs],
        default=[r.label for r in hybrid_runs[:2]],
    )

    backtest_payload = (
        _load_backtest_payload(str(selected_backtest)) if selected_backtest is not None else {}
    )
    walkforward_payload = (
        _load_walkforward_payload(str(selected_walkforward))
        if selected_walkforward is not None
        else {}
    )
    backtest_comparison = backtest_payload.get("comparison", pd.DataFrame())
    walkforward_comparison = walkforward_payload.get("comparison", pd.DataFrame())

    backtest_ids = set(_extract_strategy_ids(backtest_comparison))
    walkforward_ids = set(_extract_strategy_ids(walkforward_comparison))
    if selected_backtest is not None:
        backtest_ids.update(_strategy_ids_from_artifacts(selected_backtest))
    if selected_walkforward is not None:
        walkforward_ids.update(_strategy_ids_from_artifacts(selected_walkforward))
    strategy_options = sorted(backtest_ids | walkforward_ids)
    if not strategy_options:
        strategy_options = list(KNOWN_STRATEGY_IDS)
    selected_strategy = st.sidebar.selectbox("Strategy detail", options=strategy_options, index=0)

    tabs = st.tabs(["Overview", "Backtest", "Walk-Forward", "Hybrid"])
    with tabs[0]:
        if selected_backtest is not None:
            st.caption(f"Backtest run: `{selected_backtest}`")
            _render_comparison_block("Backtest Comparison", backtest_comparison)
        if selected_walkforward is not None:
            st.caption(f"Walk-forward run: `{selected_walkforward}`")
            _render_comparison_block("Walk-Forward Comparison", walkforward_comparison)
    with tabs[1]:
        if selected_backtest is None:
            st.info("No backtest run selected.")
        else:
            _render_backtest_strategy_details(selected_backtest, selected_strategy)
    with tabs[2]:
        if selected_walkforward is None:
            st.info("No walk-forward run selected.")
        else:
            _render_walkforward_strategy_details(selected_walkforward, selected_strategy)
    with tabs[3]:
        _render_hybrid_block(hybrid_runs, selected_hybrid)


if __name__ == "__main__":
    main()
