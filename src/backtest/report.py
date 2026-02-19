"""Report writers for backtest outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult
from src.signals.option_chain_features import option_feature_artifact_from_snapshots


def write_backtest_report(
    result: BacktestResult,
    *,
    output_dir: str,
    run_name: str = "backtest",
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{run_name}_metrics.json"
    fills_path = out_dir / f"{run_name}_fills.csv"
    equity_path = out_dir / f"{run_name}_equity.csv"
    regimes_path = out_dir / f"{run_name}_regimes.csv"
    signal_snapshots_path = out_dir / f"{run_name}_signal_snapshots.csv"
    option_features_path = out_dir / f"{run_name}_option_features.csv"
    html_path = out_dir / f"{run_name}_report.html"

    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True))
    _write_csv(result.fills, fills_path)
    _write_csv(result.equity_curve, equity_path)
    _write_csv(result.regimes, regimes_path)
    _write_csv(result.signal_snapshots, signal_snapshots_path)
    _write_csv(
        option_feature_artifact_from_snapshots(result.signal_snapshots),
        option_features_path,
    )
    html_path.write_text(_render_html(result.metrics))

    return {
        "metrics_json": str(metrics_path),
        "fills_csv": str(fills_path),
        "equity_csv": str(equity_path),
        "regimes_csv": str(regimes_path),
        "signal_snapshots_csv": str(signal_snapshots_path),
        "option_features_csv": str(option_features_path),
        "html_report": str(html_path),
    }


def write_walkforward_report(
    *,
    folds: pd.DataFrame,
    summary: dict[str, float],
    regime_table: pd.DataFrame | None = None,
    output_dir: str,
    run_name: str = "walkforward",
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_csv = out_dir / f"{run_name}_folds.csv"
    summary_json = out_dir / f"{run_name}_summary.json"
    html_path = out_dir / f"{run_name}_report.html"

    folds.to_csv(folds_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    html_path.write_text(
        _render_walkforward_html(folds=folds, summary=summary, regime_table=regime_table)
    )

    return {
        "folds_csv": str(folds_csv),
        "summary_json": str(summary_json),
        "html_report": str(html_path),
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def _render_html(metrics: dict[str, float]) -> str:
    rows = "\n".join(f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in sorted(metrics.items()))
    return (
        "<html><head><title>Backtest Report</title></head><body>"
        "<h1>Backtest Summary</h1>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Metric</th><th>Value</th></tr>"
        f"{rows}"
        "</table></body></html>"
    )


def _render_walkforward_html(
    *,
    folds: pd.DataFrame,
    summary: dict[str, float],
    regime_table: pd.DataFrame | None = None,
) -> str:
    summary_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in sorted(summary.items())
    )
    folds_table = (
        folds.to_html(index=False, border=1) if not folds.empty else "<p>No fold rows.</p>"
    )
    regime_section = "<p>No regime-segmented table.</p>"
    if regime_table is not None and not regime_table.empty:
        regime_section = regime_table.to_html(index=False, border=1)

    return (
        "<html><head><title>Walk-Forward Report</title></head><body>"
        "<h1>Walk-Forward Summary</h1>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Metric</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<h2>Fold Metrics</h2>"
        f"{folds_table}"
        "<h2>Regime-Segmented Returns</h2>"
        f"{regime_section}"
        "</body></html>"
    )
