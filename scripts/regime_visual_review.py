"""Generate replay-based visual review artifacts for manual regime validation.

Examples:
  python scripts/regime_visual_review.py --symbol NIFTY --timeframe 1d --start 2025-01-01 --end 2025-12-31
  python scripts/regime_visual_review.py --symbol NIFTY --timeframe 5m --days 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.store import DataStore
from src.regime import RegimeClassifier, RegimeThresholds
from src.regime.replay import replay_regimes_no_lookahead

REGIME_COLOR_MAP = {
    "low_vol_trending": "#2a9d8f",
    "low_vol_ranging": "#457b9d",
    "high_vol_trending": "#e76f51",
    "high_vol_choppy": "#f4a261",
    "unknown": "#8d99ae",
}


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual replay artifacts for regime review.")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol partition")
    parser.add_argument("--timeframe", default="1d", help="Candle timeframe partition")
    parser.add_argument("--start", type=parse_date, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=0, help="Shortcut: end=now, start=end-days")
    parser.add_argument(
        "--indicator-warmup-days",
        type=int,
        default=0,
        help="Extra days loaded before analysis start for indicator warmup (excluded from output).",
    )
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    parser.add_argument("--output-dir", default="data/reports", help="Directory for output artifacts")
    parser.add_argument("--run-name", default="regime_review", help="Prefix for output filenames")
    parser.add_argument(
        "--no-timestamp-subdir",
        action="store_true",
        help="Write artifacts directly in --output-dir instead of run timestamp subfolder.",
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    settings_path = REPO_ROOT / path
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text())


def resolve_windows(args: argparse.Namespace) -> tuple[datetime | None, datetime | None, datetime | None]:
    analysis_start: datetime | None
    end: datetime | None
    if args.days and args.days > 0:
        end = datetime.now(UTC).replace(tzinfo=None)
        analysis_start = end - timedelta(days=args.days)
    else:
        analysis_start = args.start
        end = args.end

    load_start = analysis_start
    if analysis_start is not None and args.indicator_warmup_days > 0:
        load_start = analysis_start - timedelta(days=args.indicator_warmup_days)
    return load_start, analysis_start, end


def build_visual_review_frame(*, candles: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Merge candles with replay snapshots for manual visual review."""
    candles_frame = candles.copy()
    candles_frame["timestamp"] = pd.to_datetime(candles_frame["timestamp"], errors="coerce")
    candles_frame = candles_frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    candles_frame["close"] = pd.to_numeric(candles_frame.get("close"), errors="coerce")

    if snapshots.empty:
        out = candles_frame[["timestamp", "close"]].copy()
        out["regime"] = "unknown"
        return out

    snap = snapshots.copy()
    snap["timestamp"] = pd.to_datetime(snap["timestamp"], errors="coerce")
    keep_cols = [
        "timestamp",
        "regime",
        "india_vix",
        "adx_14",
        "pcr",
        "fii_net_3d",
        "vix_change_5d",
        "iv_surface_parallel_shift",
        "iv_surface_tilt_change",
    ]
    keep_cols = [col for col in keep_cols if col in snap.columns]
    snap = snap[keep_cols].dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out = candles_frame.merge(snap, on="timestamp", how="left")
    out["regime"] = out.get("regime", "unknown").fillna("unknown").astype(str)
    return out


def _plot_regime_review(frame: pd.DataFrame, *, symbol: str, timeframe: str, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax_price, ax_signal = axes

    ax_price.plot(frame["timestamp"], frame["close"], color="#222222", linewidth=1.4, label=f"{symbol} close")
    for regime, part in frame.groupby("regime"):
        color = REGIME_COLOR_MAP.get(regime, REGIME_COLOR_MAP["unknown"])
        ax_price.scatter(part["timestamp"], part["close"], s=10, color=color, alpha=0.75, label=regime)
    ax_price.set_title(f"{symbol} {timeframe} replay regimes")
    ax_price.set_ylabel("Close")
    ax_price.grid(alpha=0.2)
    handles, labels = ax_price.get_legend_handles_labels()
    dedup_labels: list[str] = []
    dedup_handles = []
    for handle, label in zip(handles, labels):
        if label in dedup_labels:
            continue
        dedup_labels.append(label)
        dedup_handles.append(handle)
    ax_price.legend(dedup_handles, dedup_labels, loc="upper left", ncol=3, fontsize=8)

    if "india_vix" in frame.columns:
        ax_signal.plot(frame["timestamp"], frame["india_vix"], color="#1d3557", linewidth=1.2, label="VIX")
    if "adx_14" in frame.columns:
        ax_signal.plot(frame["timestamp"], frame["adx_14"], color="#e63946", linewidth=1.0, label="ADX14")
    ax_signal.set_ylabel("Signal")
    ax_signal.grid(alpha=0.2)
    ax_signal.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _render_html(*, review: pd.DataFrame, transitions: pd.DataFrame, chart_path: Path, out_path: Path):
    regime_counts = review["regime"].value_counts().to_dict() if "regime" in review.columns else {}
    counts_rows = "".join(
        f"<tr><td>{regime}</td><td>{int(count)}</td></tr>" for regime, count in sorted(regime_counts.items())
    )
    transition_table = transitions.to_html(index=False, border=1) if not transitions.empty else "<p>No transitions.</p>"

    html = (
        "<html><head><title>Regime Visual Review</title></head><body>"
        "<h1>Regime Visual Review</h1>"
        f"<p>Rows: {len(review)}</p>"
        "<h2>Chart</h2>"
        f"<img src='{chart_path.name}' style='max-width:100%;height:auto;'/>"
        "<h2>Regime Distribution</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Regime</th><th>Rows</th></tr>"
        f"{counts_rows}"
        "</table>"
        "<h2>Transitions</h2>"
        f"{transition_table}"
        "</body></html>"
    )
    out_path.write_text(html)


def _write_checklist(*, review_csv: Path, chart_png: Path, html_report: Path, out_path: Path):
    content = "\n".join(
        [
            "# Regime Visual Review Checklist",
            "",
            "Artifacts:",
            f"- Review CSV: `{review_csv}`",
            f"- Chart PNG: `{chart_png}`",
            f"- HTML report: `{html_report}`",
            "",
            "Manual review (check each item):",
            "- [ ] Regime labels align with major price behavior shifts.",
            "- [ ] High-vol regimes overlap with visibly unstable/high-amplitude moves.",
            "- [ ] Low-vol ranging periods align with compressed price action.",
            "- [ ] Transition clusters are plausible (not excessive flip-flopping).",
            "- [ ] Any suspicious windows are documented before strategy comparisons.",
        ]
    )
    out_path.write_text(content)


def resolve_output_dir(*, raw_output_dir: str, run_prefix: str, no_timestamp_subdir: bool) -> Path:
    base = Path(raw_output_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    if no_timestamp_subdir:
        return base
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return base / f"{run_prefix}_{stamp}"


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    load_start, analysis_start, end = resolve_windows(args)

    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")
    output_dir = resolve_output_dir(
        raw_output_dir=args.output_dir,
        run_prefix=f"{args.run_name}_{args.symbol}_{args.timeframe}",
        no_timestamp_subdir=args.no_timestamp_subdir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    store = DataStore(base_dir=str(cache_dir))
    candles = store.read_time_series("candles", symbol=args.symbol, timeframe=args.timeframe, start=load_start, end=end)
    if candles.empty:
        print("No candle data available for requested window.")
        return 1

    vix = store.read_time_series("vix", symbol="INDIAVIX", timeframe="1d", start=load_start, end=end)
    fii = store.read_time_series(
        "fii_dii",
        symbol="NSE",
        timeframe="1d",
        start=load_start,
        end=end,
        timestamp_col="date",
    )

    classifier = RegimeClassifier(thresholds=RegimeThresholds.from_config(settings.get("regime", {})))
    replay = replay_regimes_no_lookahead(
        candles=candles,
        classifier=classifier,
        vix_df=vix,
        fii_df=fii,
        analysis_start=analysis_start,
    )
    candles_for_review = candles.copy()
    if analysis_start is not None:
        candles_for_review["timestamp"] = pd.to_datetime(candles_for_review["timestamp"], errors="coerce")
        candles_for_review = candles_for_review.loc[candles_for_review["timestamp"] >= pd.Timestamp(analysis_start)]
    review = build_visual_review_frame(candles=candles_for_review, snapshots=replay.snapshots)

    prefix = f"{args.run_name}_{args.symbol}_{args.timeframe}"
    review_csv = output_dir / f"{prefix}_review.csv"
    transitions_csv = output_dir / f"{prefix}_transitions.csv"
    chart_png = output_dir / f"{prefix}_chart.png"
    html_report = output_dir / f"{prefix}_report.html"
    checklist_md = output_dir / f"{prefix}_checklist.md"
    meta_json = output_dir / f"{prefix}_meta.json"

    review.to_csv(review_csv, index=False)
    replay.transitions.to_csv(transitions_csv, index=False)
    _plot_regime_review(review, symbol=args.symbol, timeframe=args.timeframe, out_path=chart_png)
    _render_html(review=review, transitions=replay.transitions, chart_path=chart_png, out_path=html_report)
    _write_checklist(review_csv=review_csv, chart_png=chart_png, html_report=html_report, out_path=checklist_md)

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "window": {
            "load_start": load_start.isoformat() if load_start else None,
            "analysis_start": analysis_start.isoformat() if analysis_start else None,
            "end": end.isoformat() if end else None,
        },
        "indicator_warmup_days": int(args.indicator_warmup_days),
        "rows": int(len(review)),
        "transitions": int(len(replay.transitions)),
        "files": {
            "review_csv": str(review_csv.relative_to(REPO_ROOT)),
            "transitions_csv": str(transitions_csv.relative_to(REPO_ROOT)),
            "chart_png": str(chart_png.relative_to(REPO_ROOT)),
            "html_report": str(html_report.relative_to(REPO_ROOT)),
            "checklist_md": str(checklist_md.relative_to(REPO_ROOT)),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2, sort_keys=True))

    print("=" * 72)
    print("Regime Visual Review Artifact")
    print("=" * 72)
    print(
        f"symbol={args.symbol} timeframe={args.timeframe} "
        f"rows={len(review)} transitions={len(replay.transitions)}"
    )
    print(f"load_window={load_start.date() if load_start else 'begin'} -> {end.date() if end else 'latest'}")
    print(f"analysis_window={analysis_start.date() if analysis_start else 'begin'} -> {end.date() if end else 'latest'}")
    print(f"indicator_warmup_days={args.indicator_warmup_days}")
    print(f"output_dir={output_dir.relative_to(REPO_ROOT)}")
    print(f"review_csv={review_csv.relative_to(REPO_ROOT)}")
    print(f"transitions_csv={transitions_csv.relative_to(REPO_ROOT)}")
    print(f"chart_png={chart_png.relative_to(REPO_ROOT)}")
    print(f"html_report={html_report.relative_to(REPO_ROOT)}")
    print(f"checklist_md={checklist_md.relative_to(REPO_ROOT)}")
    print(f"meta_json={meta_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
