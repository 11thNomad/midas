from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.vectorbt_dashboard_data import (
    list_vectorbt_paramset_runs,
    load_detail_artifact,
    load_walkforward_folds,
    parse_artifact_filename,
    parse_fold_filename,
)


def test_list_vectorbt_paramset_runs_returns_latest_first(tmp_path: Path):
    (tmp_path / "vectorbt_paramsets_20260101_010101").mkdir()
    (tmp_path / "vectorbt_paramsets_20260102_010101").mkdir()
    (tmp_path / "vectorbt_20260102_010101").mkdir()
    names = [p.name for p in list_vectorbt_paramset_runs(tmp_path)]
    assert names == [
        "vectorbt_paramsets_20260102_010101",
        "vectorbt_paramsets_20260101_010101",
    ]


def test_parse_fold_filename_uses_known_set_and_profile(tmp_path: Path):
    path = tmp_path / "aggressive_trend_stress_walkforward_folds.csv"
    parsed = parse_fold_filename(
        csv_path=path,
        set_ids=["aggressive_trend", "baseline_trend"],
        fee_profiles=["base", "stress"],
    )
    assert parsed == ("aggressive_trend", "stress")


def test_load_walkforward_folds_attaches_set_and_profile(tmp_path: Path):
    run_dir = tmp_path / "vectorbt_paramsets_20260101_010101"
    run_dir.mkdir()
    frame = pd.DataFrame(
        {
            "fold": [1, 2],
            "total_return_pct": [1.0, -0.5],
        }
    )
    frame.to_csv(run_dir / "baseline_trend_base_walkforward_folds.csv", index=False)
    out = load_walkforward_folds(
        run_dir,
        set_ids=["baseline_trend"],
        fee_profiles=["base"],
    )
    assert len(out) == 2
    assert set(out["set_id"]) == {"baseline_trend"}
    assert set(out["fee_profile"]) == {"base"}


def test_parse_artifact_filename_for_trade_attribution(tmp_path: Path):
    path = tmp_path / "baseline_trend_base_trade_attribution.csv"
    parsed = parse_artifact_filename(
        csv_path=path,
        set_ids=["baseline_trend"],
        fee_profiles=["base", "stress"],
        suffix="trade_attribution",
    )
    assert parsed == ("baseline_trend", "base")


def test_load_detail_artifact_reads_from_details_subdir(tmp_path: Path):
    run_dir = tmp_path / "vectorbt_paramsets_20260101_010101"
    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True)
    frame = pd.DataFrame({"pnl": [100.0, -50.0]})
    frame.to_csv(details_dir / "baseline_trend_base_trade_attribution.csv", index=False)
    out = load_detail_artifact(
        run_dir,
        set_ids=["baseline_trend"],
        fee_profiles=["base"],
        suffix="trade_attribution",
    )
    assert len(out) == 2
    assert set(out["set_id"]) == {"baseline_trend"}
    assert set(out["fee_profile"]) == {"base"}


def test_load_walkforward_folds_reads_root_even_when_details_exists(tmp_path: Path):
    run_dir = tmp_path / "vectorbt_paramsets_20260101_010101"
    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True)
    pd.DataFrame({"pnl": [1.0]}).to_csv(
        details_dir / "baseline_trend_base_trade_attribution.csv",
        index=False,
    )
    pd.DataFrame({"fold": [1], "total_return_pct": [0.5]}).to_csv(
        run_dir / "baseline_trend_base_walkforward_folds.csv",
        index=False,
    )
    out = load_walkforward_folds(
        run_dir,
        set_ids=["baseline_trend"],
        fee_profiles=["base"],
    )
    assert len(out) == 1
    assert float(out.iloc[0]["total_return_pct"]) == 0.5


def test_load_walkforward_folds_skips_malformed_csv_files(tmp_path: Path):
    run_dir = tmp_path / "vectorbt_paramsets_20260101_010101"
    run_dir.mkdir(parents=True)
    bad_path = run_dir / "baseline_trend_base_walkforward_folds.csv"
    bad_path.write_text("")
    out = load_walkforward_folds(
        run_dir,
        set_ids=["baseline_trend"],
        fee_profiles=["base"],
    )
    assert out.empty
