from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_builder():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "regime_visual_review.py"
    spec = importlib.util.spec_from_file_location("regime_visual_review_module", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_visual_review_frame


build_visual_review_frame = _load_builder()


def test_build_visual_review_frame_merges_snapshot_fields():
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-02-10", "2026-02-11"]),
            "close": [22000.0, 22100.0],
        }
    )
    snapshots = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-02-10", "2026-02-11"]),
            "regime": ["low_vol_ranging", "high_vol_trending"],
            "india_vix": [13.5, 18.2],
            "adx_14": [19.0, 27.0],
        }
    )

    review = build_visual_review_frame(candles=candles, snapshots=snapshots)
    assert len(review) == 2
    assert list(review["regime"]) == ["low_vol_ranging", "high_vol_trending"]
    assert float(review.loc[0, "india_vix"]) == 13.5


def test_build_visual_review_frame_defaults_unknown_without_snapshots():
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-02-10", "2026-02-11"]),
            "close": [22000.0, 22100.0],
        }
    )

    review = build_visual_review_frame(candles=candles, snapshots=pd.DataFrame())
    assert len(review) == 2
    assert set(review["regime"].tolist()) == {"unknown"}
