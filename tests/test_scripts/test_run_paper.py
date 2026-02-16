from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_build_strategies():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "run_paper.py"
    spec = importlib.util.spec_from_file_location("run_paper_module", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_strategies


build_strategies = _load_build_strategies()


def test_build_strategies_raises_for_unknown_enabled_strategy():
    settings = {
        "strategies": {
            "iron_condor": {"enabled": True, "active_regimes": ["low_vol_ranging"]},
            "mystery_alpha": {"enabled": True},
        }
    }

    with pytest.raises(ValueError) as exc:
        build_strategies(settings)

    assert "mystery_alpha" in str(exc.value)
    assert "Known strategy ids" in str(exc.value)


def test_build_strategies_skips_unknown_disabled_strategy():
    settings = {
        "strategies": {
            "iron_condor": {"enabled": True, "active_regimes": ["low_vol_ranging"]},
            "mystery_alpha": {"enabled": False},
        }
    }

    strategies = build_strategies(settings)
    assert len(strategies) == 1
    assert strategies[0].name == "iron_condor"


def test_build_strategies_includes_jade_lizard_when_enabled():
    settings = {
        "strategies": {
            "jade_lizard": {"enabled": True, "active_regimes": ["low_vol_ranging"]},
            "iron_condor": {"enabled": True, "active_regimes": ["low_vol_ranging"]},
        }
    }

    strategies = build_strategies(settings)
    names = sorted(strategy.name for strategy in strategies)
    assert names == ["iron_condor", "jade_lizard"]


def test_build_strategies_includes_baseline_trend_when_enabled():
    settings = {
        "strategies": {
            "baseline_trend": {
                "enabled": True,
                "active_regimes": ["low_vol_trending", "high_vol_trending"],
                "adx_min": 25.0,
            },
            "iron_condor": {"enabled": True, "active_regimes": ["low_vol_ranging"]},
        }
    }

    strategies = build_strategies(settings)
    names = sorted(strategy.name for strategy in strategies)
    assert names == ["baseline_trend", "iron_condor"]
