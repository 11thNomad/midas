from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType
from src.strategies.router import StrategyRouter


def _load_run_paper_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "run_paper.py"
    spec = importlib.util.spec_from_file_location("run_paper_module", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_paper_module = _load_run_paper_module()
build_strategies = run_paper_module.build_strategies


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


def test_kite_available_cash_ignores_utilised_only_payload():
    class _DummyKite:
        @staticmethod
        def margins():
            return {"equity": {"available": {"utilised": 250000}}}

    class _DummyFeed:
        _kite = _DummyKite()

    assert run_paper_module._kite_available_cash(_DummyFeed()) is None


def test_wednesday_watchdog_trigger_uses_ist_time():
    # 2026-01-07 09:55 UTC == 15:25 IST (Wednesday).
    assert run_paper_module._is_wednesday_watchdog_trigger(datetime(2026, 1, 7, 9, 55))
    assert not run_paper_module._is_wednesday_watchdog_trigger(datetime(2026, 1, 7, 9, 54))


def test_lot_size_by_underlying_reads_enabled_option_strategies():
    settings = {
        "strategies": {
            "iron_condor": {"enabled": True, "instrument": "NIFTY", "lot_size": 75},
            "jade_lizard": {"enabled": True, "instrument": "BANKNIFTY", "lot_size": 30},
            "momentum": {"enabled": True, "instrument": "NIFTY"},
        }
    }
    out = run_paper_module._lot_size_by_underlying(settings)
    assert out == {"NIFTY": 75, "BANKNIFTY": 30}


def test_rollback_unfilled_entry_state_clears_strategy_position():
    class _DummyStrategy(BaseStrategy):
        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                strategy_name=self.name,
                instrument="NIFTY",
                timestamp=market_data.get("timestamp", datetime(2026, 1, 1)),
                regime=regime,
            )

        def get_exit_conditions(self, market_data: dict) -> Signal | None:
            return None

        def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
            return 1

    strategy = _DummyStrategy(
        name="dummy",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
    strategy.state.current_position = {"side": "LONG", "quantity": 1}
    router = StrategyRouter(strategies=[strategy])
    entry_signal = Signal(
        signal_type=SignalType.ENTRY_LONG,
        strategy_name="dummy",
        instrument="NIFTY",
        timestamp=datetime(2026, 1, 2, 9, 15),
        orders=[{"symbol": "NIFTY", "action": "BUY", "quantity": 1}],
        regime=RegimeState.LOW_VOL_RANGING,
    )

    run_paper_module._rollback_unfilled_entry_state(
        router=router,
        signals=[entry_signal],
        fills=[],
    )
    assert strategy.state.current_position is None


def test_clear_strategy_positions_for_system_fills_handles_expiry_settlement():
    class _DummyStrategy(BaseStrategy):
        def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                strategy_name=self.name,
                instrument="NIFTY",
                timestamp=market_data.get("timestamp", datetime(2026, 1, 1)),
                regime=regime,
            )

        def get_exit_conditions(self, market_data: dict) -> Signal | None:
            return None

        def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
            return 1

    strategy = _DummyStrategy(
        name="dummy",
        config={"instrument": "NIFTY", "active_regimes": [RegimeState.LOW_VOL_RANGING.value]},
    )
    strategy.state.current_position = {"side": "SHORT", "quantity": 50}
    router = StrategyRouter(strategies=[strategy])
    fills = [{"strategy_name": "expiry_settlement", "instrument": "NIFTY_20260108_22000CE"}]

    run_paper_module._clear_strategy_positions_for_system_fills(router, fills)
    assert strategy.state.current_position is None


def test_freshness_issues_uses_ist_age_with_utc_loop_time():
    now_utc_naive = datetime(2026, 1, 7, 9, 30)  # 15:00 IST
    candles = pd.DataFrame({"timestamp": [datetime(2026, 1, 7, 14, 55)]})  # 14:55 IST
    vix = pd.DataFrame({"timestamp": [datetime(2026, 1, 7, 14, 55)]})
    settings = {
        "ops": {
            "freshness": {
                "candles_runtime_max_age_minutes": 4,
                "vix_1d_max_age_minutes": 10_000,
            }
        }
    }

    issues = run_paper_module._freshness_issues(
        candles=candles,
        vix=vix,
        now=now_utc_naive,
        settings=settings,
    )
    assert any(issue.startswith("candles_stale_") for issue in issues)
