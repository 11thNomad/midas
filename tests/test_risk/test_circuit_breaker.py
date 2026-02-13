from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from src.risk.circuit_breaker import BreakerState, CircuitBreaker


def test_circuit_breaker_trips_on_daily_loss():
    breaker = CircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=2.0, max_drawdown_pct=20.0)
    breaker.update(current_equity=97000.0, realized_pnl_today=-2500.0, open_positions=1)
    assert breaker.state == BreakerState.TRIPPED_DAILY
    assert breaker.can_trade() is False


def test_circuit_breaker_trips_on_drawdown():
    breaker = CircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=10.0, max_drawdown_pct=5.0)
    breaker.update(current_equity=94000.0, realized_pnl_today=-1000.0, open_positions=1)
    assert breaker.state == BreakerState.TRIPPED_DRAWDOWN
    assert breaker.can_trade() is False


def test_circuit_breaker_warning_and_position_limit():
    breaker = CircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=5.0, max_open_positions=1)
    breaker.update(current_equity=97000.0, realized_pnl_today=-3500.0, open_positions=1)
    assert breaker.state in (BreakerState.WARNING, BreakerState.TRIPPED_DAILY, BreakerState.TRIPPED_DRAWDOWN)
    assert breaker.can_add_position() is False


def test_circuit_breaker_transitions_warning_then_trip():
    breaker = CircuitBreaker(
        initial_capital=100000.0,
        max_daily_loss_pct=10.0,
        max_drawdown_pct=50.0,
        warning_threshold_pct=70.0,
    )
    breaker.update(current_equity=94000.0, realized_pnl_today=-7000.0, open_positions=1)
    assert breaker.state == BreakerState.WARNING
    breaker.update(current_equity=89000.0, realized_pnl_today=-11000.0, open_positions=1)
    assert breaker.state == BreakerState.TRIPPED_DAILY


def test_circuit_breaker_reset_requires_confirmation():
    breaker = CircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=2.0)
    breaker.update(current_equity=97000.0, realized_pnl_today=-2500.0, open_positions=1)
    assert breaker.state == BreakerState.TRIPPED_DAILY
    breaker.reset(confirm=False)
    assert breaker.state == BreakerState.TRIPPED_DAILY
    breaker.reset(confirm=True)
    assert breaker.state == BreakerState.NORMAL


def test_circuit_breaker_auto_resets_daily_trip_on_new_day():
    breaker = CircuitBreaker(
        initial_capital=100000.0,
        max_daily_loss_pct=5.0,
        max_drawdown_pct=25.0,
        auto_reset_daily_trip=True,
    )
    breaker.state = BreakerState.TRIPPED_DAILY
    breaker.daily_pnl.date = breaker._today() - timedelta(days=1)
    breaker.update(current_equity=99500.0, realized_pnl_today=-500.0, open_positions=0)
    assert breaker.state == BreakerState.NORMAL


def test_circuit_breaker_daily_trip_requires_manual_reset_by_default():
    breaker = CircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=5.0, max_drawdown_pct=25.0)
    breaker.state = BreakerState.TRIPPED_DAILY
    breaker.daily_pnl.date = breaker._today() - timedelta(days=1)
    breaker.update(current_equity=99500.0, realized_pnl_today=-500.0, open_positions=0)
    assert breaker.state == BreakerState.TRIPPED_DAILY


def test_circuit_breaker_warning_clears_when_loss_recovers():
    breaker = CircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=10.0, warning_threshold_pct=70.0)
    breaker.update(current_equity=94000.0, realized_pnl_today=-7000.0, open_positions=1)
    assert breaker.state == BreakerState.WARNING
    breaker.update(current_equity=97000.0, realized_pnl_today=-3000.0, open_positions=1)
    assert breaker.state == BreakerState.NORMAL


def test_circuit_breaker_persists_and_restores_state(tmp_path):
    state_file = tmp_path / "breaker_state.json"
    breaker = CircuitBreaker(
        initial_capital=100000.0,
        max_daily_loss_pct=2.0,
        state_path=str(state_file),
    )
    breaker.update(current_equity=97000.0, realized_pnl_today=-2500.0, open_positions=1)
    assert breaker.state == BreakerState.TRIPPED_DAILY

    restored = CircuitBreaker(
        initial_capital=100000.0,
        max_daily_loss_pct=2.0,
        state_path=str(state_file),
    )
    assert restored.state == BreakerState.TRIPPED_DAILY
    assert restored.daily_pnl.realized == -2500.0
    assert restored.open_position_count == 1


def test_circuit_breaker_concurrent_updates_keep_valid_state():
    breaker = CircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=3.0, max_drawdown_pct=20.0)

    def _update(i: int):
        breaker.update(
            current_equity=100000.0 - float(i * 10),
            realized_pnl_today=-float(i * 5),
            open_positions=i % 3,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_update, range(50)))

    assert breaker.state in {
        BreakerState.NORMAL,
        BreakerState.WARNING,
        BreakerState.TRIPPED_DAILY,
        BreakerState.TRIPPED_DRAWDOWN,
    }


def test_circuit_breaker_uses_provided_timestamp_for_daily_rollover():
    breaker = CircuitBreaker(
        initial_capital=100000.0,
        max_daily_loss_pct=5.0,
        max_drawdown_pct=25.0,
        auto_reset_daily_trip=True,
    )
    breaker.state = BreakerState.TRIPPED_DAILY

    day_one = datetime(2026, 1, 5, 10, 0, 0)
    day_two = datetime(2026, 1, 6, 10, 0, 0)
    breaker.daily_pnl.date = day_one.date()

    breaker.update(
        current_equity=99500.0,
        realized_pnl_today=-500.0,
        open_positions=0,
        timestamp=day_one,
    )
    assert breaker.state == BreakerState.TRIPPED_DAILY

    breaker.update(
        current_equity=99800.0,
        realized_pnl_today=-100.0,
        open_positions=0,
        timestamp=day_two,
    )
    assert breaker.daily_pnl.date == day_two.date()
    assert breaker.state == BreakerState.NORMAL
