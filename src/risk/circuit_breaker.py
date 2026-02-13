"""
Circuit Breaker — the last line of defense.

This module runs INDEPENDENTLY of strategy logic and has authority to:
- Halt all new order placement
- Close all open positions immediately
- Disable the system entirely until manual reset

The circuit breaker is NOT optional. It runs even if the strategy thinks
everything is fine. Think of it as the fuse box in your house — it doesn't
care what appliance you're running, it just cuts power if the load is dangerous.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger()


class BreakerState(Enum):
    """Circuit breaker states."""

    NORMAL = "normal"  # All systems go
    WARNING = "warning"  # Approaching limits, reduce exposure
    TRIPPED_DAILY = "tripped_daily"  # Daily loss limit hit — no new trades today
    TRIPPED_DRAWDOWN = "tripped_dd"  # Max drawdown hit — full shutdown
    KILLED = "killed"  # Manual kill switch activated


@dataclass
class DailyPnL:
    """Track P&L for a single trading day."""

    date: date
    realized: float = 0.0
    unrealized: float = 0.0
    peak_equity: float = 0.0
    trades_today: int = 0

    @property
    def total(self) -> float:
        return self.realized + self.unrealized


@dataclass
class CircuitBreaker:
    """
    Monitors portfolio health and halts trading when limits are breached.

    Usage:
        breaker = CircuitBreaker(
            initial_capital=500_000,
            max_daily_loss_pct=3.0,
            max_drawdown_pct=15.0,
        )

        # On every tick / position update:
        breaker.update(current_equity=485_000, realized_pnl=-8_000)

        # Before placing any order:
        if breaker.can_trade():
            execute_order(...)
        else:
            log.warning("Circuit breaker active", state=breaker.state)
    """

    initial_capital: float
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    max_open_positions: int = 4
    warning_threshold_pct: float = 70.0  # Warn at 70% of daily limit
    trading_timezone: str = "Asia/Kolkata"
    auto_reset_daily_trip: bool = False
    state_path: str | None = None

    # Internal state
    state: BreakerState = BreakerState.NORMAL
    peak_equity: float = 0.0
    current_equity: float = 0.0
    daily_pnl: DailyPnL = field(default_factory=lambda: DailyPnL(date=date.today()))
    open_position_count: int = 0
    trip_history: list[dict] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __post_init__(self):
        with self._lock:
            self.peak_equity = self.initial_capital
            self.current_equity = self.initial_capital
            loaded = self._load_state()
            if not loaded:
                self.daily_pnl.date = self._today()

    # === Core API ===

    def can_trade(self) -> bool:
        """Should the system place new orders? The single most important method."""
        with self._lock:
            return self.state in (BreakerState.NORMAL, BreakerState.WARNING)

    def can_add_position(self) -> bool:
        """Can we open additional positions?"""
        with self._lock:
            if self.state not in (BreakerState.NORMAL, BreakerState.WARNING):
                return False
            if self.open_position_count >= self.max_open_positions:
                logger.warning(
                    "Max positions reached",
                    current=self.open_position_count,
                    max=self.max_open_positions,
                )
                return False
            return True

    def update(
        self,
        current_equity: float,
        realized_pnl_today: float,
        unrealized_pnl: float = 0.0,
        open_positions: int = 0,
        timestamp: datetime | None = None,
    ):
        """
        Called on every tick or position update.

        This is the heartbeat of risk management. Call it frequently.
        """
        with self._lock:
            self.current_equity = float(current_equity)
            self.open_position_count = int(open_positions)

            # Reset daily tracking if new day in trading timezone.
            # In historical backtests, pass bar timestamp so daily limits reset per bar-date.
            today = self._today(as_of=timestamp)
            if today != self.daily_pnl.date:
                self._reset_daily(today=today)

            self.daily_pnl.realized = float(realized_pnl_today)
            self.daily_pnl.unrealized = float(unrealized_pnl)

            # Update peak equity (for drawdown calculation).
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity

            # If manually killed, stay killed.
            if self.state == BreakerState.KILLED:
                self._save_state()
                return

            # Check limits in order of severity.
            self._check_drawdown()
            self._check_daily_loss()
            self._check_warning()
            self._save_state()

    def kill(self, reason: str = "Manual kill switch"):
        """Emergency stop. Requires manual reset to resume."""
        with self._lock:
            self.state = BreakerState.KILLED
            self._log_trip("KILLED", reason)
            logger.critical("KILL SWITCH ACTIVATED", reason=reason)
            self._save_state()

    def reset(self, confirm: bool = False):
        """
        Manual reset after a trip. Requires explicit confirmation.
        Call this only after reviewing what caused the trip.
        """
        with self._lock:
            if not confirm:
                logger.warning("Reset requires confirm=True. Review trip_history first.")
                return

            old_state = self.state
            self.state = BreakerState.NORMAL
            self.peak_equity = self.current_equity
            logger.info("Circuit breaker reset", from_state=old_state.value)
            self._save_state()

    # === Internal Checks ===

    def _check_daily_loss(self):
        """Trip if daily loss exceeds limit."""
        if self.state in (BreakerState.TRIPPED_DRAWDOWN, BreakerState.KILLED):
            return  # Already in a worse state

        max_daily_loss = self.initial_capital * (self.max_daily_loss_pct / 100)
        if (
            abs(self.daily_pnl.total) >= max_daily_loss
            and self.daily_pnl.total < 0
            and self.state != BreakerState.TRIPPED_DAILY
        ):
            self.state = BreakerState.TRIPPED_DAILY
            self._log_trip(
                "DAILY_LOSS",
                f"Daily loss {self.daily_pnl.total:.0f} exceeds limit {max_daily_loss:.0f}",
            )
            logger.error(
                "Daily loss limit breached",
                daily_pnl=self.daily_pnl.total,
                limit=max_daily_loss,
            )

    def _check_drawdown(self):
        """Trip if drawdown from peak exceeds limit."""
        if self.state == BreakerState.KILLED:
            return

        if self.peak_equity == 0:
            return

        drawdown_pct = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100

        if drawdown_pct >= self.max_drawdown_pct and self.state != BreakerState.TRIPPED_DRAWDOWN:
            self.state = BreakerState.TRIPPED_DRAWDOWN
            self._log_trip(
                "MAX_DRAWDOWN",
                f"Drawdown {drawdown_pct:.1f}% exceeds limit {self.max_drawdown_pct}%",
            )
            logger.critical(
                "Max drawdown breached — FULL SHUTDOWN",
                drawdown_pct=drawdown_pct,
                peak=self.peak_equity,
                current=self.current_equity,
            )

    def _check_warning(self):
        """Issue warning when approaching daily limit and clear when recovered."""
        if self.state in (
            BreakerState.TRIPPED_DAILY,
            BreakerState.TRIPPED_DRAWDOWN,
            BreakerState.KILLED,
        ):
            return

        max_daily_loss = self.initial_capital * (self.max_daily_loss_pct / 100)
        if max_daily_loss <= 0:
            return
        warning_threshold = max_daily_loss * (self.warning_threshold_pct / 100)
        loss = self.daily_pnl.total

        if loss < 0 and abs(loss) >= warning_threshold and abs(loss) < max_daily_loss:
            if self.state == BreakerState.NORMAL:
                self.state = BreakerState.WARNING
                logger.warning(
                    "Approaching daily loss limit",
                    daily_pnl=loss,
                    warning_at=warning_threshold,
                    limit=max_daily_loss,
                )
        elif self.state == BreakerState.WARNING and abs(loss) < warning_threshold:
            self.state = BreakerState.NORMAL
            logger.info(
                "Daily loss warning cleared",
                daily_pnl=loss,
                warning_at=warning_threshold,
            )

    def _reset_daily(self, *, today: date):
        """Reset daily tracking at start of new trading day."""
        self.daily_pnl = DailyPnL(date=today)
        # Optionally auto-reset daily trip only.
        if self.state == BreakerState.TRIPPED_DAILY and self.auto_reset_daily_trip:
            self.state = BreakerState.NORMAL
            logger.info("Daily circuit breaker auto-reset for new trading day")
        elif self.state == BreakerState.WARNING:
            self.state = BreakerState.NORMAL

    def _log_trip(self, trip_type: str, reason: str):
        """Record trip event for post-analysis."""
        self.trip_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": trip_type,
                "reason": reason,
                "equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "daily_pnl": self.daily_pnl.total,
                "state": self.state.value,
            }
        )

    # === Reporting ===

    def status(self) -> dict:
        """Current status snapshot for monitoring/dashboard."""
        with self._lock:
            drawdown_pct = 0.0
            if self.peak_equity > 0:
                drawdown_pct = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100

            max_daily_loss = self.initial_capital * (self.max_daily_loss_pct / 100)
            daily_loss_used_pct = 0.0
            if max_daily_loss > 0 and self.daily_pnl.total < 0:
                daily_loss_used_pct = (abs(self.daily_pnl.total) / max_daily_loss) * 100

            return {
                "state": self.state.value,
                "can_trade": self.state in (BreakerState.NORMAL, BreakerState.WARNING),
                "equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "drawdown_pct": round(drawdown_pct, 2),
                "max_drawdown_pct": self.max_drawdown_pct,
                "daily_pnl": round(self.daily_pnl.total, 2),
                "daily_loss_limit": round(max_daily_loss, 2),
                "daily_loss_used_pct": round(daily_loss_used_pct, 1),
                "open_positions": self.open_position_count,
                "max_positions": self.max_open_positions,
                "trips_total": len(self.trip_history),
            }

    def __repr__(self) -> str:
        drawdown_pct = self.status()["drawdown_pct"]
        return (
            f"<CircuitBreaker [{self.state.value}] "
            f"equity={self.current_equity:.0f} dd={drawdown_pct:.1f}%>"
        )

    def _today(self, *, as_of: datetime | None = None) -> date:
        try:
            zone = ZoneInfo(self.trading_timezone)
            if as_of is None:
                return datetime.now(zone).date()
            if as_of.tzinfo is None:
                return as_of.replace(tzinfo=zone).date()
            return as_of.astimezone(zone).date()
        except Exception:
            logger.warning(
                "Invalid trading timezone, falling back to UTC",
                trading_timezone=self.trading_timezone,
            )
            if as_of is None:
                return datetime.now(ZoneInfo("UTC")).date()
            if as_of.tzinfo is None:
                return as_of.date()
            return as_of.astimezone(ZoneInfo("UTC")).date()

    def _load_state(self) -> bool:
        if not self.state_path:
            return False
        path = Path(self.state_path)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to read circuit breaker state", path=str(path), error=str(exc))
            return False

        try:
            state_value = str(payload.get("state", self.state.value))
            self.state = BreakerState(state_value)
            self.peak_equity = float(payload.get("peak_equity", self.peak_equity))
            self.current_equity = float(payload.get("current_equity", self.current_equity))
            self.open_position_count = int(
                payload.get("open_position_count", self.open_position_count)
            )

            daily = payload.get("daily_pnl", {})
            daily_date_raw = daily.get("date")
            daily_date = date.fromisoformat(daily_date_raw) if daily_date_raw else self._today()
            self.daily_pnl = DailyPnL(
                date=daily_date,
                realized=float(daily.get("realized", 0.0)),
                unrealized=float(daily.get("unrealized", 0.0)),
                peak_equity=float(daily.get("peak_equity", 0.0)),
                trades_today=int(daily.get("trades_today", 0)),
            )
            history = payload.get("trip_history", [])
            if isinstance(history, list):
                self.trip_history = history
        except Exception as exc:
            logger.warning("Failed to parse circuit breaker state", path=str(path), error=str(exc))
            return False
        return True

    def _save_state(self):
        if not self.state_path:
            return
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": self.state.value,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "open_position_count": self.open_position_count,
            "daily_pnl": {
                "date": self.daily_pnl.date.isoformat(),
                "realized": self.daily_pnl.realized,
                "unrealized": self.daily_pnl.unrealized,
                "peak_equity": self.daily_pnl.peak_equity,
                "trades_today": self.daily_pnl.trades_today,
            },
            "trip_history": self.trip_history,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f"{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(payload, tmp, indent=2, sort_keys=True)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.warning(
                "Failed to persist circuit breaker state", path=str(path), error=str(exc)
            )
