"""
Abstract base class for all trading strategies.

Every strategy must implement this interface. This ensures:
- Consistent signal generation across strategies
- Uniform risk management integration
- Seamless swapping between paper and live execution
- Strategy-agnostic backtesting
"""

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Trading signal types."""

    NO_SIGNAL = "no_signal"
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT = "exit"
    ADJUST = "adjust"  # Modify existing position (e.g., roll options)


class RegimeState(Enum):
    """Market regime classifications."""

    LOW_VOL_TRENDING = "low_vol_trending"
    LOW_VOL_RANGING = "low_vol_ranging"
    HIGH_VOL_TRENDING = "high_vol_trending"
    HIGH_VOL_CHOPPY = "high_vol_choppy"
    UNKNOWN = "unknown"


@dataclass
class Signal:
    """A trading signal produced by a strategy."""

    signal_type: SignalType
    strategy_name: str
    instrument: str
    timestamp: datetime

    # What to trade (for options strategies, this includes legs)
    orders: list[dict[str, Any]] = field(default_factory=list)
    # Example order dict:
    # {
    #     "symbol": "NIFTY2430725500CE",
    #     "action": "SELL",
    #     "quantity": 50,
    #     "order_type": "LIMIT",
    #     "price": 120.50,
    # }

    # Context — logged with every trade for post-analysis
    regime: RegimeState = RegimeState.UNKNOWN
    confidence: float = 0.0  # 0.0 to 1.0
    greeks_snapshot: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, Any] = field(default_factory=dict)  # ADX, RSI, VIX, etc.
    reason: str = ""  # Human-readable entry/exit reason

    @property
    def is_actionable(self) -> bool:
        return self.signal_type != SignalType.NO_SIGNAL


@dataclass
class StrategyState:
    """Tracks the current state of a strategy instance."""

    name: str
    is_active: bool = True
    current_position: dict[str, Any] | None = None
    entry_time: datetime | None = None
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trade_count: int = 0
    consecutive_losses: int = 0


class BaseStrategy(ABC):
    """
    Abstract base class that all strategies must inherit from.

    Lifecycle:
        1. __init__() — load config, set parameters
        2. on_regime_change() — called when regime classifier detects a shift
        3. generate_signal() — called every tick/bar, produces Signal or NO_SIGNAL
        4. on_fill() — called when an order is filled
        5. on_exit() — cleanup when strategy is deactivated

    The strategy should NEVER place orders directly. It only produces Signals.
    The execution layer (broker.py or paper.py) handles order placement.
    """

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.state = StrategyState(name=name)
        self.active_regimes: list[RegimeState] = self._parse_active_regimes()

    def _parse_active_regimes(self) -> list[RegimeState]:
        """Parse regime strings from config into RegimeState enums."""
        regime_strings = self.config.get("active_regimes", [])
        regimes = []
        for r in regime_strings:
            with suppress(ValueError):
                regimes.append(RegimeState(r))
        return regimes

    def should_be_active(self, current_regime: RegimeState) -> bool:
        """Check if this strategy should be active in the current regime."""
        return current_regime in self.active_regimes

    def _entry_regime(self) -> RegimeState | None:
        current_position = self.state.current_position
        if not isinstance(current_position, dict):
            return None
        raw = current_position.get("entry_regime")
        if raw is None:
            return None
        if isinstance(raw, RegimeState):
            return raw
        with suppress(ValueError):
            return RegimeState(str(raw))
        return None

    def should_exit_on_regime_change(self, *, new_regime: RegimeState) -> bool:
        """
        Determine if a regime-transition exit should be emitted.

        Exit only when:
        1) There is an open position.
        2) The new regime is outside active_regimes.
        3) The new regime differs from the recorded entry regime (if recorded).
        """
        if self.state.current_position is None:
            return False
        if self.should_be_active(new_regime):
            return False
        entry_regime = self._entry_regime()
        return not (entry_regime is not None and new_regime == entry_regime)

    @abstractmethod
    def generate_signal(
        self,
        market_data: dict[str, Any],
        regime: RegimeState,
    ) -> Signal:
        """
        Core strategy logic. Called on every new bar/tick.

        Args:
            market_data: Dict containing at minimum:
                - "candles": DataFrame of OHLCV candles
                - "option_chain": Current option chain (for options strategies)
                - "greeks": Current Greeks (for options strategies)
                - "vix": Current India VIX value
                - "timestamp": Current time
            regime: Current market regime classification

        Returns:
            Signal object. Return Signal with NO_SIGNAL if no action needed.
        """
        ...

    @abstractmethod
    def get_exit_conditions(self, market_data: dict[str, Any]) -> Signal | None:
        """
        Check if current position should be closed.

        Called every tick when the strategy has an open position.
        Returns exit Signal if conditions are met, None otherwise.

        Typical exit conditions:
        - Profit target hit
        - Stop loss hit
        - Time-based exit (DTE for options)
        - Regime changed to unfavorable
        """
        ...

    @abstractmethod
    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        """
        Determine how many lots/units to trade.

        Args:
            capital: Available trading capital
            risk_per_trade: Maximum risk for this trade (from risk manager)

        Returns:
            Number of lots (must be positive integer, 0 means skip trade)
        """
        ...

    def on_regime_change(self, old_regime: RegimeState, new_regime: RegimeState) -> Signal | None:
        """
        Called when the regime classifier detects a shift.

        Default behavior: if new regime is not in active_regimes and we have
        an open position, generate an exit signal.

        Override for more nuanced behavior (e.g., tighten stops instead of exiting).
        """
        if self.should_exit_on_regime_change(new_regime=new_regime):
            return Signal(
                signal_type=SignalType.EXIT,
                strategy_name=self.name,
                instrument=self.config.get("instrument", "NIFTY"),
                timestamp=datetime.now(),
                reason=f"Regime changed from {old_regime.value} to {new_regime.value}",
                regime=new_regime,
            )
        return None

    def on_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: int,
        timestamp: datetime,
    ) -> None:
        """Called when an order is filled. Update internal state."""
        self.state.trade_count += 1

    def on_exit(self) -> None:
        """Cleanup when strategy is deactivated. Override if needed."""
        self.state.is_active = False

    def __repr__(self) -> str:
        status = "ACTIVE" if self.state.is_active else "INACTIVE"
        return (
            f"<{self.name} [{status}] "
            f"trades={self.state.trade_count} "
            f"pnl={self.state.realized_pnl:.0f}>"
        )
