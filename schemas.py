"""
Data schemas — canonical data models used across the entire system.

These Pydantic models define the "contract" between layers.
Every component speaks the same language.
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# === Enums ===

class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"  # NSE F&O segment


class InstrumentType(str, Enum):
    INDEX = "INDEX"
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    CALL = "CE"
    PUT = "PE"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"          # Stop-loss
    SL_M = "SL-M"      # Stop-loss market


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# === Market Data Models ===

class Candle(BaseModel):
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    oi: int = 0          # Open interest (for derivatives)

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low


class OptionGreeks(BaseModel):
    """Greeks for a single option contract."""
    iv: float = 0.0           # Implied volatility
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


class OptionContract(BaseModel):
    """Single option contract with market data and Greeks."""
    symbol: str                    # e.g., "NIFTY2430725500CE"
    instrument_type: InstrumentType
    strike: float
    expiry: datetime
    ltp: float = 0.0               # Last traded price
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    oi: int = 0
    change_in_oi: int = 0
    greeks: OptionGreeks = Field(default_factory=OptionGreeks)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ltp


class OptionChain(BaseModel):
    """Full option chain for an instrument at a single timestamp."""
    underlying: str                # e.g., "NIFTY"
    underlying_price: float
    timestamp: datetime
    expiry: datetime
    contracts: list[OptionContract] = Field(default_factory=list)

    def get_by_strike(self, strike: float, opt_type: InstrumentType) -> OptionContract | None:
        """Find a specific contract by strike and type."""
        for c in self.contracts:
            if c.strike == strike and c.instrument_type == opt_type:
                return c
        return None

    def get_atm_strike(self, step: float = 50.0) -> float:
        """Get the at-the-money strike (nearest to underlying price)."""
        return round(self.underlying_price / step) * step

    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.instrument_type == InstrumentType.CALL]

    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.instrument_type == InstrumentType.PUT]

    @property
    def pcr(self) -> float:
        """Put-Call Ratio based on open interest."""
        call_oi = sum(c.oi for c in self.calls())
        put_oi = sum(c.oi for c in self.puts())
        if call_oi == 0:
            return 0.0
        return put_oi / call_oi


# === Order Models ===

class Order(BaseModel):
    """An order to be placed or that has been placed."""
    id: str = ""                    # Assigned by broker on placement
    symbol: str
    exchange: Exchange = Exchange.NFO
    side: OrderSide
    order_type: OrderType = OrderType.LIMIT
    quantity: int
    price: float = 0.0             # For LIMIT orders
    trigger_price: float = 0.0     # For SL orders
    status: OrderStatus = OrderStatus.PENDING

    # Tracking
    strategy_name: str = ""
    placed_at: datetime | None = None
    filled_at: datetime | None = None
    fill_price: float = 0.0
    fill_quantity: int = 0

    # Context for trade journal
    reason: str = ""


class Position(BaseModel):
    """A currently held position."""
    symbol: str
    exchange: Exchange = Exchange.NFO
    side: OrderSide
    quantity: int
    average_price: float
    ltp: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Metadata
    strategy_name: str = ""
    entry_time: datetime | None = None
    entry_reason: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.ltp


# === Trade Journal Models ===

class TradeRecord(BaseModel):
    """A completed trade (entry + exit) for the journal."""
    id: str
    strategy_name: str
    instrument: str

    # Entry
    entry_time: datetime
    entry_price: float
    entry_side: OrderSide
    entry_quantity: int
    entry_reason: str = ""
    entry_regime: str = ""

    # Exit
    exit_time: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_regime: str = ""

    # Result
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_period_minutes: int = 0

    # Context snapshot at entry
    vix_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    iv_at_entry: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
