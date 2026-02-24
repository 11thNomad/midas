"""Minimal paper execution simulator for actionable strategy signals."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.store import DataStore
from src.risk.circuit_breaker import CircuitBreaker
from src.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class PaperExecutionEngine:
    """Immediate-fill paper executor with basic cost and persistence hooks."""

    base_dir: str = "data/cache"
    dataset: str = "paper_fills"
    slippage_bps: float = 5.0
    slippage_multiplier: float = 1.0
    commission_per_order: float = 20.0
    initial_capital: float = 150_000.0
    paper_capital: float | None = None
    margin_buffer_pct: float = 15.0
    available_cash_resolver: Callable[[], float | None] | None = None
    paper_log_dir: str = "data/paper_trading"
    state_db_name: str = "paper_state.sqlite3"
    circuit_breaker: CircuitBreaker | None = None
    _store: DataStore = field(init=False, repr=False)
    _fill_seq: int = field(default=0, init=False, repr=False)
    _cash: float = field(default=0.0, init=False, repr=False)
    _current_balance: float = field(default=0.0, init=False, repr=False)
    _mtm_balance: float = field(default=0.0, init=False, repr=False)
    _unrealized_pnl_current: float = field(default=0.0, init=False, repr=False)
    _positions: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _instrument_meta_by_instrument: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _avg_cost_by_instrument: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_price_by_instrument: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _realized_pnl_today: float = field(default=0.0, init=False, repr=False)
    _gross_realized_pnl_today: float = field(default=0.0, init=False, repr=False)
    _fees_paid_today: float = field(default=0.0, init=False, repr=False)
    _entries_today: int = field(default=0, init=False, repr=False)
    _exits_today: int = field(default=0, init=False, repr=False)
    _paper_log_path: Path = field(init=False, repr=False)
    _state_db_path: Path = field(init=False, repr=False)
    _summary_day_key: str | None = field(default=None, init=False, repr=False)

    _FILL_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "trade_id",
        "signal_type",
        "instrument",
        "leg",
        "action",
        "strike",
        "expiry",
        "quantity",
        "fill_price",
        "mid_price",
        "slippage_applied",
        "fees_estimated",
        "regime",
        "vix",
        "adx",
    )
    _SUMMARY_COLUMNS: tuple[str, ...] = (
        "date",
        "entries",
        "exits",
        "open_positions",
        "gross_pnl",
        "fees",
        "net_pnl",
        "realized_balance",
        "mtm_balance",
        "unrealized_pnl",
        "cash_balance",
        "margin_utilisation_pct",
    )
    _OPTION_SYMBOL_PATTERN = re.compile(
        r"^(?P<underlying>[A-Z]+)[_-]?(?P<expiry>\d{8})[_-]?(?P<strike>\d+(?:\.\d+)?)(?P<option_type>CE|PE)$"
    )

    def _starting_capital(self) -> float:
        configured = self.paper_capital
        if configured is None:
            configured = self.initial_capital
        return float(configured)

    def __post_init__(self) -> None:
        self._store = DataStore(base_dir=self.base_dir)
        starting_capital = self._starting_capital()
        self.paper_capital = float(starting_capital)
        self._cash = float(starting_capital)
        self._current_balance = float(starting_capital)
        self._mtm_balance = float(starting_capital)
        self._unrealized_pnl_current = 0.0
        self._paper_log_path = Path(self.paper_log_dir)
        self._paper_log_path.mkdir(parents=True, exist_ok=True)
        self._state_db_path = self._paper_log_path / self.state_db_name
        self._init_state_db()
        self._load_state_from_db()

    def execute_signals(
        self, signals: list[Signal], *, market_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute actionable signals as immediate paper fills and persist events."""
        market_data = market_data or {}
        self._roll_daily_counters_if_needed(market_data=market_data)
        fills: list[dict[str, Any]] = []
        executed_signals: list[Signal] = []

        for signal in signals:
            if not signal.is_actionable:
                continue
            if (
                signal.signal_type != SignalType.EXIT
                and self.circuit_breaker is not None
                and not self.circuit_breaker.can_trade()
            ):
                continue
            if signal.signal_type == SignalType.ENTRY_SHORT and not self._can_enter_short(
                signal, market_data=market_data
            ):
                continue
            signal_fills = self._fills_for_signal(signal, market_data=market_data)
            if signal_fills:
                executed_signals.append(signal)
                if signal.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
                    self._entries_today += 1
                elif signal.signal_type == SignalType.EXIT:
                    self._exits_today += 1
                # Persist state after each actionable signal to minimize restart-loss window.
                self._persist_state_to_db(fills=signal_fills)
            fills.extend(signal_fills)

        settlement_fills = self._collect_expiry_settlement_fills(market_data=market_data)
        if settlement_fills:
            fills.extend(settlement_fills)
            self._persist_state_to_db(fills=settlement_fills)

        if bool(market_data.get("force_exit_all", False)):
            watchdog_reason = str(
                market_data.get("force_exit_reason", "watchdog_wed_1525_force_exit")
            )
            watchdog_fills = self._force_exit_all_positions(
                market_data=market_data,
                reason=watchdog_reason,
            )
            if watchdog_fills:
                fills.extend(watchdog_fills)
                self._persist_state_to_db(fills=watchdog_fills)

        if fills:
            frame = pd.DataFrame(fills)
            self._store.write_time_series(
                self.dataset,
                frame,
                symbol=str(market_data.get("symbol", "NIFTY")),
                timestamp_col="timestamp",
                source="paper_executor",
            )

        self._refresh_mark_to_market(market_data=market_data)
        # Persist snapshot each loop so MTM state and day-rollover counters survive restarts.
        self._persist_state_to_db(fills=[])

        if self.circuit_breaker is not None:
            open_positions = sum(1 for _, qty in self._positions.items() if qty != 0)
            self.circuit_breaker.update(
                current_equity=self._mtm_balance,
                realized_pnl_today=self._realized_pnl_today,
                unrealized_pnl=self._unrealized_pnl_current,
                open_positions=open_positions,
            )

        self._write_csv_logs(fills=fills, signals=executed_signals, market_data=market_data)
        return fills

    def _connect_state_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._state_db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_state_db(self) -> None:
        with self._connect_state_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_fill_ledger (
                    fill_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    fees REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_state_snapshot (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    fill_seq INTEGER NOT NULL,
                    cash REAL NOT NULL,
                    current_balance REAL NOT NULL DEFAULT 0.0,
                    mtm_balance REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    positions_json TEXT NOT NULL,
                    instrument_meta_json TEXT NOT NULL DEFAULT '{}',
                    avg_cost_json TEXT NOT NULL,
                    last_price_json TEXT NOT NULL,
                    realized_pnl_today REAL NOT NULL,
                    gross_realized_pnl_today REAL NOT NULL,
                    fees_paid_today REAL NOT NULL,
                    entries_today INTEGER NOT NULL,
                    exits_today INTEGER NOT NULL,
                    summary_day_key TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cols = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(paper_state_snapshot)").fetchall()
            }
            if "current_balance" not in cols:
                conn.execute(
                    """
                    ALTER TABLE paper_state_snapshot
                    ADD COLUMN current_balance REAL NOT NULL DEFAULT 0.0
                    """
                )
            if "mtm_balance" not in cols:
                conn.execute(
                    """
                    ALTER TABLE paper_state_snapshot
                    ADD COLUMN mtm_balance REAL NOT NULL DEFAULT 0.0
                    """
                )
            if "unrealized_pnl" not in cols:
                conn.execute(
                    """
                    ALTER TABLE paper_state_snapshot
                    ADD COLUMN unrealized_pnl REAL NOT NULL DEFAULT 0.0
                    """
                )
            if "instrument_meta_json" not in cols:
                conn.execute(
                    """
                    ALTER TABLE paper_state_snapshot
                    ADD COLUMN instrument_meta_json TEXT NOT NULL DEFAULT '{}'
                    """
                )

    def _load_state_from_db(self) -> None:
        with self._connect_state_db() as conn:
            snapshot = conn.execute(
                """
                SELECT
                    fill_seq,
                    cash,
                    current_balance,
                    mtm_balance,
                    unrealized_pnl,
                    positions_json,
                    instrument_meta_json,
                    avg_cost_json,
                    last_price_json,
                    realized_pnl_today,
                    gross_realized_pnl_today,
                    fees_paid_today,
                    entries_today,
                    exits_today,
                    summary_day_key
                FROM paper_state_snapshot
                WHERE id = 1
                """
            ).fetchone()
            if snapshot is not None:
                self._fill_seq = max(
                    int(snapshot["fill_seq"]),
                    self._max_fill_seq_from_ledger(conn),
                )
                self._cash = float(snapshot["cash"])
                restored_balance = float(snapshot["current_balance"])
                if restored_balance > 0.0:
                    self._current_balance = restored_balance
                restored_mtm = float(snapshot["mtm_balance"])
                self._mtm_balance = (
                    restored_mtm if restored_mtm > 0.0 else float(self._current_balance)
                )
                self._unrealized_pnl_current = float(snapshot["unrealized_pnl"])
                self._positions = self._decode_int_map(snapshot["positions_json"])
                self._instrument_meta_by_instrument = self._decode_meta_map(
                    snapshot["instrument_meta_json"]
                )
                self._avg_cost_by_instrument = self._decode_float_map(snapshot["avg_cost_json"])
                self._last_price_by_instrument = self._decode_float_map(snapshot["last_price_json"])
                self._realized_pnl_today = float(snapshot["realized_pnl_today"])
                self._gross_realized_pnl_today = float(snapshot["gross_realized_pnl_today"])
                self._fees_paid_today = float(snapshot["fees_paid_today"])
                self._entries_today = int(snapshot["entries_today"])
                self._exits_today = int(snapshot["exits_today"])
                self._summary_day_key = str(snapshot["summary_day_key"] or "") or None
                return

            if self._max_fill_seq_from_ledger(conn) > 0:
                self._rebuild_state_from_ledger(conn)

    def _persist_state_to_db(self, *, fills: list[dict[str, Any]]) -> None:
        fill_rows = [
            (
                str(fill.get("fill_id", "")).strip(),
                pd.Timestamp(fill.get("timestamp")).isoformat(),
                str(fill.get("trade_id", "")).strip(),
                str(fill.get("signal_type", "")).strip(),
                str(fill.get("instrument", "")).strip(),
                str(fill.get("side", "")).strip().upper(),
                int(fill.get("quantity", 0) or 0),
                float(fill.get("price", 0.0) or 0.0),
                float(fill.get("notional", 0.0) or 0.0),
                float(fill.get("fees", 0.0) or 0.0),
            )
            for fill in fills
            if str(fill.get("fill_id", "")).strip()
        ]
        with self._connect_state_db() as conn:
            if fill_rows:
                conn.executemany(
                    """
                    INSERT INTO paper_fill_ledger (
                        fill_id,
                        timestamp,
                        trade_id,
                        signal_type,
                        instrument,
                        side,
                        quantity,
                        price,
                        notional,
                        fees
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(fill_id) DO NOTHING
                    """,
                    fill_rows,
                )
            conn.execute(
                """
                INSERT INTO paper_state_snapshot (
                    id,
                    fill_seq,
                    cash,
                    current_balance,
                    mtm_balance,
                    unrealized_pnl,
                    positions_json,
                    instrument_meta_json,
                    avg_cost_json,
                    last_price_json,
                    realized_pnl_today,
                    gross_realized_pnl_today,
                    fees_paid_today,
                    entries_today,
                    exits_today,
                    summary_day_key,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    fill_seq = excluded.fill_seq,
                    cash = excluded.cash,
                    current_balance = excluded.current_balance,
                    mtm_balance = excluded.mtm_balance,
                    unrealized_pnl = excluded.unrealized_pnl,
                    positions_json = excluded.positions_json,
                    instrument_meta_json = excluded.instrument_meta_json,
                    avg_cost_json = excluded.avg_cost_json,
                    last_price_json = excluded.last_price_json,
                    realized_pnl_today = excluded.realized_pnl_today,
                    gross_realized_pnl_today = excluded.gross_realized_pnl_today,
                    fees_paid_today = excluded.fees_paid_today,
                    entries_today = excluded.entries_today,
                    exits_today = excluded.exits_today,
                    summary_day_key = excluded.summary_day_key,
                    updated_at = excluded.updated_at
                """,
                (
                    1,
                    int(self._fill_seq),
                    float(self._cash),
                    float(self._current_balance),
                    float(self._mtm_balance),
                    float(self._unrealized_pnl_current),
                    json.dumps(self._positions, sort_keys=True),
                    json.dumps(self._instrument_meta_by_instrument, sort_keys=True),
                    json.dumps(self._avg_cost_by_instrument, sort_keys=True),
                    json.dumps(self._last_price_by_instrument, sort_keys=True),
                    float(self._realized_pnl_today),
                    float(self._gross_realized_pnl_today),
                    float(self._fees_paid_today),
                    int(self._entries_today),
                    int(self._exits_today),
                    self._summary_day_key,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    @staticmethod
    def _decode_int_map(raw: object) -> dict[str, int]:
        if raw is None:
            return {}
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        out: dict[str, int] = {}
        for key, value in payload.items():
            try:
                out[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _decode_float_map(raw: object) -> dict[str, float]:
        if raw is None:
            return {}
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in payload.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _decode_meta_map(raw: object) -> dict[str, dict[str, Any]]:
        if raw is None:
            return {}
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            out[str(key)] = {str(meta_key): meta_value for meta_key, meta_value in value.items()}
        return out

    @staticmethod
    def _parse_fill_sequence(fill_id: str) -> int | None:
        raw = fill_id.strip()
        if not raw.startswith("PAPER-"):
            return None
        suffix = raw.split("PAPER-", maxsplit=1)[1]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _max_fill_seq_from_ledger(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT fill_id FROM paper_fill_ledger ORDER BY fill_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0
        fill_id = str(row["fill_id"])
        parsed = self._parse_fill_sequence(fill_id)
        return int(parsed or 0)

    def _rebuild_state_from_ledger(self, conn: sqlite3.Connection) -> None:
        starting_capital = self._starting_capital()
        self._cash = float(starting_capital)
        self._current_balance = float(starting_capital)
        self._mtm_balance = float(starting_capital)
        self._unrealized_pnl_current = 0.0
        self._positions = {}
        self._instrument_meta_by_instrument = {}
        self._avg_cost_by_instrument = {}
        self._last_price_by_instrument = {}
        self._realized_pnl_today = 0.0
        self._gross_realized_pnl_today = 0.0
        self._fees_paid_today = 0.0
        self._entries_today = 0
        self._exits_today = 0
        self._summary_day_key = None

        rows = conn.execute(
            """
            SELECT
                fill_id,
                timestamp,
                trade_id,
                signal_type,
                instrument,
                side,
                quantity,
                price,
                fees
            FROM paper_fill_ledger
            ORDER BY timestamp ASC, fill_id ASC
            """
        ).fetchall()

        seen_entry_keys: set[tuple[str, str]] = set()
        seen_exit_keys: set[tuple[str, str]] = set()
        active_day_key: str | None = None

        for row in rows:
            signal_type = str(row["signal_type"]).lower()
            trade_id = str(row["trade_id"]).strip()
            ts = pd.Timestamp(row["timestamp"])
            day_key = ts.strftime("%Y%m%d")
            if active_day_key != day_key:
                active_day_key = day_key
                seen_entry_keys.clear()
                seen_exit_keys.clear()
                self._entries_today = 0
                self._exits_today = 0
                self._gross_realized_pnl_today = 0.0
                self._fees_paid_today = 0.0
                self._realized_pnl_today = 0.0
            self._summary_day_key = day_key

            signal_key = (trade_id, signal_type)
            if signal_type in (SignalType.ENTRY_LONG.value, SignalType.ENTRY_SHORT.value):
                if signal_key not in seen_entry_keys:
                    self._entries_today += 1
                    seen_entry_keys.add(signal_key)
            elif signal_type == SignalType.EXIT.value and signal_key not in seen_exit_keys:
                self._exits_today += 1
                seen_exit_keys.add(signal_key)

            self._apply_fill_accounting(
                instrument=str(row["instrument"]),
                side=str(row["side"]).upper(),
                quantity=int(row["quantity"]),
                fill_price=float(row["price"]),
                fees=float(row["fees"]),
            )
            self._remember_instrument_meta(instrument=str(row["instrument"]), order={})

            seq = self._parse_fill_sequence(str(row["fill_id"]))
            if seq is not None:
                self._fill_seq = max(self._fill_seq, seq)
        self._refresh_mark_to_market(market_data={})

    def _apply_fill_accounting(
        self,
        *,
        instrument: str,
        side: str,
        quantity: int,
        fill_price: float,
        fees: float,
    ) -> None:
        notional = abs(float(fill_price) * int(quantity))
        realized_delta = self._update_position_and_realized_pnl(
            instrument=instrument,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
        )
        self._gross_realized_pnl_today += realized_delta
        self._fees_paid_today += fees
        self._realized_pnl_today += realized_delta - fees
        self._current_balance += realized_delta - fees
        if side == "BUY":
            self._cash -= notional + fees
        else:
            self._cash += notional - fees
        self._last_price_by_instrument[instrument] = float(fill_price)
        if int(self._positions.get(instrument, 0)) == 0:
            self._instrument_meta_by_instrument.pop(instrument, None)

    def _can_enter_short(self, signal: Signal, *, market_data: dict[str, Any]) -> bool:
        required_margin = self._estimate_required_margin(signal, market_data=market_data)
        required_with_buffer = required_margin * (1.0 + (self.margin_buffer_pct / 100.0))
        available_cash, cash_source = self._resolve_available_cash()
        if available_cash >= required_with_buffer:
            return True
        logger.warning(
            "ABORT_ENTRY - insufficient margin (available: %.2f, required: %.2f, source: %s)",
            available_cash,
            required_with_buffer,
            cash_source,
        )
        return False

    def _estimate_required_margin(self, signal: Signal, *, market_data: dict[str, Any]) -> float:
        indicators = signal.indicators if isinstance(signal.indicators, dict) else {}
        wing_width = float(
            indicators.get("call_wing")
            or indicators.get("put_wing")
            or indicators.get("wing_width")
            or 100.0
        )
        orders = signal.orders or []
        quantity_units = max((int(order.get("quantity", 1) or 1) for order in orders), default=1)
        spot = self._resolve_spot(signal=signal, market_data=market_data)
        wing_based = wing_width * float(quantity_units) * 1.5
        notional_based = float(quantity_units) * spot * 0.06 if spot > 0.0 else 0.0
        minimum_floor = 100_000.0
        return max(wing_based, notional_based, minimum_floor)

    @staticmethod
    def _resolve_spot(*, signal: Signal, market_data: dict[str, Any]) -> float:
        indicators = signal.indicators if isinstance(signal.indicators, dict) else {}
        candidates = (
            indicators.get("spot"),
            indicators.get("underlying_price"),
            market_data.get("close_price"),
            market_data.get("last_price"),
        )
        for value in candidates:
            try:
                if value is None:
                    continue
                parsed = float(value)
                if parsed > 0.0:
                    return parsed
            except (TypeError, ValueError):
                continue
        return 0.0

    def _resolve_available_cash(self) -> tuple[float, str]:
        if self.available_cash_resolver is not None:
            try:
                resolved = self.available_cash_resolver()
            except Exception:  # pragma: no cover - defensive runtime guard
                resolved = None
            if resolved is not None and float(resolved) > 0.0:
                return float(resolved), "kite"
        return float(self._current_balance), "paper_balance"

    def read_fills(
        self,
        *,
        symbol: str = "NIFTY",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return self._store.read_time_series(
            self.dataset,
            symbol=symbol,
            start=start,
            end=end,
            timestamp_col="timestamp",
        )

    def _fills_for_signal(
        self, signal: Signal, *, market_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        orders = signal.orders or [self._default_order(signal)]
        ts = signal.timestamp
        trade_id = self._resolve_trade_id(signal)

        out: list[dict[str, Any]] = []
        for order in orders:
            self._fill_seq += 1
            instrument = str(order.get("symbol", signal.instrument))
            action = str(
                order.get("action", self._default_action(signal, symbol=instrument))
            ).upper()
            quantity = int(order.get("quantity", 1) or 1)
            option_type = self._resolve_option_type(order=order, instrument=instrument)
            if not self._validate_order_quantity(
                instrument=instrument,
                quantity=quantity,
                option_type=option_type,
                signal_type=signal.signal_type,
                market_data=market_data,
            ):
                continue
            raw_price = self._resolve_order_price(
                order=order,
                market_data=market_data,
                signal_type=signal.signal_type,
                action=action,
            )
            filled_price = self._apply_slippage(raw_price, action=action)
            leg = self._resolve_leg_label(
                signal_type=signal.signal_type.value,
                action=action,
                option_type=option_type,
            )
            self._remember_instrument_meta(instrument=instrument, order=order)

            notional = abs(filled_price * quantity)
            fees = float(self.commission_per_order)
            slippage_applied = float(filled_price) - float(raw_price)
            out.append(
                {
                    "timestamp": ts,
                    "fill_id": f"PAPER-{self._fill_seq:08d}",
                    "trade_id": trade_id,
                    "strategy_name": signal.strategy_name,
                    "signal_type": signal.signal_type.value,
                    "instrument": instrument,
                    "side": action,
                    "quantity": quantity,
                    "price": float(filled_price),
                    "notional": float(notional),
                    "fees": fees,
                    "mid_price": float(raw_price),
                    "slippage_applied": float(slippage_applied),
                    "leg": leg,
                    "strike": order.get("strike"),
                    "expiry": order.get("expiry"),
                    "option_type": option_type,
                    "reason": signal.reason,
                    "regime": signal.regime.value,
                    "confidence": float(signal.confidence),
                }
            )
            self._apply_fill_accounting(
                instrument=instrument,
                side=action,
                quantity=quantity,
                fill_price=filled_price,
                fees=fees,
            )
        return out

    def _write_csv_logs(
        self,
        *,
        fills: list[dict[str, Any]],
        signals: list[Signal],
        market_data: dict[str, Any],
    ) -> None:
        self._write_fills_csv(fills=fills, signals=signals, market_data=market_data)
        self._write_daily_summary_csv(market_data=market_data)

    def _write_fills_csv(
        self,
        *,
        fills: list[dict[str, Any]],
        signals: list[Signal],
        market_data: dict[str, Any],
    ) -> None:
        if not fills:
            return
        _ = signals  # consumed at source via per-fill metadata
        rows: list[dict[str, Any]] = []
        for fill in fills:
            ts = pd.Timestamp(fill.get("timestamp"))
            signal_type = str(fill.get("signal_type", "")).strip().lower()
            instrument = str(fill.get("instrument", "")).strip()
            trade_id = str(fill.get("trade_id", "")).strip()
            if not trade_id:
                trade_id = self._default_trade_id_from_fill(fill)
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "trade_id": trade_id,
                    "signal_type": signal_type,
                    "instrument": instrument,
                    "leg": str(fill.get("leg", "unknown")),
                    "action": str(fill.get("side", "")).upper(),
                    "strike": fill.get("strike"),
                    "expiry": self._serialize_expiry(fill.get("expiry")),
                    "quantity": int(fill.get("quantity", 0) or 0),
                    "fill_price": float(fill.get("price", 0.0) or 0.0),
                    "mid_price": float(fill.get("mid_price", 0.0) or 0.0),
                    "slippage_applied": float(fill.get("slippage_applied", 0.0) or 0.0),
                    "fees_estimated": float(fill.get("fees", 0.0) or 0.0),
                    "regime": str(fill.get("regime", "")),
                    "vix": self._as_float(market_data.get("vix")),
                    "adx": self._as_float(market_data.get("adx")),
                }
            )
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=list(self._FILL_COLUMNS))
        by_day = frame.groupby(pd.to_datetime(frame["timestamp"]).dt.strftime("%Y%m%d"), sort=True)
        for day_key, subset in by_day:
            path = self._paper_log_path / f"fills_{day_key}.csv"
            # Append-only audit trail by design: restarts can create duplicate logical fills
            # for the same day. Downstream reconciliation can deduplicate by fill_id/trade_id.
            subset.to_csv(path, index=False, mode="a", header=not path.exists())

    def _write_daily_summary_csv(self, *, market_data: dict[str, Any]) -> None:
        ts = pd.to_datetime(
            market_data.get("timestamp", datetime.now()),
            errors="coerce",
        )
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        day_key = ts.strftime("%Y%m%d")
        open_positions = sum(1 for _, qty in self._positions.items() if qty != 0)
        gross_pnl = float(self._gross_realized_pnl_today)
        fees = float(self._fees_paid_today)
        net_pnl = float(self._realized_pnl_today)
        summary_row = {
            "date": ts.strftime("%Y-%m-%d"),
            "entries": int(self._entries_today),
            "exits": int(self._exits_today),
            "open_positions": int(open_positions),
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "realized_balance": float(self._current_balance),
            "mtm_balance": float(self._mtm_balance),
            "unrealized_pnl": float(self._unrealized_pnl_current),
            "cash_balance": float(self._cash),
            "margin_utilisation_pct": self._estimate_margin_utilisation_pct(),
        }
        summary_path = self._paper_log_path / f"daily_summary_{day_key}.csv"
        pd.DataFrame([summary_row], columns=list(self._SUMMARY_COLUMNS)).to_csv(
            summary_path,
            index=False,
        )

    def _roll_daily_counters_if_needed(self, *, market_data: dict[str, Any]) -> None:
        ts = pd.to_datetime(
            market_data.get("timestamp", datetime.now()),
            errors="coerce",
        )
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        day_key = ts.strftime("%Y%m%d")
        if self._summary_day_key is None:
            self._summary_day_key = day_key
            return
        if day_key == self._summary_day_key:
            return
        self._summary_day_key = day_key
        self._entries_today = 0
        self._exits_today = 0
        self._gross_realized_pnl_today = 0.0
        self._fees_paid_today = 0.0
        self._realized_pnl_today = 0.0

    @staticmethod
    def _resolve_option_type(*, order: dict[str, Any], instrument: str) -> str | None:
        raw = order.get("option_type")
        if raw is not None:
            opt = str(raw).strip().upper()
            if opt in {"CE", "PE"}:
                return opt
        symbol = instrument.strip().upper()
        if symbol.endswith("CE"):
            return "CE"
        if symbol.endswith("PE"):
            return "PE"
        return None

    @staticmethod
    def _resolve_leg_label(*, signal_type: str, action: str, option_type: str | None) -> str:
        # Label indicates the original strategy leg being opened/unwound, not the trade action verb.
        opt = (option_type or "").upper()
        side = action.upper()
        st = signal_type.lower()
        if st == SignalType.ENTRY_SHORT.value:
            if opt == "CE" and side == "SELL":
                return "call_short"
            if opt == "PE" and side == "SELL":
                return "put_short"
            if opt == "CE" and side == "BUY":
                return "call_hedge"
            if opt == "PE" and side == "BUY":
                return "put_hedge"
        if st == SignalType.EXIT.value:
            if opt == "CE" and side == "BUY":
                return "call_short"
            if opt == "PE" and side == "BUY":
                return "put_short"
            if opt == "CE" and side == "SELL":
                return "call_hedge"
            if opt == "PE" and side == "SELL":
                return "put_hedge"
        return "unknown"

    @staticmethod
    def _serialize_expiry(value: Any) -> str | None:
        if value is None:
            return None
        expiry = pd.to_datetime(value, errors="coerce")
        if pd.isna(expiry):
            return str(value)
        return pd.Timestamp(expiry).strftime("%Y-%m-%d")

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_trade_id(self, signal: Signal) -> str:
        indicators = signal.indicators if isinstance(signal.indicators, dict) else {}
        for key in ("trade_id", "position_id"):
            value = indicators.get(key)
            if value is not None and str(value).strip():
                return str(value)
        ts = pd.Timestamp(signal.timestamp).strftime("%Y%m%dT%H%M%S")
        return f"{signal.strategy_name}:{signal.instrument}:{ts}"

    @staticmethod
    def _default_trade_id_from_fill(fill: dict[str, Any]) -> str:
        ts = pd.Timestamp(fill.get("timestamp")).strftime("%Y%m%dT%H%M%S")
        strategy = str(fill.get("strategy_name", "strategy")).strip() or "strategy"
        instrument = str(fill.get("instrument", "instrument")).strip() or "instrument"
        return f"{strategy}:{instrument}:{ts}"

    def _estimate_margin_utilisation_pct(self) -> float:
        capital = float(self._current_balance)
        if capital <= 0.0:
            return 0.0
        gross_notional = 0.0
        for instrument, qty in self._positions.items():
            if qty == 0:
                continue
            price = float(self._last_price_by_instrument.get(instrument, 0.0) or 0.0)
            gross_notional += abs(float(qty) * price)
        # Approximation only: uses notional*6% and can overstate utilisation for hedged spreads.
        # Broker-reported margin should be treated as source of truth for operational decisions.
        margin_estimate = gross_notional * 0.06
        return (margin_estimate / capital) * 100.0

    def _default_order(self, signal: Signal) -> dict[str, Any]:
        symbol = signal.instrument
        action = self._default_action(signal, symbol=symbol)
        quantity = abs(int(self._positions.get(symbol, 0)))
        if quantity <= 0:
            quantity = 1
        return {"symbol": symbol, "action": action, "quantity": quantity}

    def _default_action(self, signal: Signal, *, symbol: str) -> str:
        if signal.signal_type == SignalType.ENTRY_SHORT:
            return "SELL"
        if signal.signal_type == SignalType.EXIT:
            net_qty = int(self._positions.get(symbol, 0))
            if net_qty < 0:
                return "BUY"
            if net_qty > 0:
                return "SELL"
            side_hint = str(signal.indicators.get("position_side", "")).lower()
            if side_hint == "short":
                return "BUY"
            if side_hint == "long":
                return "SELL"
            return "SELL"
        return "BUY"

    def _resolve_order_price(
        self,
        *,
        order: dict[str, Any],
        market_data: dict[str, Any],
        signal_type: SignalType,
        action: str,
    ) -> float:
        instrument = str(order.get("symbol", "")).strip()
        option_type = self._resolve_option_type(order=order, instrument=instrument)
        if signal_type == SignalType.EXIT and option_type in {"CE", "PE"}:
            option_exit_quote = self._resolve_option_exit_quote(
                order=order,
                market_data=market_data,
                action=action,
            )
            if option_exit_quote is not None:
                return option_exit_quote
            if "price" in order and order["price"] is not None:
                return float(order["price"])
            if "ltp" in order and order["ltp"] is not None:
                return float(order["ltp"])
            last_mark = float(self._last_price_by_instrument.get(instrument, 0.0) or 0.0)
            if last_mark > 0.0:
                logger.warning(
                    "OPTION_EXIT_PRICE_FALLBACK last_mark used for %s action=%s", instrument, action
                )
                return last_mark
            logger.warning(
                "OPTION_EXIT_PRICE_MISSING no quote found for %s action=%s; using 1.0 fallback",
                instrument,
                action,
            )
            return 1.0
        if "price" in order and order["price"] is not None:
            return float(order["price"])
        if "ltp" in order and order["ltp"] is not None:
            return float(order["ltp"])
        if "last_price" in market_data and market_data["last_price"] is not None:
            return float(market_data["last_price"])
        if "close_price" in market_data and market_data["close_price"] is not None:
            return float(market_data["close_price"])
        return 1.0

    def _resolve_option_exit_quote(
        self,
        *,
        order: dict[str, Any],
        market_data: dict[str, Any],
        action: str,
    ) -> float | None:
        instrument = str(order.get("symbol", "")).strip()
        quote_source = market_data.get("option_quotes")
        quote: dict[str, Any] = {}
        if isinstance(quote_source, dict):
            raw = quote_source.get(instrument)
            if isinstance(raw, dict):
                quote = raw
        if not quote:
            quote = {k: order.get(k) for k in ("bid", "ask", "ltp", "mid")}
        preferred = "ask" if action.upper() == "BUY" else "bid"
        for key in (preferred, "mid", "ltp"):
            value = quote.get(key)
            try:
                if value is None:
                    continue
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed <= 0.0:
                continue
            if key != preferred:
                logger.warning(
                    "OPTION_EXIT_PRICE_FALLBACK instrument=%s action=%s preferred=%s used=%s",
                    instrument,
                    action,
                    preferred,
                    key,
                )
            return parsed
        return None

    def _validate_order_quantity(
        self,
        *,
        instrument: str,
        quantity: int,
        option_type: str | None,
        signal_type: SignalType,
        market_data: dict[str, Any],
    ) -> bool:
        if quantity <= 0:
            logger.warning(
                "SKIP_ORDER invalid quantity=%s instrument=%s signal=%s",
                quantity,
                instrument,
                signal_type.value,
            )
            return False
        if signal_type == SignalType.EXIT:
            return True
        if option_type not in {"CE", "PE"}:
            return True
        lot_sizes = market_data.get("lot_size_by_underlying")
        if not isinstance(lot_sizes, dict):
            return True
        underlying = self._infer_underlying_from_instrument(instrument)
        if underlying is None:
            return True
        raw_lot_size = lot_sizes.get(underlying)
        try:
            lot_size = int(raw_lot_size) if raw_lot_size is not None else 0
        except (TypeError, ValueError):
            lot_size = 0
        if lot_size <= 0:
            return True
        if quantity % lot_size == 0:
            return True
        logger.warning(
            "SKIP_ORDER lot mismatch instrument=%s quantity=%s lot_size=%s signal=%s",
            instrument,
            quantity,
            lot_size,
            signal_type.value,
        )
        return False

    @staticmethod
    def _infer_underlying_from_instrument(instrument: str) -> str | None:
        symbol = instrument.strip().upper()
        match = PaperExecutionEngine._OPTION_SYMBOL_PATTERN.match(symbol.replace("-", "_"))
        if match is not None:
            return str(match.group("underlying")).upper()
        if "_" in symbol:
            base = symbol.split("_", maxsplit=1)[0].strip()
            if base:
                return base.upper()
        if symbol.startswith("NIFTY"):
            return "NIFTY"
        if symbol.startswith("BANKNIFTY"):
            return "BANKNIFTY"
        return None

    def _remember_instrument_meta(self, *, instrument: str, order: dict[str, Any]) -> None:
        existing = self._instrument_meta_by_instrument.get(instrument, {}).copy()
        parsed = self._parse_option_meta_from_instrument(instrument)
        if parsed:
            existing.update(parsed)
        expiry = self._serialize_expiry(order.get("expiry"))
        if expiry is not None:
            existing["expiry"] = expiry
        strike = order.get("strike")
        try:
            if strike is not None:
                existing["strike"] = float(strike)
        except (TypeError, ValueError):
            pass
        option_type = self._resolve_option_type(order=order, instrument=instrument)
        if option_type in {"CE", "PE"}:
            existing["option_type"] = option_type
        underlying = self._infer_underlying_from_instrument(instrument)
        if underlying:
            existing["underlying"] = underlying
        if existing:
            self._instrument_meta_by_instrument[instrument] = existing

    @classmethod
    def _parse_option_meta_from_instrument(cls, instrument: str) -> dict[str, Any]:
        symbol = instrument.strip().upper().replace("-", "_")
        match = cls._OPTION_SYMBOL_PATTERN.match(symbol)
        if match is None:
            return {}
        expiry = pd.to_datetime(match.group("expiry"), format="%Y%m%d", errors="coerce")
        expiry_str = pd.Timestamp(expiry).strftime("%Y-%m-%d") if not pd.isna(expiry) else None
        out: dict[str, Any] = {
            "underlying": str(match.group("underlying")).upper(),
            "option_type": str(match.group("option_type")).upper(),
        }
        if expiry_str is not None:
            out["expiry"] = expiry_str
        with suppress(TypeError, ValueError):
            out["strike"] = float(match.group("strike"))
        return out

    def _apply_slippage(self, price: float, *, action: str) -> float:
        effective_bps = self.slippage_bps * max(float(self.slippage_multiplier), 0.0)
        slip = (effective_bps / 10_000.0) * price
        if action == "BUY":
            return price + slip
        return max(0.0, price - slip)

    def _collect_expiry_settlement_fills(
        self,
        *,
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        ts = pd.to_datetime(market_data.get("timestamp", datetime.now()), errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        now_date = ts.date()
        fills: list[dict[str, Any]] = []
        for instrument, qty in list(self._positions.items()):
            if qty == 0:
                continue
            meta = self._instrument_meta_by_instrument.get(instrument, {})
            if not meta:
                self._remember_instrument_meta(instrument=instrument, order={})
                meta = self._instrument_meta_by_instrument.get(instrument, {})
            expiry_raw = meta.get("expiry")
            option_type = str(meta.get("option_type", "")).upper()
            if expiry_raw is None or option_type not in {"CE", "PE"}:
                continue
            expiry_ts = pd.to_datetime(expiry_raw, errors="coerce")
            if pd.isna(expiry_ts) or now_date < expiry_ts.date():
                continue
            strike = float(meta.get("strike", 0.0) or 0.0)
            if strike <= 0.0:
                continue
            settlement_price, settlement_reason = self._option_settlement_price(
                market_data=market_data,
                instrument=instrument,
                option_type=option_type,
                strike=strike,
                underlying=str(meta.get("underlying", "")).upper(),
            )
            close_fill = self._close_fill_for_position(
                instrument=instrument,
                quantity=qty,
                close_price=settlement_price,
                signal_type=SignalType.EXIT.value,
                strategy_name="expiry_settlement",
                reason=settlement_reason,
                timestamp=ts.to_pydatetime(),
                apply_slippage=False,
                fees=0.0,
            )
            close_fill["expiry"] = self._serialize_expiry(expiry_raw)
            close_fill["strike"] = strike
            close_fill["option_type"] = option_type
            fills.append(close_fill)
        return fills

    def _option_settlement_price(
        self,
        *,
        market_data: dict[str, Any],
        instrument: str,
        option_type: str,
        strike: float,
        underlying: str,
    ) -> tuple[float, str]:
        spot = 0.0
        underlying_prices = market_data.get("underlying_prices")
        if isinstance(underlying_prices, dict):
            try:
                spot = float(underlying_prices.get(underlying, 0.0) or 0.0)
            except (TypeError, ValueError):
                spot = 0.0
        if spot <= 0.0:
            symbol = str(market_data.get("symbol", "")).strip().upper()
            if symbol == underlying:
                try:
                    spot = float(
                        market_data.get("close_price", market_data.get("last_price", 0.0))
                        or 0.0
                    )
                except (TypeError, ValueError):
                    spot = 0.0
        if spot <= 0.0:
            logger.warning(
                "EXPIRY_SETTLEMENT_SPOT_MISSING instrument=%s underlying=%s; settling at zero",
                instrument,
                underlying,
            )
        intrinsic = 0.0
        if option_type == "CE":
            intrinsic = max(0.0, spot - strike)
        elif option_type == "PE":
            intrinsic = max(0.0, strike - spot)
        reason = (
            "expiry_settlement_itm"
            if intrinsic > 0.0
            else "expiry_settlement_otm"
        )
        return float(intrinsic), reason

    def _force_exit_all_positions(
        self,
        *,
        market_data: dict[str, Any],
        reason: str,
    ) -> list[dict[str, Any]]:
        ts = pd.to_datetime(market_data.get("timestamp", datetime.now()), errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        fills: list[dict[str, Any]] = []
        for instrument, qty in list(self._positions.items()):
            if qty == 0:
                continue
            order = {
                "symbol": instrument,
                "action": "BUY" if qty < 0 else "SELL",
                "quantity": abs(qty),
            }
            meta = self._instrument_meta_by_instrument.get(instrument, {})
            for key in ("expiry", "strike", "option_type"):
                if key in meta:
                    order[key] = meta[key]
            close_price = self._resolve_order_price(
                order=order,
                market_data=market_data,
                signal_type=SignalType.EXIT,
                action=str(order["action"]),
            )
            close_fill = self._close_fill_for_position(
                instrument=instrument,
                quantity=qty,
                close_price=close_price,
                signal_type=SignalType.EXIT.value,
                strategy_name="watchdog",
                reason=reason,
                timestamp=ts.to_pydatetime(),
            )
            close_fill["expiry"] = self._serialize_expiry(order.get("expiry"))
            close_fill["strike"] = order.get("strike")
            close_fill["option_type"] = self._resolve_option_type(
                order=order,
                instrument=instrument,
            )
            fills.append(close_fill)
        return fills

    def _close_fill_for_position(
        self,
        *,
        instrument: str,
        quantity: int,
        close_price: float,
        signal_type: str,
        strategy_name: str,
        reason: str,
        timestamp: datetime,
        apply_slippage: bool = True,
        fees: float | None = None,
    ) -> dict[str, Any]:
        action = "BUY" if quantity < 0 else "SELL"
        absolute_qty = abs(int(quantity))
        self._fill_seq += 1
        filled_price = (
            self._apply_slippage(float(close_price), action=action)
            if apply_slippage
            else float(close_price)
        )
        notional = abs(filled_price * absolute_qty)
        resolved_fees = float(self.commission_per_order) if fees is None else float(fees)
        self._apply_fill_accounting(
            instrument=instrument,
            side=action,
            quantity=absolute_qty,
            fill_price=filled_price,
            fees=resolved_fees,
        )
        return {
            "timestamp": timestamp,
            "fill_id": f"PAPER-{self._fill_seq:08d}",
            "trade_id": f"{strategy_name}:{pd.Timestamp(timestamp).strftime('%Y%m%dT%H%M%S')}",
            "strategy_name": strategy_name,
            "signal_type": signal_type,
            "instrument": instrument,
            "side": action,
            "quantity": absolute_qty,
            "price": float(filled_price),
            "notional": float(notional),
            "fees": resolved_fees,
            "mid_price": float(close_price),
            "slippage_applied": float(filled_price) - float(close_price),
            "leg": "unknown",
            "strike": None,
            "expiry": None,
            "option_type": self._resolve_option_type(order={}, instrument=instrument),
            "reason": reason,
            "regime": "",
            "confidence": 0.0,
        }

    def _refresh_mark_to_market(self, *, market_data: dict[str, Any]) -> None:
        instrument_prices = market_data.get("instrument_prices")
        if isinstance(instrument_prices, dict):
            for raw_instrument, raw_price in instrument_prices.items():
                instrument = str(raw_instrument).strip()
                if not instrument:
                    continue
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    continue
                if price <= 0.0:
                    continue
                self._last_price_by_instrument[instrument] = price
        default_price = float(
            market_data.get("last_price", market_data.get("close_price", 0.0)) or 0.0
        )
        self._mtm_balance = self._mark_to_market(default_price=default_price)
        self._unrealized_pnl_current = self._compute_unrealized_pnl(default_price=default_price)

    def _mark_to_market(self, *, default_price: float) -> float:
        equity = float(self._cash)
        for instrument, qty in self._positions.items():
            if qty == 0:
                continue
            price = self._last_price_by_instrument.get(instrument, default_price)
            equity += float(qty) * float(price)
        return equity

    def _compute_unrealized_pnl(self, *, default_price: float) -> float:
        unrealized = 0.0
        for instrument, qty in self._positions.items():
            if qty == 0:
                continue
            mark = float(self._last_price_by_instrument.get(instrument, default_price))
            avg_cost = float(self._avg_cost_by_instrument.get(instrument, mark))
            if qty > 0:
                unrealized += (mark - avg_cost) * qty
            else:
                unrealized += (avg_cost - mark) * abs(qty)
        return float(unrealized)

    def _update_position_and_realized_pnl(
        self,
        *,
        instrument: str,
        side: str,
        quantity: int,
        fill_price: float,
    ) -> float:
        realized = 0.0
        qty_change = quantity if side == "BUY" else -quantity
        current_qty = int(self._positions.get(instrument, 0))
        current_avg = float(self._avg_cost_by_instrument.get(instrument, 0.0))

        if (
            current_qty == 0
            or (current_qty > 0 and qty_change > 0)
            or (current_qty < 0 and qty_change < 0)
        ):
            new_qty = current_qty + qty_change
            if new_qty == 0:
                self._positions[instrument] = 0
                self._avg_cost_by_instrument.pop(instrument, None)
                return 0.0
            total_qty = abs(current_qty) + abs(qty_change)
            if total_qty > 0:
                new_avg = (
                    (abs(current_qty) * current_avg) + (abs(qty_change) * fill_price)
                ) / total_qty
                self._positions[instrument] = new_qty
                self._avg_cost_by_instrument[instrument] = float(new_avg)
            return 0.0

        if current_qty > 0 and qty_change < 0:
            closing_qty = min(current_qty, abs(qty_change))
            realized += (fill_price - current_avg) * closing_qty
            remaining = current_qty - closing_qty
            new_short = abs(qty_change) - closing_qty
            if remaining > 0:
                self._positions[instrument] = remaining
                self._avg_cost_by_instrument[instrument] = current_avg
            elif new_short > 0:
                self._positions[instrument] = -new_short
                self._avg_cost_by_instrument[instrument] = fill_price
            else:
                self._positions[instrument] = 0
                self._avg_cost_by_instrument.pop(instrument, None)
            return float(realized)

        if current_qty < 0 and qty_change > 0:
            closing_qty = min(abs(current_qty), qty_change)
            realized += (current_avg - fill_price) * closing_qty
            remaining_short = abs(current_qty) - closing_qty
            new_long = qty_change - closing_qty
            if remaining_short > 0:
                self._positions[instrument] = -remaining_short
                self._avg_cost_by_instrument[instrument] = current_avg
            elif new_long > 0:
                self._positions[instrument] = new_long
                self._avg_cost_by_instrument[instrument] = fill_price
            else:
                self._positions[instrument] = 0
                self._avg_cost_by_instrument.pop(instrument, None)
            return float(realized)

        return 0.0
