"""Jade lizard scaffold strategy for regime-aware options routing."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class JadeLizardStrategy(BaseStrategy):
    """Three-leg options structure with configurable bullish/bearish bias."""

    VALID_VARIANTS = {"neutral", "bullish", "bearish"}

    def generate_signal(self, market_data: dict[str, Any], regime: RegimeState) -> Signal:
        raw_ts = market_data.get("timestamp")
        ts = raw_ts if isinstance(raw_ts, datetime) else datetime.now()
        instrument = self.config.get("instrument", "NIFTY")
        lots = self.compute_position_size(capital=0, risk_per_trade=0)
        underlying_price = float(market_data.get("underlying_price", 0.0) or 0.0)
        chain_df = self._normalize_chain(market_data.get("option_chain"))
        min_dte = int(self.config.get("dte_min", 5) or 5)
        max_dte = int(self.config.get("dte_max", 14) or 14)

        if self.state.current_position is None:
            entry_chain, entry_dte = self._select_chain_for_entry_dte(chain_df=chain_df, now_ts=ts)
            if not chain_df.empty and "expiry" in chain_df.columns and entry_chain.empty:
                return Signal(
                    signal_type=SignalType.NO_SIGNAL,
                    strategy_name=self.name,
                    instrument=instrument,
                    timestamp=ts,
                    regime=regime,
                    reason=f"No contracts within DTE bounds [{min_dte}, {max_dte}]",
                )

            variant = self._resolve_effective_variant(market_data)
            legs = self._build_lizard_legs(
                chain_df=entry_chain,
                underlying_price=underlying_price,
                instrument=instrument,
                quantity=lots,
                variant=variant,
            )
            entry_credit = self._entry_credit_from_legs(legs)
            min_entry_credit = float(self.config.get("min_entry_credit", 0.0) or 0.0)
            if entry_credit < min_entry_credit:
                return Signal(
                    signal_type=SignalType.NO_SIGNAL,
                    strategy_name=self.name,
                    instrument=instrument,
                    timestamp=ts,
                    regime=regime,
                    reason=(
                        "Entry credit "
                        f"{entry_credit:.2f} below min_entry_credit {min_entry_credit:.2f}"
                    ),
                )

            self.state.current_position = {
                "structure": "jade_lizard",
                "variant": variant,
                "quantity": lots,
                "entry_time": ts,
                "entry_regime": regime.value,
                "legs": legs,
                "entry_credit": entry_credit,
                "entry_dte": entry_dte,
            }
            return Signal(
                signal_type=SignalType.ENTRY_SHORT,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts,
                orders=legs,
                regime=regime,
                reason=f"Jade lizard entry ({variant}) in allowed regime",
            )

        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason="Position already open",
        )

    def get_exit_conditions(self, market_data: dict[str, Any]) -> Signal | None:
        if self.state.current_position is None:
            return None

        raw_ts = market_data.get("timestamp")
        ts = raw_ts if isinstance(raw_ts, datetime) else datetime.now()
        instrument = self.config.get("instrument", "NIFTY")
        chain_df = self._normalize_chain(market_data.get("option_chain"))
        if chain_df.empty:
            return None

        legs = self.state.current_position.get("legs", [])
        qty = int(self.state.current_position.get("quantity", 1))
        entry_credit = float(self.state.current_position.get("entry_credit", 0.0) or 0.0)
        if not legs or entry_credit <= 0:
            return None

        price_map = self._price_map_from_chain(chain_df)
        close_debit = self._close_debit(legs=legs, price_map=price_map)
        if close_debit is None:
            return None

        profit_target_pct = float(self.config.get("profit_target_pct", 50.0) or 50.0)
        stop_loss_pct = float(self.config.get("stop_loss_pct", 100.0) or 100.0)
        target_close = entry_credit * max(0.0, 1.0 - (profit_target_pct / 100.0))
        stop_close = entry_credit * (1.0 + (stop_loss_pct / 100.0))

        reason = None
        if close_debit <= target_close:
            reason = f"Profit target hit: close_debit={close_debit:.2f} <= {target_close:.2f}"
        elif close_debit >= stop_close:
            reason = f"Stop loss hit: close_debit={close_debit:.2f} >= {stop_close:.2f}"
        else:
            dte_exit = int(self.config.get("dte_exit", 1) or 1)
            dte = self._min_dte(chain_df, legs=legs, now_ts=ts)
            if dte is not None and dte <= dte_exit:
                reason = f"DTE exit: dte={dte} <= {dte_exit}"

        if reason is None:
            return None

        self.state.current_position = None
        return Signal(
            signal_type=SignalType.EXIT,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            orders=self._reverse_orders(legs=legs, quantity=qty),
            regime=market_data.get("regime", RegimeState.UNKNOWN),
            reason=reason,
        )

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return int(self.config.get("max_lots", 1) or 1)

    def on_regime_change(self, old_regime: RegimeState, new_regime: RegimeState) -> Signal | None:
        if not self.should_exit_on_regime_change(new_regime=new_regime):
            return None

        instrument = self.config.get("instrument", "NIFTY")
        qty = int(self.state.current_position.get("quantity", 1))
        legs = self.state.current_position.get("legs", [])
        self.state.current_position = None

        exit_orders = self._reverse_orders(legs=legs, quantity=qty)
        if not exit_orders:
            exit_orders = [
                {"symbol": f"{instrument}_CALL_SHORT", "action": "BUY", "quantity": qty},
                {"symbol": f"{instrument}_PUT_SHORT", "action": "BUY", "quantity": qty},
                {"symbol": f"{instrument}_HEDGE", "action": "SELL", "quantity": qty},
            ]

        return Signal(
            signal_type=SignalType.EXIT,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=datetime.now(),
            orders=exit_orders,
            regime=new_regime,
            reason=f"Regime changed from {old_regime.value} to {new_regime.value}",
        )

    @staticmethod
    def _normalize_chain(chain_df: pd.DataFrame | None) -> pd.DataFrame:
        if chain_df is None or chain_df.empty:
            return pd.DataFrame()
        out = chain_df.copy()
        if "option_type" not in out.columns or "strike" not in out.columns:
            return pd.DataFrame()
        out["option_type"] = out["option_type"].astype(str).str.upper()
        out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
        if "delta" in out.columns:
            out["delta"] = pd.to_numeric(out["delta"], errors="coerce")
        if "expiry" in out.columns:
            out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
        if "ltp" in out.columns:
            out["ltp"] = pd.to_numeric(out["ltp"], errors="coerce")
        out = out.dropna(subset=["strike"])
        return out.reset_index(drop=True)

    def _select_chain_for_entry_dte(
        self, *, chain_df: pd.DataFrame, now_ts: datetime
    ) -> tuple[pd.DataFrame, int | None]:
        if chain_df.empty or "expiry" not in chain_df.columns:
            return chain_df, None

        dte_min = int(self.config.get("dte_min", 5) or 5)
        dte_max = int(self.config.get("dte_max", 14) or 14)
        if dte_min > dte_max:
            dte_min, dte_max = dte_max, dte_min

        out = chain_df.copy()
        out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
        out = out.dropna(subset=["expiry"])
        if out.empty:
            return pd.DataFrame(), None

        now_day = pd.Timestamp(now_ts).normalize()
        out["_dte"] = (out["expiry"].dt.normalize() - now_day).dt.days
        valid = out.loc[(out["_dte"] >= dte_min) & (out["_dte"] <= dte_max)].copy()
        if valid.empty:
            return pd.DataFrame(), None

        target_dte = int(valid["_dte"].min())
        min_expiry = valid.loc[valid["_dte"] == target_dte, "expiry"].min()
        selected = valid.loc[valid["expiry"] == min_expiry].drop(columns=["_dte"])
        return selected.reset_index(drop=True), target_dte

    def _resolve_effective_variant(self, market_data: dict[str, Any]) -> str:
        configured = str(self.config.get("variant", "bullish")).strip().lower()
        if configured not in self.VALID_VARIANTS:
            configured = "bullish"
        if configured != "neutral":
            return configured

        bias = self._coerce_float(market_data.get("bias"))
        if bias > 0.1:
            return "bullish"
        if bias < -0.1:
            return "bearish"

        fii_net_3d = self._coerce_float(market_data.get("fii_net_3d"))
        if fii_net_3d > 0.0:
            return "bullish"
        if fii_net_3d < 0.0:
            return "bearish"

        fallback = str(self.config.get("neutral_fallback", "bullish")).strip().lower()
        if fallback in {"bullish", "bearish"}:
            return fallback
        return "bullish"

    def _build_lizard_legs(
        self,
        *,
        chain_df: pd.DataFrame,
        underlying_price: float,
        instrument: str,
        quantity: int,
        variant: str,
    ) -> list[dict[str, Any]]:
        spread_width = float(self.config.get("spread_width", 100))
        call_target = float(self.config.get("call_delta", 0.15))
        put_target = float(self.config.get("put_delta", -0.15))

        if chain_df.empty:
            if variant == "bearish":
                return [
                    {"symbol": f"{instrument}_CALL_SHORT", "action": "SELL", "quantity": quantity},
                    {"symbol": f"{instrument}_PUT_SHORT", "action": "SELL", "quantity": quantity},
                    {"symbol": f"{instrument}_PUT_HEDGE", "action": "BUY", "quantity": quantity},
                ]
            return [
                {"symbol": f"{instrument}_PUT_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_CALL_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_CALL_HEDGE", "action": "BUY", "quantity": quantity},
            ]

        calls = chain_df.loc[chain_df["option_type"] == "CE"].copy()
        puts = chain_df.loc[chain_df["option_type"] == "PE"].copy()
        if calls.empty or puts.empty:
            return [
                {"symbol": f"{instrument}_PUT_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_CALL_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_HEDGE", "action": "BUY", "quantity": quantity},
            ]

        call_short = self._pick_short_leg(
            calls, target_delta=call_target, is_call=True, underlying_price=underlying_price
        )
        put_short = self._pick_short_leg(
            puts, target_delta=put_target, is_call=False, underlying_price=underlying_price
        )

        if variant == "bearish":
            put_hedge_strike = float(put_short["strike"]) - spread_width
            put_hedge = self._pick_nearest_strike(puts, target_strike=put_hedge_strike)
            return [
                {
                    "symbol": self._row_symbol(call_short, instrument, "CE"),
                    "action": "SELL",
                    "quantity": quantity,
                    "price": self._row_price(call_short),
                },
                {
                    "symbol": self._row_symbol(put_short, instrument, "PE"),
                    "action": "SELL",
                    "quantity": quantity,
                    "price": self._row_price(put_short),
                },
                {
                    "symbol": self._row_symbol(put_hedge, instrument, "PE"),
                    "action": "BUY",
                    "quantity": quantity,
                    "price": self._row_price(put_hedge),
                },
            ]

        call_hedge_strike = float(call_short["strike"]) + spread_width
        call_hedge = self._pick_nearest_strike(calls, target_strike=call_hedge_strike)
        return [
            {
                "symbol": self._row_symbol(put_short, instrument, "PE"),
                "action": "SELL",
                "quantity": quantity,
                "price": self._row_price(put_short),
            },
            {
                "symbol": self._row_symbol(call_short, instrument, "CE"),
                "action": "SELL",
                "quantity": quantity,
                "price": self._row_price(call_short),
            },
            {
                "symbol": self._row_symbol(call_hedge, instrument, "CE"),
                "action": "BUY",
                "quantity": quantity,
                "price": self._row_price(call_hedge),
            },
        ]

    @staticmethod
    def _pick_short_leg(
        frame: pd.DataFrame,
        *,
        target_delta: float,
        is_call: bool,
        underlying_price: float,
    ) -> pd.Series:
        candidates = frame.copy()
        has_delta = "delta" in candidates.columns and candidates["delta"].notna().any()
        if has_delta:
            candidates["_delta_dist"] = (candidates["delta"] - target_delta).abs()
            return candidates.sort_values("_delta_dist").iloc[0]

        if is_call:
            otm = candidates.loc[candidates["strike"] >= underlying_price]
        else:
            otm = candidates.loc[candidates["strike"] <= underlying_price]
        if not otm.empty:
            candidates = otm
        candidates["_strike_dist"] = (candidates["strike"] - underlying_price).abs()
        return candidates.sort_values("_strike_dist").iloc[0]

    @staticmethod
    def _pick_nearest_strike(frame: pd.DataFrame, *, target_strike: float) -> pd.Series:
        candidates = frame.copy()
        candidates["_strike_dist"] = (candidates["strike"] - target_strike).abs()
        return candidates.sort_values("_strike_dist").iloc[0]

    @staticmethod
    def _row_symbol(row: pd.Series, instrument: str, opt_type: str) -> str:
        for key in ("symbol", "tradingsymbol"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value)
        strike = int(float(row.get("strike", 0.0) or 0.0))
        return f"{instrument}_{strike}{opt_type}"

    @staticmethod
    def _row_price(row: pd.Series) -> float:
        for key in ("ltp", "close", "last_price"):
            value = row.get(key)
            if pd.notna(value):
                return float(value)
        return 0.0

    @staticmethod
    def _entry_credit_from_legs(legs: list[dict[str, Any]]) -> float:
        credit = 0.0
        for leg in legs:
            price = float(leg.get("price", 0.0) or 0.0)
            action = str(leg.get("action", "")).upper()
            if action == "SELL":
                credit += price
            elif action == "BUY":
                credit -= price
        return float(credit)

    @staticmethod
    def _price_map_from_chain(chain_df: pd.DataFrame) -> dict[str, float]:
        prices: dict[str, float] = {}
        if chain_df.empty:
            return prices
        symbol_col = "symbol" if "symbol" in chain_df.columns else "tradingsymbol"
        if symbol_col in chain_df.columns:
            for _, row in chain_df.iterrows():
                symbol = str(row.get(symbol_col, "")).strip()
                if not symbol:
                    continue
                prices[symbol] = float(row.get("ltp", 0.0) or 0.0)
        return prices

    @staticmethod
    def _close_debit(legs: list[dict[str, Any]], price_map: dict[str, float]) -> float | None:
        close_debit = 0.0
        for leg in legs:
            symbol = str(leg.get("symbol", ""))
            if symbol not in price_map:
                return None
            leg_price = float(price_map[symbol])
            action = str(leg.get("action", "")).upper()
            if action == "SELL":
                close_debit += leg_price
            elif action == "BUY":
                close_debit -= leg_price
        return float(close_debit)

    @staticmethod
    def _reverse_orders(legs: list[dict[str, Any]], quantity: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for leg in legs:
            action = str(leg.get("action", "")).upper()
            if action == "BUY":
                reverse = "SELL"
            elif action == "SELL":
                reverse = "BUY"
            else:
                continue
            out.append(
                {
                    "symbol": leg.get("symbol"),
                    "action": reverse,
                    "quantity": quantity,
                }
            )
        return out

    @staticmethod
    def _min_dte(
        chain_df: pd.DataFrame,
        *,
        legs: list[dict[str, Any]],
        now_ts: datetime,
    ) -> int | None:
        if chain_df.empty or "expiry" not in chain_df.columns:
            return None
        if "symbol" not in chain_df.columns and "tradingsymbol" not in chain_df.columns:
            return None

        symbol_col = "symbol" if "symbol" in chain_df.columns else "tradingsymbol"
        needed_symbols = {str(leg.get("symbol", "")) for leg in legs if leg.get("symbol")}
        if not needed_symbols:
            return None

        match = chain_df.loc[chain_df[symbol_col].astype(str).isin(needed_symbols)].copy()
        if match.empty:
            return None
        match["expiry"] = pd.to_datetime(match["expiry"], errors="coerce")
        match = match.dropna(subset=["expiry"])
        if match.empty:
            return None

        now_day = pd.Timestamp(now_ts).normalize()
        dte = (match["expiry"].dt.normalize() - now_day).dt.days.min()
        return int(dte) if pd.notna(dte) else None

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
