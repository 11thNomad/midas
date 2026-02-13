"""Iron condor scaffold strategy for regime-aware backtest plumbing."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class IronCondorStrategy(BaseStrategy):
    """Simplified iron condor lifecycle for early backtest integration."""

    def generate_signal(self, market_data: dict, regime: RegimeState) -> Signal:
        ts = market_data.get("timestamp", datetime.now())
        instrument = self.config.get("instrument", "NIFTY")
        lots = self.compute_position_size(capital=0, risk_per_trade=0)
        underlying_price = float(market_data.get("underlying_price", 0.0) or 0.0)
        chain_df = self._normalize_chain(market_data.get("option_chain"))

        if self.state.current_position is None:
            entry_chain, entry_dte = self._select_chain_for_entry_dte(chain_df=chain_df, now_ts=ts)
            if not chain_df.empty and "expiry" in chain_df.columns and entry_chain.empty:
                return Signal(
                    signal_type=SignalType.NO_SIGNAL,
                    strategy_name=self.name,
                    instrument=instrument,
                    timestamp=ts,
                    regime=regime,
                    reason=(
                        f"No contracts within DTE bounds "
                        f"[{int(self.config.get('dte_min', 5) or 5)}, {int(self.config.get('dte_max', 14) or 14)}]"
                    ),
                )

            legs = self._build_condor_legs(
                chain_df=entry_chain,
                underlying_price=underlying_price,
                instrument=instrument,
                quantity=lots,
            )
            entry_credit = self._entry_credit_from_legs(legs)
            self.state.current_position = {
                "structure": "iron_condor",
                "quantity": lots,
                "entry_time": ts,
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
                reason="Iron condor entry in allowed low-vol regime",
            )

        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason="Position already open",
        )

    def get_exit_conditions(self, market_data: dict) -> Signal | None:
        if self.state.current_position is None:
            return None

        ts = market_data.get("timestamp", datetime.now())
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
        if self.state.current_position is None:
            return None
        if self.should_be_active(new_regime):
            return None

        instrument = self.config.get("instrument", "NIFTY")
        qty = int(self.state.current_position.get("quantity", 1))
        legs = self.state.current_position.get("legs", [])
        self.state.current_position = None

        if legs:
            exit_orders = self._reverse_orders(legs=legs, quantity=qty)
        else:
            exit_orders = [
                {"symbol": f"{instrument}_CALL_SHORT", "action": "BUY", "quantity": qty},
                {"symbol": f"{instrument}_PUT_SHORT", "action": "BUY", "quantity": qty},
                {"symbol": f"{instrument}_CALL_HEDGE", "action": "SELL", "quantity": qty},
                {"symbol": f"{instrument}_PUT_HEDGE", "action": "SELL", "quantity": qty},
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
        out = out.dropna(subset=["strike"])
        return out.reset_index(drop=True)

    def _select_chain_for_entry_dte(self, *, chain_df: pd.DataFrame, now_ts: datetime) -> tuple[pd.DataFrame, int | None]:
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

    def _build_condor_legs(
        self,
        *,
        chain_df: pd.DataFrame,
        underlying_price: float,
        instrument: str,
        quantity: int,
    ) -> list[dict]:
        wing_width = float(self.config.get("wing_width", 100))
        call_target = float(self.config.get("call_delta", 0.15))
        put_target = float(self.config.get("put_delta", -0.15))

        if chain_df.empty:
            return [
                {"symbol": f"{instrument}_CALL_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_PUT_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_CALL_HEDGE", "action": "BUY", "quantity": quantity},
                {"symbol": f"{instrument}_PUT_HEDGE", "action": "BUY", "quantity": quantity},
            ]

        calls = chain_df.loc[chain_df["option_type"] == "CE"].copy()
        puts = chain_df.loc[chain_df["option_type"] == "PE"].copy()
        if calls.empty or puts.empty:
            return [
                {"symbol": f"{instrument}_CALL_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_PUT_SHORT", "action": "SELL", "quantity": quantity},
                {"symbol": f"{instrument}_CALL_HEDGE", "action": "BUY", "quantity": quantity},
                {"symbol": f"{instrument}_PUT_HEDGE", "action": "BUY", "quantity": quantity},
            ]

        call_short = self._pick_short_leg(calls, target_delta=call_target, is_call=True, underlying_price=underlying_price)
        put_short = self._pick_short_leg(puts, target_delta=put_target, is_call=False, underlying_price=underlying_price)

        call_hedge_strike = float(call_short["strike"]) + wing_width
        put_hedge_strike = float(put_short["strike"]) - wing_width
        call_hedge = self._pick_nearest_strike(calls, target_strike=call_hedge_strike)
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
                "symbol": self._row_symbol(call_hedge, instrument, "CE"),
                "action": "BUY",
                "quantity": quantity,
                "price": self._row_price(call_hedge),
            },
            {
                "symbol": self._row_symbol(put_hedge, instrument, "PE"),
                "action": "BUY",
                "quantity": quantity,
                "price": self._row_price(put_hedge),
            },
        ]

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
        for key in ("ltp", "last_price", "close", "price"):
            value = row.get(key)
            parsed = pd.to_numeric(value, errors="coerce")
            if not pd.isna(parsed):
                return float(parsed)
        return 0.0

    def _pick_short_leg(
        self,
        frame: pd.DataFrame,
        *,
        target_delta: float,
        is_call: bool,
        underlying_price: float,
    ) -> pd.Series:
        if "delta" in frame.columns and not frame["delta"].dropna().empty:
            scored = frame.assign(_score=(frame["delta"] - target_delta).abs()).sort_values("_score")
            return scored.iloc[0]

        if underlying_price > 0:
            if is_call:
                otm = frame.loc[frame["strike"] >= underlying_price].copy()
            else:
                otm = frame.loc[frame["strike"] <= underlying_price].copy()
            if not otm.empty:
                scored = otm.assign(_score=(otm["strike"] - underlying_price).abs()).sort_values("_score")
                return scored.iloc[0]

        return self._pick_nearest_strike(frame, target_strike=float(frame["strike"].median()))

    @staticmethod
    def _pick_nearest_strike(frame: pd.DataFrame, *, target_strike: float) -> pd.Series:
        scored = frame.assign(_score=(frame["strike"] - target_strike).abs()).sort_values("_score")
        return scored.iloc[0]

    @staticmethod
    def _entry_credit_from_legs(legs: list[dict]) -> float:
        credit = 0.0
        for leg in legs:
            qty = float(leg.get("quantity", 1) or 1)
            px = float(leg.get("price", 0.0) or 0.0)
            action = str(leg.get("action", "")).upper()
            if action == "SELL":
                credit += px * qty
            elif action == "BUY":
                credit -= px * qty
        return max(credit, 0.0)

    @staticmethod
    def _price_map_from_chain(chain_df: pd.DataFrame) -> dict[str, float]:
        price_col = None
        for col in ("ltp", "last_price", "close", "price"):
            if col in chain_df.columns:
                price_col = col
                break
        if price_col is None:
            return {}
        symbol_col = "symbol" if "symbol" in chain_df.columns else "tradingsymbol" if "tradingsymbol" in chain_df.columns else None
        if symbol_col is None:
            return {}
        out: dict[str, float] = {}
        for _, row in chain_df.iterrows():
            symbol = str(row.get(symbol_col, "")).strip()
            price = pd.to_numeric(row.get(price_col), errors="coerce")
            if symbol and not pd.isna(price):
                out[symbol] = float(price)
        return out

    @staticmethod
    def _close_debit(*, legs: list[dict], price_map: dict[str, float]) -> float | None:
        close_debit = 0.0
        for leg in legs:
            symbol = str(leg.get("symbol", "")).strip()
            if symbol not in price_map:
                return None
            px = float(price_map[symbol])
            qty = float(leg.get("quantity", 1) or 1)
            action = str(leg.get("action", "")).upper()
            if action == "SELL":
                close_debit += px * qty
            elif action == "BUY":
                close_debit -= px * qty
        return float(close_debit)

    @staticmethod
    def _reverse_orders(*, legs: list[dict], quantity: int) -> list[dict]:
        exit_orders = []
        for leg in legs:
            action = str(leg.get("action", "")).upper()
            exit_action = "BUY" if action == "SELL" else "SELL"
            exit_orders.append({"symbol": leg.get("symbol", ""), "action": exit_action, "quantity": quantity})
        return exit_orders

    @staticmethod
    def _min_dte(chain_df: pd.DataFrame, *, legs: list[dict], now_ts: datetime) -> int | None:
        if "expiry" not in chain_df.columns:
            return None
        symbols = {str(leg.get("symbol", "")).strip() for leg in legs}
        symbol_col = "symbol" if "symbol" in chain_df.columns else "tradingsymbol" if "tradingsymbol" in chain_df.columns else None
        if symbol_col is None:
            return None
        subset = chain_df.loc[chain_df[symbol_col].astype(str).isin(symbols)].copy()
        if subset.empty:
            return None
        subset["expiry"] = pd.to_datetime(subset["expiry"], errors="coerce")
        subset = subset.dropna(subset=["expiry"])
        if subset.empty:
            return None
        min_expiry = pd.Timestamp(subset["expiry"].min()).to_pydatetime()
        dte = (min_expiry.date() - now_ts.date()).days
        return int(dte)
