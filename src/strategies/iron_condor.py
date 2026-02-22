"""Iron condor strategy with strict leg validation and config-gated entries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.data.option_symbols import option_lookup_keys, resolve_option_price
from src.signals import volatility
from src.signals.greeks import mibian_greeks, mibian_implied_iv
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class IronCondorStrategy(BaseStrategy):
    """Daily iron condor lifecycle with robust strike/credit guards."""

    LOT_SIZE_BY_INSTRUMENT: dict[str, int] = {"NIFTY": 50, "BANKNIFTY": 15}
    WEEKDAY_BY_NAME: dict[str, int] = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
    }

    def generate_signal(self, market_data: dict[str, Any], regime: RegimeState) -> Signal:
        raw_ts = market_data.get("timestamp")
        ts = raw_ts if isinstance(raw_ts, datetime) else datetime.now()
        instrument = str(self.config.get("instrument", "NIFTY"))
        quantity = self.compute_position_size(capital=0.0, risk_per_trade=0.0)
        underlying_price = float(market_data.get("underlying_price", 0.0) or 0.0)
        chain_df = self._normalize_chain(market_data.get("option_chain"))
        candles = market_data.get("candles")
        vix_value = float(market_data.get("vix", 0.0) or 0.0)

        if self.state.current_position is not None:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason="Position already open",
            )

        if quantity <= 0:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason="Computed quantity <= 0",
            )

        allowed_days = self._resolve_entry_weekdays()
        weekday = ts.weekday()
        if allowed_days and weekday not in allowed_days:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason=f"Entry weekday blocked: weekday={weekday} allowed={sorted(allowed_days)}",
                indicators={"weekday": weekday, "allowed_entry_weekdays": sorted(allowed_days)},
            )

        min_entry_vix = float(self.config.get("min_entry_vix", 0.0) or 0.0)
        max_entry_vix = float(self.config.get("max_entry_vix", 0.0) or 0.0)
        if vix_value <= 0.0:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason="VIX unavailable for entry gate",
                indicators={"vix": vix_value},
            )
        if vix_value < min_entry_vix or vix_value > max_entry_vix:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason=(
                    "VIX outside entry band: "
                    f"vix={vix_value:.2f}, band=[{min_entry_vix:.2f}, {max_entry_vix:.2f}]"
                ),
                indicators={
                    "vix": vix_value,
                    "min_entry_vix": min_entry_vix,
                    "max_entry_vix": max_entry_vix,
                },
            )

        min_dte = int(self.config.get("dte_min", 5) or 5)
        max_dte = int(self.config.get("dte_max", 14) or 14)
        entry_chain, entry_dte = self._select_chain_for_entry_dte(chain_df=chain_df, now_ts=ts)
        if not chain_df.empty and "expiry" in chain_df.columns and entry_chain.empty:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason=f"No contracts within DTE bounds [{min_dte}, {max_dte}]",
            )

        atr_14 = self._atr_14(candles)
        wing_width = float(self.config.get("wing_width", 100.0) or 100.0)
        atr_multiple = float(self.config.get("atr_multiple", 1.5) or 1.5)
        target_offset = max(wing_width, atr_14 * atr_multiple) if atr_14 > 0.0 else wing_width
        strike_step = self._infer_strike_step(entry_chain)

        legs, selection, error = self._build_condor_legs(
            chain_df=entry_chain,
            now_ts=ts,
            underlying_price=underlying_price,
            instrument=instrument,
            quantity=quantity,
            target_offset=target_offset,
            strike_step=max(strike_step, 1.0),
        )
        if error is not None:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason=error,
                indicators={
                    "vix": vix_value,
                    "atr_14": atr_14,
                    "target_offset": target_offset,
                    **selection,
                },
            )

        entry_credit = self._entry_credit_from_legs(legs)
        min_premium = float(self.config.get("min_premium", 0.0) or 0.0)
        credit_per_unit = (entry_credit / max(quantity, 1)) if quantity > 0 else 0.0
        if credit_per_unit < min_premium:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason=(
                    "Credit per unit below min_premium: "
                    f"{credit_per_unit:.2f} < {min_premium:.2f}"
                ),
                indicators={
                    "entry_credit": entry_credit,
                    "credit_per_unit": credit_per_unit,
                    "min_premium": min_premium,
                    **selection,
                },
            )

        issues = self._validate_structure(
            spot=underlying_price,
            selection=selection,
            wing_width=wing_width,
            strike_step=max(strike_step, 1.0),
            entry_credit=entry_credit,
        )
        if issues:
            return self._no_signal(
                ts=ts,
                regime=regime,
                instrument=instrument,
                reason=f"Entry validation failed: {'; '.join(issues)}",
                indicators={
                    "entry_credit": entry_credit,
                    "credit_per_unit": credit_per_unit,
                    **selection,
                },
            )

        self.state.current_position = {
            "structure": "iron_condor",
            "quantity": quantity,
            "entry_time": ts,
            "entry_regime": regime.value,
            "legs": legs,
            "entry_credit": entry_credit,
            "entry_dte": entry_dte,
            "entry_selection": selection,
        }
        return Signal(
            signal_type=SignalType.ENTRY_SHORT,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            orders=legs,
            regime=regime,
            indicators={
                "vix": vix_value,
                "atr_14": atr_14,
                "target_offset": target_offset,
                "entry_credit": entry_credit,
                "credit_per_unit": credit_per_unit,
                **selection,
            },
            reason="Iron condor entry passed all gates",
        )

    def get_exit_conditions(self, market_data: dict[str, Any]) -> Signal | None:
        if self.state.current_position is None:
            return None

        raw_ts = market_data.get("timestamp")
        ts = raw_ts if isinstance(raw_ts, datetime) else datetime.now()
        instrument = str(self.config.get("instrument", "NIFTY"))
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
        indicators: dict[str, Any] = {
            "close_debit": close_debit,
            "entry_credit": entry_credit,
            "target_close": target_close,
            "stop_close": stop_close,
        }

        if self._is_time_exit(ts):
            reason = "Calendar time exit gate reached"
        elif close_debit <= target_close:
            reason = f"Profit target hit: close_debit={close_debit:.2f} <= {target_close:.2f}"
        elif close_debit >= stop_close:
            reason = f"Stop loss hit: close_debit={close_debit:.2f} >= {stop_close:.2f}"
        else:
            dte_exit = int(self.config.get("dte_exit", 1) or 1)
            dte = self._min_dte(chain_df, legs=legs, now_ts=ts)
            indicators["dte"] = dte
            if dte is not None and dte <= dte_exit:
                reason = f"DTE exit: dte={dte} <= {dte_exit}"

        if reason is None:
            return None

        return Signal(
            signal_type=SignalType.EXIT,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            orders=self._reverse_orders(legs=legs, quantity=qty),
            regime=market_data.get("regime", RegimeState.UNKNOWN),
            indicators=indicators,
            reason=reason,
        )

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        max_lots = int(self.config.get("max_lots", 1) or 1)
        lot_size = self._resolve_lot_size()
        return max(max_lots * lot_size, 0)

    def on_regime_change(self, old_regime: RegimeState, new_regime: RegimeState) -> Signal | None:
        if not self.should_exit_on_regime_change(new_regime=new_regime):
            return None

        instrument = str(self.config.get("instrument", "NIFTY"))
        qty = int(self.state.current_position.get("quantity", 1))
        legs = self.state.current_position.get("legs", [])
        exit_orders = self._reverse_orders(legs=legs, quantity=qty)
        if not exit_orders:
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
        if "expiry" in out.columns:
            out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
        if "ltp" in out.columns:
            out["ltp"] = pd.to_numeric(out["ltp"], errors="coerce")
        out = out.dropna(subset=["strike"]).reset_index(drop=True)
        return out

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

    def _build_condor_legs(
        self,
        *,
        chain_df: pd.DataFrame,
        now_ts: datetime,
        underlying_price: float,
        instrument: str,
        quantity: int,
        target_offset: float,
        strike_step: float,
    ) -> tuple[list[dict[str, Any]], dict[str, float], str | None]:
        selection: dict[str, float] = {}
        if chain_df.empty:
            return [], selection, "Option chain unavailable"

        calls = chain_df.loc[chain_df["option_type"] == "CE"].copy()
        puts = chain_df.loc[chain_df["option_type"] == "PE"].copy()
        if calls.empty or puts.empty or underlying_price <= 0.0:
            return [], selection, "Insufficient chain rows for CE/PE leg construction"

        wing_width = float(self.config.get("wing_width", 100.0) or 100.0)
        call_target = float(self.config.get("call_delta", 0.15) or 0.15)
        put_target = float(self.config.get("put_delta", -0.15) or -0.15)
        rate_pct = float(self.config.get("risk_free_rate_pct", 8.0) or 8.0)
        call_target_strike = float(underlying_price + max(target_offset, strike_step))
        put_target_strike = float(underlying_price - max(target_offset, strike_step))

        call_short = self._pick_short_leg(
            frame=calls,
            now_ts=now_ts,
            underlying_price=underlying_price,
            target_delta=call_target,
            target_strike=call_target_strike,
            is_call=True,
            rate_pct=rate_pct,
            strike_step=strike_step,
        )
        put_short = self._pick_short_leg(
            frame=puts,
            now_ts=now_ts,
            underlying_price=underlying_price,
            target_delta=put_target,
            target_strike=put_target_strike,
            is_call=False,
            rate_pct=rate_pct,
            strike_step=strike_step,
        )
        if call_short is None or put_short is None:
            return [], selection, "Unable to select valid short OTM legs"

        call_hedge_target = float(call_short["strike"]) + wing_width
        put_hedge_target = float(put_short["strike"]) - wing_width
        call_hedge = self._pick_hedge_leg(
            frame=calls,
            short_strike=float(call_short["strike"]),
            target_strike=call_hedge_target,
            is_call=True,
        )
        put_hedge = self._pick_hedge_leg(
            frame=puts,
            short_strike=float(put_short["strike"]),
            target_strike=put_hedge_target,
            is_call=False,
        )
        if call_hedge is None or put_hedge is None:
            return [], selection, "Unable to select valid hedge legs"

        selection = {
            "spot": float(underlying_price),
            "call_short_strike": float(call_short["strike"]),
            "call_hedge_strike": float(call_hedge["strike"]),
            "put_short_strike": float(put_short["strike"]),
            "put_hedge_strike": float(put_hedge["strike"]),
            "call_short_delta_calc": float(call_short.get("_delta_calc", 0.0) or 0.0),
            "put_short_delta_calc": float(put_short.get("_delta_calc", 0.0) or 0.0),
            "call_short_iv_calc": float(call_short.get("_iv_calc", 0.0) or 0.0),
            "put_short_iv_calc": float(put_short.get("_iv_calc", 0.0) or 0.0),
            "call_wing": float(call_hedge["strike"]) - float(call_short["strike"]),
            "put_wing": float(put_short["strike"]) - float(put_hedge["strike"]),
            "target_offset": float(target_offset),
            "call_target_strike": float(call_target_strike),
            "put_target_strike": float(put_target_strike),
        }

        legs = [
            {
                "symbol": self._row_symbol(call_short, instrument, "CE"),
                "action": "SELL",
                "quantity": quantity,
                "price": self._row_price(call_short),
                "expiry": call_short.get("expiry"),
                "strike": float(call_short.get("strike", 0.0) or 0.0),
                "option_type": "CE",
            },
            {
                "symbol": self._row_symbol(put_short, instrument, "PE"),
                "action": "SELL",
                "quantity": quantity,
                "price": self._row_price(put_short),
                "expiry": put_short.get("expiry"),
                "strike": float(put_short.get("strike", 0.0) or 0.0),
                "option_type": "PE",
            },
            {
                "symbol": self._row_symbol(call_hedge, instrument, "CE"),
                "action": "BUY",
                "quantity": quantity,
                "price": self._row_price(call_hedge),
                "expiry": call_hedge.get("expiry"),
                "strike": float(call_hedge.get("strike", 0.0) or 0.0),
                "option_type": "CE",
            },
            {
                "symbol": self._row_symbol(put_hedge, instrument, "PE"),
                "action": "BUY",
                "quantity": quantity,
                "price": self._row_price(put_hedge),
                "expiry": put_hedge.get("expiry"),
                "strike": float(put_hedge.get("strike", 0.0) or 0.0),
                "option_type": "PE",
            },
        ]
        return legs, selection, None

    def _pick_short_leg(
        self,
        *,
        frame: pd.DataFrame,
        now_ts: datetime,
        underlying_price: float,
        target_delta: float,
        target_strike: float,
        is_call: bool,
        rate_pct: float,
        strike_step: float,
    ) -> pd.Series | None:
        if underlying_price <= 0.0 or frame.empty:
            return None

        if is_call:
            candidates = frame.loc[frame["strike"] > underlying_price].copy()
        else:
            candidates = frame.loc[frame["strike"] < underlying_price].copy()
        if candidates.empty:
            return None

        candidates["_strike_score"] = (candidates["strike"] - target_strike).abs() / max(
            strike_step, 1.0
        )
        calc = candidates.apply(
            lambda row: self._derive_delta_and_iv(
                row=row,
                now_ts=now_ts,
                spot=underlying_price,
                rate_pct=rate_pct,
            ),
            axis=1,
            result_type="expand",
        )
        calc.columns = ["_delta_calc", "_iv_calc", "_iv_mode"]
        candidates = pd.concat([candidates, calc], axis=1)

        valid_delta = pd.to_numeric(candidates["_delta_calc"], errors="coerce")
        has_delta = valid_delta.notna().any()
        if has_delta:
            candidates["_delta_score"] = (valid_delta - target_delta).abs() * 3.0
            candidates["_score"] = candidates["_delta_score"] + candidates["_strike_score"]
        else:
            candidates["_score"] = candidates["_strike_score"]

        scored = candidates.sort_values(["_score", "_strike_score"])
        if scored.empty:
            return None
        return scored.iloc[0]

    @staticmethod
    def _pick_hedge_leg(
        *,
        frame: pd.DataFrame,
        short_strike: float,
        target_strike: float,
        is_call: bool,
    ) -> pd.Series | None:
        if frame.empty:
            return None
        if is_call:
            candidates = frame.loc[frame["strike"] > short_strike].copy()
        else:
            candidates = frame.loc[frame["strike"] < short_strike].copy()
        if candidates.empty:
            return None
        candidates["_score"] = (candidates["strike"] - target_strike).abs()
        scored = candidates.sort_values(["_score", "strike"])
        if scored.empty:
            return None
        return scored.iloc[0]

    def _derive_delta_and_iv(
        self,
        *,
        row: pd.Series,
        now_ts: datetime,
        spot: float,
        rate_pct: float,
    ) -> tuple[float | None, float | None, str]:
        strike = float(self._to_float(row.get("strike")) or 0.0)
        option_type = str(row.get("option_type", "")).upper()
        if option_type not in {"CE", "PE"}:
            return None, None, "invalid_option_type"

        expiry_raw = row.get("expiry")
        if expiry_raw is None:
            return None, None, "missing_expiry"
        expiry = pd.to_datetime(str(expiry_raw), errors="coerce")
        if pd.isna(expiry):
            return None, None, "missing_expiry"
        dte = int((expiry.normalize() - pd.Timestamp(now_ts).normalize()).days)
        days_to_expiry = max(1, dte)
        price = self._row_price(row)
        iv = mibian_implied_iv(
            spot=spot,
            strike=strike,
            rate_pct=rate_pct,
            days_to_expiry=days_to_expiry,
            option_price=price,
            option_type=option_type,  # type: ignore[arg-type]
        )
        iv_mode = "implied"
        if iv is None:
            iv = float(self.config.get("fallback_iv_pct", 20.0) or 20.0)
            iv_mode = "fallback"

        greeks = mibian_greeks(
            spot=spot,
            strike=strike,
            rate_pct=rate_pct,
            days_to_expiry=days_to_expiry,
            iv_pct=iv,
            option_type=option_type,  # type: ignore[arg-type]
        )
        delta = self._to_float(greeks.get("delta"))
        if delta is None:
            return None, iv, "delta_nan"
        return float(delta), float(iv), iv_mode

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
            parsed = IronCondorStrategy._to_float(row.get(key))
            if parsed is not None:
                return parsed
        return 0.0

    def _resolve_lot_size(self) -> int:
        cfg_lot = self._to_float(self.config.get("lot_size"))
        if cfg_lot is not None and int(cfg_lot) > 0:
            return int(cfg_lot)
        instrument = str(self.config.get("instrument", "NIFTY")).upper()
        return int(self.LOT_SIZE_BY_INSTRUMENT.get(instrument, 1))

    def _resolve_entry_weekdays(self) -> set[int]:
        configured = self.config.get("entry_days", [0, 1])
        out: set[int] = set()
        if isinstance(configured, list):
            for raw in configured:
                value = self._to_float(raw)
                if value is not None:
                    weekday = int(value)
                    if 0 <= weekday <= 6:
                        out.add(weekday)
                        continue
                name = str(raw).strip().lower()
                if name in self.WEEKDAY_BY_NAME:
                    out.add(self.WEEKDAY_BY_NAME[name])
        return out

    def _resolve_time_exit_weekday(self) -> int:
        raw_weekday = self._to_float(self.config.get("time_exit_weekday"))
        if raw_weekday is not None:
            weekday = int(raw_weekday)
            if 0 <= weekday <= 6:
                return weekday
        raw_day = str(self.config.get("time_exit_day", "wednesday")).strip().lower()
        return int(self.WEEKDAY_BY_NAME.get(raw_day, 2))

    def _is_time_exit(self, ts: datetime) -> bool:
        if not bool(self.config.get("enable_time_exit", True)):
            return False
        entry_time = (
            self.state.current_position.get("entry_time") if self.state.current_position else None
        )
        if not isinstance(entry_time, datetime):
            return False
        if ts.date() <= entry_time.date():
            return False
        return ts.weekday() >= self._resolve_time_exit_weekday()

    @staticmethod
    def _entry_credit_from_legs(legs: list[dict[str, Any]]) -> float:
        credit = 0.0
        for leg in legs:
            qty = float(leg.get("quantity", 1) or 1)
            px = float(leg.get("price", 0.0) or 0.0)
            action = str(leg.get("action", "")).upper()
            if action == "SELL":
                credit += px * qty
            elif action == "BUY":
                credit -= px * qty
        return float(credit)

    @staticmethod
    def _price_map_from_chain(chain_df: pd.DataFrame) -> dict[str, float]:
        price_col = None
        for col in ("ltp", "last_price", "close", "price"):
            if col in chain_df.columns:
                price_col = col
                break
        if price_col is None:
            return {}
        symbol_col = (
            "symbol"
            if "symbol" in chain_df.columns
            else "tradingsymbol"
            if "tradingsymbol" in chain_df.columns
            else None
        )
        if symbol_col is None:
            return {}
        out: dict[str, float] = {}
        for _, row in chain_df.iterrows():
            symbol = str(row.get(symbol_col, "")).strip()
            raw_price = row.get(price_col)
            if raw_price is None:
                continue
            price = IronCondorStrategy._to_float(raw_price)
            if not symbol or price is None:
                continue
            resolved_price = float(price)
            for lookup_key in option_lookup_keys(
                symbol=symbol,
                expiry=row.get("expiry"),
                strike=row.get("strike"),
                option_type=row.get("option_type"),
            ):
                out[lookup_key] = resolved_price
        return out

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return parsed

    @staticmethod
    def _close_debit(*, legs: list[dict[str, Any]], price_map: dict[str, float]) -> float | None:
        close_debit = 0.0
        for leg in legs:
            symbol = str(leg.get("symbol", "")).strip()
            px = resolve_option_price(
                price_lookup=price_map,
                symbol=symbol,
                expiry=leg.get("expiry"),
                strike=leg.get("strike"),
                option_type=leg.get("option_type"),
            )
            if px is None:
                return None
            qty = float(leg.get("quantity", 1) or 1)
            action = str(leg.get("action", "")).upper()
            if action == "SELL":
                close_debit += px * qty
            elif action == "BUY":
                close_debit -= px * qty
        return float(close_debit)

    @staticmethod
    def _reverse_orders(*, legs: list[dict[str, Any]], quantity: int) -> list[dict[str, Any]]:
        exit_orders: list[dict[str, Any]] = []
        for leg in legs:
            action = str(leg.get("action", "")).upper()
            exit_action = "BUY" if action == "SELL" else "SELL"
            order = {
                "symbol": str(leg.get("symbol", "")),
                "action": exit_action,
                "quantity": int(leg.get("quantity", quantity) or quantity),
            }
            expiry = leg.get("expiry")
            strike = leg.get("strike")
            option_type = leg.get("option_type")
            if expiry is not None:
                order["expiry"] = expiry
            if strike is not None:
                order["strike"] = strike
            if option_type is not None:
                order["option_type"] = option_type
            exit_orders.append(order)
        return exit_orders

    @staticmethod
    def _min_dte(
        chain_df: pd.DataFrame, *, legs: list[dict[str, Any]], now_ts: datetime
    ) -> int | None:
        if "expiry" not in chain_df.columns:
            return None
        symbols = {str(leg.get("symbol", "")).strip() for leg in legs}
        symbol_col = (
            "symbol"
            if "symbol" in chain_df.columns
            else "tradingsymbol"
            if "tradingsymbol" in chain_df.columns
            else None
        )
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
        return int((min_expiry.date() - now_ts.date()).days)

    @staticmethod
    def _infer_strike_step(chain_df: pd.DataFrame) -> float:
        if chain_df.empty or "strike" not in chain_df.columns:
            return 50.0
        strikes = pd.to_numeric(chain_df["strike"], errors="coerce").dropna().sort_values().unique()
        if len(strikes) < 2:
            return 50.0
        diffs = pd.Series(strikes).diff().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return 50.0
        return float(diffs.median())

    @staticmethod
    def _validate_structure(
        *,
        spot: float,
        selection: dict[str, float],
        wing_width: float,
        strike_step: float,
        entry_credit: float,
    ) -> list[str]:
        issues: list[str] = []
        call_short = float(selection.get("call_short_strike", 0.0) or 0.0)
        call_hedge = float(selection.get("call_hedge_strike", 0.0) or 0.0)
        put_short = float(selection.get("put_short_strike", 0.0) or 0.0)
        put_hedge = float(selection.get("put_hedge_strike", 0.0) or 0.0)
        call_wing = float(selection.get("call_wing", 0.0) or 0.0)
        put_wing = float(selection.get("put_wing", 0.0) or 0.0)

        if call_short <= spot:
            issues.append(f"call short not OTM: call_short={call_short:.2f}, spot={spot:.2f}")
        if put_short >= spot:
            issues.append(f"put short not OTM: put_short={put_short:.2f}, spot={spot:.2f}")
        if call_hedge <= call_short:
            issues.append(
                f"call hedge invalid: call_hedge={call_hedge:.2f}, call_short={call_short:.2f}"
            )
        if put_hedge >= put_short:
            issues.append(
                f"put hedge invalid: put_hedge={put_hedge:.2f}, put_short={put_short:.2f}"
            )
        if call_wing <= 0.0 or put_wing <= 0.0:
            issues.append(
                f"zero/negative wing: call_wing={call_wing:.2f}, put_wing={put_wing:.2f}"
            )

        tolerance = max(strike_step, 1.0)
        if abs(call_wing - wing_width) > tolerance:
            issues.append(
                f"call wing width mismatch: actual={call_wing:.2f}, target={wing_width:.2f}"
            )
        if abs(put_wing - wing_width) > tolerance:
            issues.append(
                f"put wing width mismatch: actual={put_wing:.2f}, target={wing_width:.2f}"
            )

        if entry_credit <= 0.0:
            issues.append(f"non-positive entry credit: entry_credit={entry_credit:.2f}")
        return issues

    @staticmethod
    def _atr_14(candles: pd.DataFrame | None) -> float:
        if candles is None or candles.empty or len(candles) < 20:
            return 0.0
        required = {"high", "low", "close"}
        if not required.issubset(candles.columns):
            return 0.0
        high = candles["high"].astype("float64")
        low = candles["low"].astype("float64")
        close = candles["close"].astype("float64")
        series = volatility.atr(high=high, low=low, close=close, period=14).dropna()
        if series.empty:
            return 0.0
        return float(series.iloc[-1])

    def _no_signal(
        self,
        *,
        ts: datetime,
        regime: RegimeState,
        instrument: str,
        reason: str,
        indicators: dict[str, Any] | None = None,
    ) -> Signal:
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason=reason,
            indicators=indicators or {},
        )
