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
            legs = self._build_condor_legs(
                chain_df=chain_df,
                underlying_price=underlying_price,
                instrument=instrument,
                quantity=lots,
            )
            self.state.current_position = {
                "structure": "iron_condor",
                "quantity": lots,
                "entry_time": ts,
                "legs": legs,
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
        return None

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
            exit_orders = []
            for leg in legs:
                action = str(leg.get("action", "")).upper()
                exit_action = "BUY" if action == "SELL" else "SELL"
                exit_orders.append({"symbol": leg.get("symbol", instrument), "action": exit_action, "quantity": qty})
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
        out = out.dropna(subset=["strike"])
        return out.reset_index(drop=True)

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
            {"symbol": self._row_symbol(call_short, instrument, "CE"), "action": "SELL", "quantity": quantity},
            {"symbol": self._row_symbol(put_short, instrument, "PE"), "action": "SELL", "quantity": quantity},
            {"symbol": self._row_symbol(call_hedge, instrument, "CE"), "action": "BUY", "quantity": quantity},
            {"symbol": self._row_symbol(put_hedge, instrument, "PE"), "action": "BUY", "quantity": quantity},
        ]

    @staticmethod
    def _row_symbol(row: pd.Series, instrument: str, opt_type: str) -> str:
        for key in ("symbol", "tradingsymbol"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value)
        strike = int(float(row.get("strike", 0.0) or 0.0))
        return f"{instrument}_{strike}{opt_type}"

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
