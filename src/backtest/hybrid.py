"""Hybrid backtest utilities: vectorized schedule + event-driven execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.simulator import FillSimulator
from src.data.option_chain_quality import OptionChainQualityThresholds
from src.regime.classifier import RegimeClassifier, RegimeThresholds
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


class ScheduleDrivenStrategy(BaseStrategy):
    """Execute entry/exit events from a precomputed schedule."""

    def __init__(self, name: str, config: dict[str, Any], schedule: pd.DataFrame):
        super().__init__(name=name, config=config)
        out = schedule.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
        out["entry"] = out["entry"].astype(bool)
        out["exit"] = out["exit"].astype(bool)
        self._schedule = out.set_index("timestamp")[["entry", "exit"]]

    def generate_signal(self, market_data: dict[str, Any], regime: RegimeState) -> Signal:
        ts = pd.Timestamp(market_data["timestamp"])
        instrument = str(self.config.get("instrument", "NIFTY"))
        rec = self._schedule.loc[ts] if ts in self._schedule.index else None
        if rec is None:
            return self._no_signal(
                ts.to_pydatetime(), instrument, regime, reason="No schedule event"
            )

        if self.state.current_position is None and bool(rec["entry"]):
            qty = self.compute_position_size(capital=0.0, risk_per_trade=0.0)
            self.state.current_position = {"side": "LONG", "quantity": qty}
            return Signal(
                signal_type=SignalType.ENTRY_LONG,
                strategy_name=self.name,
                instrument=instrument,
                timestamp=ts.to_pydatetime(),
                orders=[{"symbol": instrument, "action": "BUY", "quantity": qty}],
                regime=regime,
                reason="Hybrid schedule entry",
            )

        return self._no_signal(
            ts.to_pydatetime(), instrument, regime, reason="Schedule has no actionable entry"
        )

    def get_exit_conditions(self, market_data: dict[str, Any]) -> Signal | None:
        if self.state.current_position is None:
            return None
        ts = pd.Timestamp(market_data["timestamp"])
        rec = self._schedule.loc[ts] if ts in self._schedule.index else None
        if rec is None or not bool(rec["exit"]):
            return None

        qty = int(self.state.current_position.get("quantity", 1))
        self.state.current_position = None
        instrument = str(self.config.get("instrument", "NIFTY"))
        return Signal(
            signal_type=SignalType.EXIT,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts.to_pydatetime(),
            orders=[{"symbol": instrument, "action": "SELL", "quantity": qty}],
            regime=RegimeState.UNKNOWN,
            reason="Hybrid schedule exit",
        )

    def compute_position_size(self, capital: float, risk_per_trade: float) -> int:
        return int(self.config.get("max_lots", 1) or 1)

    def _no_signal(
        self,
        ts: datetime,
        instrument: str,
        regime: RegimeState,
        *,
        reason: str,
    ) -> Signal:
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            strategy_name=self.name,
            instrument=instrument,
            timestamp=ts,
            regime=regime,
            reason=reason,
        )


@dataclass
class HybridConfig:
    symbol: str = "NIFTY"
    timeframe: str = "1d"
    initial_capital: float = 150_000.0
    max_lots: int = 1


def run_hybrid_schedule_backtest(
    *,
    candles: pd.DataFrame,
    schedule: pd.DataFrame,
    config: HybridConfig,
    simulator: FillSimulator,
    thresholds: RegimeThresholds,
    vix_df: pd.DataFrame | None = None,
    fii_df: pd.DataFrame | None = None,
    usdinr_df: pd.DataFrame | None = None,
    option_chain_df: pd.DataFrame | None = None,
    analysis_start: datetime | None = None,
    chain_quality_thresholds: OptionChainQualityThresholds | None = None,
) -> BacktestResult:
    strategy = ScheduleDrivenStrategy(
        name="hybrid_schedule",
        config={
            "instrument": config.symbol,
            "timeframe": config.timeframe,
            "max_lots": config.max_lots,
            "active_regimes": [r.value for r in RegimeState],
        },
        schedule=schedule,
    )
    engine = BacktestEngine(
        classifier=RegimeClassifier(thresholds=thresholds),
        strategy=strategy,
        simulator=simulator,
        initial_capital=config.initial_capital,
        chain_quality_thresholds=chain_quality_thresholds or OptionChainQualityThresholds(),
    )
    return engine.run(
        candles=candles,
        vix_df=vix_df,
        fii_df=fii_df,
        usdinr_df=usdinr_df,
        option_chain_df=option_chain_df,
        analysis_start=analysis_start,
    )
