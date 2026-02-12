"""Event-driven backtest engine scaffold (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.backtest.metrics import summarize_backtest
from src.backtest.simulator import FillSimulator
from src.regime.classifier import RegimeClassifier, RegimeSignals
from src.signals.regime import build_regime_signals
from src.strategies.base import BaseStrategy, RegimeState, Signal, SignalType


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    regimes: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class BacktestEngine:
    """Run a single strategy over historical bars with regime awareness."""

    classifier: RegimeClassifier
    strategy: BaseStrategy
    simulator: FillSimulator
    initial_capital: float = 1_000_000.0
    periods_per_year: int = 252

    def run(
        self,
        *,
        candles: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        fii_df: pd.DataFrame | None = None,
        option_chain_df: pd.DataFrame | None = None,
    ) -> BacktestResult:
        if candles.empty:
            empty = pd.DataFrame()
            return BacktestResult(
                equity_curve=empty,
                fills=empty,
                regimes=empty,
                metrics=summarize_backtest(equity_curve=empty, fills=empty, initial_capital=self.initial_capital),
            )

        bars = candles.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
        bars = bars.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        vix = self._prep_vix(vix_df)
        fii = self._prep_fii(fii_df)
        chain = self._prep_chain(option_chain_df)

        cash = float(self.initial_capital)
        position_qty = 0
        fill_rows: list[dict] = []
        equity_rows: list[dict] = []
        regime_rows: list[dict] = []
        previous_regime = self.classifier.current_regime

        for i in range(len(bars)):
            row = bars.iloc[i]
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            close_price = float(row["close"])
            candles_hist = bars.iloc[: i + 1]
            vix_hist = vix.loc[vix["timestamp"] <= pd.Timestamp(ts)] if not vix.empty else pd.DataFrame()
            fii_hist = fii.loc[fii["date"] <= pd.Timestamp(ts)] if not fii.empty else pd.DataFrame()
            chain_asof = self._latest_chain_asof(chain, ts)

            vix_series = vix_hist["close"].astype("float64") if not vix_hist.empty else None
            vix_value = float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else 0.0
            fii_net_3d = float(fii_hist["fii_net"].tail(3).sum()) if not fii_hist.empty else 0.0

            regime_signals = build_regime_signals(
                timestamp=ts,
                candles=candles_hist,
                vix_value=vix_value,
                vix_series=vix_series,
                fii_net_3d=fii_net_3d,
            )
            regime = self.classifier.classify(regime_signals)
            regime_rows.append(
                {"timestamp": ts, "regime": regime.value, "vix": regime_signals.india_vix, "adx": regime_signals.adx_14}
            )

            signal = self._next_signal(
                timestamp=ts,
                regime=regime,
                previous_regime=previous_regime,
                candles_hist=candles_hist,
                vix_value=vix_value,
                option_chain=chain_asof,
                underlying_price=close_price,
            )
            previous_regime = regime

            if signal is not None and signal.is_actionable:
                fills = self.simulator.simulate(signal, close_price=close_price, timestamp=ts)
                for fill in fills:
                    notional = fill["price"] * fill["quantity"]
                    if fill["side"] == "BUY":
                        cash -= notional + fill["fees"]
                        position_qty += int(fill["quantity"])
                    else:
                        cash += notional - fill["fees"]
                        position_qty -= int(fill["quantity"])
                    fill_rows.append(fill)

            equity = cash + (position_qty * close_price)
            equity_rows.append({"timestamp": ts, "cash": cash, "position_qty": position_qty, "equity": equity})

        fills_df = pd.DataFrame(fill_rows)
        equity_df = pd.DataFrame(equity_rows)
        regimes_df = pd.DataFrame(regime_rows)
        metrics = summarize_backtest(
            equity_curve=equity_df,
            fills=fills_df,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year,
        )
        return BacktestResult(equity_curve=equity_df, fills=fills_df, regimes=regimes_df, metrics=metrics)

    def _next_signal(
        self,
        *,
        timestamp: datetime,
        regime: RegimeState,
        previous_regime: RegimeState,
        candles_hist: pd.DataFrame,
        vix_value: float,
        option_chain: pd.DataFrame | None,
        underlying_price: float,
    ) -> Signal | None:
        if self.strategy.should_be_active(regime):
            return self.strategy.generate_signal(
                market_data={
                    "timestamp": timestamp,
                    "candles": candles_hist,
                    "vix": vix_value,
                    "option_chain": option_chain,
                    "underlying_price": underlying_price,
                },
                regime=regime,
            )
        return self.strategy.on_regime_change(previous_regime, regime)

    @staticmethod
    def _prep_vix(vix_df: pd.DataFrame | None) -> pd.DataFrame:
        if vix_df is None or vix_df.empty:
            return pd.DataFrame(columns=["timestamp", "close"])
        out = vix_df.copy()
        if "timestamp" not in out.columns or "close" not in out.columns:
            return pd.DataFrame(columns=["timestamp", "close"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return out.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _prep_fii(fii_df: pd.DataFrame | None) -> pd.DataFrame:
        if fii_df is None or fii_df.empty:
            return pd.DataFrame(columns=["date", "fii_net"])
        out = fii_df.copy()
        if "date" not in out.columns or "fii_net" not in out.columns:
            return pd.DataFrame(columns=["date", "fii_net"])
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["fii_net"] = pd.to_numeric(out["fii_net"], errors="coerce")
        return out.dropna(subset=["date", "fii_net"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _prep_chain(chain_df: pd.DataFrame | None) -> pd.DataFrame:
        if chain_df is None or chain_df.empty:
            return pd.DataFrame(columns=["timestamp", "option_type", "strike"])
        out = chain_df.copy()
        if "timestamp" not in out.columns:
            return pd.DataFrame(columns=["timestamp", "option_type", "strike"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        if "option_type" in out.columns:
            out["option_type"] = out["option_type"].astype(str).str.upper()
        if "strike" in out.columns:
            out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return out

    @staticmethod
    def _latest_chain_asof(chain_df: pd.DataFrame, ts: datetime) -> pd.DataFrame | None:
        if chain_df.empty:
            return None
        cutoff = pd.Timestamp(ts)
        eligible = chain_df.loc[chain_df["timestamp"] <= cutoff]
        if eligible.empty:
            return None
        latest_ts = eligible["timestamp"].max()
        snap = eligible.loc[eligible["timestamp"] == latest_ts].copy()
        return snap.reset_index(drop=True)
