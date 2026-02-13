"""
Regime Classifier — determines the current market state.

The regime classifier is the "brain" that decides which strategies should be
active. It takes in multiple signals (VIX, ADX, FII flows, PCR) and outputs
a single RegimeState that the strategy router uses.

Design philosophy:
- Simple, interpretable rules over complex ML models
- Each signal is computed independently in signals.py
- This module combines them with clear, auditable logic
- Thresholds are configurable via settings.yaml
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import structlog

from src.strategies.base import RegimeState

logger = structlog.get_logger()


@dataclass
class RegimeSignals:
    """Container for all regime-relevant signals at a point in time."""

    timestamp: datetime

    # Volatility
    india_vix: float = 0.0
    vix_change_5d: float = 0.0  # VIX change over last 5 days (rate of change matters)
    iv_surface_parallel_shift: float = 0.0
    iv_surface_tilt_change: float = 0.0

    # Trend strength
    adx_14: float = 0.0  # ADX(14) on Nifty daily

    # Sentiment
    pcr: float = 0.0  # Nifty Put-Call Ratio
    fii_net_3d: float = 0.0  # FII net flow, 3-day cumulative (crore INR)

    # Price context
    nifty_above_50dma: bool = True
    nifty_above_200dma: bool = True
    nifty_banknifty_corr: float = 0.95  # Rolling 20-day correlation


@dataclass
class RegimeThresholds:
    """Configurable thresholds for regime classification."""

    vix_low: float = 14.0
    vix_high: float = 18.0
    adx_trending: float = 25.0
    adx_ranging: float = 20.0
    fii_selling_alert: float = -6000.0
    pcr_oversold: float = 0.7
    pcr_overbought: float = 1.3
    corr_breakdown: float = 0.80  # If Nifty-BankNifty corr drops below this, something unusual
    vix_spike_5d_alert: float = 3.0
    iv_surface_shift_alert: float = 2.0
    iv_surface_tilt_alert: float = 1.5
    vix_hysteresis_buffer: float = 0.5
    adx_hysteresis_buffer: float = 2.0
    adx_smoothing_alpha: float = 0.3

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RegimeThresholds":
        """Load thresholds from settings.yaml regime section."""
        return cls(
            vix_low=config.get("vix_low", 14.0),
            vix_high=config.get("vix_high", 18.0),
            adx_trending=config.get("adx_trending", 25.0),
            adx_ranging=config.get("adx_ranging", 20.0),
            fii_selling_alert=config.get("fii_selling_alert", -6000.0),
            pcr_oversold=config.get("pcr_oversold", 0.7),
            pcr_overbought=config.get("pcr_overbought", 1.3),
            corr_breakdown=config.get("corr_breakdown", 0.80),
            vix_spike_5d_alert=config.get("vix_spike_5d_alert", 3.0),
            iv_surface_shift_alert=config.get("iv_surface_shift_alert", 2.0),
            iv_surface_tilt_alert=config.get("iv_surface_tilt_alert", 1.5),
            vix_hysteresis_buffer=config.get("vix_hysteresis_buffer", 0.5),
            adx_hysteresis_buffer=config.get("adx_hysteresis_buffer", 2.0),
            adx_smoothing_alpha=config.get("adx_smoothing_alpha", 0.3),
        )


@dataclass
class RegimeClassifier:
    """
    Combines multiple market signals into a single regime classification.

    The classification uses a simple decision tree:

        VIX < vix_low?
        ├── Yes (low vol)
        │   └── ADX > adx_trending?
        │       ├── Yes → LOW_VOL_TRENDING
        │       └── No  → LOW_VOL_RANGING
        └── No (high vol)
            └── ADX > adx_trending?
                ├── Yes → HIGH_VOL_TRENDING
                └── No  → HIGH_VOL_CHOPPY

    Additional signals (FII flows, PCR, correlation breakdown) act as
    modifiers that can override or add confidence to the base classification.
    """

    thresholds: RegimeThresholds
    current_regime: RegimeState = RegimeState.UNKNOWN
    previous_regime: RegimeState = RegimeState.UNKNOWN
    regime_since: datetime | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    _smoothed_adx: float | None = None
    _last_timestamp: datetime | None = None

    def classify(self, signals: RegimeSignals) -> RegimeState:
        """
        Determine current market regime from signals.

        Returns the new regime state. If regime changed, logs the transition.
        """
        self._last_timestamp = signals.timestamp
        self.previous_regime = self.current_regime

        if pd.isna(signals.india_vix) or pd.isna(signals.adx_14):
            logger.error(
                "Invalid regime signals: NaN detected",
                india_vix=signals.india_vix,
                adx_14=signals.adx_14,
            )
            new_regime = RegimeState.UNKNOWN
            if new_regime != self.current_regime:
                self._on_regime_change(new_regime, signals)
            self.current_regime = new_regime
            self.snapshots.append(self.snapshot(signals=signals, regime=new_regime))
            return new_regime

        # Base classification: VIX x ADX matrix with hysteresis + smoothing.
        adx_for_state = self._smooth_adx(signals.adx_14)
        low_vol, high_vol = self._resolve_vix_state(signals.india_vix)
        trending = self._resolve_trend_state(adx_for_state)

        if low_vol and trending:
            new_regime = RegimeState.LOW_VOL_TRENDING
        elif low_vol and not trending:
            new_regime = RegimeState.LOW_VOL_RANGING
        elif high_vol and trending:
            new_regime = RegimeState.HIGH_VOL_TRENDING
        elif high_vol:
            new_regime = RegimeState.HIGH_VOL_CHOPPY
        else:
            # VIX in the middle zone (between vix_low and vix_high)
            # Use ADX as tiebreaker, lean toward the previous regime for stability
            if trending:
                new_regime = (
                    RegimeState.LOW_VOL_TRENDING
                    if self.current_regime
                    in (RegimeState.LOW_VOL_TRENDING, RegimeState.LOW_VOL_RANGING)
                    else RegimeState.HIGH_VOL_TRENDING
                )
            else:
                new_regime = (
                    RegimeState.LOW_VOL_RANGING
                    if self.current_regime
                    in (RegimeState.LOW_VOL_TRENDING, RegimeState.LOW_VOL_RANGING)
                    else RegimeState.HIGH_VOL_CHOPPY
                )

        # Override: extreme FII selling can force high-vol classification
        if signals.fii_net_3d <= self.thresholds.fii_selling_alert and new_regime in (
            RegimeState.LOW_VOL_TRENDING,
            RegimeState.LOW_VOL_RANGING,
        ):
            logger.warning(
                "FII selling overriding low-vol classification",
                fii_net_3d=signals.fii_net_3d,
                original_regime=new_regime.value,
            )
            new_regime = RegimeState.HIGH_VOL_CHOPPY

        # Override: fast VIX expansion often precedes unstable/choppy tape.
        if signals.vix_change_5d >= self.thresholds.vix_spike_5d_alert and new_regime in (
            RegimeState.LOW_VOL_TRENDING,
            RegimeState.LOW_VOL_RANGING,
        ):
            logger.warning(
                "VIX 5-day spike overriding low-vol classification",
                vix_change_5d=signals.vix_change_5d,
                original_regime=new_regime.value,
            )
            new_regime = RegimeState.HIGH_VOL_CHOPPY

        # Override: abrupt IV surface shifts/tilts indicate options stress regime.
        iv_surface_stress = (
            abs(signals.iv_surface_parallel_shift) >= self.thresholds.iv_surface_shift_alert
            or abs(signals.iv_surface_tilt_change) >= self.thresholds.iv_surface_tilt_alert
        )
        if iv_surface_stress and new_regime in (
            RegimeState.LOW_VOL_TRENDING,
            RegimeState.LOW_VOL_RANGING,
        ):
            logger.warning(
                "IV surface stress overriding low-vol classification",
                iv_surface_parallel_shift=signals.iv_surface_parallel_shift,
                iv_surface_tilt_change=signals.iv_surface_tilt_change,
                original_regime=new_regime.value,
            )
            new_regime = RegimeState.HIGH_VOL_CHOPPY

        # Override: correlation breakdown signals structural change
        if signals.nifty_banknifty_corr < self.thresholds.corr_breakdown:
            logger.warning(
                "Nifty-BankNifty correlation breakdown detected",
                correlation=signals.nifty_banknifty_corr,
            )
            # Don't override, but flag it — strategies can check this

        # Detect regime change
        if new_regime != self.current_regime:
            self._on_regime_change(new_regime, signals)

        self.current_regime = new_regime
        self.snapshots.append(self.snapshot(signals=signals, regime=new_regime))
        return new_regime

    def snapshot(self, signals: RegimeSignals, regime: RegimeState | None = None) -> dict[str, Any]:
        """Return a normalized snapshot row for persistence and debugging."""
        resolved_regime = regime or self.current_regime
        return {
            "timestamp": signals.timestamp.isoformat(),
            "regime": resolved_regime.value,
            "previous_regime": self.previous_regime.value,
            "regime_since": self.regime_since.isoformat() if self.regime_since else None,
            "india_vix": signals.india_vix,
            "vix_change_5d": signals.vix_change_5d,
            "adx_14": signals.adx_14,
            "adx_14_smoothed": self._smoothed_adx
            if self._smoothed_adx is not None
            else signals.adx_14,
            "pcr": signals.pcr,
            "fii_net_3d": signals.fii_net_3d,
            "nifty_above_50dma": signals.nifty_above_50dma,
            "nifty_above_200dma": signals.nifty_above_200dma,
            "nifty_banknifty_corr": signals.nifty_banknifty_corr,
            "iv_surface_parallel_shift": signals.iv_surface_parallel_shift,
            "iv_surface_tilt_change": signals.iv_surface_tilt_change,
        }

    def _on_regime_change(self, new_regime: RegimeState, signals: RegimeSignals) -> None:
        """Log and record regime transitions."""
        transition = {
            "timestamp": signals.timestamp.isoformat(),
            "from": self.current_regime.value,
            "to": new_regime.value,
            "vix": signals.india_vix,
            "adx": signals.adx_14,
            "adx_smoothed": self._smoothed_adx
            if self._smoothed_adx is not None
            else signals.adx_14,
            "pcr": signals.pcr,
            "fii_net_3d": signals.fii_net_3d,
            "vix_change_5d": signals.vix_change_5d,
            "iv_surface_parallel_shift": signals.iv_surface_parallel_shift,
            "iv_surface_tilt_change": signals.iv_surface_tilt_change,
        }
        self.history.append(transition)
        self.regime_since = signals.timestamp

        logger.info(
            "REGIME CHANGE",
            from_regime=self.current_regime.value,
            to_regime=new_regime.value,
            vix=signals.india_vix,
            adx=signals.adx_14,
        )

    def get_regime_duration_days(self, *, as_of: datetime | None = None) -> int | None:
        """How long has the current regime been active?"""
        if self.regime_since is None:
            return None
        anchor = as_of or self._last_timestamp
        if anchor is None:
            return None
        return (anchor - self.regime_since).days

    def get_context(self) -> dict[str, Any]:
        """Current regime context for logging and dashboards."""
        return {
            "current_regime": self.current_regime.value,
            "previous_regime": self.previous_regime.value,
            "regime_since": self.regime_since.isoformat() if self.regime_since else None,
            "duration_days": self.get_regime_duration_days(),
            "total_transitions": len(self.history),
        }

    def __repr__(self) -> str:
        duration = self.get_regime_duration_days()
        dur_str = f" ({duration}d)" if duration is not None else ""
        return f"<Regime: {self.current_regime.value}{dur_str}>"

    def _smooth_adx(self, adx_raw: float) -> float:
        alpha = self.thresholds.adx_smoothing_alpha
        if self._smoothed_adx is None:
            self._smoothed_adx = float(adx_raw)
        else:
            self._smoothed_adx = alpha * float(adx_raw) + (1.0 - alpha) * self._smoothed_adx
        return float(self._smoothed_adx)

    def _resolve_vix_state(self, india_vix: float) -> tuple[bool, bool]:
        vix = float(india_vix)
        b = self.thresholds.vix_hysteresis_buffer

        low_family = self.current_regime in (
            RegimeState.LOW_VOL_TRENDING,
            RegimeState.LOW_VOL_RANGING,
        )
        high_family = self.current_regime in (
            RegimeState.HIGH_VOL_TRENDING,
            RegimeState.HIGH_VOL_CHOPPY,
        )

        if low_family:
            low_vol = vix < (self.thresholds.vix_low + b)
            high_vol = vix >= (self.thresholds.vix_high + b)
        elif high_family:
            low_vol = vix < (self.thresholds.vix_low - b)
            high_vol = vix >= (self.thresholds.vix_high - b)
        else:
            low_vol = vix < self.thresholds.vix_low
            high_vol = vix >= self.thresholds.vix_high

        return low_vol, high_vol

    def _resolve_trend_state(self, adx_value: float) -> bool:
        adx = float(adx_value)
        b = self.thresholds.adx_hysteresis_buffer
        prev_trending = self.current_regime in (
            RegimeState.LOW_VOL_TRENDING,
            RegimeState.HIGH_VOL_TRENDING,
        )
        enter_threshold = self.thresholds.adx_trending + b
        exit_threshold = max(self.thresholds.adx_ranging - b, 0.0)
        stay_threshold = max(self.thresholds.adx_trending - b, 0.0)

        if prev_trending:
            return adx >= stay_threshold
        if adx <= exit_threshold:
            return False
        return adx >= enter_threshold
