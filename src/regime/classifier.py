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

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.strategies.base import RegimeState

logger = structlog.get_logger()


@dataclass
class RegimeSignals:
    """Container for all regime-relevant signals at a point in time."""
    timestamp: datetime

    # Volatility
    india_vix: float = 0.0
    vix_change_5d: float = 0.0   # VIX change over last 5 days (rate of change matters)
    iv_surface_parallel_shift: float = 0.0
    iv_surface_tilt_change: float = 0.0

    # Trend strength
    adx_14: float = 0.0          # ADX(14) on Nifty daily

    # Sentiment
    pcr: float = 0.0             # Nifty Put-Call Ratio
    fii_net_3d: float = 0.0      # FII net flow, 3-day cumulative (crore INR)

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

    @classmethod
    def from_config(cls, config: dict) -> "RegimeThresholds":
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
    history: list[dict] = field(default_factory=list)

    def classify(self, signals: RegimeSignals) -> RegimeState:
        """
        Determine current market regime from signals.

        Returns the new regime state. If regime changed, logs the transition.
        """
        self.previous_regime = self.current_regime

        # Base classification: VIX x ADX matrix
        low_vol = signals.india_vix < self.thresholds.vix_low
        high_vol = signals.india_vix >= self.thresholds.vix_high
        trending = signals.adx_14 >= self.thresholds.adx_trending

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
                    if self.current_regime in (RegimeState.LOW_VOL_TRENDING, RegimeState.LOW_VOL_RANGING)
                    else RegimeState.HIGH_VOL_TRENDING
                )
            else:
                new_regime = (
                    RegimeState.LOW_VOL_RANGING
                    if self.current_regime in (RegimeState.LOW_VOL_TRENDING, RegimeState.LOW_VOL_RANGING)
                    else RegimeState.HIGH_VOL_CHOPPY
                )

        # Override: extreme FII selling can force high-vol classification
        if signals.fii_net_3d <= self.thresholds.fii_selling_alert:
            if new_regime in (RegimeState.LOW_VOL_TRENDING, RegimeState.LOW_VOL_RANGING):
                logger.warning(
                    "FII selling overriding low-vol classification",
                    fii_net_3d=signals.fii_net_3d,
                    original_regime=new_regime.value,
                )
                new_regime = RegimeState.HIGH_VOL_CHOPPY

        # Override: fast VIX expansion often precedes unstable/choppy tape.
        if signals.vix_change_5d >= self.thresholds.vix_spike_5d_alert:
            if new_regime in (RegimeState.LOW_VOL_TRENDING, RegimeState.LOW_VOL_RANGING):
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
        if iv_surface_stress and new_regime in (RegimeState.LOW_VOL_TRENDING, RegimeState.LOW_VOL_RANGING):
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
        return new_regime

    def _on_regime_change(self, new_regime: RegimeState, signals: RegimeSignals):
        """Log and record regime transitions."""
        transition = {
            "timestamp": signals.timestamp.isoformat(),
            "from": self.current_regime.value,
            "to": new_regime.value,
            "vix": signals.india_vix,
            "adx": signals.adx_14,
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

    def get_regime_duration_days(self) -> int | None:
        """How long has the current regime been active?"""
        if self.regime_since is None:
            return None
        return (datetime.now() - self.regime_since).days

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
