# openai-regime.md
## NIFTY Options Regime Classification & Strategy Switching Framework

---

# 1. Overview

This document defines a deterministic regime classification system for NIFTY index options trading.

The goal is to:

- Classify market state into distinct regimes
- Deploy appropriate option structures per regime
- Control risk via strict capital allocation rules
- Avoid overfitting and subjective interpretation

The system is designed for systematic implementation from scratch.

---

# 2. High-Level Architecture

The framework consists of 5 layers:

1. Data Layer
2. Indicator Calculation Layer
3. Regime Detection Engine
4. Strategy Allocation Layer
5. Risk Management Layer

Each layer must be modular and independently testable.

---

# 3. Data Requirements

Minimum required data:

## 3.1 Market Data
- NIFTY spot price (daily + intraday optional)
- OHLC candles (daily required)
- Volume (optional but recommended)

## 3.2 Volatility Data
- India VIX (daily)
- Historical implied volatility (ATM IV)
- Option chain data (optional for advanced skew logic)

## 3.3 Derived Metrics
- 20 EMA
- 50 EMA
- 200 DMA
- ATR (14 or 20)
- ADX (14)
- Realized volatility (20-day std dev)
- IV percentile (1-year rolling window)

---

# 4. Indicator Definitions

## 4.1 Trend Metrics

trend_up:
- Close > 50 DMA
- 50 DMA slope > 0
- Higher high / higher low structure confirmed

trend_down:
- Close < 50 DMA
- 50 DMA slope < 0
- Lower high / lower low structure confirmed

trend_strength:
- ADX > 20 indicates trending
- ADX < 20 indicates range

---

## 4.2 Volatility Metrics

iv_percentile:
- Rolling 252-day IV percentile
- < 30 → Low volatility
- 30–70 → Neutral
- > 70 → High volatility

realized_vol:
- 20-day annualized standard deviation

vol_expansion:
- realized_vol > 20-day moving average of realized_vol
- OR daily ATR increasing 3 consecutive days

iv_change:
- (Current IV - 5-day average IV)

---

# 5. Regime Definitions

The system classifies into 4 primary regimes.

---

## REGIME 1: LOW VOL RANGE

Conditions:

- iv_percentile < 40
- ADX < 20
- Price within 20-day high-low range
- 50 DMA slope near zero

Intent:
Market compressing. Time decay favorable.

Deploy:
- Iron condors
- Defined-risk short strangles
- Calendar spreads

Exit Conditions:
- ADX rises above 20
- iv_change > threshold
- Breakout beyond 20-day range

---

## REGIME 2: BULL TREND (CONTROLLED VOL)

Conditions:

- trend_up == true
- ADX > 20
- iv_percentile between 30 and 70
- realized_vol stable

Intent:
Directional drift upward.

Deploy:
- Bull put spreads
- Broken wing butterflies (bullish bias)
- Call debit spreads (optional)

Avoid:
- Neutral condors

Exit Conditions:
- Breakdown below 50 DMA
- ADX weakens below 20
- Volatility spike

---

## REGIME 3: BEAR TREND + VOL EXPANSION

Conditions:

- trend_down == true
- ADX > 20
- iv_change positive
- vol_expansion == true

Intent:
Directional move with expanding volatility.

Deploy:
- Bear put spreads
- Long puts (early expansion only)
- Call credit spreads

Reduce size when:
- iv_percentile > 80

Exit Conditions:
- Higher high formed
- IV plateau detected
- ADX falling

---

## REGIME 4: HIGH VOL EVENT / SHOCK

Conditions:

- iv_percentile > 80
- ATR expansion > 2x 20-day average
- Gap open > 1.5% multiple times in 5 days

Intent:
Unstable market. High gamma risk.

Deploy:
- Very small defined-risk spreads
- Gamma scalping (advanced only)
- Or stay flat

Avoid:
- Naked selling
- Large premium selling

Exit Conditions:
- iv_percentile drops below 70
- ATR normalizes

---

# 6. Regime Transition Rules

Avoid switching on single signal.

Require confirmation:

Example transition LOW_VOL → BEAR_EXPANSION:

- Two consecutive lower lows
- iv_percentile increase > 10 points
- ADX > 20

Minimum 2 confirmation signals required before switching.

---

# 7. Capital Allocation Rules

Starting capital example: ₹3,00,000

## 7.1 Risk Per Trade
Max 1–2% of total capital.

## 7.2 Max Active Capital
40–60% deployed at any time.

## 7.3 Portfolio Drawdown Controls
- At -8% monthly drawdown → reduce size 50%
- At -12% drawdown → halt new trades until recovery

---

# 8. Position Sizing Formula

risk_per_trade = capital * risk_percent

position_size = risk_per_trade / max_loss_per_spread

Always calculate using defined max loss.

Never use full margin availability.

---

# 9. Regime Engine Pseudocode

```python
def classify_regime(data):

    if iv_percentile < 40 and adx < 20:
        return "LOW_VOL_RANGE"

    elif trend_up and 30 <= iv_percentile <= 70 and adx > 20:
        return "BULL_TREND"

    elif trend_down and vol_expansion and iv_change > 0:
        return "BEAR_EXPANSION"

    elif iv_percentile > 80:
        return "HIGH_VOL_EVENT"

    else:
        return "NEUTRAL"

---

def deploy_strategy(regime):

    if regime == "LOW_VOL_RANGE":
        deploy_iron_condor()

    elif regime == "BULL_TREND":
        deploy_bull_put_spread()

    elif regime == "BEAR_EXPANSION":
        deploy_bear_put_spread()

    elif regime == "HIGH_VOL_EVENT":
        reduce_position_size()
        deploy_small_defined_risk_only()

---

Backtesting Requirements
Backtest must include:
2020 crash
Budget weeks
Election weeks
High IV spike periods
Low IV grinding markets
Include:
Slippage
Brokerage
STT
Gap risk modeling

---

System must:
Maintain expectancy > 0
Maintain max drawdown < 25%
Avoid >5 consecutive max-loss trades
If violated: Trigger risk throttle mode.

Future Enhancements (Optional)
Skew analysis (Put vs Call IV skew)
Term structure (Weekly vs Monthly IV)
Open interest clustering
FII/DII flow inputs
Intraday volatility filters

Design Principles
Survival > Growth
Regime detection > Strategy optimization
Simplicity > Overfitting
Defined risk > Margin illusion
Capital preservation first