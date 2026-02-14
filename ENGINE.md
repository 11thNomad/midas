# Engine Blueprint (Options-First)

## 1. Objective
Build a single trading engine that is options-first from day one (Iron Condor + Jade Lizard), with minimal friction to switch between:
- `backtest` (research and validation)
- `paper` (forward test with broker market data)
- `live` (broker execution)

The same strategy logic and risk logic must run in all modes.

## 1.1 Decisions Locked
- Primary launch symbol: `NIFTY`
- Secondary symbol: scaffold `BANKNIFTY` from the start (disabled by default)
- Strategy set at launch: `iron_condor`, `jade_lizard`
- Jade lizard variants: `neutral`, `bullish`, `bearish`
- FII filters: both `3d` and `5d` rolling features (used together)
- USD/INR filters: both `1d` and `3d` ROC
- Signal clock policy: `15m` primary decision clock + `5m` defensive exit checks
- Entry timing policy: market open at `09:15` IST, no new entries after `15:15` IST
- Initial capital baseline: `150000` for paper/live risk accounting
- Margin utilization policy: configurable operating band `20%` to `40%`
- Backtest mode policy: remove `research_fast`; keep only realistic modes

## 2. Core Design Principles
1. One strategy code path: no separate "paper strategy" or "live strategy" logic.
2. One signal contract: all features are computed identically in backtest/paper/live.
3. One risk engine: sizing and kill-switch behavior is mode-agnostic.
4. Pluggable adapters only for data and execution.
5. Broker data is treated as raw input; derived signals are computed in-house.

## 3. Runtime Architecture

```text
Signal Clock (15m decisions + 5m defensive checks)
  -> Feature Pipeline
  -> Regime + Trade Gate
  -> Strategy (Condor/Jade Lizard)
  -> Risk + Sizing
  -> Execution Adapter (backtest simulator / paper broker / live broker)
  -> State + Journal + Metrics
```

### 3.1 Engine Interfaces
- `MarketDataAdapter`
  - `get_candles(...)`
  - `get_option_chain(...)`
  - `get_vix(...)`
  - `get_usdinr(...)`
  - `get_fii(...)`
- `ExecutionAdapter`
  - `place_orders(...)`
  - `modify_orders(...)`
  - `cancel_orders(...)`
  - `positions()`
- `PortfolioState`
  - positions, margin used, realized/unrealized PnL, fees
- `RiskEngine`
  - max daily loss, drawdown, max open positions, per-trade sizing

## 4. Signal Stack

### 4.1 Trade-Gating Signals (enter or do not enter)
- India VIX level
- India VIX rate of change (5-day or configurable)
- ADX(14)
- FII flow (3-day rolling net; optionally 5-day for slower filter)
- USD/INR rate of change
- Option-chain sentiment: PCR, OI shape/pressure
- Optional: IV surface shift/tilt (already present in repo)

### 4.2 Sizing / Structure Signals
- ATR(14) for strike distance and stop buffers
- Bollinger Bands (20, 2) for squeeze and expansion context
- RSI(14) for stretch/filtering entries
- OI support/resistance zones
- Regime confidence score

### 4.3 Greeks
- Compute greeks locally via `mibian` from:
  - spot/underlying price
  - strike
  - time to expiry
  - implied volatility
  - risk-free rate
- If broker provides greeks, treat them as reference only (not source of truth).

## 5. Pull vs Compute Policy

### 5.1 Pull from broker/data source
- OHLCV candles
- Option chain rows (strike, expiry, option_type, LTP, OI, bid/ask)
- India VIX series (index candles)
- USD/INR candles
- FII/DII raw flow dataset (from NSE pipeline, not broker)

### 5.2 Compute in engine
- VIX ROC
- ADX, ATR, Bollinger, RSI
- PCR and OI-derived support/resistance
- FII rolling features
- USD/INR ROC
- Greeks via `mibian`
- Regime label and confidence

## 6. Strategy Specs (Initial)

### 6.1 Iron Condor
- Entry only in approved low-volatility regimes.
- Short call + short put near target delta.
- Long wings at configured width or ATR-based width.
- Exits:
  - profit target
  - stop loss
  - DTE exit
  - regime invalidation

### 6.2 Jade Lizard
- Short put spread + short call (no upside call wing).
- Constraint: total credit >= call width risk profile target.
- Variants: neutral, bullish, bearish.
- Entry depends on selected variant and regime gate.
- Exits:
  - max loss threshold
  - profit target
  - DTE exit
  - volatility shock / regime transition

### 6.3 Trade-Gate Decision Tree (How to decide and how to structure)
Gate order is strict and hierarchical. If any hard gate fails, do not trade.

1. Gate 1 (Regime): allow only approved regimes per strategy/variant.
2. Gate 2 (Volatility sanity): enforce entry VIX min/max band.
3. Gate 3 (Calendar): day-of-week and session timing checks.
4. Gate 4 (DTE window): expiry must be within configured min/max DTE.
5. Gate 5 (Position exclusivity): skip if same-structure position already open.
6. Gate 6 (Risk breaker): skip if circuit breaker disallows new risk.
7. Structure build (how to trade):
   - ATR/Bollinger determine strike distance and width context.
   - OI/PCR determine strike preference and support/resistance alignment.
   - RSI/FII/USDINR are bias adjusters (not primary hard gates unless configured).
8. Premium viability check: skip if expected credit is below configured minimum.
9. Emit signal with explicit `reason` string including gate context values.

Indicator role separation:
- Hard gate indicators: regime, VIX band, calendar/time, DTE, breaker state.
- Structure indicators: ATR, Bollinger, OI map, PCR.
- Bias/adjustment indicators: RSI, FII rolling signals, USD/INR ROC.

## 7. VectorBT Integration Plan

### 7.1 Why VectorBT here
- Fast parameter sweeps, walk-forward slicing, Monte Carlo permutations.
- Good fit for signal mask generation and portfolio-level analytics.

### 7.2 Practical approach for multi-leg options
Use a hybrid model:
1. VectorBT layer:
   - generates entry/exit timestamps from features/regime
   - runs parameter sweeps and walk-forward experiment orchestration
2. Options pricing/simulation layer (custom):
   - resolves strikes from chain snapshots
   - constructs leg orders
   - applies fees/slippage/partial-fill rules
   - computes leg-level and portfolio PnL

This preserves options realism while still using VectorBT for research speed.

### 7.3 Backtest modes
- `research_realistic`: hybrid VectorBT + event-level options simulator.
- `production_shadow`: same path as paper/live except execution is simulated.

## 8. Paper and Live Switch-Over

Switching should only change adapters and mode flags:
- `mode=backtest`: historical data adapter + simulated execution
- `mode=paper`: live broker data adapter + paper execution adapter
- `mode=live`: live broker data adapter + real broker execution adapter

Required invariant:
- identical strategy/risk code
- identical feature calculation code
- identical signal contract

## 9. Data and State Requirements
- Persist every signal snapshot with timestamp.
- Persist selected strikes and full leg definitions at entry.
- Persist execution events (submitted, acknowledged, filled, canceled).
- Persist daily risk state and circuit-breaker state.
- Persist run metadata (mode, config hash, git commit hash).

## 10. Risk and Sizing Rules (Engine-Level)
- Global circuit breaker:
  - max daily loss
  - max drawdown
  - optional manual kill switch
- Per-trade sizing:
  - capital-at-risk cap
  - margin utilization cap
  - ATR-aware width/quantity adjustment
- Position constraints:
  - max concurrent positions
  - symbol-level exposure caps
  - no fresh entries near market close

## 11. Implementation Milestones

### M1: Signal Contract Freeze
- Status: `completed (v1.0.0 schema + DTO/frame mapper + persistence store)`
- Finalize feature list and formulas.
- Lock pull-vs-compute matrix.
- Add tests for every feature function.

### M2: Feature Pipeline Hardening
- Status: `completed (shared build_feature_context in paper/backtest)`
- Add USD/INR ingestion.
- Add stable FII rolling features.
- Add mibian greeks pipeline and test vectors.

### M3: Strategy Expansion
- Keep iron condor as baseline.
- Implement jade lizard strategy with full lifecycle tests.

### M4: VectorBT Research Layer
- Status: `completed (research runner + walk-forward scaffolding)`
- Add VectorBT experiment runner over feature/signal outputs.
- Add walk-forward and sensitivity templates.

### M4.1: VectorBT Parameter-Set Lab
- Status: `completed (named parameter-set catalog + batch runner + leaderboard artifacts)`
- Maintain a versioned collection of parameter sets (`config/vectorbt_parameter_sets.yaml`).
- Run batch comparisons and rank with eligibility filters (trade count and drawdown caps).
- Validate selected candidates with walk-forward before paper promotion.

### M5: Hybrid Realistic Backtester
- Status: `completed (vector schedule -> event-driven simulator handoff)`
- Integrate custom options execution simulator with VectorBT schedule outputs.
- Validate against existing event-driven backtest on overlapping scenarios.

### M6: Paper-Ready Operations
- Status: `completed (ops runbook + freshness gates + cron-ready open/close checks)`
- Daily maintenance + data freshness gates.
- Session start/end health checks.
- Structured reporting and risk alerts.

### M7: Live Readiness Gate
- Paper pass criteria met (PnL stability, drawdown, fill quality, operational stability).
- Dry-run live sessions with zero size.
- Controlled capital ramp-up plan.

## 11.1 Immediate Next Focus
1. Trade-attribution and signal-context dashboard in Streamlit:
   - show each trade with entry/exit signal values, regime, and PnL attribution
   - add cohort views by regime and by parameter-set ID
2. Multi-leg options realism in hybrid execution:
   - replace schedule-level proxy with leg-aware fill and lifecycle accounting for iron condor/jade lizard
3. Promotion gate:
   - formal rule for moving a parameter set from research to paper candidate

## 12. Acceptance Criteria Before Live
1. Strategy and risk behavior parity between backtest and paper is demonstrated.
2. Data quality gates pass continuously.
3. No unresolved critical execution/risk incidents for a defined paper window.
4. Runbook exists for open, intraday incident, and close workflows.

## 13. Open Questions
No blocking open questions right now. Implementation can proceed.

---
This document is the implementation contract for engine migration and strategy expansion.
