# NiftyQuant — Implementation Plan

> A phased roadmap for building a modular algorithmic trading system for Indian
> markets (NSE), focused on Nifty and Bank Nifty options. This document captures
> the full scope of what we're building, why each piece matters, and the order
> in which to build it.

---

## Table of Contents

1. [Vision & Philosophy](#vision--philosophy)
2. [Phase 0 — Foundation (Weeks 1–2)](#phase-0--foundation-weeks-12)
3. [Phase 1 — Data Pipeline (Weeks 3–5)](#phase-1--data-pipeline-weeks-35)
4. [Phase 2 — Signal Library (Weeks 5–7)](#phase-2--signal-library-weeks-57)
5. [Phase 3 — Regime Classifier (Weeks 7–8)](#phase-3--regime-classifier-weeks-78)
6. [Phase 4 — Backtesting Engine (Weeks 8–11)](#phase-4--backtesting-engine-weeks-811)
7. [Phase 5 — Strategy Implementation (Weeks 11–14)](#phase-5--strategy-implementation-weeks-1114)
8. [Phase 6 — Paper Trading (Weeks 14–18)](#phase-6--paper-trading-weeks-1418)
9. [Phase 7 — Live Deployment (Weeks 18–22)](#phase-7--live-deployment-weeks-1822)
10. [Phase 8 — Strategy Pipeline & Rotation (Ongoing)](#phase-8--strategy-pipeline--rotation-ongoing)
11. [Architecture Principles](#architecture-principles)
12. [Risk Framework](#risk-framework)
13. [Cost Model](#cost-model)
14. [Regime Detection Deep Dive](#regime-detection-deep-dive)
15. [Strategy Niches for Indian Markets](#strategy-niches-for-indian-markets)
16. [Infrastructure & Deployment](#infrastructure--deployment)
17. [Failure Modes & Mitigations](#failure-modes--mitigations)
18. [What Claude Can Help With](#what-claude-can-help-with)
19. [Realistic Expectations](#realistic-expectations)
20. [Decision Log](#decision-log)

---

## Current Status Checkpoint (Updated: 2026-02-13)

### Active Phase
- **Current implementation phase:** Phase 6 integration (paper runtime + live-feed wiring), with Phases 1-5 implemented in code and tests.
- **Interpretation:** Core data contracts, signal/regime stack, backtest engine, and baseline strategies are in place. Immediate next build is real Kite option-chain integration in paper runtime.

### What Is Already Covered
- [x] Phase 1 core data interfaces (`DataFeed`, store, quality checks, DTO contracts + normalization wiring)
- [x] Phase 2 signal library modules + unit tests (including IV surface change signals)
- [x] Phase 3 baseline classifier + overrides + snapshot persistence + strategy activation router
- [x] Phase 4 backtest stack (`scripts/run_backtest.py`) with walk-forward, anti-overfitting checks, and report generation scaffolding
- [x] Phase 5 initial strategy set in code (`iron_condor`, `momentum`, `regime_probe`) with unit coverage
- [x] Paper runtime scaffold (`scripts/run_paper.py`) with persisted regime snapshots/transitions
- [x] Current automated test baseline green (`80 passed`)

### Immediate TODOs (Added)
- [x] Wire real Kite option-chain snapshots into `scripts/run_paper.py` (`chain_df` and `previous_chain_df`) and persist snapshots into cache.
- [ ] Replace remaining `NoOpStrategy` usage in `scripts/run_paper.py` with explicit strategy classes or fail-fast config validation.
- [x] Move paper/runtime and historical ingest paths to Kite-only provider wiring (`scripts/run_paper.py`, `scripts/download_historical.py`).
- [x] Replace multi-provider health check with Kite-focused checks (auth/profile + quote smoke checks).
- [x] Remove legacy `src/data/free_feed.py` module and free-data dependencies.
- [ ] Keep `src/data/truedata_feed.py` as optional standby feed; add integration hooks only when needed.
- [x] Add cron-ready daily maintenance workflow (`scripts/daily_maintenance.py` + `config/cron/daily_maintenance.crontab.example`).
- [x] Implement composite-key dedup support in `DataStore` for option-chain persistence (`timestamp + expiry + strike + option_type`).
- [ ] Add `paper_fills` daily P&L/reporting script (fills, fees, gross/net by day/strategy).
- [ ] **Visual review step:** run replay notebook/report and manually verify regime labels vs chart context before strategy comparisons.

### Required To Proceed To Phase 6 Completion
- [x] Add/confirm hysteresis + smoothing behavior tests in classifier to reduce flip-flops in transition zone.
- [x] Prove strict no-lookahead regime computation path for historical replay.
- [x] Run an end-to-end dry replay path that logs regime transitions and supports sanity validation (`scripts/replay_regime.py`).
- [x] Replace placeholder option-chain inputs in paper runtime with live Kite chain snapshots.
- [x] Implement option-chain persistence with composite-key dedup (timestamp + expiry + strike + option_type) before historical chain backtests.
- [ ] Add a Kite-backed paper-loop acceptance runbook (open, intraday, close checks).
- [ ] Convert replay outputs into final notebook artifact for visual inspection (optional packaging task; not a code blocker).

---

## Vision & Philosophy

### What we're building

A personal, modular algorithmic trading system that:

- Ingests market data from Kite Connect (primary), NSE public datasets, and optional TrueData standby adapters
- Maintains a library of composable signals and indicators
- Classifies market regimes to decide *when* to trade, not just *what* to trade
- Backtests strategies with realistic cost modeling and walk-forward validation
- Paper trades in live market conditions before risking capital
- Executes live trades via Zerodha Kite Connect with robust risk management
- Continuously develops and rotates strategies through a pipeline

### Core beliefs driving the design

1. **"Boring Alpha" wins.** Simple, interpretable strategies with excellent risk
   management beat complex ML models for solo traders. The edge is discipline,
   not sophistication.

2. **Regime detection is the real game.** A mediocre strategy in the right regime
   beats a great strategy in the wrong regime. Knowing *when not to trade* is
   more valuable than knowing when to trade.

3. **The system is a pipeline, not a product.** Strategies decay. Markets change.
   The valuable thing is the *infrastructure* for continuously developing,
   testing, and rotating strategies — not any single strategy.

4. **Risk management is not a feature; it's the foundation.** The circuit breaker
   runs independently and can override any strategy. Capital preservation comes
   before capital appreciation.

5. **Paper trading is not optional.** No strategy goes live without 50+ paper
   trades showing results consistent with backtest expectations.

---

## Phase 0 — Foundation (Weeks 1–2)

> Get the development environment working and verify all external dependencies.

### Goals
- [ ] Repository initialized, pushed to private GitHub
- [ ] Python 3.11+ virtual environment with all dependencies installed
- [ ] `.env` configured with API credentials
- [ ] `health_check.py` passing for Kite + core dependencies
- [ ] Basic familiarity with Kite Connect REST/WebSocket flows

### Tasks

```
0.1  Set up repository
     - git init, push to private GitHub repo
     - Install dependencies: pip install -e ".[dev]"
     - Verify: python scripts/health_check.py

0.2  Kite market-data setup
     - Verify Kite Connect app setup (API key/secret + redirect URL)
     - Complete auth flow and persist access token
     - Test: fetch 5 days of NIFTY 1-min/5-min candles
     - Test: fetch current NIFTY option-chain snapshot
     - Document quirks (rate limits, symbol mapping, timezone handling)

0.3  Zerodha Kite Connect setup
     - Ensure F&O segment is activated on your Zerodha account
       (requires: 18+ age, ₹2L+ annual income, pass derivatives test)
     - Pay ₹2,000 for Kite Connect API access
     - Generate API key and secret
     - Implement daily access token refresh flow
       (Kite tokens expire daily — needs login redirect each morning)
     - Test: fetch account margins and holdings

0.4  NSE companion datasets
     - Test NSE India VIX historical pulls
     - Test NSE FII/DII flow pulls
     - Document format quirks and retry rules

0.5  Notebook environment
     - Verify JupyterLab launches
     - Create 01_data_exploration.ipynb with Kite + NSE pulls
     - Plot sample NIFTY candle chart to confirm data pipeline works end-to-end
```

### Definition of Done
You can run a single Python script that fetches NIFTY data via Kite APIs,
pulls NSE companion datasets (VIX + FII/DII), and places a test order
(immediately cancelled) on Zerodha's paper trading mode.

### Estimated cost
- Kite Connect: ₹2,000 one-time
- Total Phase 0: ~₹2,000 (+ optional infra costs)

---

## Phase 1 — Data Pipeline (Weeks 3–5)

> Build the data ingestion, storage, and retrieval layer. Everything downstream
> depends on clean, reliable, well-organized data.

### Goals
- [ ] Historical data download and caching system
- [ ] Unified data interface (same API regardless of source)
- [ ] Local storage in Parquet format for fast backtesting
- [ ] Option chain historical data with Greeks
- [ ] Data quality validation and gap detection

### Tasks

```
1.1  Data source abstraction (src/data/)
     - Implement DataFeed protocol/interface with methods:
       - get_candles(symbol, timeframe, start, end) -> DataFrame
       - get_option_chain(symbol, expiry, timestamp) -> OptionChain
       - get_vix(start, end) -> DataFrame
       - get_fii_data(start, end) -> DataFrame
     - Implement KiteFeed (primary candles + option chain)
     - Implement NSE companion pipeline for FII/DII and validation

1.2  Historical data downloader (scripts/download_historical.py)
     - Bulk download NIFTY and BANKNIFTY daily candles (3–5 years)
     - Bulk download NIFTY and BANKNIFTY 5-min candles (1–2 years)
     - Download NIFTY option chain snapshots (daily EOD, 2–3 years)
       * This is the critical dataset for options backtesting
       * Must include: strike, expiry, LTP, OI, IV, Greeks
     - Download India VIX daily history (5 years)
     - Download FII/DII daily net flows from NSE (3 years)
     - Cache everything as Parquet files in data/cache/

1.3  Data storage layer (src/data/store.py)
     - Parquet-based storage with partitioning by symbol and date
     - Efficient date-range queries (avoid loading full dataset for a backtest)
     - Metadata tracking: when was each dataset last updated?
     - Auto-download missing data when backtest requests a date range

1.4  Data quality checks
     - Detect and log gaps (missing trading days, missing candles)
     - Validate OHLC integrity (high >= open/close, low <= open/close)
     - Flag truncated trading days (Muhurat, early closes)
     - Validate replay reproducibility from cached parquet snapshots
     - Create a data quality report notebook (02_data_quality.ipynb)

1.5  FII/DII flow pipeline
     - NSE publishes FII/DII data daily (provisional and revised)
     - Build scraper/downloader for this data
     - Store as time series alongside market data
     - This feeds directly into the regime classifier
```

### Data Sources Summary

| Source        | Cost         | What it provides                            | Historical depth |
|---------------|-------------|---------------------------------------------|-----------------|
| Kite Connect  | Paid app     | Intraday/daily candles, live quotes, option chain snapshots | Broker-limited by endpoint |
| TrueData      | Paid monthly | Optional secondary feed for data validation and extended coverage | Plan-dependent |
| NSE website   | Free         | India VIX, FII/DII flows, reference datasets | Varies          |
| NSE data shop | ₹10K+ once   | Full tick-by-tick, complete option history    | Comprehensive   |

### Key decision: How far back?

- **For price/index strategies:** 3–5 years (2021–2025). Covers post-COVID,
  weekly expiry era, current SEBI margin framework, multiple VIX regimes.
- **For regime detection validation:** 5–7 years. Need 2–3 full regime cycles.
- **Avoid pre-2019 for options strategies:** No weekly expiries existed, different
  lot sizes, different margin rules. That market no longer exists.
- **Weight recent data more heavily:** The options market in 2025 has far more
  retail participation than 2021. Counterparty behavior has shifted.

### Definition of Done
Running `python scripts/download_historical.py --full` populates your local
cache with all historical data. A notebook can load any date range for NIFTY
candles, option chains, VIX, and FII data in under 2 seconds.

---

## Phase 2 — Signal Library (Weeks 5–7)

> Build a composable library of technical indicators and market signals that
> strategies and the regime classifier both draw from.

### Goals
- [ ] Clean, tested implementations of all signals
- [ ] Each signal is a pure function: data in, value out
- [ ] Signals are composable (can be combined by strategies)
- [ ] Performance-optimized for backtesting (vectorized with numpy/pandas)

### Signal Inventory

```
2.1  Trend signals
     - EMA (Exponential Moving Average) — configurable period
     - EMA crossover (fast/slow) — binary signal
     - ADX (Average Directional Index, 14-period) — trend strength
     - Supertrend — trend direction with ATR-based bands
     - 50-DMA and 200-DMA position (price above/below)

2.2  Mean reversion signals
     - RSI (Relative Strength Index, 14-period)
     - Bollinger Bands (20-period, 2σ) — %B position
     - Z-score of price vs N-day mean
     - VWAP deviation — distance from VWAP in ATR units

2.3  Volatility signals
     - India VIX (absolute level + rate of change)
     - ATR (Average True Range) — absolute and as % of price
     - Historical volatility (20-day rolling)
     - Implied volatility rank (IV percentile over past year)
     - IV skew (put IV vs call IV at equal delta)

2.4  Volume & flow signals
     - Volume vs 20-day average (volume spike detection)
     - On-Balance Volume (OBV)
     - FII net flow (daily + 3-day cumulative)
     - FII cumulative direction (3-day, 5-day trend)

2.5  Options-specific signals
     - Put-Call Ratio (PCR) — OI-weighted
     - Max Pain — strike with maximum option seller profit
     - Change in Open Interest (ΔOI) — by strike
     - Total call OI vs total put OI (shift detection)
     - Implied volatility surface changes

2.6  Regime filter signals
     - VIX regime band (low / transitional / high)
     - ADX regime band (trending / transitional / ranging)
     - Nifty-BankNifty correlation (20-day rolling)
     - Composite regime score (weighted combination)

2.7  Composite / derived signals
     - "Boring Alpha" setup detector:
       ADX < 20 AND VIX < 14 AND PCR between 0.8–1.2
       → High-confidence range-bound environment for options selling
     - Breakout confirmation:
       Price breaks 20-day high AND volume > 1.5x average AND ADX rising
     - Regime transition early warning:
       VIX rate-of-change accelerating AND FII selling increasing
```

### Implementation approach

Each signal lives in `src/signals/` (create this directory) as a pure function:

```python
# src/signals/trend.py
def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def ema_crossover(series: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Returns 1 when fast EMA > slow EMA, -1 otherwise."""
    return (ema(series, fast) > ema(series, slow)).astype(int) * 2 - 1

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (0–100)."""
    # Use ta library for validated implementation
    from ta.trend import ADXIndicator
    indicator = ADXIndicator(high, low, close, window=period)
    return indicator.adx()
```

Use the `ta` library for standard indicators (already battle-tested), but wrap
them in our own functions for a consistent interface. Only hand-roll signals
that `ta` doesn't provide (FII flows, options-specific signals, composites).

### Testing strategy
- Every signal gets a unit test with known input/output
- Backtest each signal independently to verify it produces the expected
  distribution (e.g., RSI should spend ~equal time above/below 50 on random data)
- Create notebook 03_signal_exploration.ipynb to visualize each signal
  overlaid on NIFTY price data

### Definition of Done
`from src.signals import trend, mean_reversion, volatility, options_signals`
works. Each module contains pure functions that accept pandas Series/DataFrames
and return Series. All signals have unit tests and a visual notebook.

---

## Phase 3 — Regime Classifier (Weeks 7–8)

> Wire the signals into the regime classification system.

### Goals
- [ ] Regime classifier uses live signal library
- [ ] Validated against historical data (do regime labels make sense?)
- [ ] Regime transitions logged and visualized
- [ ] Backtestable — can compute regime at any historical point in time

### The Four Regimes

| Regime             | VIX        | ADX      | Character                         | Best strategies              |
|--------------------|-----------|----------|-----------------------------------|------------------------------|
| Low-vol trending   | < 14      | > 25     | Calm, directional                 | Trend following, directional options |
| Low-vol ranging    | < 14      | < 20     | Quiet, mean-reverting             | Iron condors, credit spreads, mean reversion |
| High-vol trending  | > 18      | > 25     | Volatile, directional             | Trend following (wider stops), directional options |
| High-vol choppy    | > 18      | < 20     | Violent, no direction             | Sit out or very small positions |

### Tasks

```
3.1  Wire signals into RegimeSignals dataclass
     - Compute all regime signals from raw market data
     - Handle the "VIX transition zone" (14–18) with hysteresis
       (don't flip-flop between regimes on small VIX moves)
     - Implement signal smoothing where needed (e.g., 3-day EMA of ADX
       to avoid whipsawing on daily noise)

3.2  Historical regime analysis (notebook)
     - Compute regime labels for every trading day 2021–2025
     - Visualize: Nifty price chart color-coded by regime
     - Compute: how many days in each regime? Average duration?
     - Validate: do regime labels match your intuitive reading of the market?
       (e.g., March 2020 should be HIGH_VOL_TRENDING, mid-2023 LOW_VOL_RANGING)

3.3  Override signals
     - Implement FII flow override (heavy selling → force high-vol classification)
     - Implement correlation breakdown detection (Nifty/BankNifty decoupling)
     - Test: how often do overrides trigger? Are they useful or noise?

3.4  Regime transition detection
     - Log every regime change with full signal snapshot
     - Compute: do transitions predict anything useful?
       (e.g., does LOW_VOL_RANGING → HIGH_VOL_CHOPPY predict a drop?)
     - Build: early warning system for regime transitions
       (VIX rate-of-change accelerating, ADX approaching thresholds)

3.5  Regime backtestability
     - Ensure regime can be computed at any historical timestamp
       using only data available at that time (no lookahead)
     - This is critical: the backtest engine will call
       classifier.classify(signals_at_time_t) for every bar
```

### Definition of Done
Running the regime classifier on 3 years of data produces a labeled time series
that passes visual inspection. You can overlay it on a NIFTY chart and the
labels make intuitive sense. Transition points correspond to known market events
(elections, RBI surprises, global shocks).

---

## Phase 4 — Backtesting Engine (Weeks 8–11)

> Build the backtesting infrastructure with realistic cost modeling.
> This is where most solo traders fail — not in strategy logic, but in
> backtest validity.

### Goals
- [ ] Walk-forward backtesting (not just in-sample)
- [ ] Realistic fill simulation (slippage, partial fills)
- [ ] Full Indian cost modeling (STT, GST, stamp duty, brokerage)
- [ ] Regime-aware backtesting (strategy only active in its regimes)
- [ ] Comprehensive metrics and reporting

### Tasks

```
4.1  Backtest engine core (src/backtest/engine.py)
     - Event-driven architecture: iterate through historical bars
     - For each bar:
       1. Update regime classifier with current signals
       2. Check if strategy should be active in current regime
       3. If active and no position: call strategy.generate_signal()
       4. If active and has position: call strategy.get_exit_conditions()
       5. If signal is actionable: simulate fill via simulator
       6. Update P&L, risk metrics, trade journal
       7. Check circuit breaker
     - Support multiple strategies running simultaneously
     - Support both candle-based (daily/5min) and option-chain-based backtesting

4.2  Fill simulator (src/backtest/simulator.py)
     - Market orders: fill at close + random slippage (configurable, default 0.05%)
     - Limit orders: fill only if price touches limit within the bar
     - Options: simulate bid-ask spread impact (use historical spread if available,
       else estimate from IV and time to expiry)
     - Partial fills: for larger orders, simulate partial execution
     - CRITICAL: no lookahead bias — fill price uses only data available at signal time

4.3  Cost model (integrated into simulator)
     All costs are configurable in settings.yaml:
     - Brokerage: ₹20 per F&O order (Zerodha flat fee)
     - STT: 0.0125% on sell side (options), 0.0125% on sell side (futures)
     - Exchange transaction charges: 0.053% (NSE F&O)
     - GST: 18% on (brokerage + transaction charges)
     - SEBI turnover fee: 0.0001%
     - Stamp duty: 0.003% on buy side
     - Slippage estimate: 0.05% per side (configurable)
     These add up. A round-trip on a ₹100 option premium costs roughly ₹1.50–2.00
     in friction. If your strategy's edge is 2%, half of it is eaten by costs.

4.4  Walk-forward validation
     - Split data into rolling train/test windows:
       Train: 12 months → Test: 3 months → Roll forward 3 months
     - For each window:
       1. Optimize parameters on train set (or use fixed params)
       2. Run strategy on test set with NO parameter changes
       3. Record out-of-sample performance
     - Aggregate: what's the strategy's out-of-sample Sharpe? Win rate?
     - If in-sample Sharpe is 2.0 but out-of-sample is 0.3, you've overfit.

4.5  Metrics & reporting (src/backtest/metrics.py + report.py)
     Core metrics:
     - Total return (%) and annualized return
     - Sharpe ratio (risk-free rate = India 10Y bond yield, ~7%)
     - Sortino ratio (downside deviation only)
     - Maximum drawdown (%) and duration
     - Win rate and average win/loss ratio
     - Profit factor (gross profit / gross loss)
     - Number of trades and average holding period
     - Calmar ratio (annualized return / max drawdown)
     - Monthly returns table
     - Regime-segmented performance (how does strategy do in each regime?)

     Report output:
     - HTML report with charts (equity curve, drawdown, monthly heatmap)
     - JSON metrics file for programmatic comparison across strategies
     - Trade log CSV with full context (regime, signals, Greeks at entry/exit)

4.6  Anti-overfitting checks
     - Monte Carlo permutation test: shuffle trade returns, what % of random
       orderings produce a better Sharpe? If > 5%, edge is likely noise.
     - Parameter sensitivity analysis: vary key params ±20% — does the strategy
       degrade gracefully or collapse? Cliff edges = overfitting.
     - Cross-instrument validation: if strategy works on NIFTY, does it show
       some edge on BANKNIFTY without re-tuning? If not, suspicious.
     - Minimum trade count: reject any backtest with < 50 trades. Not enough
       data for statistical significance.
```

### Definition of Done
`python scripts/run_backtest.py --strategy iron_condor --from 2022-01-01 --to 2025-01-01`
produces an HTML report showing walk-forward results with realistic costs. The
report includes regime-segmented performance and Monte Carlo significance test.

---

## Phase 5 — Strategy Implementation (Weeks 11–14)

> Implement the first set of strategies, starting simple.

### Strategy Development Order

Build in order of complexity. Each strategy must pass the full backtest
validation pipeline before moving to the next.

```
5.1  Strategy 1: Nifty Iron Condor (options selling)
     - Active regimes: LOW_VOL_RANGING, LOW_VOL_TRENDING (tighter strikes)
     - Logic:
       * Sell OTM call at ~15 delta
       * Sell OTM put at ~15 delta
       * Buy protective wings 100 points further out
       * Enter with 5–14 DTE
     - Exit conditions:
       * 50% of max profit reached → close
       * Loss equals premium received → close
       * DTE = 1 → close (avoid expiry risk)
       * Regime changes to HIGH_VOL → close
     - Position sizing: max 2 lots, ₹1,00,000 per trade
     - Expected profile: high win rate (65–75%), small wins, occasional larger losses
     - This is the "boring alpha" strategy. Start here.

5.2  Strategy 2: EMA Crossover Momentum
     - Active regimes: LOW_VOL_TRENDING, HIGH_VOL_TRENDING
     - Logic:
       * Long when 20 EMA > 50 EMA AND ADX > 25
       * Short when 20 EMA < 50 EMA AND ADX > 25
       * Use Nifty futures or ATM options (depending on capital)
     - Exit: ATR-based trailing stop (2x ATR)
     - Expected profile: lower win rate (40–50%), but winners >> losers

5.3  Strategy 3: RSI Mean Reversion
     - Active regimes: LOW_VOL_RANGING
     - Logic:
       * Buy when RSI(14) < 30 AND price within Bollinger lower band
       * Sell when RSI(14) > 70 AND price within Bollinger upper band
       * VWAP confirmation required (don't fade the institutional trend)
     - Exit: RSI reverts to 50, or z-score returns to 0.5

5.4  Strategy 4: Expiry Day Theta Decay
     - Active regimes: LOW_VOL_RANGING (Thursday Nifty / Wednesday BankNifty)
     - Logic:
       * Sell ATM straddle at 1:00 PM on expiry day
       * Buy protective wings (or use iron butterfly)
       * Profit from rapid theta decay in final 2.5 hours
     - Exit: fixed time (3:15 PM) or 50% profit, whichever comes first
     - Expected profile: very high win rate, small per-trade profit,
       occasional large loss on unexpected expiry-day moves
     - WARNING: This strategy has the highest variance. Implement last.

5.5  For each strategy, the implementation checklist:
     [ ] Inherits from BaseStrategy
     [ ] generate_signal() produces well-formed Signal objects
     [ ] get_exit_conditions() covers all exit scenarios
     [ ] compute_position_size() respects risk limits
     [ ] on_regime_change() handles graceful exit
     [ ] Unit tests with mocked market data
     [ ] Walk-forward backtest with realistic costs
     [ ] Monte Carlo significance test passes (p < 0.05)
     [ ] Parameter sensitivity analysis shows graceful degradation
     [ ] Documented in notebook with trade examples
```

### Definition of Done
At least Strategy 1 (iron condor) passes walk-forward backtesting with a
positive out-of-sample Sharpe ratio after costs. You understand exactly why
it works, when it works, and when it doesn't.

---

## Phase 6 — Paper Trading (Weeks 14–18)

> The bridge between backtest and real money. This phase is non-negotiable.

### Goals
- [ ] Paper trading engine that mirrors live execution exactly
- [ ] Same code path as live — only the execution layer swaps
- [ ] Minimum 50 paper trades before going live
- [ ] Paper results compared rigorously to backtest expectations

### Tasks

```
6.1  Paper trading engine (src/execution/paper.py)
     - Implements same interface as broker.py
     - Simulates fills with realistic slippage model
     - Tracks positions, P&L, margins in real-time
     - Logs every signal, order, and fill identically to live

6.2  Live data integration
     - Kite real-time WebSocket feed for live ticks/candles
     - Live option chain with Greeks (refreshed every 30 seconds or tick)
     - Live India VIX
     - Regime classifier running on live data

6.3  Monitoring dashboard (src/monitoring/dashboard.py)
     - Streamlit dashboard showing:
       * Current regime classification with signal values
       * Active strategies and their states
       * Open positions with live P&L
       * Circuit breaker status
       * Today's trade log
       * Equity curve (paper)
     - Telegram alerts for: trades, regime changes, circuit breaker events

6.4  Paper trading validation
     Run paper trading for 4+ weeks. Then compare:
     - Paper win rate vs backtest win rate (should be within 5–10%)
     - Paper average P&L per trade vs backtest (should be within 15–20%)
     - Paper max drawdown vs backtest max drawdown
     - Paper Sharpe vs backtest Sharpe
     If paper results are significantly worse than backtest:
       → The backtest has hidden flaws. Do NOT go live. Investigate.
     Common causes of paper-vs-backtest divergence:
       - Slippage model in backtest was too optimistic
       - Option chain data in backtest didn't capture intraday IV swings
       - Strategy was inadvertently overfit to a specific historical period
       - Regime during paper period is different from backtest period

6.5  Psychological preparation
     - Paper trading should feel real. Track it daily. Review weekly.
     - Practice NOT intervening when the bot takes a loss
     - Practice NOT tweaking parameters mid-run
     - If you can't sit through a ₹15,000 paper loss without wanting to
       override the bot, you're not ready for live trading
```

### Definition of Done
50+ paper trades completed over 4+ weeks. Paper results are statistically
consistent with backtest expectations (within 2 standard deviations on key
metrics). You have a documented record comparing paper vs backtest performance.

---

## Phase 7 — Live Deployment (Weeks 18–22)

> Real money. Small size. Maximum caution.

### Goals
- [ ] Live trading on a Mumbai VPS with failover
- [ ] Starting capital: ₹2–3 lakhs (minimum viable for options selling)
- [ ] Circuit breaker proven and trusted
- [ ] Daily monitoring routine established

### Tasks

```
7.1  Infrastructure setup
     - Provision AWS/GCP Mumbai-region VPS (~₹500/month)
       * 2 vCPU, 4GB RAM, SSD — more than enough
       * Mumbai region = 1–3ms to NSE vs 20–40ms from Kottayam
     - Install NiftyQuant on VPS
     - Set up systemd service for auto-restart
     - Configure Telegram alerts as primary notification channel
     - Set up daily health check cron job

     CRITICAL: Your Kottayam machine is the MONITORING terminal.
     The VPS is the EXECUTION machine. If your home internet drops,
     the bot continues running and the circuit breaker protects you.

7.2  Kite Connect daily auth
     - Kite access tokens expire daily
     - Options:
       a) Manual: generate token each morning before market open (tedious)
       b) Automated: use kite_auth.py with TOTP-based auto-login
          (slightly grey area with Zerodha's ToS — research this)
       c) Use a broker with persistent API tokens instead (Fyers, Dhan)
     - Whichever method: token refresh must happen before 9:15 AM IST

7.3  Go-live checklist
     Before placing the first real trade:
     [ ] Circuit breaker tested with simulated losses
     [ ] Kill switch tested manually
     [ ] Telegram alerts confirmed working
     [ ] Stop-loss orders confirmed working at exchange level
         (so they execute even if bot crashes)
     [ ] Broker mobile app installed as manual backup
     [ ] Capital deployed: ₹2–3 lakhs (no more for first month)
     [ ] Family/personal situation stable enough for this
     [ ] Emergency fund separate from trading capital

7.4  First month protocol
     - Trade at MINIMUM lot size (1 lot only)
     - Maximum 1 active strategy
     - Review every trade same evening
     - Weekly review: am I following the system or overriding it?
     - Monthly review: compare live to paper to backtest
     - DO NOT increase size for at least 30 trades

7.5  Scaling protocol (month 2+)
     - Increase only if month 1 live results are consistent with paper
     - Scale by 1 lot at a time, never doubling overnight
     - Add second strategy only after first is stable for 2 months
     - Target capital deployment: ₹5 lakhs by month 3, ₹10 lakhs by month 6
       (only if results warrant it)
```

### Definition of Done
One month of live trading completed with results consistent with paper trading.
No manual overrides. Circuit breaker was not manually triggered. System ran
autonomously on VPS with monitoring from Kottayam.

---

## Phase 8 — Strategy Pipeline & Rotation (Ongoing)

> The long game. This is what separates a project from a system.

### The Portfolio of Strategies Model

At any time, maintain strategies at three readiness levels:

| Stage          | Count | What's happening                                      |
|----------------|-------|-------------------------------------------------------|
| **Live**       | 1–2   | Deployed with real capital, monitored daily            |
| **On deck**    | 1–2   | Paper trading, accumulating out-of-sample track record |
| **In development** | 2–3 | Being backtested, walk-forward validated              |

### Ongoing rhythm

```
Weekly:
  - Review all live trades
  - Check regime classifier accuracy (did it predict the week's behavior?)
  - Compare live P&L to expected P&L range from backtest
  - Check paper trading strategies for readiness

Monthly:
  - Full performance review: live vs paper vs backtest
  - Strategy health check: is the live strategy's edge decaying?
    Signs of decay: win rate dropping, average P&L shrinking,
    drawdowns getting longer
  - Regime analysis: have regime durations or transitions changed?
  - Develop 1 new strategy idea, begin backtesting

Quarterly:
  - Full strategy rotation review
  - Is any live strategy underperforming its paper/backtest expectations
    by > 2 standard deviations? → Consider retiring
  - Is any on-deck strategy consistently outperforming? → Consider promoting
  - Re-run walk-forward validation on all strategies with updated data
  - Review and update cost model (SEBI changes regulations periodically)
  - Tax review: are trading profits/losses being tracked correctly?
```

### Strategy ideas pipeline (future development)

```
- Pairs trading (HDFC Bank / ICICI Bank, TCS / Infosys)
- VIX mean reversion (buy when VIX > 20, sell when < 12)
- Earnings straddle (buy straddle before earnings, sell before event)
- OI-based support/resistance (max pain gravitational strategy)
- Sentiment signal from news NLP (RBI announcements, budget)
- Multi-timeframe momentum (daily trend + 5-min entry)
- Calendar spreads (exploit term structure of IV)
```

---

## Architecture Principles

### 1. Separation of concerns

```
Data Layer    → NEVER places orders
Strategy      → NEVER touches the broker; only emits Signals
Execution     → ONLY follows Signals; has no market opinion
Risk          → Can OVERRIDE everything; runs independently
Monitoring    → Observes only; never modifies state
```

### 2. Same code path for backtest, paper, and live

The strategy code is identical in all three modes. Only the data source
and execution layer swap out:

```
Backtest:  HistoricalDataFeed + BacktestSimulator
Paper:     KiteDataFeed       + PaperExecutor
Live:      KiteDataFeed       + KiteBroker
```

This eliminates an entire class of "works in backtest but not live" bugs.

### 3. Configuration over code

Every tunable number lives in `settings.yaml`. If you find yourself hardcoding
a threshold, stop and move it to config. Changing strategy parameters for a new
backtest run should never require a code change or git commit.

### 4. Log everything with context

Every trade entry includes: signal values, regime classification, Greeks
snapshot, VIX level, strategy confidence score, and human-readable reason.
This trade journal is your most valuable asset for learning what works.

### 5. Fail safe, not fail open

- If Kite market-data stream disconnects or goes stale → stop generating new signals
- If Kite API errors → retry 3 times, then alert and stop
- If regime is UNKNOWN → no trades
- If circuit breaker state is unclear → assume tripped

---

## Risk Framework

### Position-level risk
- No single position risks more than 5% of capital
- Options strategies must be defined-risk (spreads, not naked)
  until capital exceeds ₹10 lakhs and you have 6+ months track record
- Stop-loss orders placed at exchange level (survive bot crashes)

### Portfolio-level risk
- Maximum 4 concurrent positions
- Maximum daily loss: 3% of capital → circuit breaker trips
- Maximum drawdown: 15% from peak → full shutdown, manual reset required

### Regime-based risk
- HIGH_VOL_CHOPPY → reduce position size to 50% of normal, or sit out entirely
- VIX > 22 → no new options selling positions (premium is high but so is risk)
- Regime transition in progress → no new entries until regime stabilizes

### Operational risk
- VPS in Mumbai for execution reliability
- Exchange-level stop-losses as failsafe
- Telegram alerts for any anomaly
- Broker mobile app as manual kill switch
- Monthly review of broker API changes / SEBI regulation changes

---

## Cost Model

Realistic cost modeling for Indian F&O (as of 2025–2026):

```
Per options trade (one leg, one side):
  Brokerage:                    ₹20 flat (Zerodha)
  STT (sell side only):         0.0125% of (premium × quantity)
  Exchange transaction charge:  0.053% of turnover
  GST:                          18% of (brokerage + exchange charges)
  SEBI turnover fee:            0.0001% of turnover
  Stamp duty (buy side only):   0.003% of turnover
  Slippage (estimated):         0.05% per side

Example: Selling 1 lot (50 qty) of NIFTY 25500 CE at ₹120
  Turnover = 120 × 50 = ₹6,000
  Brokerage:    ₹20
  STT:          ₹0.75 (sell side)
  Exchange:     ₹3.18
  GST:          ₹4.17
  SEBI:         ₹0.006
  Stamp:        ₹0.18 (buy side)
  Slippage:     ~₹3.00
  TOTAL:        ~₹31.30 per leg

  An iron condor has 4 legs × 2 sides (entry + exit) = 8 transactions
  Approximate round-trip cost: ~₹250

  If max profit on the condor is ₹3,000, costs eat ~8% of profit.
  → This is why cost modeling matters.

Tax treatment:
  F&O profits = business income, taxed at your income tax slab rate
  (NOT 15% STCG like equity delivery)
  If turnover > ₹10 crore: tax audit required
  Advance tax: pay quarterly if tax liability > ₹10,000
  → Budget for a CA who understands F&O taxation (₹5,000–15,000/year)
```

---

## Regime Detection Deep Dive

### Why regime detection is the core differentiator

Most retail traders run the same strategy in all market conditions. They make
money for 3 months, then the market changes and they give it all back. This is
the "regime change blindness" problem.

Our approach: **the strategy router is more important than any individual strategy.**

### Signal combination logic

```
Primary classification (VIX × ADX):

         ADX > 25           ADX 20-25           ADX < 20
       ┌─────────────┬───────────────────┬─────────────────┐
VIX<14 │ LOW_VOL     │ Lean toward       │ LOW_VOL         │
       │ TRENDING    │ previous regime   │ RANGING         │
       ├─────────────┼───────────────────┼─────────────────┤
VIX    │ Use ADX to  │ TRANSITIONAL      │ Use ADX to      │
14-18  │ tiebreak    │ (lean previous)   │ tiebreak        │
       ├─────────────┼───────────────────┼─────────────────┤
VIX>18 │ HIGH_VOL    │ Lean toward       │ HIGH_VOL        │
       │ TRENDING    │ high-vol side     │ CHOPPY          │
       └─────────────┴───────────────────┴─────────────────┘

Override signals:
  - FII net selling > ₹6,000 Cr (3-day) → Force HIGH_VOL classification
  - Nifty-BankNifty correlation < 0.80  → Flag structural change
  - VIX rate of change > 3 pts/day      → Early warning of regime shift

Hysteresis:
  - Don't switch regimes on a single day's data
  - Require 2 consecutive days of new-regime signals to confirm transition
  - Exception: VIX spike > 5 points in a day → immediate regime override
```

### Regime duration statistics (approximate, 2021–2025)

```
LOW_VOL_RANGING:    Most common. Average duration 15–30 trading days.
LOW_VOL_TRENDING:   Second most common. Average 10–20 days.
HIGH_VOL_TRENDING:  Less common. Average 5–15 days. Often around events.
HIGH_VOL_CHOPPY:    Least common. Average 3–10 days. Usually transitional.
```

---

## Strategy Niches for Indian Markets

### Why Nifty/BankNifty options dominate

- Highest liquidity in Indian derivatives
- Weekly expiries create frequent, recurring opportunities
- Defined-risk strategies possible with spreads
- Low capital requirements (₹25K–₹3L depending on strategy)
- Most retail quant traders in India operate here

### Why "boring" strategies win for solo traders

| Factor              | Complex ML strategy  | Simple rule-based     |
|---------------------|---------------------|-----------------------|
| Overfitting risk    | Very high           | Moderate              |
| Debugging           | Opaque              | Transparent           |
| Regime adaptability | Needs retraining    | Needs parameter shift |
| Monitoring          | Hard to tell if broken | Easy to audit      |
| Edge source         | Data/compute        | Discipline/patience   |
| Competition         | Competing with firms | Competing with retail |

### The "indicator as filter, not signal" principle

Standard indicators (RSI, MACD) on their own are **low-conviction signals**.
Everyone sees the same RSI=30 reading. The edge comes from:

1. Using them as **filters** (only trade when ADX confirms trending)
2. Combining with **regime context** (RSI mean reversion only in LOW_VOL_RANGING)
3. Adding **options-specific intelligence** (sell premium when IV rank is high)
4. Applying **disciplined position sizing** (Kelly criterion or fractional)

---

## Infrastructure & Deployment

### Development (Kottayam)
- Your local machine for coding, backtesting, analysis
- Jupyter notebooks for exploration
- Git push to private repo

### Execution (Mumbai VPS)
- AWS/GCP Mumbai region (~₹500/month)
- Runs the live/paper trading bot
- 1–3ms to NSE exchange servers
- systemd service for auto-restart
- Daily health check cron
- 300 Mbps Kottayam line is irrelevant for execution —
  the VPS handles it. Your home connection is only for monitoring.

### Monitoring (anywhere)
- Telegram alerts for trades, regime changes, errors
- Streamlit dashboard accessible via VPS
- Broker mobile app as manual kill switch

### Why Kottayam is not a disadvantage
- 20–40ms latency from Kottayam to Mumbai is irrelevant for strategies
  that hold positions for hours to days
- Even for intraday strategies, the VPS in Mumbai handles execution
- You're not doing HFT — you're doing medium-frequency options trading
- The 300 Mbps line is overkill; 1 Mbps would suffice for monitoring
- What matters: connection reliability, not speed. Keep the VPS as primary.

---

## Failure Modes & Mitigations

### 1. Overfitting (highest risk for you)
**Why it's dangerous:** You're smart enough to build complex models that
look amazing in backtest but are just memorizing noise.
**Mitigation:** Walk-forward validation, Monte Carlo tests, parameter
sensitivity analysis, minimum 50-trade requirement, cross-instrument testing.

### 2. Slippage erosion
**Why it's dangerous:** Your strategy shows 1% edge in backtest. After
realistic slippage and costs, it's net negative.
**Mitigation:** Conservative cost model in backtest (overestimate costs).
Paper trade to measure real slippage. If paper results match backtest, costs
are modeled correctly.

### 3. Regime change blindness
**Why it's dangerous:** Strategy works for 3 months then bleeds as the
market shifts to a different regime.
**Mitigation:** The regime classifier. Strategy only runs in declared regimes.
Automatic position exit on regime change.

### 4. Infrastructure failure
**Why it's dangerous:** Bot crashes with open positions during high volatility.
**Mitigation:** Mumbai VPS, exchange-level stop-losses, Telegram alerts,
broker mobile app as manual backup. Circuit breaker that kills positions if
things go wrong.

### 5. API / broker changes
**Why it's dangerous:** Zerodha updates their API, your bot silently fails.
**Mitigation:** Daily health check, version-pin dependencies, Telegram alert
on any API error, manual backup procedures documented.

### 6. Psychological override
**Why it's dangerous:** You turn off the bot during a drawdown, then the
strategy recovers and you miss the rebound. Or you increase size after wins.
**Mitigation:** No manual override capability in the bot (by design). All
parameter changes require a config file edit, git commit, and redeployment.
Drawdowns are handled by the circuit breaker, not by you.

### 7. Data integrity issues
**Why it's dangerous:** Partial/missing snapshots or token mismatches corrupt
signal quality and strategy behavior.
**Mitigation:** Kite-only schema normalization, strict data quality checks,
and immutable cached parquet snapshots for reproducible backtests.

### 8. SEBI regulatory changes
**Why it's dangerous:** SEBI restricts weekly expiries, increases lot sizes,
or changes margin rules — strategy becomes unviable.
**Mitigation:** Build modular strategies that can adapt to different instruments.
Monitor SEBI circulars quarterly. Have on-deck strategies ready.

---

## What Claude Can Help With

### High-value use cases (use freely)
- **Code review:** Share backtest code, I check for lookahead bias, bugs
- **Statistical analysis:** Sharpe ratios, drawdown analysis, Monte Carlo sims
- **Risk framework design:** Position sizing algorithms, circuit breaker logic
- **Regime detection:** Help build and validate classification logic
- **Post-trade analysis:** Share trade logs, I identify patterns and weaknesses
- **NLP pipeline for news signals:** Given your RAG background, build a pipeline
  where Claude API processes RBI announcements / earnings reports into
  structured sentiment signals

### What Claude cannot do
- Make real-time trading decisions (latency too high, no live data)
- Predict market direction (no edge here)
- Replace your judgment on "is this strategy dead or just in a drawdown"

### Best workflow
The bot runs deterministic code. You use Claude (via API or conversation) as
an offline analyst — reviewing code, analyzing results, stress-testing
assumptions, and helping think through regime changes.

---

## Realistic Expectations

### Financial

| Phase         | Duration   | Expected P&L                                     |
|---------------|-----------|--------------------------------------------------|
| Learning      | Month 1–6  | **Lose** ₹50K–₹1L (treat as tuition)             |
| Competent     | Month 6–12 | Break even to small profit (₹5–15K/month)        |
| Profitable    | Year 2+    | ₹15–40K/month on ₹5–10L capital (if edge holds)  |

These are honest median outcomes, not aspirational targets.

### Time investment

| Activity                              | Hours/week |
|---------------------------------------|-----------|
| Phase 0–5 (building)                  | 10–15     |
| Phase 6 (paper trading + monitoring)  | 5–8       |
| Phase 7+ (live + maintenance)         | 3–5       |
| Strategy development (ongoing)        | 2–4       |

### The most important thing

**Keep your job search as the priority.** This trading system is a side project
that might become supplementary income. The AI/ML engineering career is the
reliable wealth-building path. A jump from ₹12 LPA to ₹25 LPA is more
achievable and certainly more reliable than consistently profitable trading.

The engineering skills you build doing this — Python, statistics, real-time
systems, data pipelines, production deployment — make your resume stronger
regardless of whether the bot makes money.

---

## Decision Log

Track major design decisions here as the project evolves.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-13 | Standardize on Kite as primary feed, retain TrueData adapter as optional standby | Keep operational simplicity while preserving a paid secondary path for future validation/expansion |
| _start_ | Use TrueData + Kite Connect | Original multi-provider plan |
| _start_ | Parquet for local storage | Fast columnar reads for backtesting, better than CSV or SQLite for time series |
| _start_ | Regime-based strategy routing | Core thesis: knowing when NOT to trade is the real edge |
| _start_ | Walk-forward validation required | Prevents overfitting; any strategy that only works in-sample is rejected |
| _start_ | Minimum 50 paper trades before live | Statistical minimum for meaningful comparison to backtest |
| _start_ | Mumbai VPS for execution | Eliminates network reliability risk from Kottayam |
| _start_ | Iron condor as first strategy | Highest win rate, defined risk, most studied in Indian options community |
| | | |
| | | |
