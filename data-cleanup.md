# Data Cleanup and Signal Preparation Plan

## Purpose
This document defines:
- A one-time cleanup process for historical candles and option-chain data.
- An incremental process for new data (for example, 2026 onward).
- A checklist for implementation and expert review.

The goal is to make downstream backtests, walk-forward tests, and paper/live logic run on reproducible, point-in-time-correct, and quality-gated data.

---

## Current Findings (as of 2026-02-18)

### Candle duplicates (daily)
- `NIFTY 1d`: 1080 rows, 1069 unique trading days, 11 duplicate trading days (~1.03%).
- `BANKNIFTY 1d`: 1032 rows, 1024 unique trading days, 8 duplicate trading days (~0.78%).
- Duplicate pattern is concentrated in 2026 data and is mostly a timestamp convention mix:
  - One row around `18:30:00` (legacy daily timestamp convention).
  - One row around `00:00:00` (calendar-day timestamp convention).
- In most duplicate pairs, OHLC values are near-identical, but volume/precision can differ.

### Option-chain historical data
- Historical chain data is now available from NSE FO bhavcopy ingestion.
- Current normalized historical chain rows have:
  - `iv = 0`, `delta = 0`, `gamma = 0`, `theta = 0`, `vega = 0`, `rho = 0` by default.
- This means chain-derived analytics using IV/Greeks must be computed in a post-processing feature pass before relying on them.

---

## Problem Statement

### Why "take the first row" is not sufficient
- Chronological first/last may preserve a lower-quality row.
- Duplicate rows may differ in volume, precision, or corrected OHLC values.
- A fixed "first row wins" rule can silently keep weaker records and distort indicators.

### Correct approach
Use a deterministic **best-row selection policy** per `(symbol, timeframe, trade_date)` based on data quality scoring, with deterministic tie-breakers.

### Holiday-aware gap treatment
- Daily gap checks should be evaluated in **trading-day space**, not wall-clock minutes.
- Weekends and declared NSE holidays must not be treated as data failures.
- Missing trading sessions are captured via:
  - `missing_trading_days` (primary completeness signal)
  - trading-day-aware `largest_gap_minutes` (secondary severity signal)

---

## Data Architecture Decision

### Keep two layers
1. `raw` layer:
   - Immutable ingested files from sources.
   - Never overwritten by cleanup logic.
2. `curated` layer:
   - Cleaned, deduplicated, canonical timestamps, quality-gated.
   - Used by backtest/walk-forward/paper/live pipelines.

This preserves auditability and allows reprocessing when rules evolve.

---

## Candle Cleanup Policy (1d)

### Canonical key
- Deduplicate by `(symbol, timeframe, trade_date_ist)`.

### Canonical timestamp
- Use a single normalized daily timestamp convention in curated data:
  - Recommendation: `trade_date` at `00:00:00` (naive local calendar key).
- Do not mix `18:30` and `00:00` conventions in curated output.

### Best-row scoring (per duplicate group)
Rank rows by:
1. Valid OHLC shape (`high >= max(open, close)`, `low <= min(open, close)`).
2. Fewer null/invalid numeric fields.
3. Presence of informative `volume`/`oi` (non-null, non-negative).
4. Higher precision/non-rounded values (if clearly a tie-breaker signal).
5. Deterministic final tie-breaker (for example, latest timestamp).

### Important rule
- Never average OHLC across duplicates.
- Averaging can create synthetic bars that never traded and invalidates candle semantics.

---

## One-Time Historical Cleanup Workflow

1. Read raw candles for target symbols/timeframes.
2. Compute `trade_date_ist` from timestamp.
3. Group by dedupe key and apply best-row selection.
4. Rebuild curated dataset with canonical timestamps.
5. Run quality gates (duplicates, missing days, OHLC validity, monotonic order).
6. Write curated partitions and a cleanup report:
   - Input rows, output rows, dropped duplicates.
   - Duplicate percentage by symbol/year.
   - Any unresolved anomaly groups.
7. Freeze a cleanup version tag (for reproducibility).

---

## Incremental Append Workflow (2026 onward)

1. New data is appended to raw only.
2. Identify affected recent partitions (for example, current year/month).
3. Re-run dedupe and quality gates only on affected window.
4. Rewrite curated partitions for that window.
5. Emit incremental cleanup report and version marker.

This avoids full reprocessing on every daily run.

---

## Option-Chain Preparation Policy

### Current state
Historical bhavcopy chain has OI/price/volume but no reliable IV/Greeks fields populated in curated data yet.

### Required post-processing
1. Build per-snapshot chain quality checks:
   - CE/PE presence, strike continuity, expiry continuity, stale/zero price flags.
2. Select analysis expiry bucket (for example, DTE window used by strategy).
3. Compute implied volatility from option price and spot.
4. Compute Greeks from solved IV.
5. Store feature table with quality flags and all derived fields.

### Rule for missing derived values
- Prefer `NaN` + quality flag over fake zeros for unavailable IV/Greeks.

---

## Why ATM First, and When Other Strikes Matter

### ATM focus (first phase)
- Most liquid and stable quotes.
- Lower microstructure noise.
- Robust for regime-level volatility and sentiment context.

### Non-ATM usage (next phases)
- Near-OTM (10-30 delta): strike selection for iron condor/jade lizard.
- Wings/far OTM: tail risk, hedge quality, skew stress.
- Cross-expiry structure: term slope and roll/expiry behavior.
- OI walls away from ATM: support/resistance and pin risk context.

### Practical rollout
1. Regime features: ATM/near-ATM first.
2. Execution features: add non-ATM and expiry-structure once data quality is stable.

---

## Expert Review Questions

1. Is the dedupe key `(symbol, timeframe, trade_date_ist)` acceptable for NSE daily bars?
2. Is best-row ranking (quality-first, deterministic tie-break) preferable to source-priority rules?
3. Is canonical daily timestamp at `00:00:00` acceptable, or should exchange-close convention be enforced?
4. Are there regulatory/reporting constraints that require preserving original timestamp semantics in curated outputs?
5. For IV solve, should we standardize on close-based pricing, mid-price proxy, or last trade price fallback?
6. What minimum chain quality thresholds should block strategy execution?

---

## Implementation Checklist

### Phase A: Candle cleanup
- [x] Create curated candle dataset path.
- [x] Implement `trade_date_ist` derivation.
- [x] Implement duplicate grouping and best-row selection.
- [x] Normalize daily timestamp convention.
- [x] Generate cleanup report (rows in/out, drops, anomalies).
- [x] Run quality gates and archive results.

### Phase B: Chain feature readiness
- [ ] Add chain quality gate per snapshot.
- [ ] Implement IV solve and Greeks computation pass.
- [ ] Persist derived chain feature table with flags.
- [ ] Mark invalid/low-quality snapshots as non-tradable for chain-dependent logic.

### Phase C: Incremental operations
- [x] Add incremental cleanup routine for recent partitions only.
- [x] Add version metadata for curated datasets.
- [x] Add runbook command sequence for daily maintenance.

---

## Acceptance Criteria
- No duplicate `trade_date_ist` rows in curated daily candles.
- Deterministic reruns produce identical curated outputs.
- Backtest and paper use curated candles by default.
- Chain-dependent signals only activate when quality and derived features are available.
- Cleanup and feature versions are recorded per run artifact for reproducibility.
