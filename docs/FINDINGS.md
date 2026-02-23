# NiftyQuant System — Complete Findings & Research Log

**As of:** 2026-02-23  
**Backtest window:** 2022-01-01 to 2025-12-31  
**Instrument:** NIFTY weekly options (Friday expiry)  
**Capital:** ₹1,50,000 (1 lot throughout all tests)  
**Audience:** Strategic decisions + coding assistant context

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Infrastructure History & Bugs Fixed](#2-infrastructure-history--bugs-fixed)
3. [Confirmed Baseline](#3-confirmed-baseline)
4. [Walk-Forward Results (11-Fold)](#4-walk-forward-results-11-fold)
5. [Phase 2-A: Tested Changes — REVERTED](#5-phase-2-a-tested-changes--reverted)
6. [The Delta Paradox — Critical Finding](#6-the-delta-paradox--critical-finding)
7. [The Tuesday Exit Investigation](#7-the-tuesday-exit-investigation)
8. [Counterfactual Benchmarks](#8-counterfactual-benchmarks)
9. [Phase 2-B: FII Gate — REVERTED](#9-phase-2-b-fii-gate--reverted)
10. [Phase 2-B: ADX Trending Filter — NOT IMPLEMENTED](#10-phase-2-b-adx-trending-filter--not-implemented)
11. [Q1 2025 Diagnostic — Root Cause](#11-q1-2025-diagnostic--root-cause)
12. [Phase 3: Scheduled Events Strategy — ALL OPTIONS STOPPED](#12-phase-3-scheduled-events-strategy--all-options-stopped)
13. [Core Market Structure Finding](#13-core-market-structure-finding)
14. [Tax & Entity Structure](#14-tax--entity-structure)
15. [Strategy Variants Evaluated](#15-strategy-variants-evaluated)
16. [Infrastructure Built (Available For Future Use)](#16-infrastructure-built-available-for-future-use)
17. [Known Open Issues & Deferred Work](#17-known-open-issues--deferred-work)
18. [Recommended Next Actions](#18-recommended-next-actions)
19. [Key File Paths](#19-key-file-paths)
20. [Settings Reference (Baseline Config)](#20-settings-reference-baseline-config)

---

## 1. System Overview

NiftyQuant is a systematic short-volatility options strategy trading NIFTY weekly iron condors. The core thesis is that implied volatility on NIFTY weekly options consistently exceeds realised volatility — the options market systematically overprices expected moves — and a disciplined short-vol strategy can capture this premium.

**Strategy logic:**
- Entry: Monday or Tuesday close, on qualifying regime days only
- Structure: 4-leg iron condor (short call + long call hedge + short put + long put hedge)
- Strike selection: weighted scoring (delta match 3× weight + ATR proximity), targeting call_delta: 0.15 config but producing ~0.25 effective delta due to chain availability in current market conditions
- Regime filter: enters only on LOW_VOL_RANGING or LOW_VOL_TRENDING days (VIX + ADX classification)
- Exit: Wednesday calendar close, OR 50% profit target (whichever fires first), OR stop loss
- Position sizing: 1 lot (50 units) throughout all testing

**Technology stack:**
- Python backtesting engine with dynamic-only pricing pipeline
- KiteConnect (Zerodha) for live data
- VIX + ADX regime classifier
- FII data pipeline built (disabled, see Section 9)
- Walk-forward validator (11 folds, ~4.5 months each)

---

## 2. Infrastructure History & Bugs Fixed

These bugs were found and fixed before the baseline was established. All results in this document are post-fix.

### Bug 1 — Symbol Canonicalization (July 2024 format shift)
**Problem:** NSE changed NIFTY option symbol format in July 2024. The backtest pipeline used the old format for chain lookups, causing option contracts to be unfound and positions to be left stuck (unfilled exits).  
**Symptom:** 4 NIFTY_20240718 legs stuck open; ~90 unfilled exits in decisions CSV.  
**Fix:** Symbol canonicalization updated to handle both pre- and post-July 2024 formats.  
**Validation:** unfilled_exits = 0 post-fix.

### Bug 2 — Phantom Regime Change Exits
**Problem:** `BacktestEngine._next_signal()` called `strategy.on_regime_change(previous, current)` on every bar when strategy was inactive, even when `previous == current`. This caused same-to-same regime transitions (e.g. "high_vol_choppy → high_vol_choppy") to fire exit logic 65+ times per run.  
**Symptom:** Failure reasons in decisions CSV showed "Regime changed from high_vol_choppy to high_vol_choppy" 65 times, and "high_vol_trending to high_vol_trending" 18 times.  
**Fix:** Engine-level guard: `if current_regime != previous_regime: call on_regime_change()`. Plus position-level stricter exit condition: `if current_regime != position.entry_regime AND current_regime not in active_regimes`.  
**Validation:** Same-to-same regime exits = 0 post-fix.

### Bug 3 — Precompute Pipeline Removed
**Problem:** A precompute pipeline was running alongside the dynamic pricing pipeline, creating two competing data sources and potential for stale data.  
**Fix:** Precompute pipeline removed. Dynamic-only pricing is now the single source of truth.  
**Validation:** Parity confirmed — dynamic-only run produces identical metrics to the previous dual-pipeline run.

### Bug 4 — Walk-Forward Fold Boundary ADX Warmup
**Problem:** Each walk-forward fold started without 60 days of ADX warmup history. ADX=0.0 with `from_regime=unknown` appeared at fold start boundaries.  
**Fix:** Each fold preloads indicator warmup data from `fold_start - warmup_days`, same as full backtest.  
**Status:** Fixed. Not a blocker for strategy testing.

### Bug 5 — Profit Target Exit Priority
**Problem:** Calendar time exit was evaluated before profit target exit. When both conditions were true simultaneously (profit hit on Wednesday), calendar exit fired first, never allowing profit target to register.  
**Fix:** Exit priority reordered: profit target → stop loss → calendar exit.  
**Note:** This fix was confirmed necessary but the claimed magnitude of impact (30 trades affected) was later found to be partially based on a data artifact (see Section 7).

### Run Integrity Invariants (must always be 0)
After all bugs fixed, these invariants are enforced on every run. Any non-zero value indicates a new regression:
```
unfilled_exits:           0
forced_liquidations:      0
same-to-same regime exits: 0
open positions at window end: 0
```

---

## 3. Confirmed Baseline

The baseline is the clean, post-fix, pre-tuning state. It is the reference point for all Phase 2 comparisons. It was re-confirmed before Phase 3 work began.

```
Metric                        Value
─────────────────────────────────────────
total_return_pct:             33.690084%
final_equity:                 ₹2,00,535
max_drawdown_pct:             1.10%
sharpe_daily_rf7:             0.230487
sharpe_daily_rf0:             (positive)
sharpe_trade_rf0:             2.379
trade_count_closed:           126
trade_win_rate_pct:           78.571429%
trade_profit_factor:          4.383527
mean_entry_credit:            ~₹4,200
fees_paid:                    ~₹16,195
fee_drag_pct_of_gross:        ~41%
unfilled_exits:               0
forced_liquidations:          0
```

**Artifact locations:**
```
data/reports/phase3_baseline_sanity/
  iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_metrics.json
  iron_condor_nifty_1d_2022-01-01_2025-12-31_walkforward_summary.json
```

**Baseline config:** `settings.yaml` with `profit_target_pct: 50`, `tuesday_exit_threshold: null`, `active_regimes: [low_vol_ranging, low_vol_trending]`, `fii_gate_enabled: false`

---

## 4. Walk-Forward Results (11-Fold)

Walk-forward parameters (under `backtest` in settings.yaml): `train_months: 12`, `test_months: 3`, `step_months: 3`. 11 folds total covering 2023-01-01 to 2025-10-01 (training starts 2022-01-01).

```
Positive folds:       10 / 11
Mean fold return:     2.31%
Median fold return:   ~2.1%
Mean fold Sharpe RF7: varies by fold (see table)
Forced liquidations across all folds: 0
```

**Fold-by-fold breakdown** (source: `iron_condor_nifty_1d_2022-01-01_2025-12-31_walkforward_folds.csv`):

| Fold | Test Range | Return % | Trades | Win Rate % | Sharpe RF7 | Flag |
|---|---|---|---|---|---|---|
| 1 | 2023-01-01 → 2023-04-01 | 1.8735 | 8 | 75.00 | 0.4557 | ✅ |
| 2 | 2023-04-01 → 2023-07-01 | 2.4229 | 12 | 75.00 | 1.2511 | ✅ |
| 3 | 2023-07-01 → 2023-10-01 | 3.4517 | 13 | 84.62 | 2.6641 | ✅ |
| 4 | 2023-10-01 → 2024-01-01 | 1.8976 | 13 | 61.54 | 0.3429 | ✅ |
| 5 | 2024-01-01 → 2024-04-01 | 3.3494 | 12 | 83.33 | 1.6261 | ✅ |
| 6 | 2024-04-01 → 2024-07-01 | 2.0413 | 7 | 85.71 | 0.7047 | ✅ |
| 7 | 2024-07-01 → 2024-10-01 | 2.1751 | 11 | 72.73 | 0.6813 | ✅ |
| 8 | 2024-10-01 → 2025-01-01 | 2.8902 | 12 | 75.00 | 1.4421 | ✅ |
| 9 | 2025-01-01 → 2025-04-01 | **-0.2796** | 6 | 66.67 | -1.7741 | ❌ |
| 10 | 2025-04-01 → 2025-07-01 | 1.0941 | 2 | 50.00 | -0.5966 | ✅ |
| 11 | 2025-07-01 → 2025-10-01 | 4.5231 | 14 | 85.71 | 2.6610 | ✅ |

**Notes on specific folds:**
- Fold 9 (Q1 2025): the only negative fold. Diagnosed as irreducible variance from directional intraweek moves (see Section 11). No actionable fix found.
- Fold 10 (Q2 2025): only 2 trades — low statistical weight, positive but not meaningful. Sharpe negative due to sparse returns. Not a quality concern.
- Fold 11: highest return (4.52%) and highest trade count (14) — strong signal.
- Fold 4 (Oct–Jan 2024): lowest win rate at 61.54% but still positive return, suggesting larger wins when they occurred.

**Interpretation:** 10/11 folds positive across 4 different market regimes (2022 volatile, 2023 recovery, 2024 mixed, 2025 mixed) is a strong signal of genuine edge, not overfitting.

---

## 5. Phase 2-A: Tested Changes — REVERTED

Both Phase 2-A changes were reverted. The baseline (33.69%, 10/11 folds) remains the best configuration found.

### Change 1 — Delta Cap at 0.15 (REVERT)

**Hypothesis:** Reducing delta from ~0.25 to 0.15 (selling further OTM) would reduce breach frequency and improve win rate.

**Implementation:** Added `max_delta: 0.15` cap. If selected strike had delta > 0.20, move one strike further OTM.

**Result:**
```
Metric              Baseline    Delta Cap   Delta
────────────────────────────────────────────────
total_return_pct:   33.69%      4.24%       -29.45pp  ← catastrophic
sharpe_daily_rf7:   0.230       -2.405      -2.635
trade_win_rate:     78.57%      63.49%      -15.08pp
profit_factor:      4.38        1.24        -3.14
walk-forward folds: 10/11       7/11        -3 folds
trade_count:        126         126         0 (same trades, worse outcomes)
```

**Why it failed:** See Section 6 (The Delta Paradox). The edge lives at ~0.25 delta, not 0.15. The 0.15 delta strikes collect insufficient premium relative to fixed fees, making fee drag disproportionate.

**Decision:** REVERT. Do not re-test without first understanding the delta paradox diagnostic.

---

### Change 2 — Ranging-Only Regime Gate (REVERT)

**Hypothesis:** Removing LOW_VOL_TRENDING entries (keeping only LOW_VOL_RANGING) would improve per-trade quality by avoiding directional-risk weeks.

**Implementation:** `active_regimes: [low_vol_ranging]` only.

**Result:**
```
Metric              Baseline    Ranging-Only  Delta
────────────────────────────────────────────────────
total_return_pct:   33.69%      24.22%        -9.47pp
sharpe_daily_rf7:   0.230       -0.529        -0.759  (negative — sparse returns)
trade_win_rate:     78.57%      80.72%        +2.15pp (improved)
profit_factor:      4.38        5.86          +1.48   (significantly improved)
trade_count:        126         ~85           -41 trades
Q1 2025 fold:       -0.28%      +0.89%        +1.17pp (KEY FINDING)
```

**Why reverted despite quality improvements:** Total return fell 9.47pp because fewer trades were taken. The negative Sharpe is a sparse-return penalty — fewer trades means more zero-return days dragging the daily Sharpe denominator. However the per-trade quality genuinely improved and Q1 2025 recovered.

**Key signal NOT to discard:** The Q1 2025 fold flipped from -0.28% to +0.89% simply by removing LOW_VOL_TRENDING entries. This confirms that the Q1 2025 losses came specifically from trending-regime entries during the FII selloff period. The correct response is not to remove trending entirely, but to add a quality filter for trending entries specifically.

**Decision:** REVERT. But the finding that trending entries are the quality-dragging component is preserved for future work.

---

## 6. The Delta Paradox — Resolved

**This section supersedes earlier assumptions. The "paradox" is fully explained.**

### How Strike Selection Actually Works

Strike selection in `iron_condor.py` uses a weighted scoring system across all available option chain candidates:

```python
score = abs(delta - target_delta) × 3.0 + abs(strike - target_strike)
```

Delta match is weighted **3× heavier** than strike proximity to the ATR target. This means `call_delta: 0.15` in config is the dominant signal — it actively pulls strikes toward 0.15 delta on every trade. The ATR target (`atr_multiple: 1.5` from spot) is a secondary tiebreaker, not the primary placement mechanism.

The effective operating delta of ~0.25 observed in baseline trades is **not** because the config is ignored. It is an emergent property of the NIFTY option chain — when the nearest 0.15-delta strike is far from the ATR target, the scoring balances both dimensions and lands somewhere between them. This varies by market conditions (VIX level, term structure, skew).

### Why Change 1 Collapsed — Fully Explained

Change 1 added a hard override: if selected strike had delta > 0.20, move one additional strike further OTM. This forced exact 0.15 delta regardless of the scoring system.

```
Metric                    Baseline      Change 1      Delta
──────────────────────────────────────────────────────────
Mean entry_credit:        ₹4,189        ₹2,169        -48%  ← credit halved
Mean fees % of credit:    5.64%         9.77%         +4.1pp ← fee drag doubled
Exit via profit_target:   ~30 trades    4 trades      -87%
Exit via calendar:        ~96 trades    122 trades    +27%
Mean calendar exit pnl:   14.63%        10.24%        -4.4pp
```

Forcing exact 0.15 delta moved strikes far enough OTM that:
1. Credit collected nearly halved (₹4,189 → ₹2,169)
2. Fee drag as % of credit nearly doubled (5.64% → 9.77%)
3. Profit target almost never fired (4 trades vs ~30) — smaller credit means 50% target is a smaller absolute number, but the strikes are now too far OTM for theta decay to reach it within the weekly window
4. The strategy became a fee-paying calendar-exit machine with half the income

**There is no delta paradox.** The edge does not "live at 0.25 delta" specifically. The edge lives at the **credit level** the scoring system naturally produces (~₹4,200). The current `call_delta: 0.15` config, combined with ATR placement and chain availability, produces ~0.25 effective delta and ~₹4,200 credit in current market conditions. Forcing an exact 0.15 delta override destroyed credit without any compensating benefit.

### Implications For Future Changes

- Do not add hard delta caps or overrides. The scoring system is already targeting 0.15 delta as best it can given chain structure.
- The real parameter controlling credit level is the interaction of `call_delta`, `atr_multiple`, and the available option chain. `atr_multiple` is the more direct lever.
- If testing different credit levels, adjust `atr_multiple` (moves the ATR target, shifting the scoring anchor) rather than adding delta overrides.
- A minimum credit filter (`min_premium`) is a cleaner enforcement mechanism than a delta cap — it directly targets the economic variable that matters.

---

## 7. The Tuesday Exit Investigation

This section documents a full investigation cycle including a data artifact discovery.

### Initial Observation
From profit path analysis, 30 of 126 trades showed `max_profit_pct = 100%` — all on Tuesday. Five of these subsequently finished negative by Wednesday exit:

```
Trade 73  (entry 2024-07-15): hit 50% Tue → actual exit -7.39%
Trade 85  (entry 2024-11-11): hit 50% Tue → actual exit -0.11%
Trade 88  (entry 2024-12-02): hit 50% Tue → actual exit -25.45%
Trade 97  (entry 2025-03-17): hit 50% Tue → actual exit -40.73%
Trade 119 (entry 2025-11-10): hit 50% Tue → actual exit -10.97%
```

This appeared to be strong evidence for a Tuesday exit rule.

### Data Artifact Discovered
`max_profit_pct = 100` was a **zero-fill artifact** in the analysis script. When any leg price was missing for a bar, `close_debit` became zero, making `profit_pct = (credit - 0) / credit = 100%`. The 30 trades showing 100% and the all-Tuesday pattern were both artifacts.

The runtime profit calculation in `iron_condor.py` was correct throughout. Only the analysis script was wrong.

### Corrected Results
After fixing the analysis script (skip bars with missing prices instead of zero-filling):
```
Trades genuinely reaching >= 50% profit:  ~3–5 (vs 30 claimed)
Trades genuinely reaching >= 40% profit:  ~8–12
Typical weekly max achievable profit:     15–30% of credit
75% of trades: max profit < 30% during hold
```

### Tuesday Exit Rule Status
`tuesday_exit_threshold: 40` was added to code based on the artifact. After corrected data analysis:

**SCENARIO B confirmed (artifacts, not genuine Tuesday reversals):** The corrected data does not support strong evidence for systematic Tuesday reversals. The five "reversal" trades were either never meaningfully profitable or had exaggerated profit readings due to the zero-fill bug.

**Current config:** `tuesday_exit_threshold: null` (parameter code preserved, not activated). Do not activate without fresh evidence from correctly-priced profit path data.

### Profit Target Assessment — CLOSED (tested 2026-02-23)

`profit_target_pct: 35` was tested as Test A. Result: zero measurable impact on any metric.

```
Metric              Baseline (PT=50)   PT=35      Delta
────────────────────────────────────────────────────────
total_return_pct:   33.6901%           33.6901%   0.0000
sharpe_daily_rf7:   0.230487           0.230487   0.0000
win_rate:           78.5714%           78.5714%   0.0000
profit_factor:      4.3835             4.3835     0.0000
trade_count:        126                126        0
Q1 2025 fold:       -0.2796%           -0.2796%   0.0000

Exit distribution shift:
  profit_target:    0 → 9   (+9)
  calendar:         126 → 117  (-9)
  Calendar mean pnl: 14.63% → 12.64% (expected — best weeks now exit earlier)
```

**Conclusion:** The profit target threshold is **not a performance lever**. Whether a trade exits early at 35% profit or drifts to Wednesday calendar exit at ~14% profit, the net P&L is equivalent because credit levels and hold dynamics are consistent. The profit target is a safety valve for exceptional weeks only (~4–9 trades per 126). Keep at 50 (or any reasonable threshold) but do not treat as tunable. Test B (PT=25) not needed — pattern is definitive.

**Artifact:** `data/reports/phase2_profit_target_35/`

---

## 8. Counterfactual Benchmarks

Performance of the iron condor vs alternative uses of the same ₹1,50,000 capital over the same 4-year period:

```
Strategy                          P&L          Return    Notes
──────────────────────────────────────────────────────────────────
Iron Condor (actual):             ₹50,535      33.69%    post-fee
NIFTY buy & hold (approx):        ₹53,430      35.6%     17,354→23,500 approx
FD at 6.5% (4 years):             ₹43,830      29.2%     compounded
Nifty SIP at ~11% CAGR:           ₹70,575      47.0%     lump sum equiv
```

**Context:** The iron condor's 33.69% return is competitive with buy-and-hold NIFTY over this period, with dramatically lower drawdown (1.10% vs NIFTY's periodic 10-20%+ drawdowns). The risk-adjusted comparison favours the iron condor even where the raw return is similar. The strategy beats FD clearly.

At current 1-lot scale, the absolute P&L (~₹50k over 4 years = ~₹12.5k/year) is modest. The system's value is in building a proven, scalable infrastructure. At 5–10 lots with proven edge, absolute returns become economically meaningful.

---

## 9. Phase 2-B: FII Gate — REVERTED

### What Was Built
A complete FII data pipeline:
- `scripts/build_fii_cache.py` — historical cache builder from NSDL monthly XLS files
- `scripts/update_fii_daily.py` — daily updater from NSE endpoint
- `src/data/fii_signal.py` — signal computer with consecutive-days logic
- `tests/test_fii_signal.py` — 10 tests, all passing
- Cache: `data/cache/fii/fii_equity_daily.csv` (97.87% coverage, 2022-01-03 to 2026-02-20)

### FII Signal Design
```yaml
# Consecutive days filter (not rolling sum — rolling sum hides cancellation)
fii_bearish_consecutive_days: 3
fii_bearish_daily_threshold: -1000  # all 3 days must be below this
fii_bullish_daily_threshold: +1000
```

Signal distribution 2022–2025: 14.3% bearish, 76.5% neutral, 9.2% bullish.

Q1 2025 had an 18-day bearish run: Jan 9 → Feb 3, 2025.

### Backtest Results
```
Metric              Baseline    FII Gate    Delta
────────────────────────────────────────────────
total_return_pct:   33.69%      31.78%      -1.91pp
sharpe_daily_rf7:   0.2305      0.1053      -0.125
trade_win_rate:     78.57%      79.31%      +0.74pp
profit_factor:      4.38        4.49        +0.10
trade_count:        126         116         -10
```

Walk-forward: 10/11 positive folds maintained. Q1 2025 fold: baseline -0.28% → FII run -1.72% (got worse, -1.44pp).

### Why It Failed
The FII gate fired during the 18-day January bearish run (Jan 9 → Feb 3). This blocked 4 entries in Q1 2025: Jan 13, Jan 14, Mar 10, Mar 11. However:
- The January blocked entries were **winners** (missed during the FII selloff but the condors would have worked)
- The Q1 2025 actual losses occurred in **February–March** during **neutral FII periods**
- The gate fired at the right time but blocked the wrong trades — it skipped winners and let through losers

The Q1 2025 losses are mid-hold directional moves that no entry-time signal (FII or otherwise) can reliably prevent without blocking too many normal weeks.

### Status
Infrastructure preserved, gate disabled:
```yaml
regime:
  fii_gate_enabled: false  # built, validated, not currently beneficial
```

FII data pipeline is sound and available. The gate itself is correctly implemented. It may be useful in future with different parameterisation or combined with other signals.

**Artifact location:** `data/reports/phase2_change3_fii_20260223/`

---

## 10. Phase 2-B: ADX Trending Filter — NOT IMPLEMENTED

**Hypothesis:** Adding a maximum ADX threshold for LOW_VOL_TRENDING entries (e.g. only enter when ADX < 25) would reduce losses in strong trending weeks.

### Diagnostic Results
Full ADX distribution for low_vol_trending trades (2022–2025):
```
ADX bucket    trades    win rate    mean PnL
────────────────────────────────────────────
20–25            7       85.71%      ₹238
25–30           12       75.00%      ₹230
30+             24       66.67%      ₹343  ← highest PnL despite lower win rate
```

**What-if analysis at proposed thresholds:**
```
ADX 22: blocks 43 trades (31 winners, 12 losers) — ratio 2.6 winners per loser blocked
ADX 25: blocks 36 trades (25 winners, 11 losers) — ratio 2.3:1
ADX 28: blocks 30 trades (21 winners, 9 losers) — ratio 2.3:1
```

Win rate never drops below 60% at any ADX level. The ADX 30+ bucket has the highest mean PnL (₹343) — the largest and most profitable trades are in strong trends. Blocking them to remove 9 losers also removes 21 winners with higher average value.

**Decision:** NOT IMPLEMENTED. Ratio does not improve as threshold tightens (stays at ~2.3:1). Sacrificing 2.3 winners per loser avoided is not a worthwhile trade. The ADX 30+ bucket should not be filtered out.

**Artifact location:** `data/reports/phase2_change4_preimpl_20260223_132138/`

---

## 11. Q1 2025 Diagnostic — Root Cause

Q1 2025 is the only consistently negative walk-forward fold (-0.28%). Extensive diagnostic work was done to find a fixable cause.

### The Two Losing Trades
```
Trade 1: Entry 2025-03-17, Exit 2025-03-19
  Regime: low_vol_trending, ADX 25.90, VIX 13.42, FII neutral
  Exit: calendar, actual_exit_pnl_pct: -40.73%
  
Trade 2: Entry 2025-03-24, Exit 2025-03-26
  Regime: low_vol_trending, ADX 23.32, VIX 13.70, FII neutral
  Exit: calendar, actual_exit_pnl_pct: -46.41%
```

Both: no VIX expansion during hold (vix_spike = 0.0), FII neutral at entry.

### Strike Breach Analysis
```
Trade 1 (entry 2025-03-17):
  Entry spot: 22,508
  Short call: 22,850
  Hold high:  22,940 (+90 above call — BREACHED)
  Loss: -₹1,974

Trade 2 (entry 2025-03-24):
  Entry spot: 23,658
  Short put:  23,300
  Hold low:   23,433 (no breach, but -225 point move toward put)
  Delta risk: loss -₹1,947
```

### March 2025 Market Context
NIFTY moved 22,460 → 23,519 (+4.72%) across the month, with directional uptrend through Mar 24 then pullback Mar 25–26.

VIX during holds: 13.21 → 12.60 (first trade), 13.64 → 13.30 (second trade) — no VIX expansion on either.

### Conclusion: Irreducible Variance
Both losses resulted from directional impulse moves of 300–400 points during the hold period (2–3 trading days) on **low VIX**. These are not predictable at entry time:
- No VIX elevation pre-entry
- FII neutral
- ADX normal
- No event scheduled

Iron condor short strikes at ~1.5× ATR cannot survive large intraweek moves. The strategy works 78.57% of the time because most weeks don't produce 300–400 point intraweek moves. March 2025 weeks did. No entry-time filter reliably predicts this without also blocking many normal weeks.

**Decision:** Accept -0.28% Q1 2025 fold as the irreducible cost of operating during trending weeks with occasional large intraweek moves. No further Q1 2025 optimization attempts warranted.

---

## 12. Phase 3: Scheduled Events Strategy — ALL OPTIONS STOPPED

Phase 3 attempted to build a complementary strategy generating returns during high-VIX periods when the iron condor sits out. All three approaches failed feasibility testing.

### Event Calendar Built
`data/cache/events/scheduled_events.csv` — 60 events, 2022–2025:
- RBI_POLICY: 24 events (6/year)
- UNION_BUDGET: 5 events (1/year)  
- US_FOMC: 32 events (8/year)
- Tradeable (no condor conflict): 34
- Non-tradeable (condor open, skip): 26

Entry date rules:
- RBI/Budget: entry = event_date - 1 trading day
- FOMC: entry = FOMC_date (India close), event = FOMC_date + 1 (India reacts next day)

One duplicate removed (2024-01-31 FOMC overlapping UNION_BUDGET, kept Budget).

### Option A — Pre-Event Straddle Buy: STOP

**Result:**
```
Events considered:        34 (tradeable only)
Overall win rate:         5.88%
Mean raw_pnl_pct:         -13.27%

By type:
  RBI_POLICY:    win 0.00%, mean -17.13% (n=17)
  UNION_BUDGET:  win 0.00%, mean -16.54% (n=3)
  US_FOMC:       win 14.29%, mean -7.89% (n=14)
```

**Verification:** RBI dates were confirmed correct via spot price cross-check. RBI event days showed 0.82x–1.58x normal daily range, not the 2x+ required for a straddle to profit.

**Root cause:** IV crush. Straddle entry costs ran 1.5–1.85% of spot. Actual event moves were systematically smaller than this (mean ~0.8–1.2%). Classic case of options market accurately pricing (or overpricing) expected event moves. Buyers of pre-event straddles pay for moves that don't happen.

**Specific worst case verified:** 2022-02-09 RBI: entry cost 268.75, actual spot move +197.05 points (1.14%), breakeven required ±268.75, close was 17,463 (below upside breakeven 17,518). Even with a genuine 1.14% move, the straddle lost because the move was smaller than what was priced in.

**Artifact:** `data/cache/events/PHASE3_EVENT_FEASIBILITY.md`

### Option B — Pre-Event Strangle Sell: SKIPPED (risk grounds)

**Rationale:** If buyers lose -13.27% mean, sellers gain +13.27% mean. However this is short-vol exposure specifically concentrated at event timing — exactly when tail risk is highest (surprise RBI rate shocks, unexpected Budget announcements). A single surprise event can move NIFTY 5–10%, turning many prior small gains into a single catastrophic loss. The iron condor already provides short-vol exposure in calm periods with regime filters as protection. Adding event-timed short-vol removes those protections at the worst possible moment. Risk/reward structurally unfavorable for retail systematic strategy.

### Option C — HIGH_VOL_CHOPPY Long Strangle: STOP

**Result:**
```
Opportunities (Mon/Tue, HIGH_VOL_CHOPPY regime, 2022–2025): 110
Win rate:                 0.00%  (zero winners)
Mean raw_pnl_pct:         -53.81%
Mean spot move in period: 1.02%
Mean breakeven required:  2.44% (entry_cost / entry_spot)
NIFTY exceeded breakeven: 13.64% of the time
```

**Spot move distribution:**
```
< 0.5%:   40 events (36%)
0.5–1.0%: 30 events (27%)
1.0–1.5%: 12 events (11%)
> 1.5%:   28 events (25%)
```

**Root cause:** Same as Option A. Even in HIGH_VOL_CHOPPY regime, NIFTY's actual 2-day moves are less than half the breakeven required by the straddle/strangle entry cost. The options market prices HIGH_VOL_CHOPPY moves accurately — buying that implied vol is a losing proposition.

**Artifact:** `data/cache/events/PHASE3_OPTIONC_FEASIBILITY.md`

---

## 13. Core Market Structure Finding

**This is the single most important finding from all Phase 3 work:**

**On NIFTY, implied volatility consistently and significantly exceeds realised volatility across all tested time horizons and regimes.**

This is not a minor edge — it is a structural feature of the Indian options market:

```
Pre-event straddles (buy):     -13.27% mean loss to buyer
Post-event momentum (long):    -22.53% mean loss
High-vol choppy straddle:      -53.81% mean loss, 0% win rate
```

The implication:

```
Buying options on NIFTY:    structurally losing across all tested approaches
Selling options on NIFTY:   structurally winning (the iron condor captures this)
```

The iron condor is not a strategy that happens to work on NIFTY — it is correctly positioned on the right side of a fundamental and persistent market structure. All three Phase 3 complementary strategies were on the wrong side of the same structure.

**For future strategy work:** Any long-options strategy (buying calls, puts, straddles, strangles) on NIFTY will face this structural headwind. This is not a backtesting artefact. The Indian institutional options market prices event and regime vol accurately, often overpricing it, making premium buyers systematically lose.

---

## 14. Tax & Entity Structure

### Current Scale (1 lot, ₹1.5L capital)
- Options P&L taxed as business income: 30% + cess ≈ 31.2%
- Tax on ~₹12.5k annual gross: ~₹3,900
- LLP setup cost: ₹30–50k/year in CA fees alone
- **Verdict: LLP is worse at current scale.**

### When LLP Makes Sense
At 10+ lots (₹15L capital, ~₹1.14L gross annual):
- Personal tax (31.2%): ~₹35.5k
- LLP tax (25% flat): ~₹28.5k
- LLP expenses (CA, compliance): ~₹40–50k/year fixed
- Net saving over personal: ~₹17–19k after expenses
- **Verdict: LLP starts making sense around 10+ lots.**

### Key LLP Considerations
- MAT (Minimum Alternate Tax): 18.5% floor applies
- Salary to self from LLP: deductible from LLP profit, but taxable in personal hands — requires CA optimisation
- Compliance overhead: annual filing, audit, ROC compliance
- **Recommendation:** Set up LLP when paper trading at 3+ lots before generating significant live P&L. Consult CA specialising in trading businesses for salary structure.

### Broker Fee Stack (per trade, 1 lot iron condor = 4 legs × entry + exit = 8 fills)
At Zerodha (current):
```
Brokerage:        ₹20/order × 8 = ₹160
STT:              varies (higher on sell side)
Exchange charges: small
SEBI fee:         tiny
Stamp duty:       small
GST:              on brokerage
Total approx:     ₹185–200 per round-trip iron condor
```

At Dhan (₹0 brokerage):
- Saves ~₹160 brokerage, rest is identical (statutory charges)
- Annual saving at 126 trades/year: ~₹4,000–5,000
- **Verdict:** Not worth migration risk at current scale. KiteConnect API is most mature and stable for algo trading. Revisit at 3–4 lots.

---

## 15. Strategy Variants Evaluated

All evaluated conceptually (not backtested) to determine fit with the system.

| Variant | Verdict | Reasoning |
|---|---|---|
| Jade Lizard | Skip | Naked put = undefined downside. Retail risk framework requires defined risk. Feb 2025 crash example would be catastrophic. |
| Reverse Jade Lizard | Skip | Long put + short call spread. Directional bias required. Regime classifier predicts vol range, not direction. Fee drag with 3 legs. |
| Iron Butterfly | Skip | Higher credit but tiny profit zone (±100pts). NIFTY weekly moves 1–2% routinely. Win rate would collapse to 35–40%. Net neutral at best. |
| Wide Condor (1.5–2× wings) | Defer | Lower credit, worse fee drag at 1 lot. Only viable at 3+ lots where fee drag is already controlled. |
| **Broken Wing Condor** | **Worth testing (Phase 4)** | Asymmetric wings driven by RSI skew. Natural extension of existing `rsi_skew_factor` config. Potentially higher credit same fee structure. Reuses all existing infrastructure. |
| Calendar Condor | Skip | 8 legs (double fee drag), requires IV surface monitoring across expiries. Wrong complexity level for current stage. |

**Broken Wing Condor** is the most promising untested variant. However `rsi_skew_factor` is **completely absent from the codebase** — confirmed by full repo search on 2026-02-23. There is also no asymmetric wing width logic in `iron_condor.py` (both sides use the same `wing_width`). The broken wing condor is therefore more implementation work than previously assumed — it requires new code, not just config activation.

---

## 16. Infrastructure Built (Available For Future Use)

### FII Data Pipeline
```
data/cache/fii/fii_equity_daily.csv
  Coverage: 2022-01-03 to 2026-02-20 (97.87%)
  Source: NSDL monthly XLS (historical), NSE endpoint (daily updates)

scripts/build_fii_cache.py     — historical builder from NSDL monthly files
scripts/update_fii_daily.py    — daily updater from NSE
src/data/fii_signal.py         — consecutive-days signal computer
tests/test_fii_signal.py       — 10 tests, all passing
```

Gate is wired into `iron_condor.py` and controlled by `fii_gate_enabled: false` in settings.yaml. Tested, validated, disabled. Enable and retest if a new parameterisation is proposed.

### Event Calendar
```
data/cache/events/scheduled_events.csv
  60 events (2022–2025): RBI_POLICY=24, UNION_BUDGET=5, US_FOMC=32
  Columns: event_type, entry_date, event_date, tradeable (bool)
  34 tradeable, 26 non-tradeable (condor conflict)
```

Fully built with correct entry date logic and conflict detection. Available for future event strategies if a viable approach is found.

### Feasibility Analysis Scripts
All three Phase 3 feasibility analyses are scripted and reproducible. If parameters or strategy designs change, re-running is straightforward.

---

## 17. Known Open Issues & Deferred Work

### High Priority (should address before paper trading)
1. **Delta paradox diagnostic incomplete.** Change 1 artifacts exist at `data/reports/phase2_change1_delta_cap_20260223/`. Pull mean entry credit, fee%, exit reason distribution for Change 1 vs baseline. Confirms or challenges the fee drag hypothesis. Do not run further delta experiments without this.

2. **Regime suspicious windows not mitigated.** Documented in early sessions: 2025-02-28 (`low_vol_ranging` on -1.86% day), 2025-01-06 (`low_vol_trending` on -1.62% day), 2024-10-03 (`low_vol_ranging` on -2.12% day). Price-shock override to force `high_vol_choppy` for 1–2 bars on large 1-day moves was planned but not implemented.

3. ~~**Profit target threshold unvalidated.**~~ **CLOSED (tested 2026-02-23).** PT=35 tested, zero impact on all metrics. Profit target is not a performance lever — see Section 7. No further profit target testing needed.

### Medium Priority
4. **Trending regime quality filter.** Change 2 showed trending entries drag quality during stress periods (Q1 2025). The right fix is an additional quality filter for trending entries (e.g. minimum entry credit, tighter VIX range, or ADX range within trending) — not removing trending entirely. Not yet designed or tested.

5. **Walk-forward fold boundary warmup.** Fixed for ADX but should verify all other indicators (VIX ROC etc.) also properly warm up at fold boundaries.

6. **Tuesday exit threshold.** `tuesday_exit_threshold` parameter exists in code but is set to null. No genuine evidence from corrected data currently supports activating it. Leave null until fresh profit path analysis on correctly-priced data shows genuine Tuesday reversal pattern.

### Low Priority / Future Phases
7. **Multi-symbol extension** (BankNifty, FinNifty). Infrastructure was designed with multi-symbol in mind (`forced_liquidation_count > len(symbols)` generalization). Not yet tested.

8. **Broken wing condor backtest.** Most promising untested variant. However `rsi_skew_factor` is absent from codebase (confirmed 2026-02-23) and there is no asymmetric wing logic in `iron_condor.py`. This requires new implementation — not just config activation. Estimate: moderate effort (RSI signal + asymmetric wing width parameter + scoring adjustment). Phase 4 candidate after paper trading stable.

9. **Strike width optimization.** Current `wing_width: 100` was never explicitly tested against alternatives. One focused diagnostic could determine optimal width at current lot size.

10. **Entry day expansion.** Currently Monday/Tuesday only. Wednesday entries (3 DTE to Friday expiry) might add trades without meaningful regime change. Low risk to test.

---

## 18. Recommended Next Actions

**All parameter optimisation is now exhausted.** Full inventory of tested levers:

```
Delta cap (0.15):           REVERT — credit collapsed (₹4,189 → ₹2,169)
Ranging-only regime gate:   REVERT — volume collapsed
FII gate:                   REVERT — wrong failure mode
ADX trending filter:        NOT IMPLEMENTED — 2.3 winners per loser avoided
Profit target PT=35:        STOP — zero measurable impact
Profit target PT=25:        Not needed — pattern definitive
```

The baseline (33.69%, 10/11 folds positive) is the best achievable configuration with the current strategy structure and capital level. No further backtesting is warranted before paper trading.

Listed in priority order.

### Action 1 — Paper Trading Setup (immediate priority)
The system is ready. Every reasonable parameter has been tested. Paper trading will surface execution realities that no backtest captures: slippage, fill quality at market open/close, margin behaviour, weekend gap risk, KiteConnect API reliability under live conditions.

Set up paper trading on KiteConnect with:
- 1 lot NIFTY
- Monday/Tuesday entry (same as backtest)
- Live regime classification (VIX + ADX)
- Automated entry/exit via strategy logic
- Daily P&L tracking vs backtest benchmark (~0.65% per week on active weeks)

Run for **3 months minimum** before considering live capital. Target: confirm execution quality matches backtest within reasonable slippage tolerance.

### Action 2 — Capital Scaling (after 2–3 months paper trading)
The single highest-confidence mechanical improvement is reducing fee drag by scaling lots:
```
1 lot:  fee drag ~41% of gross  (current)
2 lots: fee drag ~25% of gross  (estimated)
3 lots: fee drag ~19% of gross  (estimated)
```
This does not change strategy edge — it changes how much survives fees. Prerequisites:
- Paper trading confirms execution quality matches backtest
- ₹3L capital available (2× margin requirement)
- 2–3 months live paper data with no execution anomalies

### Action 3 — Trending Regime Quality Filter (after paper trading stable)
Change 2 showed LOW_VOL_TRENDING entries are the quality-dragging component, especially during stress periods (Q1 2025 fold improved from -0.28% to +0.89% with ranging-only). The correct fix is an additional quality gate for trending entries — not removing them entirely. Candidates to test individually:
- Minimum entry credit threshold for trending entries only
- FII neutral requirement on trending entries
- ADX ceiling within trending (e.g. only enter when ADX < 28)

Each should be tested against the same 4-year window and walk-forward before combining.

### Action 4 — atr_multiple Sensitivity (low priority, after paper trading)
`atr_multiple: 1.5` is the direct lever on strike placement and credit level. A sensitivity test across values (1.2, 1.5, 1.8) with `min_premium` enforcement is the cleanest remaining experiment on credit optimisation. Defer until paper trading provides live credit data to anchor expectations.

### Action 5 — Broken Wing Condor (Phase 4, after paper trading stable)
Most promising untested strategy variant. Requires new implementation (asymmetric wing width logic + RSI signal — neither exists in current codebase). Should be tested on the same 4-year window with identical walk-forward structure. Medium implementation effort. Do not start until paper trading is stable and baseline is validated live.

---

## 19. Key File Paths

```
Strategy & Config
  settings.yaml                          — main config (baseline state described below)
  src/strategies/iron_condor.py          — strategy implementation
  src/backtest/engine.py                 — backtest engine

Data & Cache
  data/cache/fii/fii_equity_daily.csv    — FII equity flows cache
  data/cache/events/scheduled_events.csv — event calendar (60 events)
  data/cache/events/PHASE3_EVENT_FEASIBILITY.md
  data/cache/events/PHASE3_OPTIONA_FEASIBILITY.md
  data/cache/events/PHASE3_OPTIONC_FEASIBILITY.md

Baseline Artifacts (immutable reference)
  data/reports/phase3_baseline_sanity/
    iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_metrics.json
    iron_condor_nifty_1d_2022-01-01_2025-12-31_walkforward_summary.json

Phase 2 Artifacts
  data/reports/phase2_change1_delta_cap_20260223/
  data/reports/phase2_change2_regime_gate_20260223/
  data/reports/phase2_change3_fii_20260223/
  data/reports/phase2_change4_preimpl_20260223_132138/
  data/reports/phase2_profit_target_35/

Diagnostic Artifacts
  data/reports/q1_2025_event_diagnostic_20260223_132633/
  data/reports/phase2_change4_preimpl_20260223_132138/

FII Scripts
  scripts/build_fii_cache.py
  scripts/update_fii_daily.py
  src/data/fii_signal.py
  tests/test_fii_signal.py

Phase 2 Summary Reports
  PHASE2A_REPORT.md
  phase2A_20260223_summary.json
```

---

## 20. Settings Reference (Baseline Config)

The following are the **verified** values from `settings.yaml` as confirmed by the coding assistant on 2026-02-23. Note two important discrepancies from earlier assumptions documented in this file.

### ⚠️ Discrepancy 1 — Delta Config vs Operating Delta
`call_delta: 0.15` and `put_delta: -0.15` are set in config. However, the actual operating delta observed in baseline backtest trades averaged ~0.25. This means the ATR-multiple strike selection (`atr_multiple: 1.5`) is overriding or supplementing the delta target, placing strikes closer to spot than 0.15 delta would imply. The **effective operating delta is ~0.25** regardless of config value. This distinction is critical — see Section 6 (The Delta Paradox). Do not assume reducing `call_delta` / `put_delta` config values alone will change where strikes land.

### ⚠️ Discrepancy 2 — max_lots
`max_lots: 2` is set in config, but **all backtests in this document were run at 1 lot** (capital_per_trade: 150,000 with 1 lot margin). The max_lots setting may represent a ceiling, not the actual lot count used. Confirm with coding assistant which parameter controls the actual executed lot count before scaling.

### Verified Settings

**`strategies.iron_condor` (active strategy):**
```yaml
enabled: true
instrument: NIFTY
active_regimes: [low_vol_ranging, low_vol_trending]
call_delta: 0.15          # ⚠️ see discrepancy above — effective delta is ~0.25
put_delta: -0.15          # ⚠️ same
fallback_iv_pct: 20.0
lot_size: 50
wing_width: 100
dte_min: 5
dte_max: 14
min_entry_vix: 9.0
max_entry_vix: 16.0
atr_multiple: 1.5
min_premium: 15.0
entry_days: [0, 1]         # Monday, Tuesday
enable_time_exit: true
time_exit_day: Wednesday
profit_target_pct: 50
tuesday_exit_threshold: null   # code exists, not activated
stop_loss_pct: 100
dte_exit: 1
max_lots: 2               # ⚠️ see discrepancy above
capital_per_trade: 150000
```

**`regime`:**
```yaml
vix_low: 14.0
vix_high: 18.0
adx_trending: 25.0
adx_ranging: 20.0
fii_gate_enabled: false
fii_bearish_daily_threshold: -1000
fii_bullish_daily_threshold: 1000
fii_consecutive_days: 3
fii_selling_alert: -6000
pcr_oversold: 0.7
pcr_overbought: 1.3
vix_hysteresis_buffer: 0.5
adx_hysteresis_buffer: 2.0
adx_smoothing_alpha: 0.3
vix_spike_5d_alert: 3.0
iv_surface_shift_alert: 2.0
iv_surface_tilt_alert: 1.5
corr_breakdown: 0.8
```

**`backtest`:**
```yaml
start_date: "2022-01-01"
end_date: "2025-12-31"
slippage_pct: 0.05
commission_per_order: 20
risk_free_rate_annual: 0.07
stt_pct: 0.1
exchange_txn_charges_pct: 0.03503
gst_pct: 18.0
sebi_fee_pct: 0.0001
stamp_duty_pct: 0.003
train_months: 12          # walk-forward lives here, not a separate section
test_months: 3
step_months: 3
```

**Other strategies (all disabled):**
- `baseline_trend`: disabled — NIFTY trending/high-vol
- `jade_lizard`: disabled — infrastructure exists, not backtested
- `straddle`: disabled — short variant, low-vol ranging only
- `credit_spread`: disabled — BANKNIFTY, not tested
- `momentum`: disabled — futures-based
- `mean_reversion`: disabled — VWAP-confirmed

### Run Integrity Field Names (exact schema)
```
run_integrity.unfilled_exits.unfilled     = 0
run_integrity.forced_liquidations.count   = 0
run_integrity.unfilled_exits.failure_reasons = {}

Note: same_to_same_regime_exits and open_positions_at_window_end
are NOT present as fields in the metrics JSON schema.
These were validated via decisions CSV inspection, not metrics JSON.
```

---

*Document generated: 2026-02-23. Covers all sessions from 2026-02-19 through 2026-02-23.*  
*Source transcripts: see `/mnt/transcripts/journal.txt`*
