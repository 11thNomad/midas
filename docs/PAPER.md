# Paper Trading Hardening Plan

Updated: 2026-02-24

## Purpose

This document defines the paper-trading hardening phase for NiftyQuant.

Scope:
- Use live market data from Kite.
- Intercept execution in paper mode (no real orders).
- Model slippage and full options fee stack conservatively.
- Enforce deterministic exits and runtime safety controls.
- Persist data, actions, and outcomes for audit and analysis.

## Review Delta (2026-02-24)

The latest code review identified additional runtime-critical gaps that are now
explicitly in scope for this phase:
- Kite access-token lifecycle/rotation for daily expiry.
- Exit-path wiring in paper loop (`get_exit_conditions`) and strategy state sync (`on_fill`).
- Expiry auto-settlement handling for open options.
- Restart-safe position/accounting recovery (moved to highest priority group).
- Weekend and NSE-holiday market-day guard in the runtime loop.

## Current Plumbing Check

Status legend:
- `Ready`: implemented and usable now.
- `Partial`: implemented but with important gaps.
- `Missing`: not implemented.

| Requirement | Status | Notes |
|---|---|---|
| `MockBroker` class | Missing | No dedicated class exists. Closest component is `PaperExecutionEngine`. |
| Live chain fetch via real KiteConnect | Ready | `KiteFeed.get_option_chain()` is wired in `scripts/run_paper.py`. |
| Intercepted order execution | Ready | Signals are executed via `PaperExecutionEngine` (paper fills only). |
| Fee calculation from `settings.yaml` | Partial | Paper currently applies only `commission_per_order`; full fee stack exists in backtest simulator only. |
| Slippage at 1.5x backtest assumption | Ready | `paper_trading.slippage_multiplier=1.5` is wired into paper executor. |
| SQLite or CSV fill logger | Partial | CSV + parquet are present; SQLite is not implemented. |
| Wednesday 3:25 PM exit watchdog | Partial | Time-exit logic exists in strategy, but paper runtime does not currently execute strategy exit checks each loop. |
| Kite token rotation / reauth handling | Missing | Token is loaded at startup; no runtime refresh or reauth flow. |
| Restart-safe position recovery | Missing | Position/cash/fill-seq state is in-memory only. |
| Expiry settlement for open options | Missing | No explicit expiry settlement pipeline in paper executor/runtime. |
| Weekend + NSE holiday guard | Partial | Calendar utility exists, but paper runtime loop does not gate on it. |

## Hard Requirements for This Phase

1. Live data source for paper mode remains Kite.
2. No real order placement from paper runtime.
3. Exit logic must run every loop and be test-covered.
4. Full fee-stack parity across backtest and paper.
5. Persistent audit trail survives restarts.
6. Runtime ops suitable for a month-long local hardening run.
7. Runtime survives token expiry and can recover without manual firefighting.

## Proposed Architecture

### 1) Execution Boundary

Introduce `MockBroker` as the explicit paper execution boundary:
- Input: normalized orders/signals from strategy runtime.
- Behavior: fill simulation only, no broker order placement.
- Output: normalized fill events + fee components + position updates.

Implementation note:
- Reuse `PaperExecutionEngine` internals where possible.
- Either:
  - rename/refactor `PaperExecutionEngine` into `MockBroker`, or
  - keep `PaperExecutionEngine` and add `MockBroker` facade for clean interface parity with future live broker class.

### 2) Data Path

Keep live market data from `KiteFeed`:
- Option chain snapshot fetch at each decision cycle.
- Candle/VIX refresh must be incremental (not cache-stale).
- Keep option-chain persistence with composite dedup keys.
- Add runtime market-day gate: skip execution on weekends and NSE holidays.
- Add FII staleness warning gate (warn/block when data age exceeds configured threshold).
- Document VIX reality for intraday paper runs: Kite VIX is effectively daily/EOD cadence, so intraday regime uses latest available daily print.

### 3) Fee/Slippage Model

Use a shared fee utility in both backtest and paper:
- `commission_per_order`
- `stt_pct`
- `exchange_txn_charges_pct`
- `sebi_fee_pct`
- `stamp_duty_pct`
- `gst_pct`

Slippage:
- Base from `backtest.slippage_pct`.
- Paper uses `paper_trading.slippage_multiplier=1.5`.

### 4) Persistence

Required persisted artifacts:
- Raw signals (actionable and blocked).
- Fill events (per leg).
- Fee breakdown by component.
- Position snapshots.
- Daily summary metrics.
- Runtime health/status snapshots.

Storage options:
- Minimum: CSV + parquet (already partly present).
- Target for hardening: add SQLite ledger for restart-safe state and queryability.

### 5) Exit Watchdog

Add explicit watchdog in runtime:
- Wednesday force-exit check at 15:25 IST.
- Trigger exit for all open paper positions if still open.
- Log reason code: `watchdog_wed_1525_force_exit`.
- Keep strategy-level exits (PT/SL/DTE/time) as primary; watchdog is last resort.

### 6) Session Continuity

For month-long unattended hardening runs:
- Phase decision for this run: manual daily token injection (`.env`/secret) before market open, with startup auth check.
- Add runtime auth heartbeat (detect 403/session errors), then reload token source with bounded retries.
- Detect auth failures and transition to degraded state with alerts.
- Add restart-safe state recovery for open positions, cash/equity, fill sequence, and watchdog state.
- Ensure fill IDs are monotonic across restarts.

Degraded-state policy:
- Halt new entries.
- Continue exit/watchdog handling for existing open positions.
- Keep retrying auth until recovered or operator stops the run.

## Workstreams

Chronological execution order for this phase:
1. `WS1` persistence/statefulness foundation (SQLite + recovery drills).
2. `WS2` runtime correctness (exit path + settlement + pricing).
3. `WS0` continuity controls (token lifecycle + market-day guard).
4. `WS3` fee-model parity.
5. `WS4` execution-boundary cleanup (`MockBroker`).
6. `WS5` operations hardening.
7. `WS6` observability polish.

## WS1: Persistence and Restart Recovery Foundation (First)

1. Add SQLite ledger for fills, positions, cash/equity state, and watchdog events.
2. Persist and recover fill sequence to avoid duplicate `PAPER-00000001` reuse.
3. On startup, restore prior state if last session ended unexpectedly.
4. Keep CSV exports for quick manual review (secondary artifacts).
5. Set circuit-breaker policy for paper unattended runs: enable daily auto-reset of daily-trip state.

Definition of done:
- Restart mid-session resumes position state and does not duplicate ledger semantics.
- Fresh restart can reconstruct full paper account state with no manual edits.

### Recovery Drills (Mandatory)

Run these drills before any month-long hardening run:
1. `mid_position_kill`:
   - Open paper position, kill process (`SIGKILL`), restart.
   - Expected: same open position/cash/equity reconstructed; no duplicate re-entry.
2. `fill_boundary_kill`:
   - Kill process immediately after fill generation boundary.
   - Expected: idempotent recovery (no missing/duplicate ledger events after restart reconciliation).
3. `day_rollover_restart`:
   - Restart around trading-day boundary.
   - Expected: daily counters and breaker state follow defined reset policy.
4. `multiple_restarts_same_day`:
   - Perform 3-5 restarts in one session.
   - Expected: monotonic fill IDs and stable open-position accounting.

Drill pass criteria:
- Zero phantom positions.
- Zero accidental duplicate logical trades.
- Deterministic cash/equity before and after restart.
- Reproducible results across repeated drill runs.

## WS2: Correctness First (Highest Trading-Logic Priority)

1. Run strategy exit checks in paper loop every iteration before new entries.
2. Wire `on_fill` state updates so strategy internal position lifecycle stays consistent.
3. Ensure `EXIT` signals bypass circuit-breaker entry blocks (exits must always be allowed).
4. Implement explicit Wednesday 15:25 IST force-exit watchdog for all open paper positions.
5. Fix option-leg exit pricing so options never use underlying fallback price.
6. Set side-aware quote source policy for option exits:
   - BUY-to-close uses `ask`.
   - SELL-to-close uses `bid`.
   - fallback order: `mid` then `ltp`, always with warning flag.
   - apply configured slippage after quote selection.
7. Add candle/VIX freshness gate before signal generation; if stale, skip trading actions for that loop and emit warning.
8. Ensure multi-leg fills at same timestamp are persisted without collapse.
9. Add expiry settlement path for open options:
   - OTM expiry -> zero settlement.
   - ITM expiry -> intrinsic settlement.
10. Add lot-size/quantity validation guard before fill simulation.

Definition of done:
- Integration test covers entry -> prescribed exit -> fills persisted per leg.
- Integration test covers expiry settlement behavior.

## WS0: Continuity Prerequisites

1. Add explicit Kite token lifecycle strategy:
   - primary mode for this phase: manual daily token injection before open.
   - startup auth check must pass before paper loop starts.
   - runtime auth heartbeat detects failures and reloads token source with bounded retries.
2. Add market-day gate in `run_paper` (weekday + NSE holiday calendar).
3. Add startup self-check that refuses to run outside allowed schedule unless override is set.

Definition of done:
- Process does not silently die after midnight token expiry.
- Runtime does not churn on weekends/holidays.

## WS3: Fee Model Parity

1. Add shared `calculate_options_fees(...)` utility under `src/execution/fees.py` (or similar).
2. Refactor backtest simulator and paper executor to call same utility.
3. Persist fee component columns, not just total fees.

Definition of done:
- Same trade payload gives same fee result in paper and backtest paths.

## WS4: MockBroker Boundary

1. Introduce `MockBroker` interface/class for paper execution boundary.
2. Wire `scripts/run_paper.py` to use this boundary explicitly.
3. Keep live broker path separate and disabled in paper mode.

Definition of done:
- Runtime can swap paper/live execution adapters without strategy changes.

## WS5: Runtime Operations

1. Add `docker-compose.yml` with at least:
   - `paper-runner`
   - `health-check` sidecar or embedded check command
2. Configure restart policy: `unless-stopped`.
3. Add periodic health checks and liveness status logging.
4. Gate trading window to 09:00-15:30 IST weekdays.

Window control options:
- In-app gate: do not trade outside market window, keep process alive.
- Scheduler gate: start/stop container via cron/systemd by IST schedule.
- Recommended for local month run: both (defense in depth).

Definition of done:
- Process auto-recovers from crash and never places paper actions outside allowed window.

## WS6: Observability

1. Structured logs with stable event codes.
2. Daily reports:
   - fill summary
   - fee summary
   - exit reason distribution
   - blocked-entry counts
3. Optional Redis for ephemeral pubsub/metrics buffering only.

Note:
- Redis is optional for this phase.
- Do not use Redis as sole durable store for trade ledger/state.

## Fee Function Reference

Target fee logic (shared by paper and backtest):

Important semantics:
- Fee calculation is per fill/leg event.
- Strategy/trade-level fees are aggregate sums of all leg fills.
- For multi-leg structures (iron condor/jade lizard), do not compute fees once at net-trade level.

Per-fill reference function:

```python
def calculate_options_fees_for_fill(side, price, qty, config):
    turnover = price * qty
    brokerage = config["commission_per_order"]
    stt = turnover * (config["stt_pct"] / 100) if side == "SELL" else 0.0
    txn_charges = turnover * (config["exchange_txn_charges_pct"] / 100)
    sebi_charges = turnover * (config["sebi_fee_pct"] / 100)
    stamp_duty = turnover * (config["stamp_duty_pct"] / 100) if side == "BUY" else 0.0
    gst = (brokerage + txn_charges + sebi_charges) * (config["gst_pct"] / 100)
    return round(brokerage + stt + txn_charges + sebi_charges + stamp_duty + gst, 2)
```

Multi-leg aggregate example:
- `total_trade_fees = sum(calculate_options_fees_for_fill(...) for each fill leg)`

Config source:
- `config/settings.yaml` -> `backtest` fee fields.

## Milestones

M1 (Statefulness baseline):
- SQLite ledger and state restore path live.
- Fill-sequence persistence live.
- Recovery drills passing.

M2 (Correctness baseline):
- Exit checks active in runtime.
- Wednesday watchdog fallback active.
- Fill persistence fixed for multi-leg events.
- Option exit pricing fixed.
- Side-aware quote policy implemented.
- Candle/VIX freshness guard implemented.
- Expiry settlement behavior validated.

M3 (Cost parity):
- Shared fee utility used by both paper and backtest.
- Fee components persisted and reported.

M4 (Operational hardening):
- MockBroker boundary finalized.
- Docker compose + health/restart + time window controls.
- Token lifecycle + market-day guard hardened.

## Acceptance Criteria for Phase Completion

- Paper runtime uses live Kite data and never places real orders.
- Exit behavior matches strategy rules plus watchdog fallback.
- Slippage and fee model are deterministic and config-driven.
- Restart recovery preserves open positions and accounting state.
- One-month local run shows stable operations with daily audit artifacts.

## Immediate Next Actions

1. Implement WS1 SQLite ledger + startup recovery first.
2. Run WS1 recovery drills and harden until deterministic.
3. Implement WS2 exit-loop wiring and `on_fill` state sync.
4. Implement WS2 watchdog + quote-source + freshness guard + expiry settlement.
5. Implement WS0 token lifecycle handling and market-day guard, then WS3 fee parity.
