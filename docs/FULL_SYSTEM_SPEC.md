The Four Market Conditions And Their Strategies
Your regime classifier already identifies these — the gap is that only one has a deployed strategy:
LOW_VOL_RANGING:    Iron condor          ✅ implemented
LOW_VOL_TRENDING:   Iron condor (weaker) ⚠️  same strategy, worse fit
HIGH_VOL_CHOPPY:    Nothing              ❌ sitting out
HIGH_VOL_TRENDING:  Nothing              ❌ sitting out
Two of four regimes currently generate zero income. Those are not rare — from your backtest, high-vol periods covered roughly 30-40% of trading days across 2022-2025. That's significant idle time that a complementary strategy could monetize.

Strategy 1 — Trend Following For LOW_VOL_TRENDING
Why iron condor is a weak fit for trending:
Your backtest showed trending regime win rate (71.4%) slightly below ranging (76.7%) with similar mean PnL. The condor works but it's fighting the market — a trending market is trying to move directionally while you're betting on range. You're winning but leaving edge on the table.
Better fit: Bull call spread or bear put spread
In LOW_VOL_TRENDING with ADX > 25:
  ADX rising + price above 20EMA → Bull call spread
  ADX rising + price below 20EMA → Bear put spread

Example bull call spread (NIFTY trending up):
  Buy  24,000 CE @ ₹180
  Sell 24,300 CE @ ₹80
  Net debit: ₹100
  Max profit: ₹200 (if NIFTY closes above 24,300)
  Max loss:   ₹100 (premium paid)
The edge: In a confirmed uptrend with low volatility, the bought call captures directional move cheaply. Unlike the condor, time is partially your enemy but the directional move compensates.
Data you already have: The backtest covers LOW_VOL_TRENDING periods. You can extract those specific bars and check how much NIFTY actually moved directionally during them — that tells you whether the spread would have profited.

Strategy 2 — Long Straddle / Strangle For HIGH_VOL_CHOPPY
The condition: VIX elevated, ADX low — market is volatile but without clear direction. This is the worst environment for iron condors (you correctly sit out) but potentially the best for buying volatility.
Structure: Long strangle
Buy OTM call + Buy OTM put simultaneously
Example (NIFTY 23,500, VIX 18):
  Buy 23,800 CE @ ₹120
  Buy 23,200 PE @ ₹115
  Total cost: ₹235

Profit if NIFTY moves > 235 points either direction
Loss if NIFTY stays between 23,200 and 23,800
The edge in HIGH_VOL_CHOPPY: When VIX is elevated and ADX is low, the market is in a regime of large random moves without trend. A strangle profits from the large moves regardless of direction. You're now buying the volatility premium instead of selling it — appropriate because VIX elevation means the premium is justified.
Critical constraint: This only works when IV is not already overpriced. If VIX has spiked significantly before entry, you're buying expensive options. Entry timing needs a VIX mean-reversion signal — enter the strangle after VIX has been elevated for 2+ days, not on the first spike day.
Historical validation: The Feb 13, 2026 crash example from your KB would have been a perfect strangle entry — HIGH_VOL_CHOPPY regime, large directional move. The straddle example in your KB showed 112% return on that specific event. Whether that generalizes across all HIGH_VOL_CHOPPY periods is what the backtest would tell you.

Strategy 3 — Pre-Event Straddle For Scheduled Volatility
Already in your KB as a validated concept. This is the most directly actionable addition because the data already exists.
RBI policy dates:   6/year × 4 years = 24 instances
Budget dates:       4 instances
FOMC dates:         8/year × 4 years = 32 instances
Total:              ~60 data points — statistically meaningful
Structure:
Enter straddle 1-2 days before scheduled event
Exit same day as announcement (IV crush)

Buy ATM call + Buy ATM put
Entry: Tuesday/Wednesday before Thursday RBI announcement
Exit: Thursday 11 AM (after announcement, before full IV crush)

Edge: IV expands into event, collapses after
You capture the expansion, exit before collapse
Why this is separate from the iron condor system: It requires you to enter when the iron condor would be blocked (pre-event entry block). The two strategies are complementary — condor runs in normal weeks, straddle runs in event weeks.
This is the highest confidence addition because the mechanism is well understood, data points are sufficient, and your KB already documented it as tradeable.

Strategy 4 — Momentum Breakout For HIGH_VOL_TRENDING
The condition: VIX elevated, ADX high and rising — market is in a strong directional move with conviction. Think Russia-Ukraine 2022 or the 2023 rally.
Structure: Long calls or puts (directional)
HIGH_VOL_TRENDING upward (ADX > 30, price making higher highs):
  Buy slightly OTM call with 2-3 weeks DTE
  Exit at 50% profit or 1 week before expiry

HIGH_VOL_TRENDING downward:
  Buy slightly OTM put
  Same exit rules
The edge: In a confirmed strong trend with elevated volatility, bought options participate in the move while capping loss at premium paid. Theta works against you but the directional move compensates. This is the one scenario where buying options beats selling.
The difficulty: Entry timing is crucial and trend identification is harder than range identification. Your ADX override at 30+ already identifies this regime — the question is whether the directional signal (price action + EMA) is reliable enough to generate consistent edge. This needs more validation than the other strategies.
Honest assessment: This is the weakest addition of the four. Trend following with options is harder to systematize than premium selling. Include it in Phase 3+ research, not Phase 2.

Priority Sequence For Implementation
Phase 2 (current):   Iron condor optimization
                     Delta cap, FII, regime quality

Phase 3:             Pre-event straddle
                     Highest confidence, clearest mechanism
                     ~60 data points for validation
                     Fits within existing infrastructure

Phase 4:             Directional spread for LOW_VOL_TRENDING
                     Replace or complement condor in trending
                     Requires directional signal validation

Phase 5:             Long strangle for HIGH_VOL_CHOPPY
                     Complex entry timing
                     Requires IV mean-reversion signal

Phase 6+:            Momentum breakout for HIGH_VOL_TRENDING
                     Hardest to systematize
                     Only if earlier phases prove out

What This Does For The Nifty SIP Comparison
This is where the framing shifts meaningfully:
Iron condor alone:          Generates income in ~40-50% of weeks
                            (LOW_VOL regimes only)

Iron condor + event straddle: Adds 15-20 event weeks per year
                              Previously zero-income weeks now active

Iron condor + trending spread: Adds LOW_VOL_TRENDING optimization
                               Better fit for ~28 trades/year

Full multi-strategy system:   Active in 70-80% of market conditions
                              vs current 40-50%
A system active in 70-80% of weeks with appropriate strategies for each condition starts to look genuinely competitive with passive returns — not because any single strategy beats the Nifty, but because the combination generates income across almost all market environments including the flat and falling years where SIP generates nothing.
That's the long-term thesis made concrete. The iron condor is the foundation. These additions are what make it a complete system.