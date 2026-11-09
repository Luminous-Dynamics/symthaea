# Portfolio Transition Synthetic Reference v0

**Status:** public synthetic development fixture; integrity/accounting conformance only  
**Parent:** [SYM-QUANT-ORACLE-001 #11298](https://github.com/Luminous-Dynamics/symthaea/issues/11298)  
**Consumer:** [Quant Strategy Evaluation PR #11234](https://github.com/Luminous-Dynamics/symthaea/pull/11234)  
**Shared time/proof contract:** [#11241](https://github.com/Luminous-Dynamics/symthaea/issues/11241), [#11264](https://github.com/Luminous-Dynamics/symthaea/issues/11264)

## Files

- `portfolio-transition.events.json`: synthetic orders, fills, cancellation and quotes with separate event and known-at times.
- `portfolio-transition.oracle.json`: expected cash, positions, order status, portfolio value and P&L at four frontiers.

## Scenario

Opening cash is USD 10,000.00 (1,000,000 cents). The evaluator admits:
- a 10-share buy order in SYN-EQ-A, partially filled for 6 shares at USD 100.60; 4 shares are then cancelled;
- a 5-share buy order in SYN-EQ-B, fully filled at USD 49.50;
- a 10 bps fee on each fill, rounded half-up to the nearest cent;
- pre-fill reference quotes and later mark quotes.

The fill price is the transaction price. Slippage versus the pre-fill reference quote is recorded as a diagnostic embedded in that price; do not deduct it a second time from cash or P&L.

The later A/B marks have event_time 11:00 UTC but are not known until 11:05 UTC. At F2 (11:00:03), they are not admissible; at F3 (11:06), they are.

## Expected arithmetic

- Fill A notional: 6 × 10,060 = 60,360 cents; fee 10 bps = 60 cents.
- Fill B notional: 5 × 4,950 = 24,750 cents; fee 10 bps = 25 cents.
- Final cash: 1,000,000 − 60,360 − 60 − 24,750 − 25 = 914,805 cents.
- Final positions: 6 SYN-EQ-A and 5 SYN-EQ-B.
- F3 marks: A 10,160 cents; B 5,050 cents.
- F3 portfolio value: 914,805 + (6 × 10,160) + (5 × 5,050) = 1,001,015 cents.
- F3 P&L: +1,015 cents versus opening cash; total unrealized mark-to-fill gain is 1,100 cents, less 85 cents of fees.

This is a deterministic arithmetic fixture, not a realistic execution simulation. Market impact, funding, borrow, tax, settlement cash availability and capacity are not modeled; no claim of best execution, investable net return or alpha is permitted.

## Required behavior

1. F0 includes only the first A fill; O-A has 6 filled and 4 remaining.
2. F1 includes A cancellation and B fill; both orders are terminal, positions/cash reconcile.
3. F2 cannot use the later marks even though their event_time is 11:00; their known_at is 11:05.
4. F3 admits the new marks and updates portfolio valuation without creating fills.
5. Duplicate fill IDs with same bytes must be idempotent; altered bytes must conflict.
6. Marks never change quantities or cash. Orders never become fills without an admitted fill event.
7. Fee rounding and every arithmetic transition must be exact under the declared minor-unit convention.
8. This fixture's zero values for unmodeled costs are not an assumption that real costs are zero.
