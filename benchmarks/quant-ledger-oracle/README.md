# Quant Ledger Oracle v0.1

A standalone, dependency-free Rust reference implementation for synthetic quant-accounting
conformance. It is intentionally smaller than a backtest engine and does not compete with
LEAN or FinRL.

## Scope

The oracle replays already-recorded fills and marks under an inclusive known-at frontier. It
reconciles cash, long-only whole-share positions, notional, fees rounded upward to a minor
unit, market value and equity using checked integer arithmetic.

It rejects future-known fills, future fills, malformed identities, duplicate fill IDs,
ambiguous same-time fill sequences, oversells, cash shortfalls, non-positive quantities or
prices, unsupported currency conversion, missing marks, ambiguous marks and arithmetic
overflow. Fills are replayed in deterministic event-time/sequence order. Marks after the
valuation instant or known after the decision frontier are excluded and counted.

## Explicit non-goals

No broker, order submission, custody, settlement or external writes; no strategy or signal
generation; no fractional shares, shorting, margin, derivatives, corporate actions or tax
lots; no FX conversion; no spread, slippage, market impact, borrow, funding or tax model; no
realized P&L accounting policy (FIFO/LIFO/average cost); no source proof verification.

Unknown spread/slippage/impact/etc. are not assumed to be zero in a qualified performance
evaluation. A passing unit test establishes arithmetic and accounting behavior only for this
synthetic profile. It does not establish execution realism, investable performance, alpha or
SOTA.

## Run

From this directory, run cargo fmt -- --check and cargo test. No third-party dependencies.
