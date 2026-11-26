# MACE regime-aware substrate router v0.1

Research-only deterministic rule-based reference router.

## Purpose

Selects a bounded query plan over already-qualified memory substrates according to a declared DecisionNeed regime and component-wise resource budget.

It does **not**:
- admit or authenticate evidence;
- verify source authority, rights, lineage or lifecycle state;
- authorize a tool action;
- construct, mutate or delete memory;
- estimate expected utility or train a learned router;
- prove that a source is true.

Every successful plan explicitly requires canonical evidence resolution and current-at-use lifecycle revalidation. Failure to find a required substrate or fit the component-wise budget is fail-closed and non-qualifying.

## Routing rules

- Exact identifier: sparse lexical candidate lookup, or raw exact scan if sparse lookup is unavailable.
- Current mutable state: bitemporal current-state projection plus live re-observation.
- Historical as-known query: bitemporal historical projection plus raw source access.
- Multi-hop causal query: verified dependency graph plus raw source access.
- Temporal trajectory query: temporal HDC-LTC pattern lookup plus dependency-graph closure and raw-source resolution.
- Broad structured recall distinguishes typed-content HDC, temporal HDC-LTC and event-ID HDC rather than treating them as one interchangeable HDC substrate.
- Sequential action: bitemporal state; add procedural memory only when the caller declares that a qualified skill is required; always check skill preconditions.
- Broad recall: choose one candidate substrate according to a fixed, auditable preference order. Structured queries prefer typed-content HDC, then temporal HDC-LTC, dense semantic, event-ID HDC, sparse lexical, lineage-verified hierarchical summary, then raw scan. Other queries prefer dense semantic, typed-content HDC, temporal HDC-LTC, event-ID HDC, sparse lexical, lineage-verified hierarchical summary, then raw scan. A selected summary requires dependency-graph parent closure plus raw-source resolution; if either is unavailable, the route is non-qualifying.

The preference order is a transparent baseline, not an optimality claim. The next research increment should replace it with a held-out, cost-sensitive selector while preserving the same hard evidence and resource gates.

## Resource accounting

Each substrate profile contains caller-supplied amortized estimates for five phases: construction, maintenance, retrieval, verification and generation. The cost profile also pins the number of scored queries over which construction and maintenance are amortized; a zero denominator is rejected and the denominator is repeated in the resulting substrate binding. The router sums each phase and the all-phase total component-wise across selected substrates, then compares that vector against the caller-supplied budget. No weighted scalar score can trade excess latency for fewer tokens or trade failed evidence checks for lower cost.

The per-phase vector includes context bytes, context tokens, model calls, tool calls, verification operations, CPU microseconds, latency microseconds and storage bytes. Estimates are not measurements. Production qualification requires phase-attributed resource receipts and an independently pinned amortization window.

## Reproduction

From this directory, in a Rust-enabled environment:

```sh
cargo fmt --check
cargo test
```

Tests are authored but have not been executed in this environment.
