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
- Sequential action: bitemporal state; add procedural memory only when the caller declares that a qualified skill is required; always check skill preconditions.
- Broad recall: choose exactly one candidate substrate according to a fixed, auditable preference order. Structured queries prefer typed HDC; other queries prefer dense semantic search, then typed HDC, sparse lexical, then raw scan.

The preference order is a transparent baseline, not an optimality claim. The next research increment should replace it with a held-out, cost-sensitive selector while preserving the same hard evidence and resource gates.

## Resource accounting

Each substrate profile contains a caller-supplied estimated cost vector. Costs are summed component-wise across selected substrates and compared against the caller-supplied budget. No weighted scalar score can trade excess latency for fewer tokens or trade failed evidence checks for lower cost.

The cost vector includes context bytes, context tokens, model calls, tool calls, verification operations, CPU microseconds, latency microseconds and storage bytes. The estimates are not measurements; production qualification requires phase-attributed receipts for construction, maintenance, retrieval, verification and generation.

## Reproduction

From this directory, in a Rust-enabled environment:

```sh
cargo fmt --check
cargo test
```

Tests are authored but have not been executed in this environment.
