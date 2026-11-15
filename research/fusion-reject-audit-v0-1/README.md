# Fusion rejected-candidate audit v0.1

A dependency-free Rust contract for statistically bounding missed feasible candidates when exhaustive high-fidelity evaluation of every screened-out design is impractical.

This is a synthetic/research contract. It is not a plasma model, solver adapter, or physical validation method. The full-census reduced-space benchmark in [fusion-search-benchmark-v0-1](../fusion-search-benchmark-v0-1/) remains the reference for checking whether the audit bound behaves conservatively.

## Paradigm

A screening optimizer must retain every candidate and its disposition. When a full reference evaluation of every rejected candidate is too expensive, draw an independently specified, reproducible, stratified random sample from the rejected population.

For each stratum, the audit computes a one-sided upper bound on the proportion of candidates that are either:

- confirmed feasible by the declared reference evaluation; or
- unresolved because the reference evaluation failed, did not converge, was unsupported, or was inconclusive.

Unresolved outcomes are conservatively treated as potentially feasible. The per-stratum error budget is allocated with a Bonferroni correction, producing a simultaneous family-wise bound across strata. A one-sided Hoeffding bound is used without a finite-population correction; it is intentionally conservative. A full census uses the exact observed potential-feasible count instead.

The resulting upper bound on missed feasible candidates is combined with a supplied lower bound on confirmed feasible candidates that survived screening:

`recall_lower = accepted_feasible_lower / (accepted_feasible_lower + missed_feasible_upper)`

The target is certified only when this lower bound meets the declared recall target. A small or biased audit therefore fails closed rather than silently turning into an optimistic recall estimate.

## Sample-selection integrity

The crate uses a versioned deterministic SplitMix64 / partial Fisher–Yates selection algorithm. It derives the expected sample identities from:

- the frozen, lexicographically ordered rejected-candidate identities in each stratum;
- a seed declared as committed before audit outcomes were observed; and
- the stratum identity.

The supplied sample identities must match the derived identities in draw order. Stratum definition and population manifest digests, a seed-commitment reference, selection-receipt references, and per-sample evidence references are required.

These are declared identities. The crate does not fetch or hash source payloads, authenticate timestamps, prove that the seed was actually committed first, verify that the sampling process ran independently, or establish that solver labels are truthful. Those require independent artifact verification and trusted execution records. SplitMix64 is not cryptographically secure and is used only to make this research harness deterministic and replayable. A production or adversarial audit must obtain an unpredictable seed from an external cryptographically secure randomness source after the population and strata are frozen, commit/reveal it under independent timestamped control, and independently verify the selection and solver evidence.

## Outcomes

Only `ConfirmedInfeasible` counts as a negative observation. `ConfirmedFeasible`, `Failed`, `NonConverged`, `Unsupported`, and `Inconclusive` all count as potential false negatives when bounding the rejected population.

This prevents solver failures and unsupported physics from being silently recoded as infeasible.

## Run

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path Cargo.toml --bin audit_rejected_candidates
```

## Claim ceiling

A PASS means only that the declared sample identities match the deterministic selection algorithm, the declared strata satisfy the contract, unresolved results are handled conservatively, and the stated finite-population bound is calculated as specified.

It does not prove sampling independence, source-byte integrity, label truth, solver correctness, physical feasibility, plasma performance, reactor economics, safety, or a real-world 99% recall claim. A measured solver-backed audit must be independently reproduced before the bound can support an engineering decision.
