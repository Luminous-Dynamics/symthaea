# POD-Meta Lab (research-only)

A small, standalone Rust harness for the first executable slice of Proof-Obligation-Driven Metareasoning (POD-Meta).

This lab is intentionally outside the root Cargo workspace and has zero third-party dependencies. It does not define the production COG-META ABI, import the not-yet-qualified #5470 types, or modify production scheduling.

## Run

```bash
cargo test --manifest-path research/pod-meta-lab/Cargo.toml
cargo run --manifest-path research/pod-meta-lab/Cargo.toml --bin pod-meta-demo
cargo run --manifest-path research/pod-meta-lab/Cargo.toml --bin poar-v1
```

## What is implemented

- Typed synthetic obligation states.
- Finite operation vocabulary: retrieval, verification, repair, refresh, independent evidence, alternatives, clarification, and external-evidence requests.
- Deterministic lexicographic routing: admissibility, mandatory-work priority, consequence priority, number of actionable obligations, work cost, operation family, and stable ID.
- Mandatory work reserve that competitive operations cannot consume.
- Repeat guard keyed by operation ID and the relevant pre-operation state identity.
- Duplicate obligation/operation/target identity rejection before scheduling.
- Mandatory reserve and fail-closed mandatory-work exhaustion.
- Separate operation and verifier receipts, bound to the exact operation, obligation set, and pre-operation state key.
- Per-obligation current-state identity checks; a promoting receipt must match the active evidence/proof state and advance to a distinct result-state identity.
- Anti-replay behavior: a later revocation/correction advances the state identity, so an earlier valid receipt cannot resurrect a superseded conclusion.
- A declared producer/verifier identity distinction (not cryptographic proof of independence).
- Receipt validation is transactional: duplicate state IDs, missing/mismatched current-state identities, or stale receipt bindings leave the obligation state unchanged.
- A public deterministic POAR-v1 runner with 16 scheduler/receipt conformance fixtures. Fifteen fixtures assert routing/stopping choices; one exercises the valid-promotion → revocation → stale-replay receipt lifecycle. These are structural smoke fixtures, not full domain simulations.
- An AND/OR graph generator with private source-truth labels, a separate recursive oracle, disjoint source-family and source-ID splits, valid-time / known-time evidence revisions, source-independence groups, scope/authority constraints, cycle rejection, a 64-claim dependency chain, false asserted content, and an unverified-source case.
- Controlled mutations for retraction, asserted-polarity flip, known/valid-time shifts, scope/authority changes, integrity downgrade, and independence-group merging; hidden source-truth labels remain unchanged.
- 41 authored unit tests: 24 scheduler/receipt tests, one POAR routing-suite test, and 16 graph/oracle tests.
- Graph benchmarks report disposition correctness, false support/refutation, unsupported proof paths, citation completeness, temporal/scope/authority/integrity violations, retracted citations, correlated-source violations, plus ranked temporal evidence coverage@3 and precision@3; retrieval ranking is separate from final proof citations, and there is no composite score.

## Deliberate limits

This is a scheduler-mechanics prototype, not a production proof engine. The verifier receipt is supplied by the test fixture; the crate does not verify signatures, source authenticity, legal correctness, or whether a verifier is truly independent. The state SupportedUnderProfile is only a synthetic transition after an accepted, exactly-bound fixture receipt. Evidence admissibility is given by the fixture; it is not source authentication. The verifier/proposer distinction is only checked by declared IDs and does not establish true organizational or implementation independence.

The lab does not yet compute obligations from Symthaea's production ProofGraph because the #5469 inventory and #5470 typed-contract ownership sequence remains open. It also does not estimate learned value-of-computation. The first version uses a deterministic baseline so scheduler mechanics can be checked before adding a predictor.

## Next integration gates

1. Qualify #5469 and #5470 and identify the exact production producer/consumer.
2. Add an adapter from real typed inference/proof outputs into this research contract.
3. Replace synthetic verifier receipts with an independently implemented verifier adapter.
4. Run POAR against fixed-budget, generic COG-META, random cost-matched, and oracle upper-bound controls.
5. Add prospective value-of-computation forecasts only after #5476's calibration contract is available.

## Claim boundary

The source and tests are authored. Tests have not been compiled or executed here. No model comparison or SOTA result is claimed.
