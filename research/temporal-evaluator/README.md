# Symthaea Temporal Evaluator

This is a separate, evaluator-side Rust package for paired retain/evict effects and exhaustive small-subset upper bounds. It is not a candidate runtime dependency.

The current source provides:

- seeded, reproducible retain-first / evict-first order assignment using evaluator-held randomness (SplitMix64 for reproducibility only; not cryptographic secrecy);

- paired intervention checks that require identical episode/task/future-stream/model/initial-state/candidate-feature digests and identical byte/compute/refresh/observation budgets;
- componentwise wrong-belief, abstention and missing-output exposure differences;
- group interventions to expose non-additive memory interactions;
- exact subset enumeration up to 16 candidate items;
- separate best-known subsets for each exposure component and a Pareto frontier, with no weighted scalar objective;
- hard checks for duplicate IDs, unordered selections, incorrect bytes, budget overruns, incomplete subset enumeration and exposure totals exceeding episode duration.

The exact subset result is an evaluator-only clairvoyant upper bound, not a deployable candidate policy. It must never be exposed to the candidate or used to train against confirmatory outcomes.

Package separation is an architectural boundary only. It does not provide OS-level confidentiality. Before confirmatory use, complete #9437: separate OS identities and protected oracle storage, candidate sandboxing, evaluator-owned signing keys, signed receipts, replay protection and adversarial filesystem/descriptor tests.

No Cargo build, format, Clippy or test execution is claimed until a Rust toolchain is available.

## Rollback-resistant witness checkpoints

The `witness` module adds pure canonical checkpoint and witness-attestation types plus a strict verifier. Evaluator checkpoints bind the accepted receipt-chain head and previous checkpoint digest. Witness attestations bind the checkpoint digest to a fresh evaluator challenge. The verifier requires a Byzantine fault-tolerant quorum satisfying n >= 3f+1 and 2f+1 <= q <= n-f. Exact retransmissions are deduplicated; conflicting responses from one witness, malformed/invalid signatures, unknown identities and stale challenges cannot count as votes or poison a valid quorum.

This package does not implement a remote witness service, durable anti-equivocation database, TPM counter, challenge CSPRNG, RPC protocol, key rotation or recovery ceremony. A witness service must persist its high-water head outside the evaluator rollback domain and refuse to sign two different digests for the same evaluator epoch. The pure verifier is not proof that a deployed witness is independent or durable. See `docs/research/ROLLBACK_RESISTANT_RECEIPT_WITNESS_V0_1.md` and #9631.


## Joint temporal resource-control attribution

The evaluator-only `joint_budget` module enumerates complete action subsets
(up to 16) using a canonical digest of the evaluator-owned action registry (action IDs, mutual-exclusion groups, and per-action semantic-contract digests).
It validates eight hard resource dimensions and excludes incompatible
conflict-group combinations from the feasible frontier.

The evaluator-only `interaction` module derives exact paired marginal
reductions and pairwise difference-in-differences for the three non-compensatory
temporal exposure dimensions. It reports eligible-context, incompatible-context
and over-budget-context counts. Positive interaction is synergy; negative
interaction is antagonism. Values are exact signed integer sums and sample
counts, not floating-point estimates. This is a feasibility-filtered conditional
interaction measure, not a Shapley value.

Six interaction test functions and fourteen exact-oracle test functions are
authored; no Rust build or test has run for this draft.

## Higher-order temporal interaction decomposition

The evaluator-only `interaction_decomposition` module adds an exact Möbius /
Harsanyi-dividend decomposition over the complete binary action lattice.

For any non-empty action subset T, it computes:

`v(S) = E(empty) - E(S)`

`dividend(T) = sum over S subseteq T of (-1)^(|T|-|S|) * v(S)`

The decomposition reports first-order action effects, pairwise interaction,
and pure third-/fourth-/higher-order contributions after lower-order terms are
removed. Positive coefficients mean additional exposure reduction
(synergy); negative coefficients mean antagonism.

A coefficient is emitted only if every required sub-subset is compatible and
within the same episode's measured hard resource budget. Incompatible or
over-budget subset counts are reported; a missing coefficient is not silently
replaced by zero. The evaluator binds the full table to the same episode,
future stream, model, action registry and resource budget.

This is exact for the finite action lattice and the measured episode. It is
not a Shapley allocation, a population-level causal effect or evidence of
generalization. Causal interpretation requires valid randomized/matched
interventions; stochastic systems need independent paired replicates and
uncertainty intervals across held-out episodes. A single deterministic table
only characterizes its declared context.

Nine source-level tests cover additive controls, pure third-/fourth-order
synergy, higher-order antagonism, conflict-group exclusions, budget exclusions,
context/registry mismatch, and incomplete/duplicate subset tables.

The evaluator package currently has 30 authored Rust test functions across
joint-budget, pairwise-interaction and higher-order decomposition modules.
No Rust build, format, Clippy or test execution is claimed.
