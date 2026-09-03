# Affective Emergence v0.2 — Observatory State-Lifecycle and Evaluation-Isolation Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract prevents the observational layer from manufacturing history, cross-arm dependence, or order effects through hidden evaluator state.

## 1. Principle

A candidate may be prefix-causal with respect to one scenario and still be scientifically invalid if its implementation carries mutable information from a previous scenario, candidate, arm, batch, or evaluation order.

Initial v0.2 therefore treats candidate evaluation as **scenario-local and replay-determined**.

For primary `OfflinePrefixReplay` evidence, candidate output at `(candidate, scenario, cut_point)` must be determined only by:

- the locked candidate definition;
- the allowed immutable prefix through the cut point;
- locked structural/preprocessing parameters whose provenance is independently valid;
- explicitly declared numerical/environment contracts.

It must not depend on which scenario ran before it, process warmth, batch membership, semantic arm identity, prior candidate outputs, or hidden mutable global state.

## 2. Default evaluator model

The qualified initial evaluator should behave conceptually as a pure function:

`evaluate(candidate_definition, candidate_payload) -> CandidateObservation`

where `CandidatePayload` contains only allowed prefix-causal information.

Internal temporary allocations are allowed. Persistent mutable state that survives one evaluation coordinate is not part of the initial evidence-bearing authority unless it is explicitly declared, serialized, reset, and shown to be a deterministic function of the same allowed prefix.

Prefer recomputation from the frozen prefix over mutable streaming accumulators for the first study.

## 3. H1 history is within-scenario history, not cross-scenario memory

`H1ReplayedPrefixHistory` from `V02_HISTORY_STATE_SUFFICIENCY.md` allows the observatory to consume earlier events from the **same scenario prefix**.

It does not authorize:

- carrying an accumulator from scenario A into scenario B;
- sharing adaptive state across blinded arms;
- retaining prior candidate results;
- using previous scenario outcomes to alter current normalization, thresholds, forecast policy, or availability;
- warming a cache with suffix-sensitive/full-trace or semantic identities.

Cross-scenario state is an integrity failure unless a later explicitly qualified population-level analysis contract authorizes that state and keeps it outside individual candidate computation.

## 4. Proposed EvaluatorIsolationManifest

Before exploratory execution, lock an evaluator-isolation manifest binding at minimum:

- schema/version;
- evaluator source/implementation digest;
- candidate-definition set digest;
- allowed persistent-state class;
- reset policy;
- cache policy;
- concurrency policy;
- deterministic ordering/canonicalization rules;
- forbidden global/singleton/thread-local state classes;
- required isolation tests;
- canonical SHA-256.

Initial allowed persistent-state class should be `NoneAcrossEvaluationCoordinates`.

A later stateful evaluator is a new implementation/evidence identity.

## 5. Cache contract

Memoization may improve performance but must not change information authority.

A qualified cache key may include only identities already legal for the candidate computation, such as:

- candidate-definition digest;
- canonical allowed-prefix digest;
- prospectively locked structural parameter digest.

It must not include:

- full source-trace digest;
- unseen suffix digest;
- semantic arm identity;
- post-run exclusion status;
- unblinded outcome labels;
- previous candidate rank/score.

Cache hits and misses must be observationally equivalent.

A clean-process replay with an empty cache must reproduce the same artifacts as a warm-cache run.

## 6. Order/permutation invariance

For a fixed locked study, outputs must be invariant to any scheduling choice that is not part of candidate identity.

Required tests include:

- evaluate A then B vs B then A;
- reverse scenario order;
- permute candidate evaluation order;
- duplicate an evaluation coordinate and remove the duplicate afterward;
- serial vs parallel execution;
- different batch sizes/chunk boundaries;
- cold process vs warm process;
- cache enabled vs disabled where cache use is allowed;
- blind-code permutation with unchanged underlying allowed payloads.

The canonical output for each coordinate must remain identical.

## 7. Reset discipline

Every scenario-local evaluator context must have an explicit lifecycle:

`Create -> Evaluate allowed prefix coordinates -> Finalize -> Destroy`

A new scenario starts from a clean evaluator context.

If an implementation uses an incremental within-scenario state for performance, the state must:

- be derivable exactly from the canonical prefix;
- serialize under an explicit schema;
- bind candidate definition and prefix digest;
- have a deterministic rebuild path from the same prefix;
- never survive into another scenario;
- pass from-scratch vs incremental equivalence tests at every tested cut point.

If incremental and from-scratch results differ, primary evidence fails closed.

## 8. No evaluation-order adaptation

The following are forbidden in confirmatory candidate computation:

- running means/variances updated from previously evaluated confirmatory scenarios;
- online threshold tuning across arms;
- adaptive clipping based on previously observed candidate outputs;
- changing numerical tolerances after encountering difficult cases;
- dynamically selecting a faster/alternate formula based on earlier outcomes;
- early stopping that changes which locked coordinates are computed because current results look decisive.

Population summaries may be computed only after all locked individual candidate artifacts have been frozen under the prospective analysis contract.

## 9. Cross-process reproduction

A reproducer should be able to evaluate one coordinate in isolation in a fresh process and recover the same candidate artifact produced during the complete study.

The realized evidence package should support a sample or full audit binding:

- coordinate identity;
- clean-process result digest;
- study-run result digest;
- equality status;
- environment/toolchain identity.

A result that exists only under one particular process history is not qualified candidate evidence.

## 10. Malicious fixtures

The adversarial suite should include intentionally invalid evaluators that:

1. increment a global counter and use it in the result;
2. retain the prior scenario's last burden;
3. update a running cohort mean during evaluation;
4. branch on candidate evaluation order;
5. use a thread-local cache keyed by semantic arm ID;
6. use full-trace identity as a cache key;
7. produce different results on cache hit vs miss;
8. fail to reset incremental H1 state between scenarios.

Each must be detected by an explicit isolation/order/replay gate.

## 11. Evidence-root consequence

The prospective root should bind:

- evaluator-isolation contract/version;
- `EvaluatorIsolationManifest` digest;
- allowed cache/state class;
- required order/permutation test suite version.

The realized package should bind:

- evaluator-isolation report digest;
- clean-vs-warm equivalence report;
- serial-vs-parallel equivalence report;
- candidate-order/scenario-order permutation report;
- incremental-vs-from-scratch report when relevant.

An isolation failure is an `IntegrityFailure`, not a candidate negative result.

## 12. Relationship to future endogenous persistence

This contract intentionally makes the v0.2 observatory poor at pretending to have memory.

A future native-persistence/mood-like lineage must add persistence to a declared native mechanism and qualify reset, state-transfer, state-ablation, and downstream causal effects separately. It cannot inherit hidden evaluator state from v0.2 and relabel it as native affect.

## 13. Claim boundary

Passing evaluator-isolation tests can establish that candidate observations are functions of the declared scenario-local information rather than accidental process history.

It does not establish emotion, native memory, mood, subjective feeling, sentience, or consciousness.