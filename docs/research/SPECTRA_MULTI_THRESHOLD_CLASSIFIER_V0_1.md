# SPECTRA Conjunctive Multi-Threshold Classification v0.1

**Status:** deterministic Rust classifier and tests declared; not compiled or executed.
**Research parent:** #10776
**Method ID:** spectra-conjunctive-multi-threshold-classification-v1

## 1. Problem

Many engineering decisions are not "maximize a scalar score." They are "meet every noncompensatory requirement":
- performance must exceed a minimum;
- safety risk must remain below a maximum;
- latency must remain below a maximum;
- participant/user burden must remain below a maximum;
- cost and power must remain below frozen limits.

A weighted sum is dangerous here: a high performance score must not compensate for a safety violation. SPECTRA's multi-threshold classifier is a deterministic post-processing layer over a complete frozen action × constraint interval grid.

## 2. Matched literature

Jiang, Hashima, Hatano & Takimoto, *Multi-thresholding Good Arm Identification with Bandit Feedback*, ACML 2025, PMLR 304:49–64:
https://proceedings.mlr.press/v304/jiang26a.html

Their MultiTUCB setting returns an epsilon-good arm whose expected reward vector is componentwise above a predefined threshold vector (up to epsilon), or returns bottom when no such arm exists. They give a sample-complexity bound and evaluate synthetic and real datasets.

This is the closest matched good-arm comparator, but direct guarantees are not transferred to SPECTRA's mixed AtLeast/AtMost, robust-interval and external-gate composition without matching assumptions and proving the mapping.

For broader multiobjective identification:
- Chen et al., AISTATS 2025, objective-wise best-arm ID with independent objective rewards: https://proceedings.mlr.press/v258/chen25b.html
- Kone, Kaufmann & Richert, ICML 2025, constrained Pareto set identification: https://proceedings.mlr.press/v267/kone25a.html
- Xue et al., *Few-for-Many Best-Arm Identification in Multi-Objective Bandits*, arXiv:2610.16840, 12 Oct 2026: https://arxiv.org/abs/2610.16840

These solve different decision tasks and belong in separate comparator tracks.

## 3. Contract and interval rules

Freeze:
- complete action ID universe;
- complete constraint ID universe;
- direction and threshold per constraint;
- evidence scope/provenance/target/support in upstream receipts;
- a jointly simultaneous interval family across all action × constraint pairs.

AtLeast threshold τ:
- CertifiedPass iff L ≥ τ;
- CertifiedViolation iff U < τ;
- Unresolved otherwise.

AtMost threshold τ:
- CertifiedPass iff U ≤ τ;
- CertifiedViolation iff L > τ;
- Unresolved otherwise.

Action:
- CertifiedFeasible iff every constraint is CertifiedPass;
- CertifiedInfeasible iff any one constraint is CertifiedViolation (retain every other per-constraint result and all witnesses);
- Unresolved otherwise.

Family:
- GoodCandidateSetFound if at least one action is certified feasible;
- CertifiedNoGoodCandidate only if every action is certified infeasible;
- NotEstablishedAtBudget if no action is good and at least one is unresolved.

Boundary equality is intentional and conservative:
- AtLeast lower=threshold passes;
- AtMost upper=threshold passes;
- AtLeast upper=threshold with lower below threshold remains unresolved;
- AtMost lower=threshold with upper above threshold remains unresolved.

## 4. Multiplicity and selection

If one joint event covers every frozen action × constraint base interval, each action classification is deterministic post-processing of that event. A second union bound over actions or actions' constraint combinations is unnecessary.

That does not remove multiplicity control needed by the upstream simultaneous interval estimator. Nor does it validate the frozen universe, interval provenance, source/target support, model adequacy, operational mappings or other mandatory gates.

A certified infeasible action needs only one definite violation; all other constraint statuses remain in the output. A no-good family conclusion needs every action to have at least one definite violation. Any unresolved action blocks no-good.

## 5. Rust implementation

File: tools/spectra-uvsd-fit/src/multi_threshold.rs

- Frozen action and constraint universes; per-constraint AtLeast/AtMost direction.
- Complete one-interval-per-action/constraint grid required.
- Canonical action and constraint order.
- Per-constraint status and all violating/unresolved witnesses preserved.
- Per-action CertifiedFeasible / CertifiedInfeasible / Unresolved.
- Family GoodCandidateSetFound / CertifiedNoGoodCandidate / NotEstablishedAtBudget.
- Replay recomputes the complete classification and rejects any altered field.

## 6. Tests declared

- Mixed good, definitely infeasible and unresolved constraints, preserving every witness.
- No-good only when every action has at least one definite violation.
- An unresolved alternative prevents a no-good conclusion.
- Both AtLeast and AtMost boundary equality cases.
- Input-order invariance over action, constraint and interval permutations.
- Replay rejects mutated witness/disposition.
- Incomplete, duplicate, unknown action, unknown constraint and empty universes fail closed.
- Nonfinite thresholds and invalid intervals fail closed.
- 256 generated campaigns × 32 actions × 2 constraints; disjoint complete action partition.

A separate Python mirror of the classification rules exercised the same 256×32×2 generated rows, plus witness and boundary cases. This is deterministic algebra smoke only; it is not Rust execution, statistical coverage, or CI evidence.

Rust tests remain declarations. cargo, rustc and rustfmt are unavailable in the current environment, so no compile, test, formatting, clippy, CI or performance pass is claimed.

## 7. Benchmark matrix

Matched baseline first:
- MultiTUCB on its vector-feedback and threshold assumptions;
- a one-dimensional reduction against APGAI / scalar good-arm ID;
- uniform allocation;
- objective-wise independent allocation;
- vector-feedback allocation only when its feedback/correlation contract matches.

Sweep:
- actions K = 4, 8, 16, 32, 64, 128;
- constraints M = 2, 4, 8, 16, 32;
- mixed AtLeast/AtMost constraints;
- all-good, sparse-good, no-good and unresolved-at-budget populations;
- margins well above, at and below every threshold;
- independent/correlated rewards;
- hard safety thresholds combined with soft efficiency thresholds;
- target/source shift, support holes, drift and model misspecification as separate negative controls.

Measure:
- familywise false-good and false-no-good;
- complete good-set precision/recall;
- false inclusion/exclusion;
- no-good correctness and unresolved rate;
- per-constraint witness correctness;
- fully charged cost-to-good/no-good certificate;
- p50/p95/p99 allocator and replay latency;
- hard-constraint negative-control false pass (target zero).

Proposed aspirational target after matching assumptions: ≥15% lower fully charged cost to a valid multi-threshold certificate than uniform and objective-wise independent allocation in sparse-good / bottlenecked regimes, while preserving familywise coverage and zero false safety-gate passes. This is not an observed result.

## 8. Not included

- No confidence estimator or allocation policy.
- No guarantee transfer from MultiTUCB to mixed AtLeast/AtMost intervals.
- No cryptographic provenance or source/support verifier.
- No automatic relaxation of hard constraints.
- No scalarization of hard safety or governance requirements.
- No SOTA or performance claim.
