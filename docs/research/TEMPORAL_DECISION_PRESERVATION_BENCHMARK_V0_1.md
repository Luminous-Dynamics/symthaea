# Temporal Decision-Preservation for Compressed Recurrent Representations (v0.1)

**Tracking:** #13255 — REPRESENTATION-HORIZON-001  
**Single-epoch representation evaluator:** #13239  
**Decision-observability authority:** #13152 / #13165 (deterministic / mixed-policy lanes)  
**Selected-action sufficiency horizon:** #11646 / PR #11650  
**Simultaneous episode-level calibration:** #11655 / PR #11659  
**Joint resource evaluation:** #9666 / #9882 / #13176  
**Memory consolidation and transfer:** #9139  
**Evaluator isolation:** #9437  
**Status:** eleven temporal Rust unit tests authored; exact-head Rust 1.96 formatting, full evaluator tests (65 passed, 0 failed) and strict Clippy passed at `9185feae6fb775436449b8af2a64595b54bb96e1`. No external benchmark or hardware result is claimed.

## 1. Research question

Can a compressed HDC/LTC recurrent representation preserve decision-relevant distinctions across a sequence of future decision epochs, rather than only at the instant of compression?

A representation may preserve the current best action and still merge hidden states that require different actions after a later observation, contact transition, task-phase change or regime shift. A one-epoch guarantee therefore does not establish that a compressed state remains useful for future cognition.

This is a measurement and composition layer. It does not create a candidate compressor, memory allocator, world model, active-sensing policy, scheduler, calibrated loss estimator, safety model or actuator authority.

## 2. Distinguish three different horizons

Three quantities must remain separate.

### 2.1 Representation-preservation horizon

At each predeclared decision epoch \(k\), #13239 compares the full-state and compressed-state information partitions using the existing finite minimax-regret authority.

For each compressed information class \(C\), the epoch is representation-preserving only when:
1. every source full-state class was already epsilon-regret sufficient;
2. the compressed class remains epsilon-regret sufficient; and
3. its compression-induced added regret is within the task-frozen compression budget.

The representation-preservation prefix is:

\[
H_{\mathrm{repr}} =
\max\left\{h:
\text{every epoch } k\in[0,h)
\text{ is representation-preserving}\right\}.
\]

The prefix terminates at the first failed or unqualified epoch. Later success does not repair an earlier failure in this contiguous-prefix measure.

### 2.2 Selected-action decision-sufficiency horizon

#11646 / #11650 evaluates whether one previously selected action remains sufficiently good across future control ticks, consuming externally supplied calibrated expected-loss intervals, robust admissibility evidence and a task-specific regret tolerance.

It produces a selected-action decision-sufficiency horizon. It is not a recurrent-representation compression metric.

#11655 / #11659 owns simultaneous episode-level calibration across the frozen action-by-horizon rectangle. This work reuses those contracts; it adds no second calibration procedure or horizon calculator.

### 2.3 Physical commitment horizon

#11581 combines qualified physical/evidence bounds for a maximum eligible action prefix. It remains separate from both representation preservation and selected-action regret. No representation certificate grants actuation permission.

These quantities can be compared on the same paired episodes, but they must not be renamed, substituted or collapsed into one scalar.

## 3. Executable reference

The evaluator module is:

research/temporal-evaluator/src/temporal_representation_preservation.rs

For one frozen compression profile it consumes a sequence of decision epochs. Each epoch contains exactly one evaluator alternative per hidden world, including:
- stable evaluator world ID and hidden-world digest;
- full recurrent-state identity;
- candidate compressed-state digest;
- candidate-visible observation-history identity;
- frozen model/configuration identity and deterministic policy seed;
- candidate action and evaluator-owned action-loss vector.

The module:
1. requires the declared epoch count and contiguous integer epoch sequence;
2. requires the same hidden-world ID cohort at every epoch;
3. rejects a hidden-world digest change or a dropped counterfactual world;
4. freezes model/configuration identity and deterministic policy seed across the paired temporal profile;
5. allows candidate observation and recurrent-state identities to evolve with time;
6. delegates each epoch to qualify_representation_compression() from #13239;
7. reports the longest contiguous representation-preservation prefix, maximum added regret and first-failure epochs;
8. reports actual candidate-action failure separately from information-preservation failure.

It does not call or reimplement #11650. A companion runner can submit the full and compressed candidate's calibrated expected-loss records to #11650 using the matching #11659 receipt.

## 4. Why the fixed hidden-world cohort matters

A temporally compressed profile must not appear to improve by silently dropping the counterfactual worlds that become difficult later.

The same world IDs and hidden-world digests therefore recur at each epoch. Their candidate-visible observation history and recurrent state may evolve, because those are the temporal quantities under study. The model/configuration and frozen policy seed remain constant so the treatment comparison is not confounded by a changed candidate.

This is a bounded deterministic finite-world qualification. It does not prove that the supplied world cohort exhausts the real environment's possibilities; support completeness and candidate/evaluator isolation remain independent obligations.

## 5. Authored controls

Eleven Rust unit tests cover:

1. Identity compression preserves every epoch.
2. A harmless current compression collision can become a harmful future collision when the hidden worlds later require opposed actions.
3. Candidate-action failure is reported separately from representation-information loss.
4. Non-contiguous decision epochs are rejected.
5. Dropping a hidden world at a later epoch is rejected.
6. Model/configuration identity changes are rejected.
7. Policy-seed changes are rejected.
8. A supplied epoch count that differs from the frozen contract is rejected.
9. Duplicate hidden-world digests under different world IDs are rejected.
10. Recovery after a failed epoch does not extend the original contiguous preservation prefix.
11. A hidden-world digest change under a stable world ID is rejected.

The machine-readable fixture includes stable-representation, future-collision and candidate-action-failure known-answer cases, plus adversarial scenario families. The Rust 1.96 exact-head workflow has passed changed-module formatting, whole-crate tests and strict Clippy.

Exact-head Rust 1.96 verification at `9185feae6fb775436449b8af2a64595b54bb96e1` (workflow run `45085999687`):

- Changed-module rustfmt check: passed.
- Full evaluator crate test suite: 54 passed, 0 failed.
- Strict Clippy (`-D warnings`): passed.

The authoring environment itself still has no local Rust toolchain. This finite test run does not establish hardware performance, external benchmark performance, calibrated real-world support, or deployment qualification.

## 6. External prior art and fair comparisons

Decision-aware state abstraction is not a new concept. Required comparators include:

- **Online Policy-Wise State Abstraction for Reinforcement Learning Agents Leveraging Free Energy** (IEEE TNNLS, online September 30, 2026; DOI 10.1109/TNNLS.2026.3735939). It uses parallel full-state and adaptively masked learners plus free-energy-based policy selection. Compare a faithful candidate implementation; do not claim the present evaluator invents online abstraction.
- **Boosting World Models Learning via Latent-Space Value Alignment** (ICML 2026, PMLR 306, 52832–52862). It aligns latent world-model representations with task value while retaining dynamics structure. Compare its representation-learning method separately from the finite evaluator.
- **Predictive State Representations** and approximate bisimulation/state aggregation are established foundations for predictive/sufficient state and decision-preserving abstraction.
- **ConsolidationBench / governed memory consolidation**, tracked in #9139, is a direct memory-compression/transfer comparator. Compression ratio, reconstruction quality, downstream transfer, provenance, reversibility and poison resistance are separate endpoints.
- **RoboDyna** provides paired synchronous/asynchronous dynamic-manipulation evaluation and reports that inference latency flips 5–12% of episode outcomes in its evaluated tasks.
- The November 16, 2026 **Long-horizon embodied manipulation survey** identifies partial completion, error propagation, recovery cost and horizon-dependent degradation as important omissions from binary success-only evaluation.

External paper numbers remain tied to their original tasks, hardware, data and protocols. Do not compare them as if they were a shared leaderboard.

## 7. Matched experiment matrix

Each candidate profile receives the same frozen episode cohort, observation prefixes, hidden-world commitments, action/loss contracts, policy seed, deadlines and resource ceilings.

| Arm | Purpose |
|---|---|
| Full HDC/LTC recurrent state | Reference representation |
| Identity mapping | Exact evaluator/control path |
| PCA | Linear low-rank baseline |
| Random projection | Dimension-matched information-loss control |
| Fixed quantization | Precision/bit-budget baseline |
| Value-aligned world model | Task-value representation comparator |
| Online policy-wise abstraction | Free-energy/parallel-learner comparator |
| Candidate decision-aware compression | Symthaea research treatment |
| Evaluator full-information oracle | Non-deployable diagnostic upper reference |

Use paired hidden-world seeds and held-out environment/source families. Keep train, representation-selection, calibration and confirmatory splits disjoint. The episode—not a correlated decision epoch—is the independent statistical unit.

## 8. Metrics — no composite score

Report separately:

### Decision and representation
- per-epoch compression-induced minimax-regret increase;
- maximum added regret over the temporal profile;
- percentage of epochs that remain epsilon-sufficient;
- representation-preservation prefix and first failed epoch;
- selected-action decision-sufficiency horizon from #11650;
- calibrated interval coverage and unknown-evidence rate from #11659;
- candidate-action regret and action disagreement;
- wrong-belief, abstention and missing-output exposure.

### Representation quality
- reconstruction error and cosine similarity;
- representation dimension and precision;
- information recovery after refresh or re-observation;
- irreversible information loss and source-lineage preservation.

### Resources and physical timing
- persistent bytes and peak memory;
- retained byte-time;
- read/write traffic and checkpoint/refresh overhead;
- native compute work, CPU/GPU wall-clock and p50/p95/p99 end-to-end latency;
- measured energy at an explicit hardware boundary;
- observation age at decision and actuator consumption.

### Task outcomes
- task success and verified partial completion;
- failure localization and recovery cost;
- missed deadlines, stale proposals and adverse outcomes;
- worst held-out family and regime, not just a pooled mean.

A high cosine similarity is not proof of decision preservation. A low reconstruction score is not necessarily a failure if all relevant decisions remain within the declared envelope.

## 9. Adversarial scenario families

1. Current decision preserved; a later observation separates worlds and requires different actions.
2. Current decision preserved; a later contact, phase or regime transition makes discarded detail decision-critical.
3. Early compression collision followed by a delayed decision consequence.
4. Reversible compression/refresh recovers a needed distinction before its deadline.
5. Irreversible information loss: refresh cannot reconstruct a discarded source distinction.
6. Stable worlds where one compressed representation remains sufficient throughout the profile.
7. Low-confidence but epsilon-sufficient states where added detail has no decision value.
8. High reconstruction/cosine similarity but high decision regret.
9. Poor reconstruction similarity but identical decision envelope.
10. Future-event, event-ID, row-order, hidden-world-label and policy-seed leakage controls.
11. Irregular, missing and delayed observations plus bitemporal revisions.
12. Long-horizon drift where one early merge causes repeated later wrong, abstaining or missing decisions.

## 10. Prospective qualification targets (not results)

- Zero false preservation certificates on exhaustive finite counterexamples.
- Identity compression exactly reproduces the full-state partition at every epoch.
- Zero accepted future-information or nuisance-metadata leakage in the frozen attack suite.
- No compression-induced added regret above the preregistered task tolerance on qualified held-out episodes.
- Demonstrate at least one workload where lossy compression preserves the decision envelope and reduces measured resources.
- Demonstrate at least one adversarial workload where compression is rejected.
- Publish regimes where full-state retention or a simpler baseline wins.

Any memory or energy saving must be measured on the actual candidate execution path. Vector dimensionality, serialized digest size, operation count and modeled energy are not interchangeable physical resource measurements.

## 11. Claim boundary

A passing finite evaluator qualifies only the declared hidden-world/action/loss set, temporal cohort identity, compression mapping and deterministic candidate policy. It does not prove generalization, learned-compressor quality, calibration exchangeability, physical safety or measured resource savings.

Confirmatory evaluation requires #9437 isolation, #11659 calibration evidence for any selected-action horizon claims, paired episode-level experiments, held-out families and source-bound resource receipts. This protocol does not authorize actions or alter any production cognitive path.
