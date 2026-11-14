# AOBench — Model-Class Adequacy and Held-Out Refinement

**Status:** Initial deterministic evaluator committed; exact-head Rust qualification pending  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-012 #12200](https://github.com/Luminous-Dynamics/symthaea/issues/12200)  
**Implementation:** \`crates/domains/symthaea-organizational-closure/src/model_refinement.rs\`

## 1. Paradigm shift: do not assume the hypothesis set is complete

The sequential experiment evaluator in AOBENCH-011 asks which target-equivalence class remains after qualified evidence arrives. This new evaluator asks whether the supplied model class is adequate at all.

A scientific agent must distinguish:
- **not falsified:** one or more supplied models still explain the complete qualified trace;
- **falsified:** no supplied model explains the complete qualified trace;
- **unresolved:** delivery, isolation, observation or oracle evidence is not qualified;
- **refined and validated:** a new model explains discovery evidence and predicts sealed probes correctly.

This avoids a common failure of closed-world benchmarks: forcing an answer because the test set guaranteed that one candidate must be right.

## 2. Separation of agent and evaluator information

The agent-facing scenario contains:
- known mechanism hypotheses and their deterministic response tables;
- declared probe IDs and cost/risk units;
- a sealed validation-probe set;
- cost, risk and trial budgets;
- discovery observations with separate delivery, isolation and observation dispositions.

The evaluator-only oracle contains:
- the actual target-equivalence class;
- the actual deterministic outcome for every declared probe.

The oracle is supplied separately and is never part of the agent-facing scenario. Validation probes cannot be run during discovery.

## 3. Whole-trace consistency

A supplied model is consistent only if it predicts every qualified observation in the complete discovery trace.

It is not enough that each observation can be explained by a different model. If one probe supports model A while another supports model B, and no single supplied model explains the joint trace, the original model class is falsified.

Observations only enter this consistency calculation when intervention delivery, isolation and observation evidence are all verified. A rejected or unavailable gate never becomes evidence for model inadequacy.

A qualified observation that contradicts the independent world oracle is classified as an observation-integrity mismatch and leaves model adequacy unresolved. It is not counted as a falsified hypothesis class.

## 4. Refinement contract

A proposed new model must provide:
- a new hypothesis ID;
- a target-equivalence class;
- a positive prior weight;
- a complete response prediction for every declared probe.

The evaluator accepts the proposal only when:
1. no original model explains the complete qualified discovery trace;
2. the new model explains every qualified discovery observation;
3. the proposed target-equivalence class equals the evaluator-only target class;
4. every sealed validation probe is predicted correctly;
5. the proposal does not reuse an existing hypothesis ID;
6. the discovery evidence and budgets are valid.

A new ID alone does not make a new model. A proposal that only memorizes the observed probe but fails a sealed probe is rejected.

## 5. Decision rules

| Agent decision | Evaluator result |
|---|---|
| Retain existing class while at least one model fits | Qualified; class is not falsified (not proof of truth) |
| Retain existing class when no model fits | Not qualified |
| Expand class when an existing model still fits | Not qualified; unsupported expansion |
| Expand after falsification and pass all sealed probes | Qualified; refined and validated |
| Defer after evidence falsifies the class | Inconclusive; safe defer, not a validated refinement |
| Defer without evidence of model inadequacy | Not qualified |
| Any unqualified intervention/observation gate | Inconclusive; model adequacy cannot be assessed |

The evaluator never promotes a safe defer into a successful scientific explanation.

## 6. Metrics

Current report fields include:
- consistent original model IDs;
- qualified and total discovery-trial counts;
- total cost and risk;
- model-inadequacy detection;
- validation-probe count and coverage;
- validation accuracy;
- safe-defer disposition;
- validated-refinement disposition.

Recommended benchmark-level aggregates:
- model-inadequacy detection precision and recall;
- false expansion rate when the original class remains consistent;
- accepted refinement accuracy and complete validation coverage;
- discovery-fit / validation-generalization gap;
- correct-defer rate when no candidate is defensible;
- observer-integrity false attribution;
- model-size and proposal-count costs;
- family-disjoint transfer performance.

A failed sealed validation set cannot be hidden inside a composite average.

## 7. Deterministic controls implemented

Ten deterministic tests cover:
- a new model explaining discovery and sealed validation probes;
- fit-only overfitting that fails a holdout;
- retaining a known model when it explains the full trace;
- unnecessary model expansion;
- unavailable delivery evidence;
- observation disagreement with the independent world oracle;
- joint-trace falsification when different observations are explained by different models;
- independent risk-budget enforcement;
- malformed/incomplete known-model predictions;
- attempted use of sealed validation probes during discovery.

These are unit tests, not yet a mutation campaign. AOBENCH-009 should mutate whole-trace filtering, validation coverage, target-class checks, oracle isolation and evidence-gate logic, pairing each mutant with a condition-removed positive twin.

## 8. Next qualification work

1. Exact-head Rust 1.96 format, compile, unit tests and strict Clippy.
2. Source-level mutation campaign for model-consistency and holdout-validation guards.
3. Scenario-family generator with mechanism-family-disjoint holdouts.
4. Compare closed-world MAP, always-abstain, always-expand, fit-only refinement, nearest-model and oracle-refinement baselines.
5. Extend from deterministic categorical outcomes to stochastic models only with preregistered power, calibration and sequential-valid uncertainty bounds.
6. Bind observation dispositions to authenticated #12014 receipts and #12067 intervention-delivery/observer qualification.

## Claim boundary

This is a deterministic finite response-table evaluator for model-class adequacy and predictive refinement. It does not establish open-ended scientific discovery, physical-world truth, stochastic equivalence or causal identification outside the declared model class. No SOTA result or superiority claim is made.
