# MACE Decision-Value Retrieval V1

**Snapshot:** 2026-11-26  
**Status:** Decision-value evaluator prototype and authored deterministic fixtures published on `mace/decision-value-retrieval-v1`. It is a scoring library, not an autonomous tool executor, calibration service, resource meter or strategy-promotion path. Rust tests and CI have not been executed here.

## 1. The retrieval objective changes

Similarity retrieval asks: *which stored item is most similar to this query?*

Decision-value retrieval asks: *which available evidence-gathering operation is most likely to improve the decision enough to justify its cost and risk?*

These are different optimization problems. A highly similar memory can be irrelevant to the action choice. A low-similarity verification query can be valuable if it resolves a missing safety-critical requirement. Conversely, retrieving more context can be wasteful or harmful when it is stale, correlated, already redundant, or incapable of changing the decision.

The evaluator treats retrieval as an optional information-acquisition action. It can recommend the next operation, or recommend stopping when no candidate has positive net value.

## 2. Two mathematically separate modes

MACE implements two scoring modes and never averages them together.

### A. EVSI — calibrated expected value of sample information

When defensible calibrated probabilities exist:

[
operatorname{EVSI}(q)=
sum_o P(omid q)max_a mathbb{E}[U(a,S)mid o,q]
-max_a mathbb{E}[U(a,S)]
]

[
operatorname{NetEVSI}(q)=operatorname{EVSI}(q)-C(q)
]

where:
- (S) is the consumer-defined state of the world;
- (a) is a consumer-permitted decision action;
- (o) is the possible result of retrieval/verification operation (q);
- (U(a,S)) is the consumer-owned utility table;
- (C(q)) is acquisition cost expressed in the same utility units.

The prototype uses fixed-point probabilities (parts per million), avoiding floating-point nondeterminism in the score and its receipts. The calibration authority must sign a digest covering the exact prior, query-specific observation likelihoods, state/action definitions and utility profile. Each query option carries its own calibration receipt; a calibration for one query model cannot be reused for another.

If that receipt is absent, expired, revoked, bound to a different DecisionNeed, or does not match the exact probability model, EVSI evaluation fails closed.

### B. Robust value — probability-free minimax regret reduction

When probabilities are not defensible, MACE must not invent them. The alternative is a conservative support-set calculation.

For each state (s), define its best consumer-permitted utility (U^*(s)) and action regret:

[
R(a,s)=U^*(s)-U(a,s)
]

Before the query, minimax regret is:

[
R_0=min_amax_s R(a,s)
]

After query (q), each observation (o) is compatible with a consumer-declared set of possible states (S(o)):

[
R_q=max_omin_amax_{sin S(o)}R(a,s)
]

[
operatorname{RobustNetValue}(q)=R_0-R_q-C(q)
]

The support matrix is independent of the calibrated likelihood matrix. It must be conservative: a state/outcome pair is marked possible unless there is defensible evidence that it cannot occur. An observation that is possible in several states must be evaluated against all of them.

This mode may be less optimistic than EVSI, but it does not rely on a fabricated prior or posterior. The two outputs are reported separately.

## 3. Hard action constraints are not utility penalties

The consumer policy includes an explicit state-by-action admissibility matrix. A high expected utility cannot override a hard action prohibition.

For EVSI, an action may be selected after an observation only if it is permitted in every state with positive posterior mass. For robust minimax, an action must be permitted in every state compatible with that observation. If no action remains permitted, evaluation fails closed with `NoPermittedAction`.

A production DecisionNeed should include an explicit safe abstention/deferral action whenever the domain permits it. The retrieval evaluator is not an action authorizer and does not issue an execution permit.

## 4. Independent calibration and resource receipts

The module defines two verifier interfaces:

- `DecisionCalibrationReceiptVerifierV1` verifies the DecisionNeed digest, calibration profile, signed calibration receipt, exact probability-model digest and current time.
- `DecisionResourceReceiptVerifierV1` verifies the DecisionNeed digest, query option identity, action-template digest, resource-profile digest, signed resource receipt, tool-call count, latency, context tokens, inspected bytes and utility cost.

Resource fields are not trusted merely because the candidate retrieval system supplies them. The consumer must use an independent resource profiler/meter and a separately provisioned trust profile. The option's resource receipt is required for both EVSI and robust minimax scoring.

The evaluator also checks the consumer-owned tool-call, latency, context-token, inspected-byte and acquisition-cost ceilings. Options that exceed a limit are ineligible. It selects only the best positive-net-value option; if none qualifies, the result is `None` (stop/abstain from further retrieval).

The current verifier traits are interfaces. No calibration authority, resource-meter service or signing-key custody is deployed by this PR. Test verifier implementations are fixtures only.

## 5. Rust implementation

Path: `research/broca-mace-transfer-bench-v0-1/aml-adapter/src/decision_value_retrieval.rs`

The module provides:
- consumer-owned finite state/action/utility policy;
- hard action-admissibility constraints;
- query-specific observation classes and conditional likelihoods;
- a probability-free observation-support matrix;
- exact fixed-point EVSI scoring;
- worst-case minimax-regret scoring;
- independent calibration/resource verifier interfaces;
- budget-aware best-query selectors;
- deterministic tie-breaking through sorted state/action/observation identifiers;
- explicit stop when all options have non-positive net value.

The module does not execute the query. It returns a candidate choice and the scoring breakdown so that a separate, independently authorized executor can consume it.

The prototype does not yet model correlated query observations across several sequential retrieval calls, adaptive policies over multiple rounds, distribution shift after each query, dynamic resource prices, or a full causal decision graph. Those are subsequent research increments.

## 6. Relationship to existing MACE strategy learning

The existing retrieval-policy memory deliberately does not promote strategies from acquisition traces alone. Strategy promotion requires separate decision/outcome evidence and held-out counterfactual qualification.

This module preserves that boundary. A positive EVSI or robust value is a *candidate retrieval assessment*, not evidence that the operation actually improved a decision. A reusable retrieval strategy still needs:
- a qualified and independently observed outcome receipt;
- before/after decision utility and risk;
- held-out counterfactual comparison against the current policy;
- all-in measured retrieval/evaluation cost;
- calibration validity and drift monitoring;
- independent safety and governance checks.

Do not silently convert an evaluation score into a reinforcement-learning reward or strategy-promotion event.

## 7. Deterministic fixtures

The new module includes fixtures for:
1. informative query with positive EVSI after cost;
2. uninformative query with positive cost;
3. rejected calibration receipt;
4. malformed prior probability mass;
5. query-specific calibration binding to exact probabilities/likelihoods;
6. probability-free robust minimax regret reduction;
7. uninformative robust query;
8. robust evaluation with no calibrated likelihood matrix;
9. query budget exhaustion in both modes;
10. stale calibration receipt;
11. hard action prohibitions overriding utility;
12. no permitted action for a possible state;
13. highest-positive-EVSI candidate selection and stop when none is positive;
14. duplicate query-option identity rejection;
15. rejected resource receipt in both expected and robust modes;
16. robust regret computed only against permitted actions.
17. resource-receipt verifier rejects a changed validity-expiry timestamp.
18. equal-value EVSI and robust selection is invariant to candidate input order.

These test functions are authored but not executed. Cargo check/test, rustfmt and Clippy are not claimed.

## 8. Benchmark programme

External retrieval and memory benchmarks should be evaluated unchanged; decision value is a separate scorecard.

### Required retrieval baselines

- no-memory / no-additional-retrieval;
- exact-match and BM25;
- embedding similarity;
- BM25 + dense reciprocal-rank fusion;
- graph/causal retrieval;
- full-context baseline at the same total token budget;
- fixed top-k retrieval;
- information-gain/entropy-reduction retrieval (reported separately from decision value);
- EVSI selector with verified calibration;
- robust minimax-regret selector without probabilities.

### Required task families

1. **Decision-flipping evidence:** candidate evidence changes the optimal action when it is correct, and should not when irrelevant.
2. **Redundant retrieval:** additional sources share roots/common causes and should have low marginal value.
3. **Missing critical requirement:** query value comes from resolving one safety-critical need, not from similarity.
4. **Noisy/shifted calibration:** calibration error is injected and stale probability receipts are rejected.
5. **Unknown probability:** remove the prior and test robust support-set minimax regret.
6. **Misleading observation:** the retrieval result can be ambiguous or uninformative; the evaluator must not overstate value.
7. **Budget trade-off:** vary cost, latency, tokens, tool calls and inspected bytes independently.
8. **Action prohibitions:** add forbidden actions and verify no utility score can override them.
9. **Sequential retrieval:** evaluate marginal net value after each observation; stop when no positive option remains.
10. **Outcome qualification:** compare the predicted value against independently measured post-query decision quality and held-out counterfactuals.

### Proposed metrics

| Metric | Initial target |
|---|---|
| Net decision regret | ≥10% lower than strongest matched fixed-top-k/BM25/dense/full-context baseline |
| Unsafe action rate | ≤1% at ≥80% supported coverage on in-scope tasks |
| Calibration coverage | Nominal 95% risk bounds attain declared coverage, with subgroup and drift reporting |
| Retrieval efficiency | Improve the Pareto frontier of regret reduction vs all-in query cost/latency/tokens |
| Stop correctness | Zero additional retrieval when every eligible candidate has non-positive net value in deterministic fixtures |
| Hard-constraint violation | Zero on exhaustive policy fixtures |
| Probability-free correctness | 100% minimax-regret fixture agreement with an independent reference implementation |
| Calibration binding | Zero accepted model/receipt mismatch |
| Resource binding | Zero accepted resource-profile/receipt mismatch |
| Strategy promotion | Zero promotions without independent outcomes and held-out counterfactual qualification |

Use paired scenarios/seeds, cluster by source family and decision scenario, publish full selective-risk and cost curves, and report abstentions. Never claim EVSI superiority using a non-matched utility table, unverified probability model or candidate-authored costs.

## 9. External benchmark positioning

The current AML Cycle 2 textual leaders have a 13.46-point Streaming gap and 9.29-point Temporal & Event Sequencing gap; personalization/care and epistemic safety/privacy gaps are 15.88 and 13.39 points. These results justify continuous update, temporal consistency and safe personalization, but they do not measure EVSI directly.

MACE should test whether decision-value retrieval improves matched results on AML and whether improved evidence acquisition produces downstream gains on AMA-Bench causal/state-update tasks and MemoryArena interdependent agent-environment tasks. Memora/FAMA tests whether it avoids stale-memory reuse; MEMOBench and MEMORA-Bench test embodied memory and planning. Do not average these incomparable benchmarks together.

No MACE external benchmark result or SOTA victory is claimed by this module.

## 10. Production work remaining

1. Implement independent calibration and resource-profile issuers, key rotation and revocation.
2. Bind the canonical DecisionNeed and utility profile to the same consumer-owned policy registry used by sufficiency evaluation.
3. Connect candidate retrieval operations to the actual acquisition tools and exact operation receipts.
4. Collect independently observed outcomes and paired counterfactuals.
5. Add sequential/adaptive query policies with correlated observation models.
6. Add calibration drift detection, confidence intervals and subgroup coverage.
7. Integrate the evaluator with held-out strategy promotion only after outcome receipts and the existing promotion gates pass.

The current work provides a deterministic scoring kernel and its verifier interfaces. It is not yet a learned policy, production retrieval service or action authorization system.
