# MACE selective evidence sufficiency and group-conditional abstention v0.1

**Status:** research protocol and output contract; no calibration guarantee or benchmark result is claimed.  
**Tracking:** [#13332](https://github.com/Luminous-Dynamics/symthaea/issues/13332)  
**Parent programme:** [#13284 decision-sufficiency frontier and EVSI](https://github.com/Luminous-Dynamics/symthaea/issues/13284)  
**Schema:** [mace_selective_evidence_sufficiency_v0_1.schema.json](../../data/schemas/mace_selective_evidence_sufficiency_v0_1.schema.json)

## 1. Proposed contribution

MACE should not ask an LLM whether it is confident enough to act. It should estimate whether a particular evidence capsule is sufficient for a particular DecisionNeed, under a fixed decision policy, bounded loss, declared risk target and calibration profile.

The research property is:

> Among decisions that pass deterministic evidence, authority, scope, provenance and lifecycle gates, MACE should accept only those whose downstream decision risk is within a preregistered bound for the applicable calibrated group. If the bound cannot be established, it should acquire more evidence, retrieve a better capsule, abstain or escalate.

This is **Selective Evidence Sufficiency (SES)**. It complements the decision-sufficiency Pareto frontier and expected value of information (EVSI) work.

It is not a universal truth guarantee. A calibration certificate is conditional on its score definition, loss, population, sampling unit, calibration data, group hierarchy, policy, distribution assumptions and time validity.

## 2. Why confidence thresholding is insufficient

A scalar self-reported confidence or retrieval score may rank cases, but does not establish a risk bound. Similarity, support count, HDC match, NLI entailment, source reputation and an LLM's "confidence" are predictors—not certificates.

A global threshold can also hide systematic subgroup failure. A policy may satisfy a marginal risk target while accepting materially riskier cases in one assertion category, language, source channel, decision family or time horizon.

Recent research makes this a timely direction:

- **CAP: Conformalized Abstention Policies for Context-Adaptive Risk Management for LLMs and VLMs** (ACML 2025) learns instance-adaptive risk levels and chooses between point prediction, set-valued output and abstention. The paper reports improvements over static baselines under its experiments while maintaining its declared target-coverage setup. It is a selective-output baseline, not an evidence-lineage or memory benchmark.
- **Hierarchical Group-Conditional Conformal Risk Control for Selective Prediction in Language Models** (arXiv, 2026-07-27) addresses the gap between marginal risk control and per-group risk behavior. It proposes a hierarchy of group-specific thresholds with multiplicity handling and fallback to more general groups when leaf-level calibration is insufficient. Its finite-sample claims rely on the paper's assumptions; empirical zero violations are not proof of universal zero risk.
- **When to Remember, When to Abstain: Category-Conditioned Retention for Reliable Agent Memory** (arXiv, 2026-10-05) studies selective retention by assertion category. On 4,715 candidate assertions from 100 synthetic personas, the paper reports a support asymmetry between value/belief assertions and other assertion categories; category-specific thresholds improved unsupported-retention and coverage trade-offs under its protocol.
- **Do LLMs Know When Evidence is Insufficient? An Evidence Sufficiency Benchmark for Answer-Abstention Calibration in Retrieval-Augmented Generation** (Computers, Materials & Continua, published 2026-08-13) explicitly evaluates abstention under insufficient, irrelevant and contradictory evidence.

Primary references:
- [CAP, PMLR 304](https://proceedings.mlr.press/v304/tayebati26a.html).
- [Hierarchical Group-Conditional CRC](https://arxiv.org/abs/2607.24562).
- [Category-Conditioned Retention](https://arxiv.org/abs/2610.07100).
- [Evidence Insufficiency Benchmark](https://www.techscience.com/cmc/v89n1/68467/html).

These are method references and external protocols. Their scores and guarantees do not transfer automatically to MACE.

## 3. Formal objective

For a candidate-visible decision instance x, define:
- s(x): a frozen sufficiency score derived from candidate-visible features;
- t: a threshold selected on the calibration split only;
- A_t = {x : s(x) >= t}: the accepted set;
- L(x): a preregistered bounded decision loss in [0,1].

Selective risk at threshold t is:

`Risk(t) = E[L(X) | X in A_t]`

The calibrator should choose the most useful threshold whose one-sided upper confidence bound for selective risk is no greater than the preregistered target alpha.

The loss must distinguish:
- wrong action;
- unsupported consequential action;
- use of inadmissible, stale or revoked evidence;
- missed opportunity;
- abstention/escalation;
- acquisition delay and acquisition cost.

Hard evidence constraints are not converted into a soft loss. Known revoked, unauthorized, wrong-scope or unresolved mandatory-lineage evidence fails the deterministic gate even when the sufficiency score is high.

### 3.1 Separate ranking from certification

The score s(x) may be learned or engineered from candidate-visible evidence features. It can rank cases for acceptance, but it is not a guarantee by itself.

Certification uses calibration outcomes and a preregistered statistical procedure. The same calibration rows must not be used to report held-out risk performance. A test set is evaluated once after score construction, threshold selection and group policy are frozen.

### 3.2 Group-conditional policy

Define a hierarchy such as:

1. decision family / decision-risk tier;
2. assertion category: fact, preference, value/belief, instruction, state;
3. source type and ingestion channel;
4. provenance/lineage completeness and contradiction state;
5. temporal horizon and lifecycle status;
6. language and modality;
7. permitted privacy-safe user/tenant/agent strata.

Each group certificate should contain its calibration cluster count, accepted count, empirical selective risk, upper risk bound, coverage, threshold, multiplicity adjustment and status.

Use the most specific certified group that legitimately contains the current case. If a leaf group has insufficient support:
- fall back to a certified parent only if the parent group's scope includes the case and the fallback is recorded;
- otherwise retrieve more evidence, acquire another observation, abstain or escalate.

Never quietly reuse an optimistic global threshold for a group whose risk is not estimable.

## 4. Calibration score design

The score must describe evidence sufficiency, not fluency or answer confidence. Candidate-visible features may include:

- exact DecisionNeed scope and estimand match;
- completeness of canonical evidence references;
- transitive dependency closure completeness;
- independently verified source-root/family count;
- contradiction and source-conflict state;
- temporal freshness and lifecycle status;
- current-at-use revalidation result;
- source authority and rights profile;
- retrieval diversity and raw-source availability;
- agreement between raw excerpts and extracted facts;
- calibration-set-supported evidence-type features.

Potential representation arms include:
- raw evidence + structured facts (Redis Remis/Instruct-style);
- lexical/dense/hybrid retrieval;
- hierarchical abstraction and cue anchors;
- causal evidence graph;
- typed-content HDC and event-ID HDC.

Do not include hidden evaluator outcomes, intervention labels, answer keys or future observations in the score. Do not treat repeated summaries from one common root as independent corroboration.

The calibrator itself should be separately versioned and frozen. Its learned weights, score definition, feature contract and training data must be bound by digests in the calibration manifest.

## 5. Four-way controller: accept, acquire, retrieve, abstain

For each DecisionNeed, the controller should consider:

**Accept/act.** Hard constraints pass and the applicable group certificate demonstrates a risk upper bound no greater than alpha.

**Acquire evidence.** A candidate query has positive EVSI after acquisition, privacy, delay and operational costs, and the observation can arrive before the decision deadline.

**Retrieve/consolidate.** Search existing raw evidence or rebuild a more complete capsule when the evidence may already exist but is missing from the current representation.

**Abstain/escalate.** The hard gate fails, the risk bound exceeds alpha, the group is uncertified, or distribution shift invalidates the certificate.

Expected utility is optimized only among actions allowed by deterministic gates and the certified risk policy. A high retrieval score must never bypass authority or lifecycle checks.

## 6. Statistical assumptions and failure behavior

### 6.1 Dependence unit

Memory interventions, turns, assertions and derived summaries are not independent samples. Split calibration and evaluation by independent world/source/task clusters at the true dependence boundary. Where model-fit seeds are nested, preserve the nesting during resampling.

Do not inflate calibration sample size by counting each event knockout, memory view or intervention row as independent.

### 6.2 Risk bounds

Possible methods include:
- Clopper–Pearson or other exact binomial bounds for an appropriate independent Bernoulli cluster-level violation endpoint;
- Hoeffding or empirical-Bernstein upper bounds for bounded losses under their assumptions;
- cluster-level bootstrap upper bounds as uncertainty summaries, clearly distinguished from finite-sample conformal risk guarantees;
- conformal risk control or a policy-calibrated abstention procedure when its exchangeability assumptions and loss formulation are justified.

Do not label a bootstrap interval a conformal guarantee. Do not claim a distribution-free conditional guarantee from a marginal method.

### 6.3 Multiplicity and group fallback

When certifying many groups, use a preregistered multiplicity correction or a justified hierarchical procedure. Record:
- full group hierarchy digest;
- the number of groups considered;
- correction method and adjusted alpha;
- groups that could not be certified;
- every parent fallback and reason.

If a group lacks enough independent calibration clusters, output an explicit not-estimable status. Never emit a zero-risk estimate or a narrow interval simply because a subgroup has one or two observations.

### 6.4 Distribution shift

Standard exchangeability-based conformal guarantees do not automatically survive arbitrary temporal, source, language, model or task shift.

Monitor:
- score distribution drift;
- source-family and evidence-category mixture changes;
- changing rates of correction, retraction and contradiction;
- changes in task/action mix;
- reader, policy, embedding or memory representation revisions.

On detected or unqualified shift, suspend the affected certificate. Recalibrate with independent data or retrieve/acquire/abstain. Do not continue presenting a stale certificate as current.

## 7. Metrics

Primary endpoints:
- selective risk at a preregistered coverage;
- coverage at a preregistered risk upper bound;
- worst-group risk among certified groups;
- false-sufficiency rate: accepted decisions whose regret or unsupported-action loss exceeds the declared limit;
- hard-gate violation rate among accepted decisions.

Secondary diagnostics:
- risk–coverage and utility–coverage curves;
- area under risk–coverage curve (AURC) and excess AURC against an oracle ranking;
- Brier score, calibration slope, reliability plots and calibration error;
- correct/incorrect abstention rates;
- group sample sizes, coverage and risk bounds;
- parent-fallback frequency;
- not-estimable group count;
- drift detection delay;
- acquisition and verification cost/latency;
- clean-task utility preservation.

ECE and Brier score are descriptive calibration diagnostics; neither alone proves selective risk control.

## 8. Baselines and ablations

Compare:
1. Always act after hard gates.
2. Immediate abstention.
3. Global LLM self-confidence threshold.
4. Global retrieval-similarity threshold.
5. Global source/support-count threshold.
6. Category-conditioned retention threshold.
7. CAP-like context-adaptive abstention.
8. Hierarchical group-conditional conformal risk control.
9. EVSI-only acquisition and random acquisition.
10. Evaluator-only oracle policy as an upper bound, never a deployable baseline.

Ablate raw evidence, structured facts, lineage completeness, contradiction features, lifecycle freshness, assertion category, source type, language, modality and HDC similarity. Keep reader backbone, policy, task split and all-in budget matched where comparisons require it.

## 9. Proposed qualification gates

These are research targets, not achieved results:

| Capability | Proposed gate |
|---|---|
| Hard evidence/authority/lifecycle failures among accepted actions | 0 on deterministic adversarial fixtures |
| Held-out selective risk bound | <= preregistered alpha overall and for every group claimed certified |
| Not-estimable groups hidden from output | 0 |
| Worst-group coverage and false-sufficiency | Always reported |
| Multiplicity and dependence | Preregistered and auditable |
| Matched-risk utility | Positive improvement over strongest simple selective baseline at matched all-in cost |
| Clean-task success loss | <=1 percentage point at locked operating point, unless the risk target requires more abstention |

A certification result must state the population and assumptions it covers. "Certified" does not mean zero risk or universal validity.

## 10. Versioned contract

The companion JSON Schema defines three record types:
- calibration manifest: data/split, score/loss/policy/code digests, calibration method, risk target, cluster unit and evaluator custody;
- group certificate: group path, status, cluster support, threshold, risk upper bound, coverage, multiplicity adjustment and fallback state;
- decision assessment: DecisionNeed/evidence capsule, certificate identity, hard-gate result, sufficiency score, recommended action and risk bound.

It forces the output to distinguish certified, parent-fallback, not-estimable, shifted and hard-gate-rejected states. It is an output contract only; the calibration algorithm and certificate verifier remain unimplemented.

## 11. Implementation sequence

1. Freeze bounded DecisionLoss and SufficiencyScore contracts.
2. Add synthetic sufficient/insufficient/contradictory/unauthorized fixtures crossed with assertion category and source family.
3. Implement a standalone offline calibrator and group-certificate generator.
4. Integrate with #13284's regret frontier and EVSI controller.
5. Add hierarchical group calibration, multiplicity correction and drift-triggered certificate invalidation.
6. Verify with an isolated evaluator under #13287.
7. Reproduce CAP/HG-CRC-style methods and the RAG evidence-insufficiency benchmark on their original protocols.

## 12. Claim boundary

MACE currently has no demonstrated group-conditional selective-risk guarantee, conformal sufficiency certificate or calibrated decision-risk bound. The cited papers' reported results are not MACE results. Their guarantees do not automatically transfer to memory systems, especially when observations share source roots, task clusters or temporal dependence.

The intended contribution is not merely a better confidence score. It is an auditable controller that accepts only when the evidence and the calibration assumptions justify acceptance, otherwise acquiring more evidence or abstaining.
