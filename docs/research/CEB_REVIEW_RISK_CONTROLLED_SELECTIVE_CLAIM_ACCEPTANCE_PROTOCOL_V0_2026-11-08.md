# CEB Review Risk-Controlled Selective Claim Acceptance Protocol V0 — 2026-11-08

**Status:** Public research design; no calibrated CEB policy, model evaluation, task admission or production acceptance rule is claimed.
**Composes with:** CEB sequential evidence acquisition, causal identifiability, dual-annotator adjudication, preservation-aware repair verification and the existing human-escalation policy.
**Primary measures:** unchanged — defect_recall_at_5, actionable_precision, reviewer_burden.
**Execution boundary:** candidate-controlled execution remains blocked by #10064; independent pristine regrade remains blocked by #10369.

## 1. Paradigm shift: from confidence scores to risk-controlled acceptance

A model's verbal confidence, self-consistency or agreement with another model is not a probability guarantee that a software finding is correct. The decision that matters operationally is conditional:

> Among the claims this policy accepts without human adjudication, what fraction are wrong, and at what useful coverage?

CEB should study a **selective review policy** with four dispositions:

- **accept for normal review presentation:** the claim meets a frozen evidence/authority contract and the calibrated acceptance rule;
- **gather discriminating evidence:** one authorized, low-cost evidence action can separate material alternatives;
- **human escalation:** uncertainty, severity or claim authority requires a qualified reviewer;
- **abstain / unresolved:** the available evidence cannot support the claim within the declared scope.

These are dispositions within the existing sequential evidence and escalation policies, not a new scheduler or authority class.

## 2. False-discovery risk is not false-positive rate

For an evaluated set of candidate claims:

- **False-positive rate (FPR):** among truly false claims, the fraction accepted.
- **False-discovery rate (FDR):** among accepted claims, the fraction that are false.
- **Coverage:** among eligible candidate claims, the fraction accepted automatically.
- **Recall preservation:** whether the selective policy still surfaces true target defects or appropriately routes them for evidence/human review.

FDR is the operationally relevant quantity when developers receive only accepted findings. FPR is also important, but it answers a different question. Do not substitute one for the other or report one as the other.

A low FDR can be achieved by accepting almost nothing. Therefore, every risk statement must be paired with coverage, absolute counts, severity distribution, missed-target analysis, reviewer burden and all-in cost.

## 3. Calibration unit and population definition

The calibration unit is a **unique adjudicated claim**, bound to its source revision, originating defect/repository family, severity and claim kind. Multiple prompts, model retries, paraphrases, counterfactual variants or repeated comments about one defect do not create independent calibration examples.

Record at least:

- repository and defect-family cluster;
- source tree/revision and build-world where applicable;
- claim kind: source possibility, build selection, execution realization, actual-cause attribution, interventional-effect identifiability or incident attribution;
- severity and protected invariant;
- positive/negative/adjudicated-unresolved truth;
- candidate model/harness/prompt/tool versions;
- evidence authority and preservation/target witness status;
- whether the finding was accepted, escalated, evidence-routed or abstained.

If clusters are dependent, either use a conservative independent cluster as the calibration unit or use a statistical procedure whose assumptions explicitly handle clustering. Do not count sibling variants as independent n.

## 4. Three disjoint data roles

Use separate data for three functions:

1. **Development / policy selection:** choose candidate uncertainty features, evidence gates, action rules and thresholds. This set may be used repeatedly and provides no final risk guarantee.
2. **Calibration / risk certification:** evaluate a frozen policy and certify a selected threshold on a disjoint, representative set. Do not tune repeatedly on this same set and then quote a fixed-threshold guarantee.
3. **Confirmatory evaluation:** estimate risk, coverage, recall and burden on a repository-/defect-family-disjoint test population that remains untouched until the policy and analysis are frozen.

The calibration population must represent the claim types, severities, languages, repository conditions and evidence authorities to which the policy will be applied. If it does not, limit the certified scope accordingly.

## 5. Risk-certification rule

For a fixed threshold selected before certification, let:

- n = number of independently adjudicated calibration claims accepted by the frozen policy;
- k = number of those accepted claims adjudicated false;
- delta = one-sided error probability for the risk bound.

For independent/exchangeable Bernoulli error indicators, use a one-sided Clopper–Pearson upper confidence bound:

\[
U(k,n,\delta) = \operatorname{Beta}^{-1}(1-\delta;\ k+1,\ n-k)
\]

with the standard edge-case definitions. For k=0:

\[
U(0,n,\delta)=1-\delta^{1/n}.
\]

Certify only when U is at or below the predeclared risk target. If the risk target is not met, lower the acceptance threshold using a new development/calibration cycle, gather more independent calibration claims, or keep that scope in human review. Do not relabel the calibration examples or change their denominator after seeing the result.

### 5.1 Why sample size matters

With zero false discoveries and a one-sided 95% confidence bound:

- **59 accepted independent calibration claims** are needed for an upper bound below 5%;
- **299 accepted independent calibration claims** are needed for an upper bound below 1%.

These are sample-size illustrations under the stated independent/exchangeable Bernoulli assumption. They do not establish risk control for clustered, shifted or selectively sampled production claims. When the needed sample is unavailable, the correct status is **uncertified**, not “probably safe.”

### 5.2 Multiple claim groups

Calibrate and report separately for security-critical findings, ordinary correctness findings, source-only structural claims, execution-backed findings and incident-attribution claims where their error profiles or authorities differ.

If certifying multiple groups simultaneously, predeclare how the error budget is allocated (for example, Bonferroni alpha splitting or a valid joint procedure). Do not pool a small high-risk group into a large low-risk group to make its average look acceptable. If a group lacks enough independent examples, leave it uncertified and route it to human review.

## 6. What can contribute to an uncertainty/risk score

Potential features for a development-only selective policy include:

- presence and strength of exact source/contract anchors;
- whether the causal claim kind matches the evidence authority;
- number and materiality of evidence-compatible alternative models;
- existence of a concrete disagreement witness;
- target-witness and preservation-witness completeness;
- independent reviewer/judge disagreement, measured on matched claims;
- retrieval/evidence sufficiency and provenance quality;
- known prompt, model, repository, language and severity strata;
- unresolved oracle coverage or build/configuration ambiguity.

Do not treat chain-of-thought length, verbal confidence, token probability, self-reported certainty or model consensus as correctness evidence. Features and policy must be calibrated against adjudicated outcomes, not chosen because they correlate with confidence.

## 7. Compose with the existing sequential evidence policy

The new selective disposition must reuse the existing CEB evidence-action ladder:

1. accept only when the claim/authority/scope contract and calibrated policy both pass;
2. otherwise choose the smallest authorized observation that discriminates the unresolved models;
3. re-evaluate using that evidence and the frozen claim rubric;
4. escalate when the unresolved claim is critical, evidence is unsafe/unavailable, or the risk rule remains uncertified;
5. preserve an unresolved/abstain outcome rather than forcing a binary verdict.

No new action scheduler, receipt or evidence-authority class is introduced. The action selected must remain an existing, permitted evidence action under the applicable execution gate.

## 8. Development comparison arms

On identical development claims and matched budgets, compare:

- **S0 — accept-all baseline:** present all candidate claims without selective filtering;
- **S1 — verbal-confidence threshold:** use the model's self-reported confidence (a deliberately weak baseline);
- **S2 — disagreement/evidence route:** use independent judge disagreement or one discriminating evidence action, then human escalation as needed;
- **S3 — calibrated selective policy:** freeze a risk score and threshold on disjoint calibration material; accept only when the one-sided upper risk bound meets the predeclared target.

This is an exploratory development comparison. It does not create a confirmatory CEB Phase A/B/C hypothesis or a new primary metric.

## 9. Required reporting

Report the following separately, with absolute numerators and denominators:

- accepted false-discovery count and proportion;
- one-sided risk upper bound, delta, certification unit and assumptions;
- coverage of eligible claims;
- true target-defect recall and misses after abstention/routing;
- per-severity and per-claim-kind risk/coverage;
- repository/defect-family cluster composition;
- abstention, evidence-action and escalation rates;
- reviewer minutes, model/tool/judge cost and evidence-acquisition cost;
- calibration drift and performance on temporal/repository/language shifts.

A single overall FDR can conceal an unsafe group. If any critical group exceeds its predeclared bound or lacks adequate calibration, disable automatic acceptance for that group and escalate.

## 10. Prospective research targets

These are design aspirations, not observed performance or current product guarantees:

- ordinary accepted claims: one-sided 95% upper FDR bound at or below 5%;
- critical security/authorization claims: one-sided 95% upper FDR bound at or below 1%, or no automatic acceptance where the independent calibration sample is inadequate;
- at least 70% coverage on eligible non-critical claims once the risk target is satisfied;
- no material reduction in CEB defect recall/actionable precision and no increase in critical misses;
- calibration and reporting by claim kind, severity and repository/defect family;
- all-in cost and reviewer burden reported alongside coverage.

The risk threshold takes precedence over the coverage target. Do not raise risk to meet a coverage aspiration.

## 11. Failure modes and claim boundary

- Exchangeability can fail when repositories, languages, models, tooling or defect families shift.
- A selected threshold can overfit if it is tuned and certified on the same data without a valid adaptive guarantee.
- A model judge can share systematic errors with the candidate; agreement is not independent truth.
- Incomplete gold or missed open-set discoveries can underestimate the denominator of false findings.
- Multiple findings within one PR/repository can be dependent.
- A low-risk average cannot justify automatic handling of an uncertified critical subgroup.
- A source-only claim cannot be risk-certified as an incident attribution claim by reusing the same label.
- A model's own tests and proposed fix cannot be the sole oracle for certifying its own finding.

This protocol creates no new task schema, receipt, metric lane, grader, oracle authority, hidden holdout or execution permission. Candidate-controlled execution remains gated by #10064 and independent pristine regrade by #10369. No risk guarantee is claimed until the calibration population, statistical owner, analysis and independent adjudication are qualified.
