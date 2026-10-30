# SCI-REFERENCE-ASSURANCE-001 — Qualifying reference decisions

**Status:** protocol proposal under development. This document is a research specification, not an implemented validator, empirical result, or scoring authorization.

**Tracking:** [Issue #9081](https://github.com/Luminous-Dynamics/symthaea/issues/9081). This work composes with [#9065 — judge robustness](https://github.com/Luminous-Dynamics/symthaea/pull/9065), [#9005 — evidence behavior contracts](https://github.com/Luminous-Dynamics/symthaea/pull/9005), and [#8993 — paired outcome analysis](https://github.com/Luminous-Dynamics/symthaea/pull/8993).

## 1. The missing measurement layer

A judge audit can bind a reference decision by digest and measure whether a judge agrees with it. That is necessary, but it does not establish that the reference was produced independently, was supported by admissible evidence, used the right scope and time interval, or represented genuine uncertainty.

A separate confidentiality hazard applies to low-entropy labels: the current #9065 implementation hashes one of four short decision strings directly. That digest is trivially dictionary-recoverable to anyone who can read the plan. It is a binding commitment, not a hiding commitment. Either enforce and independently evidence evaluator-only plan access, or use a high-entropy per-case/per-condition nonce in a domain-separated commitment and reveal the nonce only after judge execution. Never claim a precommitted label is hidden merely because it is hashed. This issue is tracked in the review discussion on #9065.

A hash establishes byte identity. It does not establish:
- who produced the label or whether annotators were independent;
- whether a cited source exists, is authentic, or is independent of another cited source;
- whether the cited span actually supports the claim;
- whether the source applies to the relevant jurisdiction, population, unit, version or time;
- whether contrary evidence or dissenting adjudicators were omitted;
- whether a reference package was hidden from the evaluated system and judge;
- whether the reference is correct outside the fixed adjudication packet.

The reference itself must therefore become an object of measurement. The construct is not "ground truth" in the absolute sense. The target is **reference validity under a declared evidence, scope, adjudication and custody protocol**.

## 2. Core principles

1. **Atomicity before scoring.** Split compound answers into material claims and criteria. A correct sentence must not hide an unsupported clause.
2. **Evidence before consensus.** Record source snapshots, exact spans, scope and evidence relations before a reference label is finalized.
3. **Independent first judgments.** Collect blinded initial adjudications separately. Do not expose model identity, candidate identity, judge output or other raters' labels during the initial pass.
4. **Disagreement is data.** Preserve every initial vote, abstention, confidence representation and evidence link. A later resolution is an additional event, never an overwrite.
5. **Uncertainty is a valid outcome.** A forced binary label is not appropriate when sources conflict, evidence is missing, or the question is underspecified.
6. **Source count is not source independence.** Mirrored, syndicated or commonly derived sources must be linked to a shared ancestry group.
7. **Scope and time are first-class.** A source may support a claim in one setting or time interval and not another.
8. **Cryptographic binding is not external truth.** Hashes and signatures prove specific byte/identity claims only within their trust model. Qualified use requires independent custody and execution evidence.
9. **Reference quality and candidate quality are separate.** Do not silently count uncertain reference items as candidate errors or candidate successes.

## 3. Evidence graph model

The evaluator-side reference packet should be represented as a directed, versioned graph. The graph has four primary node classes:

- **Claim:** atomic proposition, criterion or rubric dimension; bound to task ID, exact claim bytes, claim version and scope.
- **Source snapshot:** immutable source bytes (or a privacy-preserving controlled artifact), digest, canonicalization profile, publisher/source identity commitment, publication/retrieval/as-of times when available, and ancestry/independence group.
- **Evidence span:** exact range within a source snapshot, using a declared offset unit and canonicalization version; bound to a span-content digest.
- **Adjudication:** an initial or later decision by a pseudonymous rater, bound to the rubric revision, blinding state, declared expertise/role, conflict-of-interest disclosure commitment, confidence representation, and evidence links.

Typed edges must distinguish:
- `supports`, `refutes`, `qualifies`, `contradicts`, and `irrelevant` (evidence-to-claim);
- `derived_from` / `same_ancestry_as` (source provenance);
- `adjudicates` (rater-to-claim);
- `cites` (claim-to-span);
- `resolves` (later adjudication-to-prior disagreement).

Do not collapse these into one generic "grounded" edge. Citation presence, source authenticity, span correctness, entailment, scope compatibility and truth status are distinct questions.

### Claim and scope record

A material claim should carry:
- stable claim ID and task ID;
- exact claim-text digest and a versioned atomicity policy;
- criterion/rubric item ID and rubric revision digest;
- domain and relevant jurisdiction/setting;
- population/entity and unit/measurement context where applicable;
- valid-from / valid-until or an explicit time-independent declaration;
- the evidence graph references and a declared scope boundary.

A validator should reject missing or contradictory scope metadata for qualified items. Do not infer time validity from a current webpage timestamp alone.

### Source and span record

Every source snapshot should include:
- exact raw snapshot digest and media/encoding metadata;
- a pinned canonicalization profile if normalized bytes are also used;
- publisher/source identity commitment and source ancestry group commitment;
- publication, retrieval and asserted-validity times where known;
- version, correction, deletion or retraction state;
- access classification and evaluator-only custody reference.

Every span should include the snapshot ID, start/end offsets, declared offset unit (prefer byte offsets into the committed bytes), and digest of the exact span bytes. A URL is a locator, not an immutable source snapshot. Do not publish protected source text or personal information in public reports.

### Independence groups

Multiple URLs do not imply multiple independent sources. Syndication, mirrored pages, press-release republication and sources derived from the same upstream dataset should share a declared ancestry or dependence group when known. Unknown provenance stays unknown; it must not be silently treated as independent.

An ancestry commitment can identify a declared relationship but cannot prove it. Independent source review and external provenance evidence remain separate requirements. A negative control must show that different sources with a shared template or schema are not automatically merged.

## 4. Adjudication protocol

### Initial independent pass

For each material claim, collect independently produced judgments under one frozen rubric. Record the panel size and actual coverage; do not imply that all items received the same number of raters if they did not.

Each initial record should bind:
- a pseudonymous annotator commitment (not a public identity);
- role/expertise category and a conflict-of-interest disclosure commitment;
- exact rubric revision and source-packet digest;
- whether the annotator was blind to model/system identity, candidate order, judge output and other raters;
- label, abstention reason and confidence representation;
- evidence graph links and a rationale digest, without publishing private rationale;
- adjudication time and protocol version.

For an initial qualification study, target at least three genuinely independent initial judgments for each scored material claim. Any smaller or incomplete panel must be reported as such and must not be upgraded to "independent consensus" by wording. For high-consequence claims, use qualified domain experts and independently review the expertise/COI process.

### Separate resolution stage

A separate adjudicator may review disagreements after the initial votes are committed. The resolution stage must:
- remain blind to candidate/judge identity where feasible;
- show the original evidence packet and rubric, not the model's preferred answer;
- retain all original labels unchanged;
- record a resolution event, the evidence/rule used, and whether uncertainty remains;
- permit unresolved disagreement and not-evaluable outcomes.

Majority vote is a descriptive aggregation, not a truth oracle. A high agreement rate among raters sharing the same assumptions or source packet can still encode common error.

### Reference states

The minimum reference-state vocabulary should include:
- `supported_within_scope`;
- `contradicted_within_scope`;
- `mixed_or_contested`;
- `insufficient_evidence`;
- `scope_ambiguous`;
- `temporally_stale`;
- `not_applicable`;
- `not_evaluable`.

These states describe the evidence and adjudication result under the declared protocol. They do not assert metaphysical certainty. Keep "no evidence found" distinct from "evidence of absence".

## 5. Metrics and statistical analysis

Report all metrics with integer numerators, denominators, missingness, and uncertainty. Never serialize a zero denominator as a zero rate.

### Reference-process metrics

- **Panel coverage:** claims with the required number of independent initial adjudications / material claims.
- **Initial disagreement:** per-item vote distribution, pairwise disagreements, abstentions and unresolved cases.
- **Inter-rater reliability:** nominal Krippendorff's alpha or another justified statistic for the annotation scale, with a task/group-aware uncertainty procedure. Report the number of items and raters; do not treat multiple judgments of one item as independent test items.
- **Resolution rate:** claims receiving a later resolution, and claims remaining mixed/contested or not-evaluable.
- **Conflict-of-interest and blinding coverage:** missing or failed declarations / planned records.
- **Reference stability:** decisions changed after source correction, deletion, expiry or rubric revision; keep each version in the lineage.

Agreement is not accuracy. Report human-human agreement alongside judge-human agreement, and retain disagreement distributions.

### Evidence-graph metrics

- **Material-claim traceability:** material claims with complete evidence/scope records / all material claims.
- **Span integrity:** spans whose snapshot digest, offset range and span digest all verify.
- **Citation correctness:** cited span exists and is the span actually referenced.
- **Evidence relation agreement:** independent agreement on support/refute/qualify/contradict/irrelevant relations.
- **Scope compatibility:** evidence that matches the declared domain, population, jurisdiction, unit and valid-time interval.
- **Source-independence coverage:** claims whose corroborating sources have reviewed provenance/ancestry; unknown ancestry remains a separate count.
- **Contradiction and staleness handling:** conflicts, superseded sources, expired claims and retractions represented rather than silently discarded.

Do not turn these diagnostics into one opaque "ground truth confidence" score.

### Candidate/judge comparison

Report judge agreement separately against:
1. each initial human vote;
2. the empirical distribution of votes;
3. the post-resolution reference state;
4. a strict subset of well-supported, in-scope items.

Where a judge produces calibrated probabilities, use proper scoring rules such as Brier or log loss on an explicitly defined label distribution. Where it produces only a categorical decision, report class-conditional agreement and abstention; do not invent probabilities from a deterministic label.

Use leave-one-rater-out or held-out-rater comparisons when measuring judge-to-panel alignment, to avoid evaluating against an aggregate that includes the rater used as the target. For generalized claims, use task-family/group-aware bootstrap or a justified hierarchical model. Publish fixed-suite results separately from estimates for a broader task population.

## 6. Hostile and positive control matrix

The synthetic suite must include at least:

| Control | Expected behavior |
|---|---|
| Several genuinely independent sources support an in-scope claim | Accept traceable support |
| Two independent sources materially conflict | Preserve conflict; no forced certainty |
| No admissible source exists | Mark insufficient evidence |
| Valid URL but cited span does not support claim | Reject evidence edge |
| Correct source, stale publication/version for target time | Mark temporal mismatch |
| Claim extrapolates beyond source population/domain | Reject scope compatibility |
| Mirrored/syndicated pages share one upstream source | Do not count as independent corroboration |
| One initial rater dissents | Preserve dissent and vote distribution |
| Rater abstains with a reason | Preserve abstention; do not impute a vote |
| Rater sees model identity or judge answer | Fail the required blindness condition |
| Post-hoc rubric/source/claim mutation | Reject commitment or mark new version |
| Correlated panel unanimously gives unsupported label | Do not treat unanimity as evidence sufficiency |
| Source corrected, deleted or retracted after freeze | Preserve original snapshot and append a supersession event |
| Distinct sources share a common template but not ancestry | Negative control: do not collapse them automatically |
| Missing/duplicate/orphan vote, source or span | Reject the incomplete artifact |
| Hash-valid reference with no external custody evidence | Verify byte binding, but deny qualification |

## 7. Competitive baselines and research comparisons

Compare the resulting process with established and current approaches, not just another prompt:

- **ARES (NAACL 2024):** small human-annotated calibration set plus prediction-powered inference for RAG evaluation. Use it as a statistical baseline where assumptions and annotation design fit.
- **RULERS (EMNLP 2026):** frozen task-level rubric, structured evidence-grounded judgments and calibration to human score boundaries. Compare rubric transfer and score-distribution behavior.
- **LLM-as-a-Jury / BT-σ (ICML 2026):** judge-aware estimation of rankings/reliability from pairwise judgments. Compare against simple majority and single-judge baselines, while keeping independent human anchors.
- **Judge geometry (EMNLP Findings 2026):** compare inter-judge agreement with human-human and judge-human agreement; shared judge consensus must not be treated as independent confirmation.
- **NIST AI 800-3 (February 2026):** distinguish fixed-suite accuracy from generalized accuracy, model task/item variation, and make assumptions explicit.

Relevant publications:
- https://aclanthology.org/2024.naacl-long.20/
- https://aclanthology.org/2026.emnlp-main.566/
- https://proceedings.mlr.press/v306/qian26i.html
- https://aclanthology.org/2026.findings-emnlp.1696/
- https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models

These are comparison targets and design references, not evidence that this protocol already outperforms them.

## 8. Engineering qualification gates

1. Freeze the reference protocol, rubric, evidence packet manifest, panel commitment, blindness policy and analysis plan before judge outcomes are inspected.
2. Add strict versioned plan/report schemas with unknown fields denied.
3. Implement typed Rust commitments and fail-closed coverage/lineage/scope validation.
4. Add independent test-oracle fixtures and hostile/negative controls above.
5. Require deterministic repeated verification and exact source/span digest binding.
6. Keep evaluator-only material out of candidate-visible inputs and public reports.
7. Require external custody/identity and runner-isolation evidence for qualified scoring; the validator alone cannot prove either.
8. Run a blinded human adjudication study and publish item-level uncertainty, panel agreement and judge alignment before making performance claims.

### Qualification targets (not achieved results)

- Reject 100% of enumerated deterministic malformed/tampered/missing/duplicate/orphan/wrong-revision and invalid-scope synthetic controls.
- Accept all committed clean positive and negative controls.
- Preserve every initial vote, abstention and dissent in the report.
- Every scoring-eligible material claim has traceable source-span, scope and adjudication records, or is explicitly excluded/not-evaluable under a frozen rule.
- No panel is declared independent solely because it has multiple raters or distinct hashes.
- The report continues to declare `runner_execution_proven=false` and `external_anchor_required=true` unless independent evidence is actually supplied.

There is no universal inter-rater threshold for every construct. Set application-specific qualification thresholds before observing outcomes; publish the full distribution and uncertainty rather than selecting a threshold after results are known.

## 9. Integration sequence

- **#9065:** consume the reference-assurance package; report judge stability, intervention consistency, human alignment and reference quality as separate dimensions.
- **#9005:** use atomic claims and evidence-span relations for support, contradiction, deletion, distraction and temporal interventions.
- **#8993:** preregister how contested, stale and not-evaluable items enter paired contrasts. Never drop them post hoc to improve an effect estimate.
- **SCI-BENCH-REFRESH-001:** reference assurance becomes an admission gate for high-confidence benchmark claims, alongside task regeneration, leakage and runner isolation.

The first implementation should be limited to one domain and a small, fully inspectable reference packet. Expand only after the schema, independent oracle and adversarial controls are stable.

## 10. Explicit limitations

This protocol can make reference construction more inspectable and test whether its declared process was followed. It cannot prove universal truth, eliminate shared human assumptions, infer independent expertise from pseudonyms, establish source independence from hashes alone, prove that a foundation model never saw a source during pretraining, or prove a runner withheld evaluator-only data without external execution evidence.

A reference receipt can be valid while its reference is poor. Preserve that distinction just as #9065 preserves the distinction between a valid judge-audit receipt and a good judge.
