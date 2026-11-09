# REA — evidence-force calibration contract and development fixtures (2026 Q4)

**Status:** research contract and original synthetic development fixtures only. No human-adjudicated benchmark, production evaluator, model score, or SOTA result is claimed.

**Owners:** [#10621 — calibrate claim strength to evidence warrant](https://github.com/Luminous-Dynamics/symthaea/issues/10621), [#8995 — evidence-responsive behavioral contracts](https://github.com/Luminous-Dynamics/symthaea/issues/8995).

**Files:**
- `docs/research/rea-evidence-force-pairs-v0.schema.json`
- `docs/research/rea-evidence-force-pairs-v0.json`

## Executive finding

A cited passage can be relevant, correctly quoted, current, and still fail to warrant the strength of the claim attached to it. This is not merely a citation-recall problem. It is a mismatch between the proposition actually supported by the evidence and the proposition asserted by the system.

The central contract is:

> With the evidence bytes and scope held fixed, a claim that makes a stronger unsupported commitment must not receive a stronger warrant assessment than a minimally edited, evidence-calibrated claim.

The converse is equally important:

> When independently qualified evidence is added and resolves the missing proof obligation, a stronger proposition must become eligible for stronger support.

Without the second law, a system that always hedges, refuses, or abstains can appear safe while failing to use evidence.

## 1. Research anchors and what they establish

### ForceBench — evidence-force contrast

Qian et al., *Relevant Is Not Warranted: Evidence-Force Calibration for Cited RAG*, Findings of EMNLP 2026, introduce a fixed-passage contrastive evaluation across relation, modality, scope, temporal validity and numeric specificity. Their paper reports 198 locality-filtered pairs and an aggregate monotonicity-violation rate of 47.2% for generic support prompting, reduced to 24.5% with explicit warrant-strength prompting. Those are results for the paper's models, judges, prompts and set—not REA results or universal field constants.

Reference: https://aclanthology.org/2026.findings-emnlp.1577/

### Behavioral contracts — test interventions, not one answer

Zeng et al., *Beyond Static Grounding: Behavioral Contracts for Trustworthy Retrieval-Augmented Generation*, Findings of EMNLP 2026, specify expected behavior under evidence deletion, conflict, distraction and permutation. This supports measuring how a system changes when its evidence changes, rather than scoring only a static answer.

Reference: https://aclanthology.org/2026.emnlp-main.1494/

### Retrieval is a separate failure source

Usmanova et al., *How Robust Are Automated Fact-Checking Systems? A Cross-Benchmark Evaluation*, EMNLP 2026, report that replacing retrieved evidence with gold evidence improved veracity accuracy by 14–22 points across the tested models and datasets; rankings also varied by domain and metric. Therefore oracle-evidence warrant reasoning must be measured separately from the full retrieve-then-verify system.

Reference: https://aclanthology.org/2026.emnlp-main.1915/

Kawada et al., *Recall Is Not Completeness: Retrieving All the Evidence a Question Needs*, NLLP 2026, show why average retrieval recall is not enough when a question requires a complete set of premises. In one multi-statute subset, top-ten retrieval averaged 39.3% of gold items while the complete evidence set was present for only 2.3% of questions. REA must report evidence-set completeness and per-question all-premises coverage in addition to recall.

Reference: https://aclanthology.org/2026.nllp-1.22/

Akhtar et al., *Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking*, TACL 2026, combine reference alignment with verdict-support evaluation and test metric robustness against human ratings and adversarial examples. REA should compare its adjudication protocol with such construct-specific evidence metrics rather than assume one NLI or overlap score is a universal warrant judge.

Reference: https://aclanthology.org/2026.tacl-1.25/

### Statistical generalization

NIST AI 800-3 distinguishes benchmark accuracy on a fixed test set from generalized accuracy over a wider population of similar items. It demonstrates generalized linear mixed models to account for item difficulty and variance across models and benchmarks. REA must preserve that distinction: a clean finite fixture result is not a generalized capability estimate.

Reference: https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models

## 2. Paradigm shifts

### 2.1 Replace one groundedness score with a typed warrant vector

A claim is not one indivisible string. For evaluation, decompose it into atomic propositions and record separate commitments:

- relation: association, causation, identity, mechanism, attribution;
- modality: possible, probable, necessary, certain, negated;
- scope: population, environment, geography, version, exclusions and denominator;
- temporal validity: valid time, publication time, acquisition time, supersession and target time;
- numeric specificity: value, unit, precision, uncertainty, denominator and rounding;
- attribution: who observed, who reported, who inferred and who endorsed;
- source lineage: original, derivative, mirror, translation, syndicated or unknown;
- population completeness: declared universe, enumeration, acquisition, processing, participation and temporal coverage.

These dimensions form a vector of proof obligations, not a scalar confidence score. The system may be strong on source attribution but weak on population completeness; that distinction must survive into the final report.

### 2.2 Separate evidence truth, source truth and claim warrant

Maintain distinct questions:

1. Did the source say this?
2. Is the source or artifact authentic under the declared trust policy?
3. Was the observation correctly acquired and interpreted?
4. Does this observation entail or warrant this exact proposition at this scope and time?
5. Is the evidence set complete enough for the proposition?

A signature, transparency receipt, citation, parser pass, or source-family count cannot silently answer all five.

### 2.3 Make evidence interventions first-class benchmark operations

For each base scenario, define expected behavior under:
- deletion of a necessary premise;
- addition of a genuine contradiction;
- addition of an irrelevant decoy;
- order or formatting permutation;
- replacement with a superseding source;
- addition of independent evidence;
- removal of one member of an allegedly independent source family;
- loss of one required population item or processing result.

Each intervention has a predeclared expected relation. Preserve the original answer, intervention, resulting answer, adjudication, and source-family lineage as separate records.

### 2.4 Measure completeness of required evidence sets

A system can retrieve many relevant documents yet miss one indispensable premise. For tasks requiring multiple premises, define an independently adjudicated required-evidence set and report:
- evidence recall;
- all-required-premises completeness;
- missing-premise rate;
- per-claim evidence coverage;
- source-family diversity after lineage collapse;
- completeness conditioned on the number of required premises.

Do not equate a high average recall with a complete evidence set.

### 2.5 Treat abstention as a selective decision, not a virtue score

Report supported-answer coverage, correct qualification, false abstention, unsupported-force acceptance, and selective risk at multiple coverage levels. A system that abstains on every item must fail usefulness and coverage measures even if it has zero overclaiming.

## 3. Experimental tracks — never conflate them

| Track | What is held fixed | What it measures |
|---|---|---|
| A. Oracle-evidence warrant | Exact evidence text/bytes and scope are identical across claim variants | Claim decomposition and force calibration |
| B. Fixed-candidate extraction | Same retrieved document pool for every system | Selection, extraction and evidence-span identification |
| C. End-to-end retrieval | Query, corpus snapshot, budget and source policy are fixed | Retrieval + extraction + warrant as an integrated system |
| D. Evidence intervention | A known evidence deletion, conflict, decoy, permutation or update is applied | Behavioral compliance under evidence change |
| E. Evidence growth | Claim held fixed while genuinely stronger independent evidence is added | Ability to increase support when the proof obligation is met |

Track A isolates warrant reasoning. Track C is the operational RAG/reverse-engineering result. Neither substitutes for the other. Every run must preserve corpus snapshot, query, candidate set, exact passage bytes, source family, tool budget, model/prompt versions and retrieval settings.

## 4. Development fixture v0

The JSON fixture contains:
- 24 original synthetic contrast pairs: three per axis across the five ForceBench axes and three REA extensions;
- eight evidence-growth controls: one per axis;
- source-family identifiers, domain, exact passage text, calibrated and force-raised claims, the single intended mutation, expected design relation, and rationale;
- explicit design-only status requiring independent human review.

The five primary axes are relation, modality, scope, temporal validity and numeric specificity. The three REA extensions are attribution, source lineage and population completeness.

These fixtures are public development material, not a hidden test set and not human-adjudicated ground truth. Their labels are hypotheses for review. They must not be used to report model accuracy, MVR, calibration, or SOTA.

### Required independent review

Before any qualification run:
1. Two independent annotators review each passage and both claims without seeing model outputs.
2. A domain-qualified adjudicator resolves disagreements or marks the pair not adjudicable.
3. Annotators record the exact span that warrants the calibrated claim and the precise unsupported commitment introduced by the raised claim.
4. A minimal-edit audit confirms only the declared axis changed. Any pair changing multiple semantic dimensions is split or excluded from the single-axis denominator.
5. Source-family and passage overlap are checked before train/dev/test assignment.
6. A held-out set is frozen before candidate model evaluation; hidden labels stay outside prompts, retrieval indexes and training data.

Use weighted Krippendorff's alpha (or another justified ordinal agreement statistic) for ordinal warrant labels, report the full disagreement distribution, and retain adjudication changes in version history. Automated judges are candidate measurement instruments, not gold truth.

## 5. Warrant labels and metrics

Use the following item-level labels:
- `adequate_as_written`: all material commitments are warranted at the requested scope and time;
- `adequate_only_with_qualification`: a narrower or more qualified formulation is warranted;
- `insufficient`: evidence does not establish the proposition;
- `contradicted`: evidence supports an incompatible proposition;
- `not_adjudicable`: evidence, rights, language or domain context prevents a defensible label.

Do not collapse these labels into a universal truth probability.

### Primary monotonicity violation rate

For an adjudicated pair where the calibrated claim is adequate as written and the force-raised claim is not, record a violation when the evaluated system assigns the raised claim warrant that is equal to or stronger than the calibrated claim under the same evidence. Ties count as violations because the system failed to respond to the unsupported increase in force.

`MVR = violating eligible pairs / all eligible strict-order pairs`

Publish the denominator and each axis-specific MVR. Report not-adjudicable, missing-run, and technically invalid trials separately; do not silently remove them.

### Complementary metrics

| Metric | Failure it reveals |
|---|---|
| Per-claim warrant accuracy | A good pairwise result can still label both claims incorrectly |
| Unsupported-force acceptance | Strong claims accepted beyond the evidence |
| Calibrated-claim acceptance | Correctly warranted claims rejected or over-hedged |
| Qualification adequacy | Whether the system finds the strongest defensible narrower wording |
| Evidence-set completeness | Missing one or more required premises |
| Attribution accuracy | Provider report mistaken for local verification or direct observation |
| Source-family independence error | Mirrors and shared upstream feeds counted as independent |
| Temporal validity error | Historical evidence promoted into current claims |
| Numeric force error | Approximate, rounded or sample values rendered as exact/population values |
| False abstention and coverage | Over-cautious systems rewarded for refusing useful answers |
| Human/automated judge agreement | Whether the scorer itself measures the intended construct |

Do not publish one composite leaderboard score as the principal result. If a composite is used for a specific decision, publish its weights, the entire metric vector, and sensitivity to the weights.

## 6. Proposed qualification targets

These are targets for future qualification, not achieved results.

| Qualification layer | Target |
|---|---|
| Fixture and schema integrity | 100% structural validation; all pair IDs unique; all expected axes represented |
| Deterministic mutation controls | 100% correct axis identification and no accidental multi-axis edits |
| Synthetic protocol controls | Zero monotonicity violations in a deterministic reference implementation |
| Human adjudication | Two independent labels plus resolved or explicitly unresolved disagreements for every scored item |
| Held-out source-family split | Zero family overlap between train/dev/test |
| Evidence integrity | 100% exact passage and corpus-manifest binding |
| Force calibration | Report MVR with source-family-clustered confidence intervals per axis; establish target only after a pilot baseline |
| Usefulness | Jointly report calibrated-claim acceptance, coverage, false abstention and unsupported-force acceptance |
| Retrieval | Report recall, all-required-premises completeness, missing-premise rate and retrieval cost separately |
| Generalization | Estimate fixed-suite and generalized accuracy separately; use item/source-family effects and disclose model assumptions |
| Reproduction | Independent exact-head reproduction of fixture, split, prompts, model versions, adjudication and analysis digests |

Do not set an arbitrary universal MVR threshold from a 24-pair synthetic pilot. First establish human agreement, scorer reliability and baseline distributions; then preregister a held-out target.

## 7. Baselines and fair comparisons

Compare on identical passages and budgets:
1. Citation-presence and token/entity-overlap heuristics.
2. Conventional NLI / entailment baselines.
3. Generic retrieval-augmented support prompting.
4. Explicit warrant-strength prompting.
5. Structured claim decomposition with axis-specific evidence assessments.
6. REA typed Evidence + source-lineage projection + independent replay.
7. Human/domain adjudicators as the reference measurement, with disagreement retained.

For integrated system comparisons, also report retrieval, extraction and tool-use budgets. Freeze model version, decoding settings, system prompt, source access, maximum context, latency/cost ceiling and retries. Do not compare an oracle-evidence run against an end-to-end RAG result.

For generalization, sample multiple source families, domains and task difficulties. Model item and source-family variation with an explicitly justified hierarchical method where sample size permits; use cluster-aware uncertainty estimates. A fixed fixture score is benchmark accuracy, not generalized accuracy.

## 8. Industry boundary

| System / research | Strong contribution | What REA must not infer |
|---|---|---|
| ForceBench | Contrastive evidence-force test | That its reported MVR is REA performance or universal SOTA |
| Dynamic Evidence Contracts | Behavioral testing under evidence intervention | That intervention compliance alone proves truth |
| Ev2R | Evidence/reference alignment plus verdict support | That one scorer transfers to every domain or claim type |
| Cross-benchmark fact-checking | Retrieval/verification separation and domain sensitivity | That a gold-evidence result represents production retrieval |
| SLSA v1.2 | Build and Source provenance tracks | Source correctness or runtime safety |
| SCITT RFC 9943 | Signed-statement registration and transparency receipts | Statement accuracy or complete issuer participation |
| TUF | Update metadata freshness, threshold trust and rollback/freeze defenses | Semantic safety of delivered bytes |
| RFC 8785 JCS | Deterministic JSON representation for commitments | Exact decimal intent or source authenticity |
| I-JSON RFC 7493 | Interoperable binary64 bounds and safe-integer warning | Application-specific numeric meaning |

REA should compose these systems as specialized evidence sources. It should not recreate provenance, transparency, update trust, or canonicalization. Its distinctive layer is the exact proposition-to-evidence warrant contract, with explicit limits and independent reproduction.

## 9. Numeric intent and exact source representation

The existing numeric-intent work should remain separate from evidence-force labels.

RFC 8785 JCS builds on ECMAScript serialization and deterministic property ordering. I-JSON RFC 7493 warns that receivers cannot be expected to treat integers outside ±(2^53−1) as exact and recommends strings for exact interchange beyond that boundary. JSON Schema Draft 2020-12 supports structural constraints but does not itself create application-specific exact-decimal semantics.

Therefore:
- preserve source bytes and exact raw numeric lexemes where available;
- bind numeric intent to exact schema paths and a versioned domain contract;
- keep exact decimal, integer, measurement, money, percentage, sample count and opaque values distinct;
- never allow numeric precision warnings to act as semantic evidence;
- retain unit, denominator, uncertainty and time scope as separate claim obligations.

References:
- https://www.rfc-editor.org/rfc/rfc8785.html
- https://www.rfc-editor.org/rfc/rfc7493.html
- https://json-schema.org/draft/2020-12/json-schema-validation

## 10. Executable fixture integrity gate

The companion checker at `tools/rea-evidence-force-fixture-check` validates the Draft 2020-12 schema and the development fixture's structural invariants: 24 unique contrast IDs, three pairs per axis, eight unique evidence-growth controls, distinct source-family roots, exact declared substring anchors, axis/mutation agreement, and the development-only / human-review-required labels. Its deterministic mutation tests exercise the rejection paths.

This gate verifies fixture structure and declared edit anchors only. It does not prove the edits are semantically single-axis, adjudicate warrant labels, qualify automated judges, or measure a model. Human/domain review remains mandatory.

## 11. Integration sequence

1. Review this contract and the synthetic pair set; correct defects before adding any scorer.
2. Independently validate the Draft 2020-12 schema and fixture counts, uniqueness, axis coverage and one-axis mutation conditions.
3. Human-adjudicate a small pilot and revise the rubric under a new fixture version.
4. Pin and license-audit the external ForceBench release before reuse; treat it as a separate external test set, not material to silently copy into the synthetic corpus.
5. Build oracle-evidence and fixed-candidate-pool runs, then add end-to-end retrieval.
6. Add source-family split enforcement, blind labels, human/automated judge calibration and run receipts.
7. Only then compare REA/Symthaea with open and frontier baselines under equal budgets.

The current default branch does not contain the `tools/knowledge-benchmark-harness` path referenced in the original research issue. This PR deliberately does not invent a parallel scoring framework or claim that a runtime harness exists. The fixture and schema are prepared for integration after the canonical benchmark-harness location and ownership are reconciled.

## Claim ceiling

This work defines a research contract and synthetic development fixtures. It does not implement a force-calibration evaluator, authenticate sources, establish real-world claim truth, prove completeness, measure Symthaea capability, validate an LLM judge, or establish superiority over ForceBench or any industry system.
