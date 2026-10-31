# SCI-INTERVENTION-QUALITY-001 — Independently measure generated test quality

**Status:** Rust plan/receipt auditor, schemas, synthetic fixture and CI controls under development. No human study or empirical generator result is claimed.

**Tracking:** [Issue #9270](https://github.com/Luminous-Dynamics/symthaea/issues/9270).  
**Stack:** after SCI-FACTORIAL-JUDGE-BIAS-001 / the judge-bias research branch.

## 1. Research motivation

The previous layers measure evaluator behavior:
- judge robustness checks repeatability, invariance, position sensitivity and reference agreement;
- metamorphic evaluation checks expected relations between paired evaluator observations;
- the factorial design measures interactions among candidate order, score-option order, criterion order, provenance cues and presentation format.

They leave a separate question unanswered:

> Are the generated tests themselves valid, informative and worth their cost?

A test generator can inflate apparent productivity by emitting near-duplicates, meaning-preserving cases that actually change meaning, or meaning-changing cases mislabeled as invariance controls. An evaluator may be penalized for correctly ignoring invalid transformations. A benchmark may also spend most of its budget rediscovering the same semantic case.

Recent work supports treating this as a first-class measurement problem:

- **RULERS** (EMNLP 2026) locks task-level rubric specifications, grounds judgments in traceable evidence and calibrates scores to human boundaries. It argues that faithful operationalization matters beyond prompt-level scoring: [ACL Anthology](https://aclanthology.org/2026.emnlp-main.566/).
- **PReMISE** (EMNLP 2026) audits rubrics on structural adequacy, reliability, preference fit and adversarial robustness. Its reported results show that inter-judge agreement and resistance to exploitation are distinct dimensions: [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2653/).
- **Explanation-Guided Metamorphic Testing** (ESEM 2026) evaluates semantic validity and testing efficiency as well as failure discovery. Its reported 2.30× verified-failure finding is specific to its models, datasets and experimental budget: [Dagstuhl / ESEM 2026](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESEM.2026.5).

This workstream adapts those lessons to the quality of Symthaea's evaluator-intervention generator. It is not a reproduction of those papers.

## 2. Paradigm shift: test generation is a selective measurement process

Treat the intervention pipeline as a classifier over generated candidates:

1. A frozen generator emits a complete candidate census.
2. A frozen semantic-verification pipeline accepts, rejects or abstains on each candidate.
3. A separately blinded panel adjudicates the intended semantic relation.
4. The auditor measures precision, false acceptance, false rejection, ambiguity, coverage, diversity and resource yield.

This separates four claims which should never be collapsed:
- a candidate was generated;
- a pipeline accepted it;
- a blinded panel supported its intended relation;
- an evaluated judge later behaved as predicted under that intervention.

SCI-INTERVENTION-QUALITY-001 measures the first three. It does **not** measure whether a judge actually exhibits a targeted bias; the metamorphic and judge-execution layers own that experiment.

## 3. Candidate contract

Files:
- `src/intervention_quality.rs`
- `schema/intervention-quality-plan.schema.json`
- `schema/intervention-quality-receipt.schema.json`
- `schema/intervention-quality-report.schema.json`
- `tools/intervention_quality_fixture.py`

The preregistered plan pins:
- analyzer source revision;
- generator source revision and prompt digest;
- semantic-verifier source revision and policy digest;
- full-census sampling policy;
- adjudication panel size, consensus threshold and blindness obligations;
- every generated candidate's ID, transformation family, target failure mode and semantic-cluster key;
- intended relation, pipeline disposition and baseline/variant input digests;
- generation tokens, estimated generation cost and wall-clock accounting.

The intended relation is either:
- `semantic_invariant`: the semantic decision should remain invariant;
- `semantic_change`: the transformation is intended to change the task's meaning or evidence state.

The pipeline disposition is `accepted`, `rejected` or `abstained`. All three remain in the denominator.

V1 requires `candidate_sampling_policy = full_census`. A sampled audit can produce verification bias if only pipeline-accepted cases are adjudicated. A future sampled/adaptive protocol must preregister its sampling mechanism and inclusion probabilities before replacing census accounting.

## 4. Blinded semantic adjudication

Every candidate receives exactly the preregistered fixed number of distinct reviewers (3–7), recorded as `reviewers_per_candidate` rather than a minimum. Each reviewer is blind to the declared intended relation as well as generator identity, pipeline disposition and evaluator outcome. This matters because a panel that knows the generator's claim can unconsciously confirm it. Each reviewer supplies one of:
- `meaning_preserving`;
- `meaning_changing`;
- `ambiguous`.

Reviewers must be blind to generator identity, the declared intended relation, pipeline disposition and evaluator outcome. Each review carries a reviewer pseudonym digest and an adjudication-artifact digest.

A relation is adjudicated only when one verdict reaches the preregistered consensus threshold. A 666-permille threshold permits a two-of-three panel majority. A tie or insufficient consensus yields `ambiguous`.

The report does not turn ambiguous labels into valid transformations. Accepted, rejected and abstained candidates are each split by adjudicated semantic class. A valid candidate that the pipeline abstains on is different from an ambiguous candidate, and both differ from an invalid candidate that is correctly rejected. For accepted cases:
- **accepted-valid** means the panel supports the intended relation;
- **accepted-invalid** means the panel supports the opposite semantic relation;
- **accepted-ambiguous** means the panel cannot support either conclusion.

The same valid/invalid/ambiguous breakdown is retained for rejected and abstained candidates. This makes false rejection, justified rejection and unresolved abstention separately auditable.

Important v1 boundary: `semantic_change` means the panel supports that some material semantic change occurred; it does not yet prove that the *specific intended proposition, direction, scope or target failure mechanism* changed as planned. V2 should add a blinded target-delta adjudication with explicit target proposition, expected direction and scope, plus an independent reference. Do not treat generic semantic change as proof of targeted counterfactual validity.

The fixed panel size enables both raw pairwise agreement and Fleiss' κ. The chance-adjusted statistic is descriptive, not evidence of independence or correctness. A panel can agree and still be wrong.

Blinding flags and unique reviewer digests are checked as declarations. The analyzer cannot prove real reviewer independence, conflicts of interest, blinding in practice, panel competence or authenticity of source artifacts.

## 5. Metrics and denominators

### Semantic validity of pipeline decisions

- accepted-valid precision lower bound = accepted-valid / all accepted;
- confirmed-invalid acceptance rate = accepted-invalid / all accepted;
- ambiguous acceptance rate = accepted-ambiguous / all accepted;
- valid-candidate recall = accepted-valid / all panel-adjudicated valid candidates;
- pipeline acceptance rate = accepted / all generated candidates.

The precision measure is deliberately conservative: ambiguous accepted candidates remain in the denominator and are not counted as success. Invalid and ambiguous accepted candidates are reported separately, so a low precision score can be diagnosed rather than hidden.

### Reliability of adjudication

- raw pairwise reviewer agreement;
- Fleiss' κ for the fixed-size panel design;
- number of reviewer pairwise comparisons;
- candidate-level valid, invalid and ambiguous counts.

A future human-validation study should also report disagreement by domain, task type and reviewer expertise. A single pooled coefficient must not erase subgroup failure.

### Diversity and resource yield

- unique transformation families;
- unique target failure modes;
- unique semantic clusters;
- unique accepted-valid test units;
- generator tokens, micro-USD and wall-clock;
- accepted-valid unique test units per 1,000 generation tokens and per USD;
- per-family and per-failure-mode validity/disposition breakdowns.

A valid accepted unit is deduplicated by semantic-cluster and target-failure-mode identity within its transformation family. This is a pragmatic coverage unit, not proof that two distinct hashes encode distinct meanings.

## 6. Qualification targets

### Engineering controls

- 100% acceptance of the clean synthetic census;
- 100% rejection of missing, duplicated, unknown or extra candidate adjudications;
- 100% rejection of duplicate reviewers within a candidate panel;
- 100% rejection of insufficient or non-blinded panels;
- 100% rejection of wrong plan, generator, verifier or analyzer revision bindings;
- all generated candidates retained in the denominator;
- ambiguity never converted into a semantic pass;
- checked overflow for token, cost and wall-clock totals;
- byte-deterministic report for identical plan and receipt bytes.

These qualify the measurement implementation, not the semantic verifier.

### Empirical mutation-quality targets

Targets to preregister after pilot variance estimates:
- accepted semantic-validity precision lower bound ≥ 0.98;
- confirmed-invalid accepted rate ≤ 0.01;
- ambiguous accepted rate ≤ 0.01;
- adjudication panel Fleiss' κ reported with uncertainty; target ≥ 0.70 for domains where a single relation is expected;
- unique adjudicated-valid test units per 1,000 tokens and per dollar reported by failure family;
- no single transformation family supplies more than 40% of valid unique yield unless the domain's actual risk distribution justifies it;
- publish coverage/precision curves as the pipeline acceptance threshold changes, rather than hiding the trade-off in a single score.

These are proposed qualification gates, not achieved results. The κ threshold is a screening target, not a universal definition of semantic truth.

## 7. Competitive benchmark protocol

Compare mutation strategies against a common held-out candidate pool, frozen task snapshots, identical adjudication policy and equal resource budgets:

1. hand-authored expert transformations;
2. heuristic mutation (formatting, paraphrase, position, distractor and evidence operators);
3. unconstrained LLM rewrite generation;
4. attribution/explanation-guided mutation with independent semantic verification;
5. rubric/evidence-grounded generation using RULERS/PReMISE-inspired specifications;
6. Symthaea's generator and semantic-verification pipeline.

Report the complete quality vector: accepted-valid precision, confirmed false acceptance, ambiguous acceptance, valid-case recall, abstention, reviewer agreement, unique semantic-cluster coverage, failure-family coverage, tokens, cost and wall-clock.

The fair comparison is **valid unique test yield at matched cost**, with precision/coverage trade-offs visible. Raw number of generated cases is not a useful success metric.

### The stronger end-to-end experiment

After mutation quality is independently established, execute each qualified suite against:
- a neutral/reference evaluator expected to satisfy the metamorphic relations;
- seeded faulty evaluators for position, formatting, authority, confidence and evidence-use behaviors;
- multiple real judge families;
- blinded human reference labels for task correctness where applicable.

Then measure fault-detection sensitivity, neutral-control false alarms and marginal fault detections per dollar. This is a separate test-execution layer and is intentionally not inferred from the current adjudication receipt.

## 8. Adaptive test selection — future v2, not silently mixed into v1

Once the census audit is established, use active test selection to spend more budget on high-information candidates:

- maintain a failure-mode coverage posterior;
- select candidates by expected information gain per unit cost;
- preserve a randomized exploration reserve;
- commit the selection policy and random seed before observing outcomes;
- record selection probabilities;
- report both unweighted census results and appropriately weighted estimates where sampling is used;
- freeze a final untouched suite for confirmatory evaluation.

Do not allow the evaluated judge's output to leak into the generator before candidate selection is frozen. Otherwise the benchmark can overfit its own discovered failures and overstate generalization.

## 9. Explicit claim boundary

Every report sets:
- `reviewer_independence_proven=false`;
- `raw_adjudication_custody_proven=false`;
- `semantic_truth_proven=false`;
- `execution_authenticity_proven=false`;
- `external_anchor_required=true`;
- `scoring_eligible_declared=false`.

A panel vote is an adjudication signal, not an oracle. A hash proves declared-byte binding, not authentic evidence. A synthetic positive control is not a human study. A large number of accepted mutations is not evidence of test effectiveness unless semantic validity, diversity and downstream fault-detection sensitivity are all measured separately.
