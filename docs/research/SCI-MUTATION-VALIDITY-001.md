# SCI-MUTATION-VALIDITY-001 — Semantic qualification of generated evaluator interventions

**Status:** Rust plan/receipt validator and analyzer, strict JSON Schemas, deterministic synthetic fixture and CI controls under development. No human semantic-validity study or empirical evaluator result is claimed.

**Tracking:** [Issue #9343](https://github.com/Luminous-Dynamics/symthaea/issues/9343).  
**Stack:** after SCI-FACTORIAL-JUDGE-BIAS-001 / [PR #9243](https://github.com/Luminous-Dynamics/symthaea/pull/9243).

## 1. The missing measurement

A metamorphic evaluator can verify that an observed pair obeys a declared relation, but it cannot establish that the mutation generator actually preserved or changed the intended meaning.

A beautifully detected evaluator failure on an invalid mutation is not a valid robustness discovery. A generator can look productive by producing many variants that silently change task semantics, contradict the declared intervention, or duplicate the same failure.

This layer separates three questions:

1. **Mutation validity:** did the variant actually satisfy its preregistered semantic relation?
2. **Evaluator behavior:** did the evaluator pass or fail the metamorphic relation?
3. **Discovery value:** did a semantically confirmed mutation expose a unique evaluator failure, and at what total test/review cost?

These are separate denominators. The output never promotes an unresolved human review to a confirmed label and never counts a failure on an invalid or unresolved mutation as a verified discovery.

## 2. Research baseline

### Explanation-guided metamorphic testing

The ESEM 2026 study *Explanation-Guided Metamorphic Testing of Specialized Language Models* evaluates failure discovery, semantic validity and testing efficiency. It combines attribution-guided mutation with automated semantic verification and human annotation. It reports 2.30× more verified failure-inducing cases than heuristic mutation in its tested settings.

Source: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESEM.2026.5

**Symthaea objective:** reproduce the comparison on a matched subset under equal model calls, token budgets, mutation budgets, human review effort and semantic-verification budgets. The 2.30× figure is a replication target from one study, not a universal SOTA threshold.

### MM-JudgeBias

The ACL 2026 benchmark *MM-JudgeBias* studies compositional bias in multimodal judges by perturbing query, image and response cues, and separates sensitivity from stability. It reports more than 1,800 curated/refined multimodal samples from 29 source benchmarks and nine bias types.

Source: https://aclanthology.org/2026.acl-long.1162/

**Symthaea objective:** use modality-aware artifact commitments for text, image, audio, video, structured and multimodal variants. The current validator binds declared artifact digests and byte lengths; it does not inspect media or certify semantic equivalence.

### Noisy but Valid

The ICLR 2026 paper *Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges* derives finite-sample Type-I error control using human-labelled calibration data and explicit estimates of judge true-positive and false-positive rates.

Source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/08a67eb74bfb9ded9e949dd52973e997-Abstract-Conference.html

**Symthaea objective:** never treat a noisy judge outcome as an unquestioned reference. Keep human semantic review, evaluator outcomes and statistical certification as separate layers.

### NIST AI 800-3

NIST's 2026 AI 800-3 describes how implicit assumptions, conflated performance targets and inadequate uncertainty analysis undermine benchmark interpretation.

Source: https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models

**Symthaea objective:** preserve planned, observed, reviewed, disputed, invalid and verified counts independently, with all claim limits encoded in the report.

## 3. Plan and artifact contract

Files:
- schema/mutation-validity-plan.schema.json
- schema/mutation-validity-receipt.schema.json
- schema/mutation-validity-report.schema.json
- src/mutation_validity.rs
- tools/mutation_validity_fixture.py

The frozen plan binds:
- exact analyzer, mutation-generator and evaluator source revisions;
- generator policy, task-set, review protocol, metamorphic plan, evaluator prompt and rubric digests;
- reviewer slots per variant;
- a unique mutation ID and linked metamorphic pair ID;
- declared relation (semantics-preserving or meaning-changing), mutation family and target dimension for meaning-changing variants;
- baseline and variant input digests;
- modality-aware artifact commitments containing artifact ID, modality, content digest and byte length.

The exact plan bytes and a domain-separated plan commitment are bound by the receipt. Every observation must echo the baseline/variant input digests. This establishes byte-level consistency of declared commitments; it does **not** mean the raw artifacts are stored or externally authenticated.

A meaning-changing mutation must declare the semantic target dimension. A semantics-preserving mutation must not declare a changed target dimension.

## 4. Independent semantic adjudication

Each mutation reserves a preregistered number of independent reviewer slots (2–7). Each observed vote retains:
- pseudonymous adjudicator ID commitment;
- independence-group commitment;
- decision: preserves semantics, changes semantics, indeterminate or abstain;
- confidence in millionths;
- rationale digest.

The validator rejects duplicate reviewer IDs or independence groups within one mutation. It does not authenticate identities or prove blinding.

A mutation is classified:
- **confirmed valid:** all required reviewers agree with the declared relation;
- **confirmed invalid:** all required reviewers agree with the opposite relation;
- **disputed:** the panel contains both preserves and changes decisions;
- **insufficient:** reviews are missing or include indeterminate outcomes without a decisive panel;
- **abstained:** all observed reviewers abstain;
- **unreviewed:** no adjudication is present.

There is no majority-vote shortcut. A 2–1 split is disputed, not confirmed. Missing reviewer slots remain in the denominator. The report's adjudicable panel concordance is the fraction of unanimously adjudicated variants whose unanimous panel agrees with the declared relation; it is **not** proof that the panel is objectively correct or representative.

Raw pairwise reviewer agreement and disagreement counts are reported separately. They are not chance-corrected and must not be interpreted as a reliability coefficient or human-reference validity claim.

## 5. Linking evaluator failures to valid mutations

Every variant carries an evaluator status:
- completed;
- timed out;
- blocked;
- invalid receipt;
- not started.

Completed observations require an explicit relation pass/fail. A failed relation must carry a nonzero failure fingerprint; a passing relation must not. Ineligible executions cannot carry a partial relation outcome.

The critical accounting rule is:

**Verified failure observation = evaluator relation failed AND semantic mutation was unanimously confirmed valid.**

Therefore:
- a failure on a unanimously invalid mutation is counted as an excluded invalid-mutation failure;
- a failure on a disputed, insufficient, abstained or unreviewed mutation is counted as an excluded unconfirmed-mutation failure;
- a failure fingerprint is unique only once among confirmed-valid mutations; repeated fingerprints are counted as duplicate observations, not new discoveries.

The report distinguishes verified failure observations from unique verified failure fingerprints.

## 6. Resource accounting

Every mutation observation reports nonnegative counters for:
- generator calls and tokens;
- semantic-assistance calls and tokens;
- evaluator calls and tokens;
- human review seconds;
- wall-clock milliseconds.

Aggregates use checked integer addition; overflow is an error, not wraparound. Derived cost fields include total tokens per confirmed-valid mutation, total tokens per unique verified failure, and human-review seconds per unique verified failure. Null is used when a denominator is zero.

These are resource-accounting fields supplied by the receipt. The validator does not prove that a model call occurred or independently measure wall time.

## 7. Qualification targets and competitive protocol

### Engineering qualification

- 100% acceptance of the clean synthetic preserving/changing controls;
- 100% rejection of malformed/duplicate/cross-reviewer/digest/denominator controls;
- zero unreviewed or unresolved mutations counted as confirmed valid;
- zero evaluator failures on invalid/unresolved mutations counted as verified discoveries;
- zero duplicate failure fingerprints counted as unique discoveries;
- exact plan/receipt and analyzer-revision binding;
- deterministic reports with verifiable commitments;
- exact resource accounting with overflow rejection;
- no raw custody, identity, blinding, causal, generalized-validity or scoring claim from synthetic receipts.

### Empirical comparison

Compare under matched conditions:
1. heuristic mutation baseline;
2. explanation/attribution-guided mutation;
3. explanation-guided mutation plus automated semantic checking;
4. independent human semantic adjudication;
5. Symthaea's evidence-linked mutation pipeline.

Freeze generator revisions, task sets, evaluator/review protocols, model calls, total tokens, semantic-verification budgets, human review time, mutation count and deduplication policy.

Report:
- confirmed-valid mutation yield over all planned variants;
- adjudicable panel concordance with disputed/insufficient/abstained counts;
- verified failure observations and unique verified failure fingerprints;
- invalid/unconfirmed mutation failures excluded;
- mutation diversity and duplicate rate;
- total tokens, calls, human-review seconds and wall time;
- valid unique failure discoveries per equal total budget.

Do not maximize mutation volume. A larger raw failure count is not better if semantic validity, independent confirmation or cost worsens.

## 8. Explicit claim boundary

The report unconditionally records:
- raw_artifact_custody_proven=false;
- reviewer_identity_authenticated=false;
- reviewer_blinding_proven=false;
- panel_reference_validity_proven=false;
- evaluator_execution_proven=false;
- causal_effect_identified=false;
- generalized_validity_estimated=false;
- external_anchor_required=true;
- scoring_eligible_declared=false.

Hashes bind declared bytes; they do not store or authenticate raw artifacts. Pseudonymous identifiers are not real-world identity proof. Agreement is not semantic truth. A receipt is not an execution attestation. A fixed synthetic fixture is not an empirical robustness study.

The next execution-custody layer should externally anchor the plan before generation, retain encrypted or independently retrievable artifact bytes, sign runner receipts, prove prompt/model revision and execution lineage, blind independent reviewers, and preregister sampling/inference. That must be a separately qualified layer, not inferred from this validator.

## 9. Integration

- **#9152 / SCI-METAMORPHIC-EVALUATOR-001:** supplies paired relation outcomes.
- **#9243 / SCI-FACTORIAL-JUDGE-BIAS-001:** measures interacting nuisance effects over balanced designs.
- **#9343 / SCI-MUTATION-VALIDITY-001:** qualifies the semantic validity of generated interventions and ensures only confirmed-valid mutations contribute verified failure discoveries.

Together, these layers distinguish whether an evaluator behaves consistently, whether its behavior is biased by interacting nuisance factors, and whether the test cases used to expose failures are actually valid.
