# SCI-METAMORPHIC-EVALUATOR-001 — Intervention-based evaluator consistency

**Status:** deterministic paired-receipt auditor and synthetic fixture under development. No judge execution, causal effect, human semantic-validity study or empirical Symthaea robustness result is claimed.

**Tracking:** [Issue #9150](https://github.com/Luminous-Dynamics/symthaea/issues/9150).

## 1. The paradigm shift

Static benchmark evaluation asks whether one output matches one expected answer. Metamorphic evaluation asks whether a system's outputs obey necessary relations across controlled transformations.

The key change is from:
`input -> output -> score`

to:
`frozen input -> controlled intervention -> paired executions -> predeclared relation -> auditable violation`

This is especially useful where exact expected outputs are expensive or incomplete, but the relationship between two outputs is known. It does not eliminate the oracle problem: relation validity itself must be independently justified. It does, however, create testable invariants and directional expectations without requiring a new absolute gold label for every transformation.

The name "intervention-based" is deliberate. This tool audits paired interventions; it does **not** identify causal effects in the statistical sense, prove the intervention changed only one latent factor, or estimate population-level effects.

## 2. Research baseline

### LGMT — logic-grounded metamorphic testing

LGMT (May 2026) derives metamorphic relations from first-order-logic equivalences. Its experiments on six state-of-the-art models report reasoning defects that static reference-based evaluations missed, particularly sensitivity to symbol-level and conclusion-level variations.

Source: https://arxiv.org/abs/2605.23965

**Symthaea opportunity:** extend equivalence testing from logical reasoning to claim/evidence/rubric/evaluator contracts, with reference-linked interventions and locality checks.

### LLMSafetyChoice metamorphic study

The May 2026 safety study defines seven metamorphic relations for safety evaluation. It reports 1,408 scenarios across 11 models, two languages, eight safety categories and seven transformations in addition to its constructed benchmark.

Source: https://doi.org/10.1007/s11390-026-5705-z

**Symthaea opportunity:** unify safety and epistemic relations in one receipt format, while preserving task-family strata and hard failures.

### Explanation-guided metamorphic testing

The 2026 ESEM study combines attribution-guided mutation with semantic verification. In its tested settings, it reports 2.30× more verified failure-inducing cases than heuristic mutation and identifies shortcut dependence on named entities and formatting cues.

Source: https://arxiv.org/abs/2608.07076

**Symthaea opportunity:** measure verified failure discovery per equal mutation/verification budget, and keep semantic validity separate from mutation volume.

### FairJudge

FairJudge (ICML 2026) models judging as an adaptive regularized policy. Its paper targets position, length, format and model-provenance bias, as well as inconsistency between pointwise and pairwise judging modes.

Source: https://proceedings.mlr.press/v306/yang26n.html

**Symthaea opportunity:** evaluate pointwise/pairwise agreement and metamorphic behavior as independent qualification axes, rather than assuming training or ensembling makes a judge consistent.

These papers provide research baselines and ideas. Their reported performance is not directly comparable to Symthaea until data, tasks, budgets and adjudication protocols are matched.

## 3. Seven relation families

### A. Invariance

Transform only a declared non-semantic property: formatting, whitespace, neutral identifiers, equivalent paraphrase, harmless serialization or irrelevant verbosity.

Expected relation: same decision, hard-veto state, selected candidate, source-origin count and criterion scores within a preregistered tolerance.

Do not label a transformation meaning-preserving based on superficial similarity alone. Equivalence must be justified by the plan owner and, in empirical work, checked independently.

### B. Directional sensitivity

Modify material evidence or scope in a declared direction:
- remove or weaken decisive evidence;
- add relevant admissible support;
- retract/supersede a source;
- alter a material unit, target date, population or jurisdiction.

Expected relation: support and affected criterion scores move only in the preregistered direction. Evidence withdrawal must not raise the affected evidence score or make the claim more supported.

The v1 support ordering is a coarse diagnostic, not a probabilistic truth scale. Mixed, insufficient, ambiguous, stale and abstaining states remain distinct in the original receipt.

### C. Locality

Intervene on one criterion or one declared construct dimension.

Expected relation: the affected criterion changes observably; unaffected criteria remain within the preregistered score tolerance. A future version should consume explicit dependency rules from the structured rubric spec so declared overrides are not incorrectly flagged as leakage.

### D. Order equivariance

Swap candidate positions while preserving candidate content and identity.

Expected relation: the same candidate identity remains selected, and its position changes from first to second or second to first. Merely requiring the selected position to be unchanged would encode the wrong property.

### E. Provenance non-duplication

Add a mirror, repost or duplicate snapshot whose declared source ancestry is unchanged.

Expected relation: unique source-origin count must not increase; decision and criterion scores remain within tolerance. This is a structural relation over declared source ancestry, not authentication that the ancestry declaration is complete or true.

### F. Cross-mode consistency

Compare pointwise candidate scoring with pairwise preference under the same rubric/reference.

Expected relation: the pairwise preference must be among the pointwise top-scoring candidate(s). Ties are preserved as ties; the checker does not force a unique winner.

### G. Counterfactual consistency

Change a material semantic value and its corresponding evidence together—for example a measurement unit, target date or entity—so the expected conclusion changes. The plan declares the expected variant decision.

Expected relation: the variant decision equals the preregistered decision and differs from the baseline decision. A counterfactual pair is invalid unless the evidence and claim transformation are both specified and semantically verified.

## 4. Executable v1

Files:
- `schema/metamorphic-evaluator-plan.schema.json`
- `schema/metamorphic-evaluator-receipt.schema.json`
- `tools/metamorphic_evaluator_audit.py`
- `tools/metamorphic_evaluator_fixture.py`
- `tools/test_metamorphic_evaluator_audit.py`

Generate and audit the example:
```sh
python3 research/science-bench-refresh/tools/metamorphic_evaluator_fixture.py /tmp/metamorphic-example
python3 research/science-bench-refresh/tools/metamorphic_evaluator_audit.py \
  /tmp/metamorphic-example/metamorphic-plan.example.json \
  /tmp/metamorphic-example/metamorphic-receipt.example.json
python3 -m unittest discover -s research/science-bench-refresh/tools -p 'test_metamorphic_evaluator_audit.py' -v
```

The plan commits to the evaluator revision, prompt digest, rubric digest, criterion set, score tolerance, per-pair input digests and relation rule. The receipt binds the exact plan bytes and contains both baseline and variant structured outputs.

The report preserves pair-level failures, criterion deltas, per-family denominators and maximum observed drift. Structural receipt validity is separate from relation success: a correctly formed receipt can fail a metamorphic relation. Missing, extra or duplicate pair receipts are rejected rather than treated as omitted observations.

The example contains seven clean synthetic pairs, one per relation family. It is generated deterministically and explicitly declares no evaluator execution or scoring qualification.

## 5. Metrics and targets

Report:
- relation pass rate by family and exact denominator;
- invariance violation rate and maximum criterion drift;
- directional sensitivity rate, separated by positive support and evidence withdrawal;
- locality leakage across unaffected criteria;
- candidate-identity consistency under position swap;
- source-origin inflation under mirror/duplicate interventions;
- pointwise/pairwise contradiction rate;
- counterfactual decision transition accuracy;
- later: verified failure discoveries per mutation budget, semantic-validity precision and evaluation cost per unique failure.

Engineering qualification targets for the tool:
- 100% acceptance of the clean seven-relation synthetic fixture;
- 100% detection of the enumerated hostile relation mutations;
- 100% rejection of malformed, duplicate, missing, extra and plan-tampered receipts;
- deterministic report bytes for identical plan/receipt bytes;
- no pass when any planned pair is absent;
- no averaging away of hard relation violations.

These targets qualify the audit implementation, not the evaluated model.

### Competitive comparison

Under matched conditions, compare:
- static exact-reference evaluation;
- LGMT-style logical-equivalence testing;
- LLMSafetyChoice-style safety metamorphic relations;
- heuristic mutation generation;
- explanation-guided mutation plus semantic verification;
- FairJudge-style bias/cross-mode consistency controls;
- Symthaea intervention ledger and evidence-aware metamorphic evaluator.

Match model versions, prompts, candidate sets, task families, mutation budgets, semantic-verification budgets, human review effort and number of judge calls. Report both failure discovery and the fraction of mutations independently accepted as semantically valid. A larger number of invalid mutations is not an improvement.

A practical target is to reproduce the published 2.30× failure-discovery comparison on a matched subset before attempting to beat it. This is a replication target from one study, not a universal SOTA threshold. The stronger Symthaea target is more verified failures per equal total budget, with equal or higher independently confirmed semantic validity.

## 6. Explicit claim boundary

The report permanently sets:
- `evaluator_execution_proven=false`;
- `receipt_authenticity_proven=false`;
- `causal_effect_identified=false`;
- `generalized_accuracy_estimated=false`;
- `construct_validity_proven=false`;
- `external_anchor_required=true`;
- `scoring_eligible_declared=false`.

Hashes bind bytes; they do not prove who generated the receipt, that an evaluator actually ran, that the intervention changed only one semantic factor, or that a human-validated causal effect exists.

A later runner should provide signed/anchored execution attestations, precommitted randomness, repeated executions, independently reviewed semantic-validity labels and a declared sampling design. That is a separate layer and must not be silently inferred from this static auditor.

## 7. Integration

- **SCI-REFERENCE-ASSURANCE-001 / #9121:** evidence removal, source lifecycle and ancestry interventions.
- **SCI-RUBRIC-EXPLOITABILITY-001 / #9124:** rubric-preserving and rubric-changing interventions.
- **SCI-JUDGE-GEOMETRY-001 / #9131:** matched outcome matrices can feed the geometry report; this checker does not replace score-space analysis.
- **SCI-JUDGE-ROBUSTNESS-001 / #9065:** position/length/format/provenance and pointwise/pairwise test families.

The long-term objective is one evidence-linked intervention protocol whose plans, executions, receipts, references and analysis reports can be independently reconstructed across Symthaea, Mycelix and Symtropy.
