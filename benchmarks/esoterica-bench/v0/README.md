# EsotericaBench v0 seed pack

This directory contains the initial claim schema and adversarial starter cases for the Luminous Dynamics Esoterica Knowledge Fabric.

- `claim.schema.json`: JSON Schema Draft 2020-12 for evidence-linked claim records.
- `cases.jsonl`: one JSON object per case. Most current cases are development fixtures, not a held-out evaluation set.
- `../../docs/research/ESOTERICA_KNOWLEDGE_FABRIC.md` (repository-relative path from this directory requires navigating up to docs separately): design, baseline protocol, metrics and nonclaims.

## Important status

This pack is not yet an executable benchmark runner, a fully expert-annotated corpus, or evidence of superiority over any model. A fixture passing schema validation proves shape, not epistemic correctness. The first next implementation step is a small deterministic Rust validator/scorer and tests for schema-version, source identity, rights status, evidence relation, and expected decision fields.

## Required before reporting a score

1. Validate every line against a versioned JSON Schema and validate all case-level contracts.
2. Pin a source-pack manifest and the exact source editions/locators used.
3. Have domain reviewers adjudicate the historical and symbolic cases; retain disagreement labels.
4. Keep synthetic controls clearly separated from factual source-grounded questions.
5. Freeze an evaluation split, publish model/retrieval/config identities, and retain per-case outputs.
6. Compare baselines under equal source, prompt, context and compute budgets.
7. Report risk/coverage, citation precision, false lineage, abstention and rights-policy failures separately.

Critical access/provenance failures are release blockers. No queued CI result or planned test may be reported as a pass.
