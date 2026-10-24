# EsotericaBench v0 seed pack

This directory contains the initial claim schema and adversarial starter cases for the Luminous Dynamics Esoterica Knowledge Fabric.

- `claim.schema.json`: JSON Schema Draft 2020-12 for evidence-linked claim records.
- `case.schema.json`: JSON Schema Draft 2020-12 for evaluation case records.
- `cases.jsonl`: 12 development case fixtures; not a held-out evaluation set.
- `claim-fixtures.jsonl`: four synthetic claim records that exercise distinct epistemic lanes; they are not historical evidence.
- Shared source identities, source-family groups, split status and rights-review status: [source catalog](../../source-catalog/v0/README.md).
- Design and evaluation protocol: [ESOTERICA_KNOWLEDGE_FABRIC.md](../../../docs/research/ESOTERICA_KNOWLEDGE_FABRIC.md).

## Important status

This pack is not yet an executable benchmark runner, a fully expert-annotated corpus, or evidence of superiority over any model. A fixture passing schema validation proves shape, not epistemic correctness. The first next implementation step is a small deterministic Rust validator/scorer and tests for schema-version, source identity, rights status, evidence relation, and expected decision fields.

## Required before reporting a score

1. Validate every line against the versioned case schema and validate claim records against the claim schema.
2. Pin a source-pack manifest and the exact source editions/locators used.
3. Have domain reviewers adjudicate the historical and symbolic cases; retain disagreement labels.
4. Keep synthetic controls clearly separated from factual source-grounded questions.
5. Freeze an evaluation split, publish model/retrieval/config identities, and retain per-case outputs.
6. Compare baselines under equal source, prompt, context and compute budgets.
7. Report risk/coverage, citation precision, false lineage, abstention and rights-policy failures separately.

Critical access/provenance failures are release blockers. No queued CI result or planned test may be reported as a pass.


## Related module: MythologyBench

Mythology is a first-class adjacent domain, not a synonym for esotericism. The Egypt-first development pack is in [`benchmarks/mythology-bench/v0/`](../../mythology-bench/v0/README.md). It reuses the overall evidence/provenance design but keeps mythology-specific evaluation separate so results are not conflated with esotericism.


## Local validator

The isolated Rust validator lives in [`tools/knowledge-benchmark-harness/`](../../../tools/knowledge-benchmark-harness/). Run [`scripts/validate-knowledge-benchmarks.sh`](../../../scripts/validate-knowledge-benchmarks.sh) from a repository checkout. The script validates ten input groups: Esoterica cases, synthetic claim controls, Mythology cases, paired metamorphic interventions, the shared source catalog, synthetic source-pack fixtures, blinded annotation fixtures, comparison-plan fixtures, paired-result label fixtures, and narrative graph fixtures. It then runs `score-annotation-agreement` on synthetic review labels and `score-paired-results` on fabricated paired observations. Both reports are qualification artifacts only, not model-performance or real reviewer-reliability claims. The harness checks Draft 2020-12 schemas, local schema references, domain contracts, source IDs/URIs and split agreement, source-family split leakage, rights/access combinations, and paired-intervention integrity; it hashes the exact input/schema/lock bytes and never fetches source URLs. Cargo runs use `--locked`.

All current cases remain development fixtures; all catalog sources are metadata-only with unknown rights and pending review. No held-out model evaluation or SOTA comparison has been run. A contract validator passing would establish data and invariant checks only—not scholarly correctness, rights, empirical truth, or model quality.
