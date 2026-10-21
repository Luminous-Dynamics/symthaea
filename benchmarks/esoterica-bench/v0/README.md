# EsotericaBench v0 seed pack

This directory contains the initial claim schema and adversarial starter cases for the Luminous Dynamics Esoterica Knowledge Fabric.

- `claim.schema.json`: JSON Schema Draft 2020-12 for evidence-linked claim records.
- `case.schema.json`: JSON Schema Draft 2020-12 for evaluation case records.
- `cases.jsonl`: one JSON object per case. Most current cases are development fixtures, not a held-out evaluation set.
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

The isolated Rust validator lives in [`tools/knowledge-benchmark-harness/`](../../../tools/knowledge-benchmark-harness/). Run [`scripts/validate-knowledge-benchmarks.sh`](../../../scripts/validate-knowledge-benchmarks.sh) from a repository checkout. It validates the schema against the Draft 2020-12 meta-schema, validates instances with format checking enabled, refuses remote/file schema references, verifies graph IDs and evidence references, hashes exact input bytes and the dependency lockfile, and emits one JSON report per fixture set. It never fetches the source URLs.

The harness has been authored on the research branch but has **not been compiled or executed in this environment**; this container has no Rust/Cargo toolchain and cannot resolve GitHub network hosts. Do not interpret the source code or CI workflow as a passing qualification result. The generated lockfile must be reviewed and committed before calling repeated runs fully dependency-reproducible.
