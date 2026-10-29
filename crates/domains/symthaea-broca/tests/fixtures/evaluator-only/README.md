# Counterfactual evidence audit fixtures

- `counterfactual-evidence-audit-v1.jsonl` is the **input fixture corpus**. It contains paired query/evidence cases and a mutation contract so the structural validator can check that the stated single mutation matches the actual evidence diff.
- `evaluator-only/counterfactual-evidence-audit-labels-v1.jsonl` contains expected answerability, decisions, claims, and supporting evidence IDs. It is for tests/evaluation only and must never be passed to memory ingestion, retrieval, query planning, semantic role extraction, refinement, or the reader.
- `symthaea_broca::counterfactual_evidence_audit::CounterfactualPairFixture::runtime_case(index)` emits only the query, as-of timestamp, and evidence list. It strips pair IDs, mutation descriptions, and expected decision relations before a case is passed downstream.

The examples are synthetic regression fixtures, not benchmark examples and not model-evaluation results. They exercise ten families: polarity, temporal validity, entity binding, scope/premise, source dependence, missing requirements, ordered transitions, correction updates, qualifier context, and memory influence. The memory-influence pair adds a high-overlap but non-decisive proposition and labels each atom's target influence as Ignore, Bound, or Control. These labels are evaluation-only and are not model inputs.

Run the Rust structural validator from the workspace root:

```sh
cargo run -p symthaea-broca --bin broca-counterfactual-audit -- crates/domains/symthaea-broca/tests/fixtures/counterfactual-evidence-audit-v1.jsonl
cargo test -p symthaea-broca counterfactual_evidence_audit
```

The CLI prints `semantic_support=NOT_EVALUATED` by design. A structural pass only establishes schema/diff integrity; it does not prove that an evidence span entails a claim or that the expected label is semantically correct. The test-only label linkage checks each case has exactly one evaluator label and that labels are associated with the correct pair.

## Atom-ablation scoring protocol

The new offline evaluator is exposed as:

```sh
cargo run -p symthaea-broca --bin broca-memory-influence-score -- \
  path/to/observations.jsonl \
  crates/domains/symthaea-broca/tests/fixtures/counterfactual-evidence-audit-v1.jsonl \
  crates/domains/symthaea-broca/tests/fixtures/evaluator-only/counterfactual-evidence-audit-labels-v1.jsonl
```

The observations file must be produced by an actual frozen reader adapter using schema V2. Each row records one case/atom ablation: full and ablated structured material fields, exact serialized context token counts, run ID, reader revision, fixture SHA-256, prompt SHA-256, tokenizer SHA-256, system commit SHA, ordered full/ablated evidence-ID manifests, and SHA-256 digests for the canonical full/ablated `CounterfactualRuntimeCase` JSON projections. The scorer reconstructs these projections from the fixture and rejects any ablation that is not exactly the fixture case minus the target atom. That validates the declared request payload but does not, by itself, attest that a remote serving endpoint consumed those exact bytes; the adapter should bind request logs/receipts to the payload. Material fields use stable paths under `claim.`, `action.`, or `decision.`; missing values are represented by absent keys, not empty strings.

The scorer recomputes the fixture hash, validates every case/evidence ID against the input corpus, requires an exact one-to-one match between labelled targets and ablation observations, and reports over-use and under-use separately for Ignore, Bound, and Control. It does not infer semantic equivalence from free-form answer strings: the run adapter must supply stable structured claim/action fields and the target-field policy must be independently reviewed.

No real reader observations are included yet. The Rust scoring tests use explicitly synthetic outputs to check the metric mechanics only; their rates are not model performance. Never put this evaluator-only label file into the memory backend or answer prompt.
