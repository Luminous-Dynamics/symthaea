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
