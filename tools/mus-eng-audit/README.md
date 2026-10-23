# mus-eng-audit

A standalone Rust audit utility for the proposed MUS-ENG benchmark corpus.

It validates:
1. the schema document against the JSON Schema meta-schema;
2. the corpus against the actual Draft 2020-12 schema;
3. cross-field policy invariants which JSON Schema alone cannot express;
4. known fault-to-disposition/evidence-ceiling expectations;
5. unique case identities and complete accounting of missing evidence.

The report is explicitly limited to **synthetic fixture consistency**. It is not a calibration verifier, physical measurement verifier, product qualification, release authority, or human-preference evaluator.

## Run

From the repository root, with Cargo available:

```sh
cargo run --manifest-path tools/mus-eng-audit/Cargo.toml -- \
  --schema schemas/instrument-engineering/mus_eng_corpus_v0_1.schema.json \
  --corpus fixtures/instrument-engineering/mus_eng_corpus_v0_1.json
```

The tool prints a deterministic JSON report to stdout. Exit codes:
- `0`: schema + fixture semantics are internally consistent; claims remain unqualified.
- `1`: schema-instance or policy invariant rejected.
- `2`: I/O, JSON parse, schema compilation, or CLI error.

The tool has no network retrieval feature enabled for JSON Schema references. The current schema uses only a local `#/$defs/case` reference.

## Safety boundary

The validator checks the *declared expected outcomes of synthetic fixtures*. It does not independently inspect physical evidence or prove that a real measurement, calibration certificate, assembly, test, license grant, or repair actually exists. A production evidence resolver and independent trust policy would be separate work.

## Tests

```sh
cargo test --manifest-path tools/mus-eng-audit/Cargo.toml
```
