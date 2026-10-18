# mus-doe-audit

Standalone Rust audit utility for the synthetic MUS-DOE-001 experiment-design corpus.

It checks Draft 2020-12 schema meta-validity and corpus conformance, then checks policy semantics that schema validation alone cannot establish:

- unique case IDs;
- positive-control group separation across declared independence axes;
- same-unit/batch/builder/environment/session/lot leakage;
- consistent declared fault modes and expected dispositions;
- prohibition on locked-test outcomes entering selection or stopping;
- admissibility of selected experiments;
- a declared seed for seeded-random policies;
- explicit rejection of post-hoc policy changes.

The corpus is synthetic. This tool does not estimate actual information gain, execute a DOE policy, authenticate evidence references, fit a physical model, verify calibration, or qualify instruments.

## Run from repository root

```sh
cargo run --manifest-path tools/mus-doe-audit/Cargo.toml -- \
  --schema schemas/instrument-engineering/mus_doe_split_corpus_v0_1.schema.json \
  --corpus fixtures/instrument-engineering/mus_doe_split_corpus_v0_1.json
cargo test --manifest-path tools/mus-doe-audit/Cargo.toml
```

Exit codes: 0 = internally consistent synthetic policy corpus only; 1 = schema/policy rejection; 2 = I/O, parse, schema-compilation or CLI error.

## Evidence boundary

A split manifest cannot establish that a source record is physically independent, that a group identifier is truthful, or that an observation really belongs to the claimed unit/batch. The validator checks supplied synthetic group assignments. Real campaigns require authoritative design/build/unit identities and integration with existing FIELD/ObservationEnvelope and artifact-provenance owners.
