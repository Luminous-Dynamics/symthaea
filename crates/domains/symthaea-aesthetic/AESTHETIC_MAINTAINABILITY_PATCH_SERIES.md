# Symthaea Aesthetic Maintainability Patch Series

This series follows the production-assurance snapshot. It consolidates repeated
infrastructure and gives downstream creative systems one safe path through the
crate without changing the aesthetic scoring model.

## Bundle A — Core foundations

1. `refactor: centralize stable integrity identifiers`
2. `refactor: share atomic JSON persistence`
3. `feat: publish persisted schema catalog`

### Why

Registry snapshots, evaluation envelopes, replay partitioning, memory, study
ledgers, and governance manifests previously implemented their own hashing or
atomic-write logic. Centralization makes compatibility behavior inspectable and
prevents those implementations from drifting apart.

### Important integrity boundary

`fnv1a64:*` identifiers are deterministic compatibility identifiers, not
cryptographic security digests. They must be authenticated by a stronger outer
evidence or signature layer when tamper resistance matters.

## Bundle B — Runtime integration

4. `fix: bind reference extractors to registry descriptors`
5. `compat: read legacy reference extractor versions`
6. `feat: add verified evaluation pipeline facade`
7. `feat: add explicit batch failure semantics`

### Why

The reference extractors emitted the non-semantic version string `1`, while the
registry required `major.minor.patch`. Each reference extractor now owns a
matching descriptor and emits `1.0.0`.

`AestheticPipeline` guarantees that one evaluation performs all of the following:

- verifies the registry and exact extractor descriptor;
- rejects runtime extractor identity or modality mismatches;
- rejects reports below the configured evidence threshold;
- persists the exact assessment policy;
- creates a replayable `EvaluationEnvelope`.

`evaluate_batch` makes partial failure explicit through `Continue` and
`FailFast` modes. It never represents an unattempted artifact as a failure.

## Bundle C — Verification ergonomics

8. `test: add deterministic downstream fixtures`
9. `feat: add independent pipeline consistency audit`
10. `test: add deterministic core invariant sweeps`

### Why

Downstream crates can enable the `test-support` feature to reuse deterministic
music, visual, text, registry, request, and pipeline fixtures.

`audit_pipeline_output` independently rechecks registry identity, report
validity, report digest, evidence lineage, policy replay, envelope validity, and
registry digest. It is intended for CI and release replay, not only unit tests.

`run_core_invariant_sweep` performs deterministic dependency-free property-style
checks across pathological numeric inputs, all creative modes, all primary
modalities, optional preference evidence, utility bounds, confidence bounds, and
assessment replay.

## Example integration

```rust
use symthaea_aesthetic::{
    AestheticModality, CreativeMode, MusicEvidenceExtractor,
    audit_pipeline_output,
};
use symthaea_aesthetic::test_support::{
    evaluation_request, music_frame, production_music_pipeline,
};

let pipeline = production_music_pipeline()?;
let extractor = MusicEvidenceExtractor::default();
let output = pipeline.evaluate(
    &extractor,
    &music_frame(64),
    evaluation_request(
        "muse-render-42",
        AestheticModality::Music,
        CreativeMode::Refine,
    ),
)?;

let audit = audit_pipeline_output(
    &output,
    pipeline.registry(),
    pipeline.descriptor(),
    1e-5,
);
assert!(audit.passes());
```

The example requires the `test-support` feature because it uses deterministic
fixtures. Production applications should supply their own artifacts and request
metadata while retaining the same pipeline and audit path.

## Compatibility notes

- Existing FNV registry and envelope identifiers remain byte-compatible.
- Existing persisted memory, study, and governance JSON formats are unchanged.
- New reference reports emit `1.0.0`. The registry reader accepts legacy `1`
  and `1.0` report strings as migration shorthand, while new registry entries
  should use the canonical semantic version.
- The default crate feature set remains dependency-neutral. `test-support` adds
  only fixture APIs and no external dependencies.
- No aesthetic weights or feedback dynamics are changed by this series.

## Recommended workspace verification

```text
cargo fmt --all -- --check
cargo test -p symthaea-aesthetic --all-features
cargo clippy -p symthaea-aesthetic --all-targets --all-features -- -D warnings
cargo test -p symthaea-muse
cargo test -p symthaea-canvas
```

Also run the release invariant sweep with at least 50,000 cases and archive its
serialized report alongside the schema-catalog digest and pipeline audit output.
