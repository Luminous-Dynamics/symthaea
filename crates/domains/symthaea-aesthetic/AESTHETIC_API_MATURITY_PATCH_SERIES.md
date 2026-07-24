# Symthaea Aesthetic API Maturity Patch Series

This series follows the maintainability snapshot. It does not add new aesthetic
weights or theoretical priors. It makes the existing evidence pipeline easier to
adopt, pin, benchmark, archive, and certify across Symthaea creative systems.

## Bundle A — Public API contracts

1. `feat: publish stable public API contract`
2. `feat: add curated prelude and legacy compatibility module`
3. `feat: add profile-aware pipeline builder`
4. `feat: bind API schemas and extractor into contract snapshots`

### Why

Package versions, persisted schema versions, and extractor versions are separate
compatibility dimensions. `ApiContract`, `SchemaCatalog`, and `ContractSnapshot`
make those dimensions explicit and digestible. The curated prelude gives new
integrations a deliberately small supported surface while the original scalar
API remains available through `legacy`.

## Bundle B — Portable evaluation evidence

5. `feat: add portable evaluation receipts`
6. `feat: add self-verifying evaluation archives`
7. `feat: close archive creation into verified pipeline`

### Why

A production evaluation should be independently reviewable without needing the
source artifact or reconstructing mutable runtime state. Receipts bind report,
assessment, envelope, registry, API, schema catalog, extractor version, and
build fingerprint. Archives package the exact output, contract snapshot, and
receipt and use the crate's atomic JSON persistence path.

## Bundle C — Performance evidence

8. `feat: add dependency-free benchmark reports and budgets`
9. `feat: benchmark verified pipelines over reusable corpora`

### Why

Performance claims are operational evidence, not aesthetic truth. The benchmark
harness records latency percentiles, throughput, failures, and explicit budgets
without adding a Criterion dependency to the production crate. Pipeline corpora
exercise the same verified path used by downstream applications.

## Bundle D — Adoption certification

10. `feat: add explicit downstream integration readiness profiles`
11. `feat: certify downstream adoption from complete release evidence`
12. `fix: harden schema and archive compatibility boundaries`
13. `test: add cross-modality downstream adoption fixtures`
14. `docs: document API maturity and adoption workflow`

### Why

Muse, Canvas, voice, poetry, and the game director have different modality and
extractor requirements. Integration profiles fail closed on API generation,
modality, channels, extractor version, determinism, confidence threshold, and
portable-archive support. Release certification combines contract, archive,
independent audit, integration readiness, and benchmark-budget evidence.

## Recommended adoption path

```rust
use symthaea_aesthetic::prelude::*;
use symthaea_aesthetic::{
    ContractSnapshot, EvaluationArchive, IntegrationProfile,
    evaluate_integration_readiness,
};

let extractor = symthaea_aesthetic::MusicEvidenceExtractor::default();
let pipeline = AestheticPipelineBuilder::for_descriptor(extractor.descriptor())?
    .profile(PipelineProfile::Production)
    .build()?;

let archive = pipeline.evaluate_archived(&extractor, &artifact, request)?;
archive.validate()?;

let readiness = evaluate_integration_readiness(
    &IntegrationProfile::muse(),
    &ApiContract::current(),
    &archive.contract,
    &pipeline,
)?;
assert!(readiness.is_ready());
```

## Compatibility notes

- No aesthetic policy weights or feedback dynamics change.
- The original crate-root exports remain available.
- New integrations should use `symthaea_aesthetic::prelude::*`.
- `AESTHETIC_API_VERSION` is independent of the Cargo package version.
- Contract snapshots, receipts, archives, benchmark reports, and release
  certifications use schema version 1 and are listed in `SchemaCatalog`.
- Reference extractor legacy version strings remain readable through the prior
  compatibility path; new reports continue to emit canonical `1.0.0`.
- Internal `fnv1a64:*` identifiers remain deterministic compatibility keys, not
  cryptographic authentication. Sign the outer evidence package when tamper
  resistance matters.

## Workspace verification

```text
cargo fmt --all -- --check
cargo check -p symthaea-aesthetic --all-features
cargo test -p symthaea-aesthetic --all-features
cargo clippy -p symthaea-aesthetic --all-targets --all-features -- -D warnings
cargo test -p symthaea-muse
cargo test -p symthaea-canvas
```

For release evidence, archive the API contract digest, schema-catalog digest,
contract snapshot, evaluation archive, independent pipeline audit, benchmark
run, integration readiness report, and release certification report together.
