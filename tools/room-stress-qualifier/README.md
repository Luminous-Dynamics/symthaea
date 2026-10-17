# Room Stress Qualifier

Independent Rust CLI for validating the synthetic room-imagination stress manifest. It checks JSON Schema Draft 2020-12 and additional semantic invariants; it is deliberately outside the main Symthaea workspace.

## Run from repository root

```sh
cargo test --manifest-path tools/room-stress-qualifier/Cargo.toml
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml -- --json
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml -- --evaluate RUN_REPORT.json EVIDENCE_ROOT
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml -- --evaluate-json RUN_REPORT.json EVIDENCE_ROOT
```

The default paths point to the checked-in benchmark and schema relative to the repository root. To validate alternate files, pass both paths, optionally after `--json`. Invalid input exits nonzero.

In JSON mode the tool emits a deterministic, content-addressed receipt with SHA-256 digests of the exact benchmark, benchmark schema, run-report schema, validator source and Cargo manifest, plus the validator version, case/family/risk counts and explicit claim ceiling. The receipt proves only that this validator accepted those manifest inputs; it is not a signed attestation and does not qualify spatial geometry or building performance.

The per-case expected observations and stable failure codes are stored in `ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json`. Candidate reports must conform to `ROOM_IMAGINATION_STRESS_RUN_REPORT_V0_1.schema.json` and bind to the exact benchmark bytes with SHA-256. The evaluator checks that every expected case is present exactly once, observations match their oracles, and each required observation has exactly one evidence-reference record. Each reference must point to a real file beneath the explicit `EVIDENCE_ROOT`; the evaluator canonicalizes the root and artifact, rejects absolute paths, dot segments and symlink escapes, limits each artifact to 64 MiB, and recomputes its SHA-256 digest. Artifact bytes are checked for identity/integrity, but **their content is not semantically inspected to prove the observation**. The caller-supplied `artifact_revision` is metadata, not independent currentness proof.

`--evaluate-json` emits structured JSON with machine-readable case IDs, assertion IDs, severities, stable failure codes, and reasons; it returns a non-zero exit on failed assertions. This enables downstream tools and CI to consume results without scraping terminal text.

The evaluator's synthetic golden-answer unit fixture points at a checked-in fixture file and tests evaluator mechanics only. It does not represent a real Symthaea output and must not be presented as a system benchmark score.

## What this does — and does not — prove

It checks the schema itself, the manifest/schema contract, exact case ordering and IDs, non-empty required lists, allowed risk values, unknown-field rejection and duplicate IDs. Unit tests include negative controls for malformed JSON, missing/extra fields, unsupported risk values, empty required lists, sequence gaps, wrong counts, invalid schemas, deterministic receipts, incorrect observations, missing evidence references, benchmark digest mismatches, duplicate result cases, incorrect artifact hashes, missing artifacts, path traversal, schema drift and structured JSON output. The synthetic golden-answer fixture tests evaluator mechanics only; it is not a simulated design system.

It does **not** run geometry predicates, daylight/energy/thermal/acoustic simulation, accessibility assessments, code compliance, safety reviews, professional approval or human studies. Its success means only that the fixture manifest passes these checks. A future geometry/simulation adapter needs its own qualified contract, golden corpus, uncertainty model and independent checker.
