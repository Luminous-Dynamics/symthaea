# Luminous Evidence Profile conformance tool

This isolated Rust workspace validates the LEP v0.1 predicate embedded in an in-toto Statement v1. It exists as a focused tool so it can be built without bypassing the repository's unrelated root-workspace member-discovery problem.

## Run locally

From the repository root:

    cargo run --manifest-path tools/lep-conformance/Cargo.toml -- docs/research/fixtures/luminous-evidence-v0.1/valid-passed.json

Run tests with:

    cargo test --manifest-path tools/lep-conformance/Cargo.toml

The CLI prints a JSON report with `conforms` and `scope: LEP_SCHEMA_AND_SEMANTIC_CHECKS_ONLY`. Exit code 0 means the supplied record conformed to this schema and these implemented semantic checks; 1 means validation failed; 2 means the input could not be read or parsed. `conforms: true` is not an authenticated evidence verdict and does not mean the claim is true.

## Validation performed

The CLI first parses JSON with duplicate object keys rejected at every nesting level, then validates the predicate against the bundled JSON Schema Draft 2020-12 schema, with format assertions enabled and unknown formats rejected, then performs semantic validation. The schema uses only in-document references; no remote schema retrieval is required.

Semantic checks include:

- Exact in-toto Statement v1 and LEP v0.1 type/version.
- Duplicate-key-free JSON parsing before schema validation.
- Non-empty subject list and supported digest algorithms/shapes.
- Immutable lowercase 40- or 64-character hexadecimal source revision.
- Execution status/result consistency and valid run-id semantics.
- Exact structured equality of declared target and verifier-checked target.
- Verifier identity, non-empty executed checks, known outcomes and unique check IDs.
- Aggregate outcome agreement with check outcomes.
- A passing verification requires completed/passed execution and all-passing checks.
- An unrun verification cannot claim a verifier, checked target, or checks.

## Security and scientific limits

A passing report establishes only that the bundled schema and implemented consistency rules accepted the provided JSON. It does not authenticate DSSE signatures, authorize signers, fetch artifacts, recompute artifact bytes, prove the reported command really executed, independently rerun checks, or prove any scientific or normative claim true. Production admission must add those verifiers and the domain-specific evidence-resolution policy.

Fixture digests are synthetic placeholders. The fixtures are conformance fixtures, not evidence of production runs. The duplicate-key fixture is intentionally rejected by the strict parser before it reaches schema validation.

A one-off local cross-check of the current schema and four structural fixtures was run with Python `jsonschema` 4.26.0, `Draft202012Validator.check_schema`, and `FormatChecker`: the valid and stale-target predicates were accepted; queued-but-passed and missing-target predicates were rejected. A strict Python `object_pairs_hook` also rejected the committed duplicate-key fixture. These are independent schema/fixture checks only; they do not run the Rust implementation or its tests.

The source is a prototype. Rust compilation and tests have not been run in the current authoring environment, which has no rustc/cargo/Nix toolchain. A standalone Cargo.lock is checked in and its transitive dependency closure was extracted from a public lockfile using matching direct dependency versions and `jsonschema` default features disabled; it was not generated or validated by Cargo here. The new CI workflow uses `--locked` for test and lint steps so a lockfile mismatch fails instead of silently rewriting dependency resolution.
