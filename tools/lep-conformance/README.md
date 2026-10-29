# Luminous Evidence Profile conformance tool

This isolated Rust workspace validates the LEP v0.1 predicate embedded in an in-toto Statement v1. It exists as a focused tool so it can be built without bypassing the repository's unrelated root-workspace member-discovery problem.

## Run locally

From the repository root:

    cargo run --manifest-path tools/lep-conformance/Cargo.toml -- docs/research/fixtures/luminous-evidence-v0.1/valid-passed.json

Run tests with:

    cargo test --manifest-path tools/lep-conformance/Cargo.toml

The CLI prints a JSON report. Exit code 0 means schema and semantic checks passed, 1 means validation failed, and 2 means the input could not be read or parsed.

## Validation performed

The CLI validates the predicate against the bundled JSON Schema Draft 2020-12 schema, with format assertions enabled and unknown formats rejected, then performs semantic validation. The schema uses only in-document references; no remote schema retrieval is required.

Semantic checks include:

- Exact in-toto Statement v1 and LEP v0.1 type/version.
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

Fixture digests are synthetic placeholders. The fixtures are conformance fixtures, not evidence of production runs.

The source is a prototype. Rust compilation and tests have not been run in the current authoring environment, which has no rustc/cargo/Nix toolchain. A generated standalone Cargo.lock and executable test result are required before treating it as release-ready.
