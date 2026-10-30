# Luminous Evidence Profile conformance tool

This isolated Rust workspace validates the LEP v0.1 predicate embedded in an in-toto Statement v1. It exists as a focused tool so it can be built without bypassing the repository's unrelated root-workspace member-discovery problem.

## Run locally

From the repository root:

    cargo run --manifest-path tools/lep-conformance/Cargo.toml --bin lep-conformance -- docs/research/fixtures/luminous-evidence-v0.1/valid-passed.json

Run tests with:

    cargo test --manifest-path tools/lep-conformance/Cargo.toml

The CLI prints a JSON report with `conforms` and `scope: LEP_SCHEMA_AND_SEMANTIC_CHECKS_ONLY`. Exit code 0 means the supplied record conformed to this schema and these implemented semantic checks; 1 means validation failed; 2 means the input could not be read or parsed. `conforms: true` is not an authenticated evidence verdict and does not mean the claim is true.

## Performance microbenchmark

The benchmark binary compares strict parsing + schema validation with strict parsing + schema + semantic validation on a synthetic valid statement. It caches schema compilation via the library, warms the process-local cache, and reports p50/p95/mean latency and a p50-derived throughput estimate. It makes no claims about signature verification, artifact I/O, run authenticity, or scientific truth.

Example:

    cargo run --release --locked --manifest-path tools/lep-conformance/Cargo.toml --bin lep-benchmark -- 10240 1000

Repeat on controlled hardware and record the actual input size and exact commit; do not compare noisy CI timings as a SOTA result.

## Validation performed

The CLI performs a bounded streaming read (16 MiB maximum), then parses JSON with duplicate object keys rejected at every nesting level and serde_json's default recursion limit, then validates the predicate against the bundled JSON Schema Draft 2020-12 schema, with format assertions enabled and unknown formats rejected, then performs semantic validation. The schema uses only in-document references; no remote schema retrieval is required.

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

The parser rejects oversized statements before deserialization; excessive JSON nesting is rejected by serde_json's recursion limit. These are parser resource-safety controls, not a complete CPU/memory sandbox.

The standalone lockfile is Cargo-generated and has been verified by the hosted `cargo test --locked` and Clippy steps. The latest successful dedicated LEP workflow run is run 45 (run ID `41035270018`), on PR source head `b7378d6641d23ac0f803ac737eb47493cedfaa31`. It passed formatting, all 23 Rust unit tests, Clippy with warnings denied, positive/negative CLI fixture checks, the 16 MiB oversize boundary check, benchmark collection, and artifact upload. The 23 tests include the executable conformance manifest, 1,000 deterministic invalid mutations, duplicate-key rejection, oversized-input rejection and deep-nesting rejection. Five independent benchmark-command batches per input size were collected; the aggregation and measured results are recorded in `docs/research/LEP_CONFORMANCE_BENCHMARK_RESULTS_2026-10-30.md` and `docs/research/LEP_CONFORMANCE_BENCHMARK_PLAN.md`. The hosted result validates the prototype's current conformance scope only; it does not authenticate DSSE signatures, resolve/re-hash artifacts, prove a run occurred, or establish scientific truth.

The latest completed successful code-test revision was `b7378d6641d23ac0f803ac737eb47493cedfaa31` (LEP workflow run 45). Documentation has since been reconciled with the repeated-process report; a fresh workflow is triggered for the settled PR head so the check receipt can bind to the complete review revision, not just the earlier code revision.
