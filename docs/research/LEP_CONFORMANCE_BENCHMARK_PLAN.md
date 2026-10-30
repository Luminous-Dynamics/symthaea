# LEP Conformance Benchmark Plan

**Status:** First hosted synthetic smoke baseline recorded; no SOTA or production-performance result is claimed  
**Version under test:** LEP predicate v0.1.0 over in-toto Statement v1  
**Implementation:** \`tools/lep-conformance\`

## Research position

LEP should complement, not replace, existing standards:

- [in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/blob/main/spec/README.md) provides the interoperable layers for predicate, statement, envelope, and bundle. LEP is a domain predicate profile, not a competing envelope.
- [SLSA Provenance v1](https://slsa.dev/spec/v1.2/provenance) is the comparison baseline for software-build provenance. LEP should reuse SLSA where a claim is specifically about build provenance and should not relabel a generic LEP record as SLSA provenance.
- [Rust \`jsonschema\`](https://docs.rs/jsonschema/0.59.0/jsonschema/) is the structural validation engine. LEP's added value must be measured in semantic checks (such as exact target binding and execution/result consistency), not in JSON Schema validation performance alone.
- DSSE/in-toto signature verification and signer identity policy remain separate gates. A structurally conforming LEP statement is not authenticated merely because this checker accepts it.

The standards do not establish a single SOTA score for general evidence profiles. Compare measurable claims: conformance, false acceptance, compatibility, authentication, target integrity, performance, and usability. Do not publish a global "beats SOTA" claim from a small synthetic fixture corpus.

## Benchmark layers and acceptance targets

| Layer | Measurement | Proposed release gate | Required evidence |
|---|---|---|---|
| Schema | Bundled schema passes its declared Draft 2020-12 meta-schema | 100% | Local result plus CI receipt bound to exact commit |
| Positive fixtures | Correctly formed synthetic statements accepted | 100% | Fixture ID, input digest, checker revision |
| Negative fixtures | Invalid status/result, missing target, malformed digest, duplicate IDs, stale target, and duplicate JSON keys rejected | 100% of labeled cases | One expected failure code per case; test must fail if that invariant stops firing |
| Mutational adversaries | Generate single-field and compound mutations of valid fixture | Zero false accepts across at least 1,000 deterministic mutations before v0.2 | Frozen mutation seed and per-case manifest |
| Parser differential | Strict LEP parser vs an independent strict JSON parser | 100% agreement on acceptance/rejection for shared test corpus | Include duplicate keys at root and nested levels, invalid UTF-8, malformed escapes, trailing values, extreme numbers |
| Cross-implementation | Rust checker vs a separately implemented validator | 100% agreement on published conformance corpus | Independent implementation, not a wrapper around the Rust checker |
| Target binding | Change exactly one target field after the verifier records its checked target | 100% rejection for source revision, configuration digest, and each input artifact mutation | One mutation per field; preserve valid syntax and digest shapes |
| Authentication | DSSE payload/signature tampering, unknown signer, wrong identity, and wrong trust root | 100% rejection when required verification fails; no "unsigned pass" fallback | Signature/key/identity test fixtures; separate from LEP-only status |
| Reproducibility | Repeated verification of the same bytes, schema, and tool build | Byte-identical normalized report and stable failure codes | Frozen test vector; deterministic ordering of diagnostics |
| Offline safety | Validate local records while network access is disabled | Zero network attempts or remote-reference retrieval | Schema has only local references; test with network sandbox or socket-denial harness |
| Performance | Warm validation latency and throughput at 1 KiB, 10 KiB, 100 KiB, and 1 MiB | First smoke p50 ratio: 1.139× at 10 KiB; provisional target remains ≤2× structural-only p50. Full gate requires five independent process samples per size | 20 warm-up iterations per path and 100 timed iterations per process, five process repeats, p50/p95, bytes/s, CPU/toolchain metadata |
| Memory / DoS | Deep nesting, oversized arrays, huge strings, extreme nesting, repeated malformed inputs | CLI and library reject serialized statements above 16 MiB; serde_json's default recursion limit rejects deeply nested structures; no unbounded resource claims | Parser tests, boundary cases, reject/limit results and peak RSS measurements |
| Operational safety | Queued or failed jobs cannot be promoted to passed; inconclusive results cannot hide failed checks | 100% of transition invariants covered by tests | Decision table plus negative controls |

The 1,000-mutation target is an engineering gate, not a statistical proof of zero production risk. Mutation generation must be reproducible, and compound mutations should not replace one-fault controls because they can hide which invariant failed.

## Required comparison matrix

| System / baseline | Use it for | Do not claim it already does |
|---|---|---|
| in-toto Statement / DSSE | Interoperable statement and authentication layers | LEP-specific semantic validation |
| SLSA Provenance v1 | Build provenance semantics and supply-chain provenance | General scientific, policy, physical-simulation, or energy-model truth |
| JSON Schema Draft 2020-12 / Rust \`jsonschema\` | Portable structural validation | Runtime provenance, trusted signer identity, or domain truth |
| Cedar or OPA/Rego (policy workstream) | Authorization decisions, policy expressiveness, evaluation behavior | Artifact integrity or proof that a physical simulation actually ran |
| LEP | Domain-neutral evidence envelope profile *inside* the existing statement format, with explicit semantic checks and a bounded report scope | Signature authentication, artifact-byte verification, actual execution, or scientific truth until those gates are independently implemented |

## Benchmark hygiene

1. Pin the exact source commit, schema digest, fixture-set digest, toolchain, dependency lock digest, and command line in every receipt.
2. Keep positive controls, one-fault negative controls, and compound adversaries separate.
3. Run on the exact commit. \`queued\` is not pass; a completed check for an older SHA is not evidence for the current SHA.
4. Report failures and denominators, not just pass percentages. Publish false-accept count, false-reject count, per-case expected/observed codes, and skipped tests.
5. Keep synthetic fixtures explicitly synthetic. Do not describe placeholder digests or example run IDs as observed measurements.
6. Report structural-validation speed separately from full semantic validation, signature verification, and artifact-byte verification.

## Reproducible microbenchmark command

The standalone harness creates a valid synthetic LEP statement, pads only its unique `limitations` text to approach the requested size, warms each path 20 times, and reports p50/p95/mean latency plus p50-derived throughput. It compares strict parsing + schema validation against strict parsing + schema + semantic validation. It does **not** measure signatures, network access, artifact I/O, or scientific workloads.

From the repository root, collect the four input sizes with the same iteration count:

```sh
for size in 1024 10240 102400 1048576; do
  for repeat in 1 2 3 4 5; do
    cargo run --release --locked --manifest-path tools/lep-conformance/Cargo.toml --bin lep-benchmark -- "$size" 100
  done
done
```

Use the emitted `actual_input_bytes` (not just the requested size) in reports. Do not compare numbers from different hardware/toolchains as if they were controlled measurements; record OS, CPU, rustc version, dependency-lock digest, and commit SHA alongside the JSON output. This is the first one-process-per-size smoke result; the repeated-process gate is now defined in CI and still needs a fresh exact-head run. The raw JSON and reproducibility record for the first run are preserved under `docs/research/benchmarks/LEP-PERF-2026-10-30/`.

## Deterministic adversarial suite now authored

The Rust conformance library now includes a deterministic 1,000-record mutation
test generated from the checked-in positive fixture. It repeats ten single-fault
operators 100 times each: mutable source revision, stale checked target,
malformed digest, queued-but-passed execution, missing completed run ID, empty
executed-check set, duplicate check ID, failed check hidden by an inconclusive
outcome, missing target, and an unknown core field. Each case asserts both
rejection and its expected diagnostic code.

This is a checked-in test definition, **not a 1,000-case execution result**.
The first successful Rust test run must confirm that the implementation rejects
all 1,000 cases on the exact commit. The suite is deterministic and is intended
as a minimum adversarial regression layer; later suites should add parser
differential tests, compound attacks, fuzzing, and resource-exhaustion bounds.

## First hosted smoke baseline

The first successful hosted LEP workflow ran on PR source head
`9f120920153c78374fb782c493046d38d8f2c2b6` (workflow merge commit
`3d89fe67866905474fb782c493046d38d8f2c2b6`), using Rust
`1.96.0 (ac68faa20 2026-05-25)`, 20 warm-up iterations per path, and 100
timed iterations. The semantic path was measured against strict parse + schema:

| Actual input | Structural p50 | Full p50 | p50 overhead | Full p95 |
|---:|---:|---:|---:|---:|
| 1,247 B (requested 1 KiB) | 5.709 µs | 6.590 µs | 15.4% | 7.071 µs |
| 10 KiB | 6.870 µs | 7.822 µs | 13.9% | 8.012 µs |
| 100 KiB | 19.169 µs | 20.021 µs | 4.4% | 21.662 µs |
| 1 MiB | 157.547 µs | 160.511 µs | 1.9% | 181.343 µs |

These figures are **single-process synthetic smoke results**, not a statistically
stable cross-machine comparison and not a comparison to Cedar, OPA, or an
industry validator. They cover LEP strict JSON parsing + local schema and
semantic checks only; no signature, artifact, execution, or network costs are
included. Raw per-size JSON, Cargo.lock digest, runner metadata, PR head, and
workflow merge SHA are preserved under
`docs/research/benchmarks/LEP-PERF-2026-10-30/`.

The implementation caches the compiled schema per process. These measurements
are the first quantitative baseline for the parser/schema/semantic split; they
do not establish a measured speedup over an uncached validator. The next CI run
collects five independent process samples per size to quantify run-to-run
variance before setting any absolute latency budget.

## Out of scope for the current prototype

- DSSE envelope parsing/signature verification and trusted identity policy.
- Resolving subject/input URIs and re-hashing artifact bytes.
- Attesting that a runner or command actually executed.
- Cross-language implementation parity.
- Full cross-machine performance characterization and memory/DoS profiling.
- Any claim that a conforming statement's claim is true.

These remain explicit follow-up gates rather than implied capabilities.
