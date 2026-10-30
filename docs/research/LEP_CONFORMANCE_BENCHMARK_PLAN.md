# LEP Conformance Benchmark Plan

**Status:** Proposed benchmark gates; no performance/SOTA result is claimed  
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
| Performance | Warm validation latency and throughput at 1 KiB, 10 KiB, 100 KiB, and 1 MiB | Establish a measured baseline before setting absolute budgets; semantic path should remain within 2× the same-version structural-only baseline on 10 KiB records | 5 warm-up runs, at least 30 timed repetitions per size, p50/p95, bytes/s, CPU/toolchain metadata |
| Memory / DoS | Deep nesting, oversized arrays, huge strings, extreme nesting, repeated malformed inputs | Defined byte/depth/item limits or documented deployment limits; no unbounded resource claims | Reject/limit cases and peak RSS measurements |
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
  cargo run --release --locked --manifest-path tools/lep-conformance/Cargo.toml --bin lep-benchmark -- "$size" 1000
done
```

Use the emitted `actual_input_bytes` (not just the requested size) in reports. Do not compare numbers from different hardware/toolchains as if they were controlled measurements; record OS, CPU, rustc version, dependency-lock digest, and commit SHA alongside the JSON output. This harness currently establishes a measurement method only; its first run must still pass compile and validity checks.

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

## Baseline to collect next

The implementation now caches the compiled schema per process so repeated validation need not parse and compile the schema for every record. This is a design optimization, **not a measured speedup**. First collect the table above on the current implementation; use that result to set a realistic absolute latency/throughput budget and to determine whether further optimization is justified.

## Out of scope for the current prototype

- DSSE envelope parsing/signature verification and trusted identity policy.
- Resolving subject/input URIs and re-hashing artifact bytes.
- Attesting that a runner or command actually executed.
- Cross-language implementation parity.
- Performance and resource-exhaustion measurements.
- Any claim that a conforming statement's claim is true.

These remain explicit follow-up gates rather than implied capabilities.
