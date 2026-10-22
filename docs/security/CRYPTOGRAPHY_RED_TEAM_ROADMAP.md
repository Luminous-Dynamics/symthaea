# Cryptography Engineering and Red-Team Roadmap

**Status:** engineering plan; not an audit or a claim of cryptographic security  
**Last reviewed:** 2026-10-22  
**Scope:** Symthaea's own cryptographic wrappers, proof/signature boundaries, test vectors, and CI. Only owned code and explicitly authorized test fixtures are in scope.

## Operating principle

Symthaea should help **verify and misuse-test** cryptography before helping invent it. Production primitives should come from maintained, specification-aligned libraries. Symthaea's differentiator should be protocol analysis, adversarial test generation, reproducible counterexamples, proof/implementation cross-checking, and assurance evidence—not home-grown signature or encryption algorithms.

Every finding should include: affected commit and code path; attacker preconditions; violated invariant; a minimal deterministic reproducer; expected versus observed result; severity rationale; and a regression test. An unexecuted hypothesis is a lead, not a confirmed vulnerability.

## Findings to address first

1. **The current wrapper dependency is an obsolete implementation line.** The standalone mycelix-zkp-core stub depends on pqcrypto-dilithium 0.5 and pqcrypto-traits 0.3. RustSec marks pqcrypto-dilithium unmaintained and points to its ML-DSA successor; the broader pqcrypto family was subsequently marked unmaintained after the upstream PQClean project announced it would be archived. Do not treat a package rename as remediation. Evaluate a maintained FIPS 204 implementation, pin the chosen version, and require conformance and interoperability evidence before migration. See [RUSTSEC-2024-0380](https://rustsec.org/advisories/RUSTSEC-2024-0380.html), [RUSTSEC-2026-0164](https://rustsec.org/advisories/RUSTSEC-2026-0164.html), and [NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final).
2. **Key import must be fail-closed.** Before the initial hardening change, DilithiumKeypair::from_bytes checked key byte lengths but did not prove that the public and secret keys formed a pair. The first regression patch adds an import-time sign/verify consistency check and negative tests. This is defense in depth, not a substitute for replacing the obsolete dependency.
3. **The existing wrapper tests are not a complete cryptographic conformance suite.** Round-trip tests alone do not establish standard conformance, side-channel resistance, protocol safety, or production readiness. Keep those claims separate.

## Workstream A — standards-aligned primitive migration (P0)

- Evaluate RustCrypto's ml-dsa implementation against NIST FIPS 204, its published API/security caveats, maintenance activity, MSRV, no-std/WASM needs, and transitive dependency posture. The upstream crate explicitly says it has not been independently audited; document that limitation rather than silently converting it into a trust claim.
- Do not simply substitute the crate name. Record the current legacy wire format, algorithm identity, key/signature sizes, attached-versus-detached signature semantics, and every serialization/storage consumer.
- Introduce an explicit algorithm identifier and format version for newly written public keys and signed envelopes. Reject unknown identifiers and malformed lengths before expensive parsing. Never infer the algorithm from length alone.
- Define an explicit migration policy for existing keys and signatures. Do not silently reinterpret legacy Dilithium encodings as FIPS 204 ML-DSA, and do not claim backward interoperability unless vectors prove it.
- Keep a small adapter boundary so downstream code depends on Symthaea's typed API, not provider-specific byte wrappers.

**Exit gate:** applicable NIST ACVP known-answer vectors pass; negative vectors are rejected; cross-implementation valid signatures verify in both supported directions; key/signature serialization is versioned; old/new format behavior is covered by tests; dependency audit and independent review findings are documented. Passing vectors does not equal a third-party audit or certification.

## Workstream B — adversarial unit and property tests (P0/P1)

Required negative cases for signature/key APIs:

- Matching key import succeeds; mixed public/secret pairs fail.
- Malformed, truncated, extended, and wrong-size key/signature inputs never panic and never verify as valid.
- One-bit signature changes, message changes, wrong public keys, and substituted client identities are rejected.
- Signed envelopes bind a canonical, versioned, domain-separated transcript: algorithm, protocol/version, purpose, signer identity, and payload. Audit construct_signed_message and every serialization boundary before declaring this invariant satisfied.
- Verification of an envelope is not treated as verification of the underlying zero-knowledge proof. Keep signature validity and proof-system validity separate.
- Randomized property tests preserve invariants; every discovered failure is minimized, checked in as a regression fixture, and linked to the finding.

## Workstream C — vectors, differential testing, fuzzing (P1)

- Import applicable NIST ACVP vectors and current Project Wycheproof vectors where the algorithm/schema is covered. Pin the source revision and hash the corpus so CI results are reproducible. [Project Wycheproof](https://github.com/C2SP/wycheproof)
- Differential-test against an independently implemented, specification-aligned provider; compare semantics and encodings, not just whether the same wrapper accepts its own output.
- Build Rust fuzz targets for key parsing, signature verification, authenticated-envelope verification, and canonical message encoding. The fuzz boundary should accept arbitrary bytes and assert no panic, no acceptance of known-invalid input, and no unbounded allocation from attacker-controlled lengths.
- Use cargo-fuzz/libFuzzer on a scheduled or dedicated job, with a small deterministic corpus on every PR. Preserve crashes and minimized counterexamples in the repository. [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
- Keep fuzz builds and optional expensive checks separately labelled; queued or absent CI results are never recorded as passing evidence.

## Workstream D — protocol red team (P1/P2)

Analyze how primitives are composed and used, not only the primitive APIs:

- **Transcript integrity:** ambiguity, concatenation collisions, serialization normalization, field omission, algorithm confusion, context confusion, and cross-protocol signature replay.
- **Identity lifecycle:** key rotation, revocation, key substitution, stale public-key caches, duplicate client IDs, and recovery after key loss.
- **Proof boundary:** signed statement versus proof payload mismatch, unverified proof accepted after a valid signature, statement-version downgrade, and verifier-policy disagreement.
- **Error behavior:** consistent fail-closed outcomes; bounded parsing work; no secret values in logs/errors; distinguish invalid user input from internal verifier failures without leaking sensitive detail.
- **Secrets:** minimize long-lived secret-key copies, remove unnecessary Clone where callers permit it, evaluate zeroization support in the selected provider, and audit persistence, crash dumps, and serialization paths.
- **Side channels:** rely on provider documentation and independent review; supplement with compiler/assembly inspection and appropriately designed timing tests. Timing tests can detect regressions but cannot prove constant-time behavior.

## Workstream E — reproducible evidence and release gates (P1)

Each run should record the exact commit SHA, compiler/toolchain, dependency lockfile hash, vector corpus hash, commands, exit codes, elapsed time, and artifacts. Retain failing seeds and negative controls. Separate these evidence levels:

1. **Unit-tested** — local invariants and regression tests pass.
2. **Vector-conformant** — applicable published vectors pass.
3. **Differentially checked** — independent implementations agree on declared cases.
4. **Fuzzed** — bounded corpus/duration/platform and crash count are recorded.
5. **Independently reviewed/audited** — scope, reviewer, date, findings, and remediation are published or retained as appropriate.

Do not collapse these states into a generic “secure” badge.

### Initial benchmark matrix

| Measure | Required reporting |
|---|---|
| Key generation / signing / verification | p50, p95, p99 latency; throughput; CPU target and build profile |
| Parsing and rejection | latency for valid and malformed inputs; maximum accepted lengths; memory peak |
| Standard conformance | applicable vector count, pass/fail/skip with reasons |
| Negative-control efficacy | count of known-invalid fixtures correctly rejected; accidental-accept count must be zero |
| Differential consistency | test cases, versions, encoding agreement, disagreements |
| Fuzzing | executions, duration, corpus revision, coverage trend, unique crashes, minimized reproducers |
| Secret handling | copies/lifetimes reviewed; zeroization and debug/log tests; limitations |
| Reproducibility | exact SHA/toolchain/lockfile/corpus digests and replay instructions |

Establish a measured baseline before imposing throughput budgets. Never optimize away checks or weaken negative tests to improve the numbers.

## Symthaea-assisted red-team loop

1. Build an inventory of cryptographic entry points and the invariant each is supposed to enforce.
2. Generate attack hypotheses from code and protocol context; label each hypothesis unconfirmed.
3. Convert each hypothesis into a minimal adversarial test against an owned fixture.
4. Run tests and retain exact-head evidence; failed or unavailable tools remain explicit.
5. Minimize failures, patch the smallest verified cause, and add a permanent regression.
6. Ask an independent reviewer/model to challenge the fix and its tests, including false-positive and false-negative cases.
7. Re-run the original counterexample, the complete focused suite, applicable vectors, and the negative controls.

The success metric is not the number of “vulnerabilities” generated. It is the number of reproducible defects fixed, invalid inputs rejected, false claims prevented, and assurance gaps closed without introducing regressions.
