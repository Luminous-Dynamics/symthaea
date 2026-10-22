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

- Evaluate a provider matrix instead of selecting a library on brand or throughput alone:
  - **Current legacy baseline:** pqcrypto-dilithium, retained only long enough to measure compatibility and migration cost; RustSec marks the pqcrypto family unmaintained.
  - **Pure-Rust candidate:** RustCrypto ml-dsa 0.1.1 implements final NIST FIPS 204 and exposes ML-DSA-44, -65, and -87. Its maintainers explicitly state that it has not been independently audited. Treat it as a candidate to validate, not as an automatically trusted production replacement: [crate documentation](https://docs.rs/ml-dsa/latest/ml_dsa/).
  - **Formally verified-components candidate:** Cryspen's libcrux-ml-dsa 0.0.10 supports ML-DSA-44, -65, and -87, with portable and AVX2 implementations. Its documentation says selected field arithmetic, NTT polynomial arithmetic, and serialization components are formally verified in F*; this must not be inflated into a claim that the complete implementation or end-to-end protocol is formally verified or independently audited. Evaluate the exact crate version, proof scope, generated-code correspondence, performance, and remaining unverified components: [crate source and README](https://docs.rs/crate/libcrux-ml-dsa/latest/source/README.md).
  - **Independent differential reference:** OpenSSL 3.5 added ML-DSA support and is an LTS line supported until April 8, 2030. Evaluate it as a separate implementation where the target platforms can pin an appropriate OpenSSL version and provider configuration: [OpenSSL 3.5 release announcement](https://openssl-library.org/post/2025-04-08-openssl-35-final-release/). Do not imply that using OpenSSL alone means a particular build/provider is FIPS-validated.
  - **Adversarial corpus:** use current Project Wycheproof ML-DSA vectors where applicable, alongside applicable NIST ACVP vectors. Pin the exact corpus revision and digest: [Project Wycheproof](https://github.com/C2SP/wycheproof).
- Record MSRV, no-std/WASM feasibility, license and transitive dependencies, reproducible-build posture, secret-lifecycle behavior, side-channel evidence, maintenance response, and independent audit status for each candidate. Do not use a single provider as both implementation and sole oracle.
- Do not simply substitute the crate name. Record the current legacy wire format, algorithm identity, key/signature sizes, attached-versus-detached signature semantics, and every serialization/storage consumer.
- Introduce an explicit algorithm identifier and format version for newly written public keys and signed envelopes. Reject unknown identifiers and malformed lengths before expensive parsing. Never infer the algorithm from length alone.
- Define an explicit migration policy for existing keys and signatures. Do not silently reinterpret legacy Dilithium encodings as FIPS 204 ML-DSA, and do not claim backward interoperability unless vectors prove it.
- Keep a small adapter boundary so downstream code depends on Symthaea's typed API, not provider-specific byte wrappers.

**Exit gate:** applicable NIST ACVP known-answer vectors pass; negative vectors are rejected; cross-implementation valid signatures verify in both supported directions where format compatibility is specified; key/signature serialization is versioned; old/new format behavior is covered by tests; dependency audit and independent review findings are documented. Passing vectors does not equal a third-party audit or certification.

### Fair benchmark protocol

- First compare **the same security level and the same workload**. Retain a legacy Dilithium5 measurement only as a migration baseline; compare ML-DSA-87 for the closest high-security-level migration study, and measure ML-DSA-65 separately as a practical performance/security trade-off—not as a like-for-like replacement for level 5.
- Run correctness and negative controls before measuring speed. A provider that rejects valid inputs or accepts invalid ones cannot win on performance.
- On the same pinned machine, toolchain, compiler flags, and message-size distribution, report key-generation/signing/verification p50, p95, p99, throughput, key/signature bytes, peak memory, import/rejection latency, and binary size. Record CPU feature flags and OpenSSL/provider build configuration.
- Run repeated samples, preserve raw output, and include confidence intervals or dispersion. Compare like-for-like local results; do not claim superiority by comparing unlike hardware or quoting numbers from different benchmarks.
- Treat zero known-invalid acceptances, applicable vector pass rate, cross-implementation consistency, and absence of crashes as hard correctness gates. Optimize only after they pass.

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

## Workstream E — standardized build provenance and trust policy (P1)

Do not grow a bespoke signature envelope into a replacement for ecosystem provenance standards. Symthaea's domain-specific attestation can record proof-specific claims, but release/build provenance should compose with the [SLSA v1.2 specification](https://slsa.dev/spec/v1.2/), in-toto statement/predicate formats, and a well-defined signature/identity verification policy.

- **Immediate baseline:** produce provenance for a release/build that binds repository URI, exact source revision, build definition, builder identity, dependency materials, and output digest.
- **Target Build L2:** use hosted CI to generate and sign provenance and make consumers verify its authenticity. Do not let the ordinary build step freely manufacture the trusted CI provenance.
- **Longer-term Build L3:** move towards an appropriately hardened builder where build steps cannot access provenance-signing material and separate build runs cannot influence one another. A valid signature generated inside an untrusted build step alone does not establish trustworthy provenance.
- **Source controls:** progressively adopt SLSA Source Track requirements for retained history, source provenance, enforced controls, and review. Do not claim a source/build level until the concrete controls are in place and independently checked.
- **Domain claims:** for an SMT or ZK proof attestation, bind a canonical statement digest, proof/verifier algorithm and version, verifier identity/version, result, and associated artifact digests. Keep “signature valid,” “trusted signer,” “proof verified,” “source-to-binary provenance verified,” and “artifact executed in a sandbox” as distinct claims.
- **Trust roots:** distribute trusted signer identities through policy/configuration or an established transparency/trust channel, never by trusting a key that appears only inside the object being verified.
- **Evidence bundle:** attach SBOM/dependency inventory and the exact test/vector/fuzz/negative-control results as linked evidence. These records support decisions; they do not turn an unreviewed cryptographic primitive into an audited one.

The [OpenSSF Scorecard](https://www.scorecard.dev/) offers a complementary repository hygiene baseline: review enforcement, branch protection, pinned dependencies/actions, least-privilege CI token permissions, dangerous workflow patterns, fuzzing, vulnerability remediation, and signed releases. Use its checks to expose missing controls, not as a substitute for a protocol-specific threat model or a crypto audit.

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

## Executable red-team corpus: first protocol-boundary benchmark

Convert each confirmed bug class into a deterministic fixture and require a paired positive control. The corpus should be a security regression benchmark, not a count of hypothetical attacks. The table below is a prioritized initial suite assembled from source review; proposed fixes and unexecuted tests must remain labelled unverified until they run on the exact commit.

| Boundary | Adversarial mutation | Oracle / positive control |
|---|---|---|
| Key import | Combine valid-length public and secret keys from different keypairs; malformed encodings | Matching generated pair imports and signs/verifies; mixed pair fails import |
| Authenticated proof | Change proof backend or energy after signing; truncate the signature | Original envelope verifies; any bound-field mutation fails |
| Sovereign attestation | Move variable-field boundaries while preserving legacy concatenated bytes; substitute self-supplied key; omit a required artifact | Legacy version rejected; untrusted key rejected; exact binary verifies while missing/wrong binary fails |
| WASM AOT loader | Supply a self-signed artifact; mutate precompiled bytes or compatibility fingerprint; use a different trusted key | Only the locally trusted key's v2 artifact loads; bytes and compatibility changes invalidate the signature |
| Disabled runtime | Call execution when Wasmtime feature is absent | Returns explicit unavailable error; must not return success or claim verification |
| WASM cache/path | Use traversal/TOML-syntax plugin names; collide or reuse a name/source cache identity; supply a non-digest extension path | Invalid names/IDs are rejected before filesystem use; identical inputs yield the same SHA-256 key, differing name/source changes it |
| MFDI signed request | Change timestamp while keeping the signature; submit a valid self-signed untrusted key; reuse a nonce | Tampered freshness field fails; untrusted key is rejected; a trusted fresh request succeeds once |
| MFDI signed output | Change the agent DID, timestamp, cycle ID, or output values | Original output verifies against the matching key; each mutation fails, and the DID must be derived from that key |
| Swarm attestation diagnostic | Use a non-ASCII untrusted signer key whose byte 16 splits a UTF-8 character | Verification returns a trust error without panicking |
| Pairing challenge | Reuse key material while resetting the cycle counter; replay an old response | Challenge nonce comes from OS randomness; replay fixture is rejected or no longer repeats the challenge nonce |

### Benchmark scoring rules

- Count **true positives** only when a test demonstrates that the unpatched implementation violated a stated invariant and the patched implementation rejects the same reproducer.
- Pair each attack fixture with a valid control so an implementation that rejects all inputs cannot score well.
- Report detection recall, false-positive rate, minimized reproduction rate, and regressions introduced. Use exact fixture IDs and hashes; do not count a generated hypothesis as a vulnerability.
- Split scores by assurance class: parsing robustness, signature correctness, signer authorization, protocol integrity, replay/freshness, artifact provenance, and execution isolation. A strong score in one class must not mask a missing class.
- Report the number of tests and mutation operators covered, not only a percentage. The 95% curated-fault target is meaningful only once the corpus, fault families, and exclusions are pinned and published.

## Symthaea-assisted red-team loop

1. Build an inventory of cryptographic entry points and the invariant each is supposed to enforce.
2. Generate attack hypotheses from code and protocol context; label each hypothesis unconfirmed.
3. Convert each hypothesis into a minimal adversarial test against an owned fixture.
4. Run tests and retain exact-head evidence; failed or unavailable tools remain explicit.
5. Minimize failures, patch the smallest verified cause, and add a permanent regression.
6. Ask an independent reviewer/model to challenge the fix and its tests, including false-positive and false-negative cases.
7. Re-run the original counterexample, the complete focused suite, applicable vectors, and the negative controls.

The success metric is not the number of “vulnerabilities” generated. It is the number of reproducible defects fixed, invalid inputs rejected, false claims prevented, and assurance gaps closed without introducing regressions.
