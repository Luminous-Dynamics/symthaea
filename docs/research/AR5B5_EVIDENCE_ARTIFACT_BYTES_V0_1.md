# AR-5B5 Content-Addressed Evidence Artifact Byte Verification v0.1

**Status:** pure Rust byte-verification primitive committed; resolver and anytime integration outstanding  
**Module:** \`crates/domains/symthaea-architecture-engineering/src/evidence_artifact.rs\`  
**Tracking:** Issue #9608; parent Issue #9469

## 1. Purpose

An evidence digest is a reference to expected bytes. It is not proof that the bytes are available, correct, complete, or semantically sufficient. This module establishes a small, deterministic boundary for verifying that a set of bytes supplied by a resolver exactly matches a frozen artifact manifest.

It intentionally separates:
- manifest construction and provenance binding;
- retrieval of bytes from a content-addressed store;
- byte-integrity checking;
- semantic verification of what those bytes mean.

Only the third is implemented here.

## 2. Implemented API

### \`EvidenceArtifactManifest\`

Binds run ID, candidate ID, source commit, and a canonical sorted list of artifact references. Each reference contains:
- stable artifact ID;
- semantic role (candidate source, build provenance, evaluator output/log, constraint report, verifier receipt, held-out result, reproduction bundle, other);
- media type;
- storage locator;
- exact byte length;
- SHA-256 digest.

Manifest ordering is canonical by artifact ID. The manifest digest is independent of the order supplied to the constructor.

### \`verify_evidence_artifact_bytes\`

Accepts a frozen manifest, verifier identity/build digest, and a fully resolved byte set. It rejects:
- missing or extra artifact entries;
- duplicate or empty artifact IDs;
- byte-length mismatch;
- digest mismatch;
- malformed verifier build digest.

The output receipt binds run/candidate/source-commit identity, verifier identity/build, manifest digest, and the exact sorted set of byte-verification results.

### \`verify_evidence_artifact_byte_receipt\`

Re-runs byte verification from the same manifest and supplied bytes, then compares the complete reconstructed receipt to the stored receipt. Tampered receipt fields fail.

## 3. Claim ceiling

This module does **not**:
- retrieve bytes from an object store or Holochain/CAS;
- perform network access;
- authenticate a storage provider or transport;
- verify digital signatures;
- prove that the declared source commit is the commit that actually produced the bytes;
- establish semantic correctness, scientific novelty, evaluator correctness, or held-out superiority.

The storage locator is metadata. It is not automatically fetched and it is not a trusted URL. The caller must implement a bounded, authenticated resolver and supply the exact bytes.

A byte-valid source file may still be wrong. A byte-valid test log may still come from an inadequate test. Byte integrity is a prerequisite for evidence verification, not the same thing as evidence validity.

## 4. Authored tests

- Canonical manifest ordering and digest invariance.
- Exact byte-set verification and deterministic receipt reconstruction.
- Tampered receipt rejection.
- Same-length byte mutation rejection.
- Missing, extra, and duplicate resolved artifact rejection.
- Invalid manifest and verifier build identity rejection.

Tests are authored but have not been compiled or executed in this environment.

## 5. Integration sequence

1. Add a bounded CAS resolver with explicit scheme/namespace allowlisting, maximum artifact size, timeout and no implicit URL fetch.
2. Retrieve and retain the exact byte blobs for every manifest reference.
3. Verify all bytes and persist the receipt.
4. Bind the receipt digest to the matching \`AnytimeAction::CandidateVerification\` event.
5. Reconstruct the receipt during independent campaign audit.
6. Only then allow the artifact-integrity portion of an \`ArtifactVerified\` evidence promotion.
7. Run domain-specific semantic validation separately; only it may establish scientific correctness or held-out validity.

Tracked in [Issue #9608](https://github.com/Luminous-Dynamics/symthaea/issues/9608).

## 6. Exact-head acceptance

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering evidence_artifact -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_anytime -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

No compilation, test, clippy, or benchmark pass is claimed until exact-head evidence is retained.