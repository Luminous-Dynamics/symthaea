# Culinary content-addressed evidence qualification benchmark v1

**Status:** content-integrity contract and synthetic byte fixtures. It does not authenticate an issuer, establish that randomization occurred, or validate scientific claims.

## Paradigm shift

A label such as `evidence_status = qualified` is a declaration—not proof that the supporting source exists or that its contents have remained unchanged. The evidence bundle therefore pins the exact manifest bytes using a separately supplied SHA-256 digest, verifies every artifact's byte count and SHA-256, and binds every process-trace ID and outcome evidence source/version pair to a named artifact.

This implements a narrow, concrete slice of provenance interoperability inspired by W3C PROV's distinction among entities (artifacts), activities (process runs) and agents/responsibility. PROV can describe provenance; the Culinary bundle additionally checks the exact payload bytes referenced by its manifest:
- https://www.w3.org/TR/prov-primer/
- https://www.w3.org/TR/prov-constraints/

## API

`verify_culinary_evidence_bundle_v1(experiment, manifest_bytes, expected_manifest_sha256, artifact_bytes)` returns an opaque `VerifiedCulinaryEvidenceBundleV1` only when:
- the digest of the exact manifest byte sequence equals a digest supplied separately by the caller;
- strict typed JSON deserialization succeeds with unknown/duplicate fields rejected;
- experiment ID and preregistered protocol digest match the manifest;
- exactly one protocol artifact exists, and its declared/verified SHA-256 matches the experiment's protocol digest;
- every artifact has unique ID, source reference and path, safe normalized relative path, positive byte length and canonical lowercase SHA-256;
- exactly the manifest-declared payload IDs are supplied, with no missing/extraneous bytes, and every byte length and SHA-256 matches;
- each run's process trace ID resolves to its own unique process-trace artifact;
- each outcome evidence source reference and version resolves to an artifact of an appropriate measurement/outcome kind;
- all artifacts are reachable from the protocol, run trace or outcome evidence; unreferenced artifacts fail closed.

`analyze_paired_outcomes_with_verified_bundle_v1` additionally fingerprints the in-memory experiment value and rejects a changed object before running the existing paired-outcome analysis.

## Acceptance gates

1. 100% of one-byte payload changes rejected by hash or byte-length check.
2. 100% of manifest-byte changes rejected when the expected digest is frozen separately.
3. 100% of missing and undeclared payload entries rejected.
4. 100% of duplicate artifact IDs, paths, source references, duplicate JSON fields and unsafe paths rejected.
5. 100% of process-trace IDs and outcome source/version pairs resolve to the exact declared artifact kind.
6. Zero unreferenced manifest artifacts accepted.
7. Any mutation of the experiment after verification invalidates the opaque verified-bundle capability.
8. A byte-verified file remains only a byte-verified file: SHA-256 is not issuer authentication, calibration proof, evidence truth, safety assessment or causal attribution.
9. Rust tests, formatting, compilation and Clippy must complete against the exact head. Skipped, queued or absent workflows are not pass evidence.

## Trust model and residual risks

- The expected manifest digest must come from a separate trusted record, such as a reviewed immutable reference or signed run receipt. Comparing a manifest with a digest read only from the same untrusted manifest would prove nothing about the manifest's integrity.
- The caller resolves paths to payload bytes. It must reject symlinks and ensure canonical resolution remains within the authorized artifact root. This library applies lexical path rules and checks the bytes once supplied; it does not do filesystem I/O.
- Hash correctness proves equality to the pinned digest; it does not prove who captured the sample or whether a thermometer was calibrated correctly. Issuer identity needs a separate signature/key trust policy.
- The bundle verifier binds the paired experiment's protocol, process trace IDs and outcome evidence. It does not claim that raw source artifacts support the observed values scientifically or that the reported variant effect is causal.
- The previous metadata-only analyzer remains available for contract/oracle tests. Source-bound analysis must use `analyze_paired_outcomes_with_verified_bundle_v1`; test-only evidence locators never count as experimental evidence.

## Synthetic negative controls

The test module creates 1 protocol, 4 process-trace and 12 outcome payloads in memory. These bytes are synthetic and authored only to test exact-byte plumbing. The 15 Rust tests cover successful binding, byte tampering, manifest digest mismatch, protocol digest mismatch, missing/extra payloads, unsafe paths, missing process traces, source-version drift, stale verified capabilities, duplicate manifest fields, synthetic-source spoofing, duplicate paths and unreferenced artifacts.

## Relationship to experimental reporting

After content integrity is established, statistical design remains a separate gate. CONSORT 2025 calls for primary and secondary outcomes to be defined by measurement variable, analysis metric, aggregation and time point, and for outcome effects to be reported with precision. This evidence bundle does not replace that design/analysis work:
- https://www.bmj.com/content/389/bmj-2024-081124

## Qualification boundary

The implementation currently provides a byte-verification API, an opaque experiment-bound verification receipt, a JSON Schema, and adversarial unit tests. The tests have **not** been compiled or executed in this environment. No claim of full schema validation, exact-head CI pass, real evidence authenticity, sensor calibration, food safety, or scientific benefit is made.
