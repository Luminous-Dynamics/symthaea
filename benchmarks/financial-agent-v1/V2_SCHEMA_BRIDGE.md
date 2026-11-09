# Finance evidence schema v2 bridge

**Status:** migration draft / schema contract only. This directory does not yet provide a trusted JSON-to-Rust adapter.

## Why v2 exists

The v1 research evidence record contains a self-asserted `known_at`, a retrieved timestamp and a rights-profile string. It does not bind either availability or authorization to a retained proof artifact, evaluated principal and verifier identity. The v1 quant run similarly contains whole-manifest flags but no per-observation proof binding.

The v2 schemas introduce one reusable evidence-version object for financial research and quantitative observations. A consumer must not promote a v1 `known_at` or a v2 producer-declared proof status into a qualified `Verified` Rust proof by schema validation alone.

## New files

- `evidence-v2.schema.json`: exact evidence identity, temporal semantics, actor-scoped availability, rights authorization, provenance-family lineage and revision ancestry.
- `evidence-proof-binding-v1.schema.json`: separate domain-separated availability and rights binding preimages; both include the exact `source_version_id`.
- `evidence-manifest-v2.schema.json`: shared principal, rights profile, evaluation mode and decision frontier for a research evidence set.
- `quant-observation-manifest-v2.schema.json`: per-observation link to the same evidence-v2 object; no whole-run point-in-time Boolean substitutes for per-observation proof.
- `dev-example.evidence-v2.json` and `dev-example.quant-observations-v2.json`: synthetic development-only examples. Their proof references and digests are intentionally fixture markers, not valid authorization or availability proofs.

## Canonical proof-binding preimages

Each v2 attestation carries its typed `binding` preimage and a `binding_sha256`. The preimage schema is in `evidence-proof-binding-v1.schema.json`.

- Availability uses domain `luminous.financial.availability-proof-binding.v1`.
- Rights uses domain `luminous.financial.rights-proof-binding.v1`.
- Both include `evidence_id`, `source_id`, `source_record_id`, exact `source_version_id`, `content_sha256`, `principal_id` and `rights_profile_id`.
- Availability additionally binds precision, exact `known_at` or a bounded interval, and `retrieved_at`.
- Rights additionally binds `disposition`, `valid_from` and `valid_until`.

`binding_sha256` is SHA-256 of the RFC 8785 JCS UTF-8 serialization of the binding object, with no newline appended. Binding objects use only ASCII strings, nulls and fixed enumerations; they carry no floating-point numbers. The verifier receipt bytes must independently record the same binding digest, proof-artifact digest, verifier identity and verifier-build digest. The receipt is itself content-addressed and checked by the existing evidence-byte verification path.

Availability and rights are separate attestations. Neither digest can be reused for the other domain. A producer-supplied `binding_sha256` is not trusted until the consumer recomputes it from the embedded preimage and validates the referenced proof/receipt artifacts. RFC 8785 defines JCS as deterministic, hashable JSON serialization with recursive property sorting: https://www.rfc-editor.org/rfc/rfc8785.html.

## Important trust boundary

The JSON `proof.state` is an assertion supplied by a manifest producer. It is **not** authoritative. A trusted adapter must:
1. Resolve the proof artifact reference and verify its exact bytes through the existing Mycelix / evidence-byte verification path (#9608 / #9429).
2. Resolve and validate the independent verifier receipt, including verifier identity and build digest.
3. Verify that the receipt binds the exact evidence ID, source ID, source-record ID, exact source-version ID, content digest, principal, rights profile and instant/interval.
4. Verify the entitlement disposition and validity at the run's decision frontier.
5. Construct `AvailabilityProof::Verified` and `RightsProof::Verified` only after those checks pass. Missing, unresolved, mismatched or revoked proof stays Missing/Unverifiable and therefore cannot PASS.
6. For `bounded_interval`, use the #11256 interval semantics. Until that extension exists, do not convert an interval to an exact instant; keep it Indeterminate.
7. For `synthetic_fixture_only`, accept only under an explicitly non-qualifying development profile. It must never be admitted to historical confirmation or prospective shadow cohorts.

The schemas enforce structure, not source truth, signatures, permissions or entitlement. No private credentials or raw provider-account identifiers belong in these files.

## Compatibility policy

- v1 remains readable for public development/regression only.
- v1 evidence with `known_at` but no exact actor-scoped proof is never qualified for historical confirmation or prospective shadow.
- v2 can represent exact instants, bounded intervals or unknown precision. Unknown precision is not a pass.
- Do not migrate public examples into sealed/confirmation cohorts.
- Do not mark the kernel integrated with #11233 or #11234 until the typed adapter, byte-verifier receipt binding and cross-schema round-trip tests exist.

## Acceptance tests still required

- V1 data rejected/indeterminate in qualified profiles.
- Exact v2 proof round-trip into the Rust `EvidenceVersion`.
- Availability proof bound to the wrong evidence/version/principal rejected.
- Rights proof wrong-profile, expired, revoked or mismatched rejected.
- Bounded availability interval entirely before / after / straddling the frontier has PASS / FAIL / INDETERMINATE under #11256.
- Quant observations each carry an evidence object; no global Boolean can qualify missing proofs.
- Mutated proof bytes, receipt bytes, verifier build, revision ancestry or frontier produce a different canonical input and fail/review.
- Same canonical v2 inputs produce deterministic adapter output.

## Claim ceiling

This PR is a schema migration foundation. It is not the adapter, a proof verifier, a live financial data connector, a strategy engine or a financial authorization mechanism.
