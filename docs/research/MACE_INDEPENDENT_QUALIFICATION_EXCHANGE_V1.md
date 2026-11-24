# MACE independent qualification exchange v1

**Status:** consumer-side protocol adapter and contract tests. This does not implement or operate the independent qualifier, issuer registry, source resolver, rights engine, lifecycle ledger, DSSE envelope, or production transport.

## Why this boundary exists

A retrieval backend emits a structurally validated but untrusted candidate batch. The consumer must not expose those records to a reader merely because an Add/Search request succeeded or the candidate set is non-empty.

The exchange order is:

1. Consumer reconstructs canonical candidate JSONL and verifies the candidate batch commitment.
2. Consumer binds the complete candidate ID set to run, query invocation, domain, effective subject scope, environment, tool profile, DecisionNeed, purpose, qualification policy, lifecycle snapshot, pinned verifier profile, and trusted time.
3. An injected transport sends the request to a separately governed issuer.
4. The issuer independently parses the candidate batch, recomputes all canonical digests, resolves source payloads and provenance, evaluates rights/authority/purpose/scope and currentness, and returns a signed qualification receipt.
5. The consumer checks the receipt against the exact request and independently pinned Ed25519 trust profile.
6. The consumer projects only admissible records and dependency-closes the result. Rejected and indeterminate records are removed; missing required parents and cycles fail closed.
7. The caller receives a qualified-only projection, explicit projection abstention, or an error. The original candidate batch is never returned by this helper.

## API

- `QualificationUseContextV1` is consumer-owned context; it is not populated from the issuer response.
- `EvidenceQualificationRequestV1` carries canonical candidate JSONL and a strict complete use binding. Its request digest is a transport/idempotency identity, not a trust assertion.
- `IndependentEvidenceQualificationIssuerV1` is an injected transport contract. A real implementation must connect to a different authority and map timeout/ambiguous outcomes to `Unavailable`.
- `independently_qualify_candidate_batch_v1` is the fail-closed consumer path. It verifies the returned receipt with the caller's pinned verifier and applies `project_qualified_candidate_evidence_v1`.

Wire schema: `data/schemas/mace_evidence_qualification_request_v1.schema.json`. Rust consumers may decode with `EvidenceQualificationRequestV1::from_json_value`; this rejects unknown/missing fields, non-canonical JSONL, candidate/binding mismatches, unsorted IDs and request-digest mismatches.

### Request digest preimage

The `request_sha256` field is SHA-256 over the following exact sequence:

1. The domain prefix `luminous:mace:evidence-qualification-request:v1\0`.
2. A u64 big-endian byte length followed by UTF-8 bytes for the schema version.
3. The same length-prefixed encoding of the exact canonical candidate JSONL string.
4. Length-prefixed UTF-8 values, in order: `run_id`, `query_invocation_id`, `domain_id`, `subject_scope_sha256`, `environment_revision_sha256`, `tool_profile_sha256`, `query_sha256`, `candidate_set_sha256`.
5. A u64 big-endian candidate-ID count, then each sorted candidate ID as a length-prefixed UTF-8 value.
6. Length-prefixed `decision_need_sha256`, `purpose_sha256`, `qualification_policy_sha256`, `lifecycle_snapshot_sha256`, `verifier_profile_sha256`.
7. `now_unix` as u64 big-endian.

The request digest is an idempotency/canonicalization aid only. It is not signed, does not authenticate the caller, and does not replace issuer-side parsing, candidate/source digest recomputation, or signature verification. The Rust test suite includes a fixed digest vector for cross-implementation parity.

## Independence requirements not supplied by this module

The interface alone does not establish process, credential, storage, trust-root or rollback independence. A production issuer must:

- run under a separately governed OS principal/service identity;
- use credentials and private signing keys unavailable to the candidate backend;
- resolve sources independently rather than trust caller-supplied payload references;
- recompute every digest and verify complete provenance closure, including transformation identity and required parent relationships;
- check issuer identity, rights, purpose, effective scope, source validity, correction/revocation/supersession and erasure state;
- bind its lifecycle snapshot to a source of truth with rollback protection, and return explicit `admissible`, `rejected`, or `indeterminate` for every candidate;
- sign the typed receipt with a key represented in a trust profile provisioned outside the candidate backend and protected by a monotonic profile-version floor;
- expose authenticated, bounded request/response transport with replay protection, timeout handling and auditable service health.

The verifier profile in the consumer must be pinned out of band; copying the profile digest from the same request or candidate database is not trust bootstrap. A local subprocess under the same user account is not, by itself, sufficient separation for a hostile-host threat model.

## Fail-closed semantics

- Candidate batch mismatch: reject before contacting the issuer.
- Issuer unavailable, timeout, malformed response, signature failure, or any binding mismatch: return an error and expose no candidate payload.
- Signed receipt with no admissible candidates: return explicit abstention.
- Admissible derived record whose required parent is rejected, indeterminate, absent, or cyclic: remove the dependent record; if no evidence remains, abstain.
- Qualification receipt expiry or use-context change: reject at use.
- A qualified projection is not an action authorization. Live authorization and lifecycle currentness must be rechecked at consequential use. Decision sufficiency, decision, action authorization, consume-once state and outcome receipts remain separate objects.

## Claim boundary

The tests use a real Ed25519 signing key and a separately pinned public verification profile as deterministic fixtures. They exercise request binding, signed-response verification, projection, abstention, issuer-unavailable behavior and purpose-rebinding rejection. They do not prove a deployed independent authority or its source-verification correctness. Rust tests and exact-head CI must still run on the pinned toolchain.
