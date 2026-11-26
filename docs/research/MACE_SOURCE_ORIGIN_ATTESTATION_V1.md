# MACE Source-Origin Attestation V1

**Status:** Consumer-side source-origin receipt verification and byte-recheck interfaces are implemented on `mace/source-origin-attestation-v1`. The new Rust module, schema and graph-node binding are draft code. Rust tests are authored but not executed in this environment. This is not a deployed fetch service, a production origin authority, or authorization to execute a consequential action.

## 1. The gap this closes

Graph replay (#13705) establishes that the disclosed source-lineage graph is internally consistent with the qualified candidate set and the independent lineage resolver's signed commitments. It does not establish that the source bytes, extracted payloads, ancestor records or derivation receipts actually match the declared digests.

This increment adds a fourth authority role:

`qualification → sufficiency → lineage resolution → source-origin verification`

The source-origin verifier has a separately pinned trust profile and signer set. It is distinct from the qualifier, sufficiency evaluator and lineage resolver. It signs one DSSE/in-toto Statement for every node in the replayed graph—including ancestors that are not in the qualified candidate set.

## 2. What the consumer checks

The new `source_origin_receipt` module:

1. Requires an already replayed `VerifiedSourceLineageGraphV1` token. It cannot accept a free-form graph digest or caller-built node list.
2. Requires exact artifact coverage for every retained graph node. Missing, extra or duplicate nodes fail closed.
3. Hashes the supplied exact source bytes, payload bytes and (for non-root nodes) derivation-receipt bytes itself. All three must match the graph node's SHA-256 commitments.
4. Requires one independently signed origin receipt per node. Each receipt binds the exact graph digest, run/invocation, evidence/source identifiers, source/payload digests, evidence class, parent list, derivation-receipt digest, common-cause groups and measured byte lengths.
5. Checks the DSSE PAE signature, in-toto Statement type, predicate type, exact subject, receipt status, time interval and pinned verifier profile.
6. Rejects an origin profile digest or signer key already used by any of the prior three authorities.
7. Rechecks the source, payload and derivation bytes and revalidates every origin signature at use time.

The origin receipt schema is `data/schemas/mace_source_origin_receipt_v1.schema.json`. The consumer's independent trust-profile digest must be provisioned out-of-band; it must not be copied from candidate memory, the lineage resolver, or the receipt being verified.

## 3. Artifact and receipt protocol

For each graph node, the acquisition boundary supplies:

- exact raw source bytes, whose SHA-256 must equal `source_sha256`;
- exact evidence payload bytes, whose SHA-256 must equal `payload_sha256`;
- exact derivation-receipt bytes when the node has parents, whose SHA-256 must equal `derivation_receipt_sha256`.

Root nodes must have no parents and no derivation receipt. Non-root nodes must have both. The receipt also commits to the exact parent-evidence IDs and common-cause-group IDs from the replayed graph. The runtime imposes per-artifact and aggregate byte limits.

The independent origin service signs a DSSE envelope containing an in-toto Statement v1. The Statement subject is the exact evidence node and its source digest. The predicate carries the complete origin binding and `status = origin_verified`. The consumer does not hash a reserialized envelope or predicate in place of the DSSE PAE; it verifies the signed payload bytes supplied in the envelope.

## 4. Use-time protocol

A consequential consumer must not treat a previously verified origin token as permanent:

1. Obtain a fresh trusted time sample.
2. Revalidate qualification, sufficiency and lineage receipts at that same time.
3. Replay the exact graph again, retaining its private verified node projection.
4. Re-acquire the source, payload and derivation bytes through the independent acquisition boundary.
5. Call `VerifiedSourceOriginSetV1::validate_at_use` with that replayed graph, the exact current bytes, the current origin trust profile and the same time.
6. Only then may a separately implemented action-authority layer consider the evidence. This module does not provide an action permit.

Unknown, unavailable or mismatched bytes are not converted into independent roots. The consumer must abstain or request re-acquisition.

## 5. Critical remaining boundary

This is **byte integrity plus a separately signed origin-verification assertion**, not proof of universal source truth.

The module receives already acquired byte slices; it does not perform HTTP, archive, database or filesystem retrieval. It proves that those supplied bytes hash to the values in the graph and that a separately trusted origin verifier signed matching metadata. The fourth authority must independently acquire the source records and validate the source's identity and publisher trust, verify transformation receipts and parent relationships, and document how it handles mirrors, syndicated sources, shared instruments, shared datasets, model-generated summaries and undisclosed upstream dependencies.

A content hash proves byte equality, not authorship, accuracy, licensing, recency or independence. A source-origin receipt proves which pinned verifier attested; it does not make that verifier infallible. The completeness problem therefore remains testable only through independent acquisition channels, external publisher/transparency metadata, cross-resolver disagreement tests and adversarial hidden-parent fixtures.

The origin-verifier profile and keys must live in a separate administrative and rollback domain. Production must not reuse the deterministic public test keys from any other MACE protocol fixture.

## 6. Adversarial tests and next work

The next test increment should include end-to-end signed fixtures for:

- exact source and payload bytes; changed bytes; truncated bytes; same digest field with different supplied data;
- missing, extra and duplicate ancestor artifacts/receipts;
- changed parent lists, derivation digests and common-cause groups;
- a valid but stale, expired, revoked or profile-rolled-back origin receipt;
- source-origin signer/profile reuse with each of the three existing authorities;
- a resolver that omits a hidden common-cause parent while the origin service independently discovers it;
- inconsistent mirrors and source revisions fetched through two independent paths;
- receipt replay across runs, invocations or graph digests;
- oversized artifacts and aggregate resource-budget exhaustion.

The final two completeness cases require an independent acquisition service and external fixtures; the current consumer verifier cannot manufacture that evidence.

## 7. Validation state

- Source changes are committed to the public branch named above.
- Schema syntax and Rust compilation/tests have not been executed here.
- No hosted CI pass, production deployment, independent origin operator, external benchmark result or consequential-use authorization is claimed.
