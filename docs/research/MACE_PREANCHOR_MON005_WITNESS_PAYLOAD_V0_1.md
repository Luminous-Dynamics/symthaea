# MACE pre-anchor payload for MON-005 witnessing v0.1

**Status:** a typed Rust pre-anchor payload and consumer-side digest verifier are authored in the AML adapter. Nine Rust tests are authored but have not been compiled or executed. No live MON-005 adapter, distributed witness deployment, transparency-log publication, or external security qualification is claimed.

## Purpose

MACE now has a verified sealed evaluator manifest and a separate consumer-side verifier for its assignments, evaluator artifacts, preregistration and exact public episode-set membership. The remaining anchoring seam is to commit those exact MACE semantics into the repository's existing MON-005 witness protocol without creating a circular hash.

New module:

- \`research/broca-mace-transfer-bench-v0-1/aml-adapter/src/mace_evaluation_anchor_payload.rs\`

Export:

- \`pub mod mace_evaluation_anchor_payload;\`

The module builds a MACE-specific pre-anchor statement only from:

1. The sealed evaluator manifest supplied for digest recomputation.
2. A private-field \`VerifiedSealedEvaluationManifestV1\` capability.
3. A private-field \`VerifiedEvaluationManifestCommitmentV1\` capability.
4. Consumer-owned expected evaluator, verifier, access-control, sealed-store, candidate-access, reveal and trace-seal profile digests.

It rechecks the manifest digest, suite/version, episode-set root/count/split, assignment root, preregistration root, artifact root, custody profiles, candidate-access and reveal profiles, trace-seal profile and anti-leakage condition. It refuses candidate readability before trace seal and requires the reveal condition to be \`all_candidate_traces_sealed\`.

## Why the pre-anchor payload excludes log coordinates

A transparency log assigns its log ID, entry ID and checkpoint coordinates only after an entry is accepted. If those post-append values were included in the payload whose digest is submitted for append, the commitment would be self-referential.

Therefore the MACE pre-anchor statement contains:

- exact suite and protocol identity;
- exact held-out/development/validation split and public episode-set root/count;
- sealed evaluator-manifest digest;
- exact assignment, preregistration and sealed-artifact roots;
- consumer-pinned evaluator/verifier/access-control/sealed-store profiles;
- candidate-access, reveal and trace-seal profile digests;
- explicit anti-leakage condition.

It deliberately contains **no** anchor log ID, anchor entry ID, post-append checkpoint size, witness signature or commit timestamp. Those belong in the detached post-append witness receipt.

The pre-anchor MACE digest is:

\`SHA-256("luminous:mace:evaluation-anchor-payload:v1" || 0x00 || JCS(payload))\`

This digest is carried as the MACE payload value in an outer MON-005 evidence record. MON-005 remains the owner of its outer record format and hashes the complete canonical outer record (including tenant, sequence, predecessor and payload) to form its Merkle leaf. The MACE payload digest is **not** itself the MON-005 leaf hash.

## Correct verification order

1. Freeze the public decision episodes and build the sealed evaluator manifest.
2. Independently verify the exact assignments, artifacts, preregistration and sealed-manifest roots using #13625's consumer-side verifier.
3. Build the MACE pre-anchor payload from the verified manifest and detached evaluator-witness capabilities.
4. Canonicalize and compute the MACE payload digest; retain that digest outside candidate custody.
5. Submit the pre-anchor payload through the MON-005 evidence-record API. MON-005 canonicalizes and hashes its complete outer record, appends it, and returns the detached witness receipt with the log coordinates.
6. Verify the MON-005 checkpoint signature/quorum and pinned roster, then verify the outer record inclusion proof and consistency from the consumer's previously pinned checkpoint.
7. Resolve the exact MACE payload bytes from the included outer record and call \`parse_and_verify_mace_evaluation_anchor_payload_v1\` against the independently retained MACE payload digest.
8. Only after the independent witness and MACE payload verifications succeed may the benchmark runner expose candidate episodes.

This order avoids circularity and prevents an arbitrary hash string from being substituted for a parsed/verified MACE payload. It does not itself implement MON-005 network transport or authenticate a production witness operator.

## Consumer-side API

- \`build_mace_evaluation_anchor_payload_v1\` emits a private-field \`VerifiedMaceEvaluationAnchorPayloadV1\` only from verified manifest/witness capabilities and consumer-pinned profile bindings.
- \`mon005_payload_sha256()\` returns the digest to place in the MACE payload field of the MON-005 outer evidence record.
- \`parse_and_verify_mace_evaluation_anchor_payload_v1\` strictly decodes a retrieved MACE payload, rejects unknown/duplicate fields, recomputes its JCS/domain-separated digest, compares it with the consumer's expected digest and returns a private-field verified capability.
- The caller must still verify the outer MON-005 record, checkpoint, signatures/quorum, inclusion proof and append-only consistency proof separately. A valid MACE payload digest is not proof that anything was externally logged.

## Fixed reference vector

The deterministic two-episode T9/T10 fixture was independently reconstructed in Python using canonical JSON key ordering and compact UTF-8 encoding. The 1,499-byte canonical payload produces:

\`117579a8803ecedd6b5eee4efbc73c4fe66ce9088c7687a5b8785ac4815d1154\`

The Rust test pins this value and checks that the serialized pre-anchor payload contains no \`anchor_log_id\`, \`anchor_entry_id\` or \`committed_at_unix\`. The vector is source-level cross-language evidence; the Rust assertion has not been executed.

## Tests authored

Nine Rust tests cover:

- building the payload from verified evaluator and witness capabilities;
- fixed independent JCS/domain-separated digest;
- absence of post-append log coordinates;
- round-trip retrieval and consumer-side digest verification;
- payload mutation and digest mismatch;
- unknown and duplicate JSON fields;
- altered manifest rejection;
- a valid but wrong episode-set witness root;
- custody, candidate-access and reveal profile pinning.

Tests are authored but not executed. No compile, rustfmt, Clippy, CI, MON-005 integration, live transparency publication, benchmark result or security qualification is claimed.

## Current integration boundary and next gates

The repository's MON-005 work (#13515 / #13518) already has RFC 9162 inclusion and consistency proof semantics, 3-of-4 test-witness quorum and a SQLite reference state adapter. Its published local 49/49 results and 496 inclusion / 2,016 consistency checks are useful reference evidence, but all witness simulations remain same-host/same-process with fixture keys and no independent operator deployment.

Next steps:

1. Implement the provider adapter that maps the exact MACE pre-anchor payload bytes into the existing MON-005 outer-record DTO. Do not assume the MACE payload digest is the outer record leaf.
2. Retain the exact returned outer record bytes and detached witness receipt; verify the checkpoint/quorum, inclusion and consistency proofs from a consumer-pinned prior checkpoint.
3. Add a cross-language test that computes the outer MON-005 record leaf from the exact payload bytes and proves inclusion using MON-005's proof generator.
4. Test crash, lost acknowledgement, duplicate append, checkpoint rollback, split-view and witness unavailability as separate outcomes.
5. Deploy witness identities in independently administered failure/backup domains with production key custody and external transparency monitoring.
6. Integrate the verified pre-anchor capability into the frozen-decision runner before any candidate exposure.

This increment is a non-circular protocol bridge and a consumer-side payload verifier—not an operationally independent witness service.
