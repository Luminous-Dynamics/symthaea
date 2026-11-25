# MACE sealed assignment and evaluator-artifact verifier v0.1

**Status:** consumer-side Rust verifier authored on top of #13600. Tests are authored but have not been compiled or executed in the current authoring environment. This is not a deployed evaluator, sealed store, witness service, transparency-log implementation, or benchmark result.

## Purpose

#13600 establishes a verified detached commitment to a complete public episode set and a sealed evaluation manifest. The remaining gap is to recompute the sealed manifest's internal roots and prove that its hidden assignment set corresponds exactly to the public episodes.

This increment adds \`sealed_evaluation_manifest.rs\` to the standalone Broca AML adapter. It returns a private-field \`VerifiedSealedEvaluationManifestV1\` capability only after the following cross-checks pass:

1. Strictly decode the complete sealed manifest, rejecting unknown and duplicate fields.
2. Recompute the public episode set through the existing canonical frozen-episode verifier and compare its root, suite/version, split and count with the detached witness and sealed manifest.
3. Require exactly one assignment for every public episode, in strict ascending UTF-8 episode-ID order. Reject missing, duplicate, surplus or substituted episode IDs.
4. Recompute each per-episode arm-assignment commitment from the exact tuple \`{episode_id, arm_id, seed_commitment_sha256, treatment_configuration_sha256}\` and compare it with the commitment exposed in that public episode.
5. Recompute the assignment-set, preregistration and sealed-artifact roots and compare all three with the independently verified detached witness.
6. Recompute the complete sealed-manifest digest and compare it with the witness.
7. Bind evaluator, verifier, access-control and sealed-store profile hashes to consumer-owned expected values. Bind the candidate-access and reveal-protocol hashes to the verified witness.
8. Reject a manifest that permits candidate readability before trace sealing or reveals evaluator data before all candidate traces are sealed.
9. Require the sealed custody commitment to follow the latest public episode freeze and precede the detached witness commitment.

## Canonical roots

All digests are lowercase SHA-256 hex. Hash preimages are domain-separated; member digests are decoded to their raw 32-byte value before inclusion in set roots.

### Per-episode assignment commitment

\`SHA-256("luminous:mace:arm-assignment:v1" || 0x00 || JCS({episode_id, arm_id, seed_commitment_sha256, treatment_configuration_sha256}))\`

The object uses exactly those four field names and the exact UTF-8 values supplied by the sealed assignment and public episode commitment.

### Assignment-set root

\`SHA-256("luminous:mace:arm-assignment-set:v1" || 0x00 || u64_be(count) || Σ(u64_be(id_utf8_len) || id_utf8 || assignment_commitment_raw32))\`

Assignments must be in the same strict ascending UTF-8 episode-ID order as the complete public episode set. Exactly one entry is required for each public episode.

### Pre-registration root

\`SHA-256("luminous:mace:pre-registration:v1" || 0x00 || JCS(pre_registration))\`

The committed object has six required digests: analysis plan, utility scale, hard-constraint oracle, metric definitions, exclusion policy and stopping rule.

### Sealed-artifact root

\`SHA-256("luminous:mace:sealed-artifact-set:v1" || 0x00 || u64_be(count) || Σ(u64_be(id_utf8_len) || id_utf8 || role_utf8_len_prefixed || content_sha256_raw32 || sealed_object_commitment_sha256_raw32))\`

Artifact IDs must be unique and strictly ordered by UTF-8 bytes. Each leaf binds its ID, role, content digest and sealed-object commitment. The supported roles remain the schema's enumerated roles; arbitrary role strings are rejected.

### Complete sealed-manifest digest

\`SHA-256("luminous:mace:sealed-evaluation-manifest:v1" || 0x00 || JCS(sealed_manifest))\`

The detached witness independently commits to this complete digest and to the separately recomputed assignment, preregistration and artifact roots. The formulas above are protocol v0.1 and must remain frozen for this schema version; any incompatible change requires a new version and new vectors.

## T9 and T10 are separate evaluator cases

The sealed-artifact protocol preserves separate attack identities:

- **T9 — whole-store rollback:** restore a valid older store snapshot that predates a later genuine write.
- **T10 — erasure resurrection:** restore a valid pre-erasure snapshot after the system has acknowledged erasure.

They must have separate episode assignments, attack labels, fixtures and scored outcomes. A T9 pass is not evidence of T10 resistance. A system without a supported erasure interface or restore capability is reported as Not Applicable for the corresponding case, never as Pass. The candidate must not receive the attack label or expected outcome before its complete trace is sealed.

## Anti-leakage boundary

The manifest's \`candidate_readable_before_trace_seal\` field must be false and \`reveal_condition\` must be exactly \`all_candidate_traces_sealed\`. These values are part of the full manifest digest already committed by the detached witness.

This is a contract check, not an access-control implementation. A separate file, process, or directory under candidate credentials is not independent custody. Production still requires evaluator and witness principals, credentials, KMS keys, stores, ACLs, protected clocks and backup/restore domains to be independently administered. The concrete append-only log verifier must validate inclusion and consistency against a separately pinned checkpoint and detect forks or rollback.

## Tests authored

The module includes negative and positive tests for:

- exact assignment/artifact/witness binding;
- missing/surplus assignments, ordering and duplicate episode identifiers;
- seed commitment, treatment configuration and arm-ID rebinding;
- separate T9 and T10 artifacts, and T10-artifact removal against the original witness;
- candidate-readable evaluator state even when the modified manifest is otherwise newly committed;
- artifact ordering/duplicate IDs;
- consumer-pinned custody profiles and custody/witness time ordering;
- unknown and duplicate JSON fields.

These tests are source additions only. They have **not** been executed. No compilation, rustfmt, Clippy, CI, real sealed-store isolation, concrete transparency-log verification, external benchmark run, or SOTA improvement is claimed.

## Next qualification gates

1. Run the AML adapter's complete locked test suite, rustfmt and strict Clippy at the exact PR head.
2. Add independent published digest vectors for assignment, preregistration, artifact and full-manifest roots, generated by a second implementation.
3. Implement the concrete append-only witness client: pinned checkpoints, inclusion and consistency proofs, fork detection, protected time and trust-anchor recovery.
4. Implement independently administered sealed storage and trace sealing/delayed reveal. Verify candidate principals cannot read assignments, artifacts, evaluator logs or decrypted labels before trace seal.
5. Connect the verified manifest capability to the paired decision runner, then run separate T9 and T10 arms with raw receipts and independent evaluator custody.

No benchmark score or production security qualification follows from this reference verifier alone.
