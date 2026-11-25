# MACE Frozen Decision and Sealed Evaluation Manifests
## Protocol contract v0.1 — 2026-11-25

**Status:** research schemas and protocol semantics. This increment defines manifest shapes, digest construction and evaluator-custody invariants. It does not implement the independent witness service, sealed artifact store, manifest verifier, or benchmark runner.

## 1. Why manifests are a security boundary

An outcome receipt is only as reproducible as the episode and evaluator configuration it names. A hash is not enough unless the object being hashed has a stable schema, an unambiguous canonicalization rule, and a custody boundary that prevents the evaluator from changing hidden labels after observing candidate results.

MACE therefore separates:

1. **Public frozen decision episode** — what the candidate may know before execution: task/input commitments, source snapshot, decision requirements, options, constraints, model/tool/runtime profiles, resource budget, and opaque commitments to randomization.
2. **Sealed evaluation manifest** — what the candidate must not know before every candidate trace is irreversibly sealed: arm assignments, random seeds, attack labels, ground truth, expected decisions, utility parameters, hard-constraint oracles, repair oracles, adjudication evidence and exclusion details.
3. **Detached commitment witness receipt** — a separately signed, independently anchored proof that the sealed manifest and the episode set were committed before the run.

These are different objects. They must not include one another's own hash in a way that creates a circular commitment.

## 2. Public episode manifest v1

Schema: `data/schemas/mace_decision_episode_manifest_v1.schema.json`

The public manifest binds:
- suite ID/version, task family, split and episode index;
- task-input, source-snapshot and provenance-graph digests;
- decision invocation, DecisionNeed, purpose, subject scope, option set, constraint set and decision policy;
- environment, reader, tool, retrieval-policy and runner profiles;
- resource-budget profile;
- episode-seed and arm-assignment commitments;
- freeze timestamp.

It intentionally contains **no arm label, random seed, expected answer, hidden attack label or evaluator oracle**. Public consumers may validate all public fields and the opaque commitments, but cannot derive the hidden assignment.

The manifest's JCS digest is:

`SHA-256("luminous:mace:decision-episode-manifest:v1" || 0x00 || JCS(manifest))`

The public episode-set root is a deterministic ordered commitment, not a self-referential field. Sort by UTF-8 episode ID; reject duplicates. Compute:

`SHA-256("luminous:mace:decision-episode-set:v1" || 0x00 || u64_be(count) || Σ(u64_be(id_bytes_len) || id_utf8 || manifest_sha256_raw32))`

The public `episode_seed_commitment_sha256` is computed from JCS of `{episode_id, seed}` under `luminous:mace:episode-seed:v1`. The public `arm_assignment_commitment_sha256` is computed from JCS of `{episode_id, arm_id, seed_commitment_sha256, treatment_configuration_sha256}` under `luminous:mace:arm-assignment:v1`. Use the exact same field names and UTF-8 values in the sealed assignment record.

## 3. Sealed evaluation manifest v1

Schema: `data/schemas/mace_sealed_evaluation_manifest_v1.schema.json`

The sealed manifest contains:
- suite and protocol identity;
- root digest and count of the exact frozen public episode set;
- exactly one treatment-arm assignment for each episode;
- preregistered analysis plan, utility scale, hard-constraint oracle, metric definitions, exclusion policy and stopping rule;
- hashes of evaluator-only artifacts, including hidden ground truth, attack labels, outcome and repair oracles, constraint oracles, metric parameters and adjudication evidence;
- evaluator, verifier, access-control and sealed-store profile hashes plus the commit timestamp and key ID;
- a protocol digest that requires candidate unreadability until all candidate traces are sealed.

The assignment array MUST have one unique entry for every public episode, sorted by UTF-8 episode ID. Runtime validation MUST reject missing, duplicate, surplus, or mismatched episode IDs and recompute every per-episode seed and arm-assignment commitment against the public manifest. The assignment root and artifact root must be recomputed from their canonical, ordered member sets. JSON Schema alone cannot enforce uniqueness by a nested property, root recomputation or correspondence between the two manifests.

The sealed manifest's digest is:

`SHA-256("luminous:mace:sealed-evaluation-manifest:v1" || 0x00 || JCS(sealed_manifest))`

The sealed store is a separate evaluator-controlled principal and data-access domain. A separate file or process under the candidate's credentials is not an isolation boundary. Candidate code, model context, retrieval-policy memory, logs accessible to the candidate, and online learning must not receive the hidden artifacts or decrypted assignments before trace sealing.

## 4. Detached commitment witness receipt v1

Schema: `data/schemas/mace_evaluation_manifest_commitment_receipt_v1.schema.json`

The sealed manifest cannot contain the digest of the witness receipt that signs it; that would be circular. The witness receipt is detached and binds:
- the public episode-set root;
- the sealed evaluation-manifest digest;
- roots for arm assignments, preregistration and sealed artifacts;
- the commit timestamp;
- the candidate access profile and reveal protocol;
- append-only witness log ID and entry ID.

The witness is a separate service with a separately pinned trust profile and signing identity. It MUST verify the sealed-manifest and public-episode roots before issuing a receipt, and MUST anchor the commitment in append-only storage or an independently witnessed transparency log. A self-signed evaluator statement without independently protected ordering/time does not establish commit-before-run.

The witness receipt payload digest is:

`SHA-256("luminous:mace:evaluation-manifest-commitment-payload:v1" || 0x00 || JCS(payload))`

The attestation preimage binds the payload digest, witness/verifier profile hashes, key ID and issue time. Its digest uses:

`SHA-256("luminous:mace:evaluation-manifest-commitment-attestation:v1" || 0x00 || JCS(attestation_preimage))`

The Ed25519 signature covers the UTF-8 bytes of the 64-character lowercase-hex attestation digest, matching the existing MACE pinned verifier convention. The full witness receipt digest is:

`SHA-256("luminous:mace:evaluation-manifest-commitment-receipt:v1" || 0x00 || JCS(full_receipt))`

The witness MUST establish that `committed_at_unix` precedes the first episode's execution start, that the commitment predates candidate access, and that the sealed object has not changed since anchoring. Clock trust and append-only ordering must be independent of the candidate host.

## 5. Binding into the outcome receipt

Decision Outcome Receipt v1 now includes `evaluation_manifest_commitment_receipt_sha256` in addition to `evaluation_manifest_sha256`.

The AML adapter now contains `evaluation_manifest_commitment_receipt.rs`, a strict consumer-side detached-witness verifier. It checks payload and attestation digest recomputation, the witness signature under a separately pinned MACE Ed25519 verifier profile, exact consumer-owned suite/manifest/root/access/reveal bindings, trusted issue time, and strict commit-before-candidate-access / commit-before-first-episode ordering.

Append-only log inclusion and consistency are mandatory through the `AppendOnlyManifestWitnessLogVerifierV1` interface. The adapter deliberately does not implement a concrete log client or trust anchor: production must provide a separately administered verifier that checks inclusion against an anchored checkpoint and consistency from a previously pinned checkpoint. A local signature without that proof is insufficient.

The verifier returns a private-field `VerifiedEvaluationManifestCommitmentV1` capability only after all checks, including the injected log verifier, pass. The outcome verifier now requires this capability rather than a caller-supplied hash string; it derives the exact sealed-manifest and witness-receipt digests from the capability. This prevents an application from accidentally passing an arbitrary digest without first verifying the detached witness.

The AML adapter also contains `frozen_decision_episode_manifest.rs`. It performs strict typed decoding, per-manifest JCS digest computation, ordered/unique episode-set validation, suite-identity consistency, and the canonical length-prefixed episode-set root computation. Its `VerifiedFrozenDecisionEpisodeSetV1` checks that root and suite against the verified witness, then issues a private `VerifiedDecisionEpisodeMembershipV1` capability for one exact episode. The outcome verifier requires both the witness capability and this per-episode membership capability; it checks the episode digest, episode ID, suite identity and set root. This closes the gap between a suite-level pre-run commitment and the exact episode whose outcome is being attested.

## 6. Commit, run, seal, reveal protocol

1. Freeze the public episode manifests and compute their ordered root.
2. Build the sealed evaluation manifest, including hidden assignments, random seeds, ground truth, oracle artifacts and preregistered analysis plan.
3. Independently recompute all public/hidden cross-bindings and canonical roots.
4. Submit the sealed manifest digest and roots to the separate witness service. Obtain a detached, signed, append-only commitment receipt.
5. Verify the witness receipt using a consumer-pinned witness trust profile. Only now may the runner expose public episode inputs to candidate systems.
6. Execute all arms under frozen reader/tool/runtime/resource profiles. Keep evaluator labels, treatment IDs, random seeds and oracles outside candidate-readable state.
7. Seal candidate traces, execution receipts and cost telemetry. Produce a trace-seal root and anchor it independently.
8. Only after every candidate trace is sealed may the evaluator reveal labels and decrypt evaluator artifacts.
9. Score the preregistered metrics, retain every applicable attack/control result, and issue outcome receipts that bind both manifest digests and the commitment-receipt digest.

No result may be excluded after reveal except under the preregistered exclusion policy. Any exclusion must carry a reason and its own evidence. Missing, not applicable, not evaluable, queued and skipped remain distinct outcomes.

## 7. Anti-gaming and independence tests

Required negative controls include:
- post-run mutation of the sealed manifest;
- swapping an arm label while retaining the old commitment;
- assignment duplication, omission or cross-episode substitution;
- seed reuse where the preregistration requires independent seeds;
- evaluator artifact access from the candidate principal;
- candidate retrieval or strategy-learning ingestion of hidden labels before trace seal;
- witness timestamp later than execution start;
- rollback or fork of the witness log;
- reveal before trace seal;
- post-hoc change to utility scale, exclusion policy or stopping rule.

For each test, preserve the raw witness response, storage/object digest, ACL decision, candidate trace-seal root, verification result, timestamps from protected clocks and exact code/profile SHAs.

## 8. Claim boundary and implementation sequence

This increment supplies three JSON Schemas, protocol semantics, a strict public episode-manifest/set validator, and the consumer-side detached-witness receipt verifier. The outcome verifier consumes private witness and per-episode membership capabilities. It does not implement a concrete transparency-log client/checkpoint trust, sealed assignment/artifact root builder/validator, independent witness service, sealed-store isolation, trace-sealing service, or full benchmark runner. Rust compilation/tests remain unexecuted in the current environment.

Next engineering steps:
1. Implement the sealed assignment/artifact canonical root builder and strict exact episode-to-assignment coverage validator.
2. Implement a concrete append-only witness-log verifier with pinned checkpoints, inclusion and consistency proofs, fork detection and rollback controls.
3. Deploy evaluator/witness principals with separate credentials, KMS keys, ACLs, storage and protected clocks.
4. Implement trace sealing and delayed reveal so evaluator labels never enter candidate-readable state.
5. Run paired arms and publish externally reproducible manifests, receipts and per-case results.

Unit tests are authored for manifest ordering, duplicate IDs, mixed suite identity and malformed fields, in addition to witness signature, timing, log-interface and binding cases. These tests have not been executed because this environment lacks the Rust toolchain. No benchmark outcome is claimed by these schemas. The goal is to make evaluator independence and pre-registration verifiable properties of the run, rather than promises in a benchmark README.
