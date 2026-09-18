# EKM-057 — Mutation-Time Evidence Seal Wire Envelope

## Purpose

EKM-056 retains the complete EKM-028 mutation-time claim/evidence census as a typed persistence sidecar. EKM-057 gives that sidecar a bounded deterministic wire representation without granting any construction, hydration, or activation authority.

The central invariant is:

**mutation-time evidence seals may cross a storage/process boundary only as bounded untrusted data; checksum validity is not proof that the seals belong to the accompanying restart mutation history.**

## Envelope

`BeliefMutationSealWireV1` uses:

- fixed magic and version framing;
- explicit encoding tag;
- declared payload length;
- 256 MiB payload ceiling;
- 16 MiB per-string ceiling;
- one-million seal-record ceiling;
- one-million aggregate sealed-evidence ceiling;
- checked integer conversion and offset arithmetic;
- UTF-8 validation;
- stable explicit claim/evidence/polarity tags;
- strict optional-value tags;
- rejection of unknown tags;
- rejection of trailing bytes;
- a domain-separated BLAKE3 corruption checksum.

## Encoded semantics

Every wire record carries:

- mutation receipt ID;
- source revision receipt ID;
- claim ID;
- seal/evaluation cycle;
- mutation application cycle;
- complete sealed claim metadata;
- complete ordered mutation-time evidence census;
- the EKM-056 record digest.

The envelope also carries:

- seal-capsule capture cycle;
- linked EKM-030 mutation-capsule capture cycle;
- record inventory;
- claimed EKM-056 capsule digest;
- wire checksum.

## Intentional cross-component dependency

The internal EKM-056 record digest binds the exact EKM-030 mutation receipt, including support transitions, authorization identity and application chronology.

EKM-057 deliberately does **not** duplicate all of those mutation fields inside the seal sidecar. The corresponding base restart mutation record remains the independent source from which the mutation-binding digest must later be recomputed.

Therefore:

- EKM-057 checksum validation can prove the sidecar bytes were not accidentally altered;
- EKM-057 parsing can prove the payload is structurally bounded and typed;
- EKM-057 alone cannot prove a seal belongs to a particular base mutation;
- a later cross-component validator must recompute the mutation binding from the base restart wire and then reproduce each EKM-056 record/capsule digest.

This avoids making the sidecar self-authenticating.

## Decoder boundary

Decoding returns only `BeliefMutationSealWireSnapshotV1`.

It does not construct:

- `PersistedBeliefMutationEvidenceSealV1`;
- `BeliefMutationEvidenceSealCapsuleV1`;
- a restart capsule;
- a quarantine object;
- an EKM-055 hydration sandbox;
- any mutation or activation capability.

## Tests

The module includes bounded envelope tests for:

- deterministic empty-capsule round-trip to an untrusted DTO;
- rejection of trailing envelope bytes.

Cross-component semantic tests belong in the next validator tranche because this envelope intentionally does not contain the base mutation history.

## Non-authorities

EKM-057 does not mutate belief support, evidence, revision history, trust checkpoints, legacy confidence, causal state, world-model state, or action policy. It performs no file/network I/O and no key custody.

## Qualification status

This tranche is stacked on EKM-056. GitHub Actions remains the executable authority. At creation time EKM-056 CI #7257 remains queued; no format, compile, Clippy, test or runtime PASS is inferred from static review or queue state.
