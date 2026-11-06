# MEL-RMI Studio artifact evidence boundary v1

**Date:** 6 November 2026  
**Implementation:** crates/domains/symthaea-muse/src/rmi_artifact_evidence.rs  
**Feature:** studio  
**Status:** additive verifier contract; no production identity/consent provider is wired.

## Why this layer exists

A synthetic RMI event ID is not an artifact identity. A storage URI, keeper key, candidate number, renderer label, or caller-supplied digest is not proof that the evaluated bytes are the bytes Studio committed.

RMI evaluation needs three independent joins:

1. event identity — the exact partner input or agent output event;
2. artifact identity — content-addressed bytes and a recomputed digest;
3. authority — an authenticated actor and a purpose-specific, unrevoked Studio consent receipt.

The verifier keeps these claims separate. It does not infer artifact identity from a URI and it does not equate a valid digest with authorship, quality, or playback.

## Implemented

- RmiTrialArtifactManifestV1 binds each declared input/output event to one or more typed artifact slots.
- Artifact identity must be sha256:<lowercase digest> and must match the declared byte count and recomputed SHA-256.
- The supplied byte set must match the manifest exactly. Missing bytes, duplicate payload IDs, unreferenced bytes, wrong event sides, duplicate slots, digest substitution, and byte-length mismatch are rejected.
- Manifest and receipt-body digests use the existing canonical JSON commitment helper.
- verify_piece_provenance_bytes independently rehashes the exact serialized recipe, symbolic score and rendered audio bytes against PieceProvenanceBundle.
- MIDI is verified only when its exact bytes and a provenance MIDI digest are both present. A MIDI URI or midi_exact boolean cannot manufacture a digest.
- The signed receipt envelope is bound to session, turn, task, policy, canonical manifest digest, exact artifact-ID set, purpose, actor, consent receipt references, commit sequence, idempotency key and commit time.
- verify_rmi_studio_evidence requires a configured RmiStudioReceiptVerifier. The UnavailableRmiStudioReceiptVerifier always fails closed.

## Production integration still required

No production signature, identity or consent authority is implemented by this PR. The trust-provider interface must be backed by the real Studio/Mycelix identity and consent boundary before a production receipt can be accepted. Its implementation must:

- authenticate the actor using a trusted key and identity-provider mapping;
- verify the receipt signature against a trusted key registry;
- verify that every referenced consent receipt is authentic, unexpired and not revoked at the verification time;
- verify that consent covers this session, exact artifact set, action and EvaluateSession purpose;
- reject use for personal memory, model training, publication or federation unless a separate exact-purpose authorization exists;
- bind the receipt to the durable Studio transaction and idempotency key, not a caller-constructed claim.

The current module does not read files from disk, access SQLite, authenticate actors, verify signatures, or resolve revocation. Studio's integration adapter must load exact bytes from authoritative storage, construct the manifest from the committed transaction, then call the verifier. Until the real provider is wired, use the unavailable verifier and accept no fully verified receipt.

## Evidence and claim ceiling

Passing the byte tests proves only that a supplied set of bytes matches the declared hashes and event bindings. Passing with a configured receipt verifier additionally means that the injected authority accepted the receipt/signature/consent contract. It does not prove that an instrument physically played the bytes, that a human heard them, that the musical change was useful, or that anyone legally authored the result.

For RMI-Bench, retain separate columns for:
- action requested;
- artifact committed;
- exact bytes independently rehashed;
- receipt signature accepted;
- purpose-specific consent accepted;
- callback-admitted / predicted playback / loopback-observed output.

Never collapse those into executed=true.

## Tests authored

The module tests exact input/output bytes, altered and equal-length-substituted payloads, missing and extra payloads, duplicate identities, event coverage and side mismatch, content-ID derivation, PieceProvenanceBundle recipe/score/audio checks, missing MIDI digest handling, receipt/manifest equality, purpose substitution, receipt-body tampering, trusted-time ordering, and the unavailable-verifier fail-closed path.

These tests exercise a test-only receipt verifier; they do not constitute a production signature or consent-provider test.
