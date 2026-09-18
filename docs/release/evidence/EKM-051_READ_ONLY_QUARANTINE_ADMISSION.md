# EKM-051 — Read-Only Quarantine Admission

## Purpose

EKM-051 is the first restart tranche allowed to turn an externally decoded restart-v2 snapshot into a verified in-memory quarantine object.

The central invariant is:

**external bytes may become read-only epistemic state only after the exact candidate independently re-passes semantic validation, staged anchor review, verifier-provenance review, and verifier-trust continuity.**

The resulting quarantine is still not writable or activatable.

## Admission path

`AdmittedEpistemicRestartQuarantineV2::validate_and_construct(...)` does not accept caller-supplied PASS booleans or detached review decisions.

For the exact candidate snapshot it:

1. re-runs EKM-049 verifier-provenance review;
2. EKM-049 internally re-runs EKM-048 staged admission;
3. EKM-048 internally re-runs EKM-044 validation, EKM-047 trust policy, EKM-046 proof verification, tracker preview and EKM-045 continuity;
4. re-runs EKM-050 verifier-trust continuity from the exact EKM-049 provenance receipt;
5. independently re-runs EKM-043 V2 semantic validation immediately before reconstruction;
6. reconstructs a fresh `EpistemicLedger` only through ordinary append APIs;
7. requires stable provenance/claim/evidence IDs to re-emerge naturally;
8. compares every reconstructed ledger record with the decoded wire record;
9. binds the source wire checksum/V2 digest, staged-review digest, verifier-provenance digest, continuity disposition and pre-review trust sequences into a new quarantine-admission digest.

## Read-only state

The admitted quarantine contains:

- a private reconstructed `EpistemicLedger`;
- a private clone of the complete typed restart-v2 wire snapshot;
- the EKM-048 admission-review digest;
- the EKM-049 verifier-provenance digest;
- EKM-050 verifier-continuity disposition;
- the trusted anchor sequence observed before review;
- the trusted verifier trust-snapshot sequence observed before review;
- a domain-separated quarantine-admission digest.

Public access to the ledger and source snapshot is immutable only.

## Authority boundary

EKM-051 does **not**:

- hydrate an `EpistemicSupportStore`;
- reconstruct a writable `BeliefRevisionHistory`;
- expose an `into_live`, `activate`, `hydrate`, `commit`, or mutable-ledger API;
- advance the real restart-anchor tracker;
- advance the trusted verifier checkpoint;
- update evidence, confidence, uncertainty, causal state, world-model state, or action policy;
- perform file/network I/O;
- manage keys or signatures.

The quarantine hard-codes:

- `anchor_tracker_mutated = false`
- `trusted_verifier_state_mutated = false`
- `writable_hydration_authorized = false`
- `activation_authorized = false`

## Mix-and-match resistance

EKM-051 deliberately re-runs EKM-049 and EKM-050 internally. A caller cannot take a continuity PASS from candidate A and attach it to candidate B, because the verifier provenance and continuity result are derived in the same call from the exact snapshot being quarantined.

## Remaining trust-context limitation

The trusted restart-anchor tracker and trusted verifier checkpoint are still supplied as two distinct trusted inputs. EKM-051 records both histories, but V1 does not prove that the two checkpoints were committed atomically as one deployment trust epoch.

A later tranche should introduce a joint trusted-restart context that binds:

- anchor sequence/digest/receipt/capture cycle;
- verifier profile/provenance digest and trust-snapshot sequence;
- deployment/trust-domain identity;
- commit epoch;
- external persistence/attestation identity.

That joint context should be reviewed before writable hydration or activation.

## Qualification status

This tranche is stacked on EKM-050. At authoring time the EKM-050 exact-head CI run and the forced EKM-046 Format Check remain queued. Queue/cancellation status is not format, compile, Clippy, test, or runtime evidence.
