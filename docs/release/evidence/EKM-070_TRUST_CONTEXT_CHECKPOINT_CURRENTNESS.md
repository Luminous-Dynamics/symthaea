# EKM-070 — Trust-Context Checkpoint Currentness

## Purpose

EKM-054 proves an exact joint restart trust-context checkpoint is externally protected and internally consistent. Its continuity gate proves ordered predecessor-chain forward progress between supplied checkpoints. Neither fact independently proves that a particular supplied checkpoint is the deployment's **current/latest head**.

A generic signature is especially insufficient: checkpoint N can remain authentically signed after checkpoint N+1 exists.

The central invariant is:

**trust-context currentness is accepted only when a deployment provider whose evidence contract explicitly means current monotonic/head state attests to the exact EKM-054 checkpoint statement.**

## Evidence classes

`RestartTrustContextCurrentnessEvidenceKindV1` intentionally contains only:

- `MonotonicProtectedState`
- `HardwareMonotonicCounter`
- `TransparencyLogHead`
- `CurrentHeadWitnessQuorum`

There is deliberately no generic `SignedCheckpoint` evidence kind.

## Canonical statement

The current-head statement binds:

- deployment ID
- trust-domain ID
- exact EKM-054 checkpoint sequence
- exact EKM-054 checkpoint statement digest
- exact joint trust-context digest
- context commit cycle
- embedded restart-anchor sequence/capture cycle
- verifier trust-snapshot sequence
- EKM-054 verification cycle and expiry
- currentness attestation cycle and expiry
- authority ID
- current-head evidence kind

The currentness window cannot outlive the EKM-054 checkpoint it attests.

## Provider boundary

`RestartTrustContextCurrentnessVerifierV1` receives the canonical statement digest and bounded opaque proof bytes.

Returning `true` means the provider attests that this exact EKM-054 checkpoint is the current monotonic/head state under the selected evidence mechanism at the attestation cycle.

The knowledge layer does not implement TPM counters, protected storage, transparency logs, or witness consensus itself.

## Verified receipt

`VerifiedRestartTrustContextCurrentnessV1` records:

- canonical statement
- statement digest
- proof digest
- verification cycle
- `current_head_proven = true`

while retaining:

- `trusted_state_mutated = false`
- `activation_preflight_authorized = false`
- `activation_authorized = false`
- `trusted_checkpoint_commit_authorized = false`

## Relationship to EKM-054 continuity

These are separate claims:

- **EKM-054 continuity:** candidate checkpoint correctly succeeds the previous supplied checkpoint.
- **EKM-070 currentness:** an external monotonic/current-head provider says this exact checkpoint is the current head now.

A deployment can require both. One does not substitute for the other.

## Authority boundary

EKM-070 does not:

- mutate the protected checkpoint chain;
- advance the joint trust context;
- expose sandbox writable state;
- authorize live-state swap;
- authorize activation;
- authorize rollback;
- authorize checkpoint commit;
- touch legacy confidence, causal/world-model state, or action state.

## Qualification status

This tranche is stacked on EKM-069 / PR #4011. Parent EKM-069 exact-head CI #7450 was queued when EKM-070 was prepared.

GitHub Actions remains the executable authority. Static/API review is not rustfmt, compilation, Clippy, unit-test, integration-test, runtime or deployment evidence.

## Next boundary

A later preflight-completion receipt may combine the exact EKM-069 composition receipt with an EKM-070 current-head receipt bound to the same EKM-054 checkpoint. Only then may it report `activation_transaction_review_eligible = true`.

Even that eligibility must not authorize live-state swap, rollback, activation, or trusted-checkpoint commit. Those remain responsibilities of a later atomic activation transaction.