# EKM-048 — Staged Restart Admission Review

## Purpose

EKM-044 through EKM-047 establish validation, continuity, external anchor evidence and local trust policy. A restore candidate can satisfy all of those checks while still not being safe to commit into trusted restart history.

EKM-048 binds the complete review into one immutable receipt **without advancing the real anchor tracker**.

The central invariant is:

**candidate review may preview trusted-anchor advancement, but the trusted anchor must not advance until a later successful activation/commit boundary.**

Advancing the anchor during review would create a failure mode where an unsuccessful restore attempt makes the still-running prior state appear stale or rolled back.

## Review sequence

`EpistemicRestartAdmissionReviewReceiptV1::validate_and_review` performs:

1. EKM-044 semantic validation and validation-receipt capture;
2. EKM-047 local trust-policy admission;
3. EKM-046 external proof verification bound to the exact validation receipt;
4. a full EKM-046 tracker `accept` against a **clone** of the trusted tracker;
5. EKM-045 continuity evaluation against the actual prior trusted receipt;
6. a requirement that the candidate is genuine `ForwardProgress`;
7. canonical policy hashing and immutable admission-review receipt generation.

The original tracker is borrowed immutably and cannot be advanced by this path.

## Receipt bindings

The review receipt binds:

- exact candidate validation-receipt digest and capture cycle;
- canonical EKM-047 trust-policy digest;
- prior trusted anchor sequence, anchor digest, receipt digest and capture cycle;
- candidate anchor sequence, statement digest and proof digest;
- candidate anchor verification cycle;
- exact EKM-045 continuity disposition;
- review observation cycle;
- explicit `anchor_tracker_mutated = false`;
- explicit `quarantine_construction_authorized = false`;
- explicit `activation_authorized = false`.

The receipt itself is domain-separated with BLAKE3 and is an audit identity, not a signature or capability token.

## Rejections

A receipt cannot be produced for:

- an uninitialized trusted tracker;
- EKM-044 validation failure;
- EKM-047 policy rejection;
- EKM-046 provider/receipt/freshness rejection;
- a candidate that would fail EKM-046 tracker progression;
- rollback or same-cycle substitution;
- anything other than monotonic forward progress.

## Authority boundary

EKM-048 does **not**:

- mutate the trusted anchor tracker;
- construct quarantine state;
- hydrate writable epistemic state;
- activate restored state;
- persist or sign trust state;
- perform file/network I/O;
- mutate evidence, belief, causal, world-model or action state.

A future quarantine-construction layer may require an exact EKM-048 receipt, but EKM-048 itself grants no restore authority.

## Qualification boundary

EKM-048 is stacked on EKM-047. The EKM-046 full CI run was uniformly cancelled; its explicitly re-run exact-head Format Check is still the smallest executable probe and remains authoritative only if it actually executes. No queued/cancelled state is treated as PASS or product failure.
