# EKM-047 — Restart Anchor Trust Policy

## Purpose

EKM-046 provides a deployment-specific verification-provider boundary for restart anchor evidence. A provider returning `true`, however, should not by itself decide whether that authority or assurance mechanism is acceptable for a particular deployment.

EKM-047 inserts a pure policy gate before provider verification.

The central invariant is:

**cryptographic/protected-state validity does not imply deployment trust-policy admissibility.**

## Policy dimensions

`RestartAnchorTrustPolicyV1` canonically binds:

- the exact allowed authority identifiers;
- the allowed anchor-evidence mechanisms;
- the maximum anchor validity-window length;
- the maximum delay from anchor issue to verification.

Authority identifiers are trimmed, bounded, sorted and duplicate-rejected. Evidence kinds are sorted into a stable canonical order and duplicate-rejected. Empty authority/evidence sets and zero policy windows fail closed.

## Two-stage admission

`RestartAnchorTrustGateV1::evaluate` is pure and does not call the verification provider.

It can reject:

- an authority not present in policy;
- an assurance mechanism not present in policy;
- an excessive issue/expiry window;
- verification attempted before issue;
- already-expired anchor evidence;
- verification delayed beyond policy.

Only an eligible decision may proceed to `verify_restart_anchor_evidence_under_policy`, which then invokes the EKM-046 external verifier and retains all of EKM-046's exact receipt binding and freshness checks.

This makes the order explicit:

`local trust policy -> external proof verification -> monotonic anchor tracking -> continuity review`

rather than allowing provider acceptance to override local trust policy.

## Authority boundary

A successful EKM-047 policy decision means only **eligible for provider verification**.

A successfully verified anchor still has:

`quarantine_construction_authorized = false`

`activation_authorized = false`

EKM-047 does not construct quarantine state, hydrate writable state, activate restored state, persist trust state, perform file/network I/O, manage keys, or mutate evidence/belief/causal/world-model/action state.

## Qualification boundary

EKM-047 is stacked on EKM-046. The previously forced exact-head Format Check for EKM-046 entered the queue after the original workflow was uniformly cancelled, but queued execution is still not PASS evidence. GitHub Actions remains authoritative for format, compile, Clippy, tests and runtime qualification.
