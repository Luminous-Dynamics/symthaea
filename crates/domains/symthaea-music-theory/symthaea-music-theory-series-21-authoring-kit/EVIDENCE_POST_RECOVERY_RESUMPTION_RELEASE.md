# Post-Recovery Publication Resumption Contract

## Purpose

This contract prevents an exceptional recovery decision from being treated as if ordinary publication had already become safe. It establishes a second, independently auditable boundary: **resumption**.

## State machine

```text
IncidentDetected
  -> Contained
  -> RecoveryAuthorized
  -> RecoveredAnchorEstablished
  -> AwaitingFreshWitnesses
  -> ResumptionReady
  -> ResumptionAuthorized
  -> PublicationResumed
```

No transition may be skipped. Recomputing outer SHA-256 values cannot repair a missing or invalid predecessor.

## Freshness rule

A checkpoint qualifies as fresh only when:

- its logical issuance epoch is strictly greater than the recovered anchor epoch;
- its catalog ordinal and event ordinal do not regress;
- its catalog extends the selected recovery checkpoint through a valid explicit lineage;
- its witness statements are produced for the recovered policy epoch;
- every mirror observation used for readiness is no earlier than checkpoint issuance;
- no authenticated rollback, equivocation, or fork proof affects the candidate lineage.

Logical freshness is not wall-clock freshness.

## Authority rule

The resumption authority is configured externally and cannot be inferred from the recovered witness policy. Implementations may choose the same organizations, but the identities and thresholds must be explicitly declared and hashed.

A verifier must receive the **expected locally trusted resumption policy**. It must not accept a threshold merely because the artifact embeds one.

## Mutation boundary

Every first post-recovery publication call must provide the exact authenticated resumption authorization. The catalog implementation must rerun:

- structural audit;
- expected-policy equality;
- external signature verification;
- trust-segment equality;
- head/lineage equality;
- logical-epoch monotonicity;
- allowance non-carryover checks.

A cached boolean or prior CLI decision is not authority.

## Delegation rule

Existing delegations remain historical evidence but are not active in a new trust segment. The default is:

- no unused allowance transfer;
- no implicit renewal;
- no identity reuse as authorization;
- new delegation issuance under the resumed segment.

A future explicit carryover protocol would require its own authority and is outside Series 21.

## Cross-segment status rule

A post-recovery catalog may refer to pre-recovery records. It may not supersede, revoke, or otherwise mutate their effective status through an ordinary segment-local event. Such a change requires an authenticated `CalibrationPublicationSegmentBridge`.

## Acceptance classes

- `StructurallyValid`: hashes, ordering, identities, and local invariants pass.
- `FreshlyWitnessed`: the recovered-policy threshold accepts a post-anchor checkpoint.
- `ResumptionReady`: all configured witness, mirror, conflict, lineage, and quarantine gates pass.
- `ResumptionAuthorized`: the expected external resumption policy authenticates the exact readiness/head payload.
- `PublicationResumed`: the catalog mutation boundary consumed that authorization exactly once for the first new publication.

These classes must not collapse into one boolean in persisted audit output.
