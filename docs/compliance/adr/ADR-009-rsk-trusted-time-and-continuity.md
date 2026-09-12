# ADR-009: RSK Trusted Time and Continuity Boundary

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

RSK authority depends on temporal facts: grant not-before/expiry, evidence freshness, monitor freshness, policy/trust snapshot validity, authorization evaluation time, and recovery/checkpoint continuity.

The reference API currently represents time as caller-provided Unix-second scalars. #1726 improves the reference state machine by enforcing:

```text
evaluated_at <= commit_time < expires_at
```

but that does not establish that `commit_time` is trustworthy.

A production boundary must resist clock rollback, restart discontinuity, stale sources, spoofed/untrusted sources, large forward jumps, and conflicting time references without letting the requester choose whichever clock makes authority valid.

This ADR contains no physical replication mechanism.

## Decision

Production RSK consumes an opaque verifier-produced time/continuity capability rather than an arbitrary scalar wall-clock assertion.

Time is represented conservatively as an interval:

```text
VerifiedTimeInterval {
    earliest_possible_time,
    latest_possible_time,
    continuity_epoch,
    continuity_sequence,
    source/provenance evidence,
    uncertainty/profile metadata,
}
```

The exact type names are non-normative. The semantics are normative.

## Why an interval

A scalar can hide uncertainty. An interval makes uncertainty explicit and monotone toward denial.

To prove an authorization evaluated at `E` and expiring at `X` may commit, RSK requires:

```text
verified_time.earliest >= E
verified_time.latest   < X
```

If either inequality cannot be proven for the entire accepted interval, new positive authority is denied/frozen.

The authority path does not pick a favorable point inside the interval.

## Freshness arithmetic

Freshness checks use conservative maximum age.

If current verified time is `[now_min, now_max]` and an observation's verified occurrence interval is `[obs_min, obs_max]`, a conservative maximum age is based on the latest possible current time and earliest possible observation time:

```text
max_age = now_max - obs_min
```

subject to checked arithmetic and continuity compatibility.

Authority requiring `max_age <= policy_limit` denies if that cannot be established.

## Authentication is necessary but not sufficient

An authenticated time protocol can establish source/message authenticity and replay properties. It does not alone prove that the source is sufficiently accurate, independent, current, uncompromised, or continuous for an RSK deployment.

Production verification therefore combines:

- source identity/authentication where applicable;
- source lifecycle/trust state;
- source uncertainty bounds;
- monotonic local continuity state;
- policy-defined source/failure-domain requirements;
- disagreement/fusion rules;
- restart continuity evidence;
- explicit authority-time interval output.

## Time-source independence

Multiple sources are not automatically independent.

A deployment profile may reason about common-mode dimensions such as:

- same GNSS constellation/receiver chain;
- same upstream network time server/root;
- same local oscillator/discipline loop;
- same host/process;
- same administrator/configuration domain;
- same network path/infrastructure;
- same external reference institution/system.

Source independence uses the same trusted failure-domain philosophy as #1762/#1766.

## Multi-source disagreement

When configured sources disagree beyond the policy's verified tolerance/combination model:

- do not select the source favorable to authority;
- do not silently average incompatible observations into a trusted scalar;
- mark time continuity uncertain;
- freeze new positive replication authority unless a policy-approved robust fusion method can still establish a conservative interval under the declared compromise budget.

The exact robust-fusion algorithm is deployment-specific and must state its adversary/failure assumptions.

## Continuity epoch

Wall-clock value and continuity are separate facts.

A `continuity_epoch` identifies one verified temporal continuity lineage. Within an epoch, accepted observations advance a monotonic sequence/state.

A process restart does not create a new trusted continuity epoch merely because the process restarted.

If accepted continuity cannot be reconstructed from durable trusted state, RSK freezes new authority until:

- continuity is re-established according to policy; or
- externally governed recovery creates a fresh authority/recovery epoch with no implicit positive-authority carryover.

## Rollback rule

A verified time observation cannot be accepted if it would roll accepted continuity backward according to the configured temporal model.

Rollback includes more than wall-clock `< previous_wall_clock`; it can include:

- continuity sequence regression;
- monotonic-counter regression;
- accepted-source state rollback;
- old time-attestation replay;
- restart from older checkpoint;
- trust/policy snapshot rollback affecting time-source acceptance.

Any unresolved rollback condition freezes new positive authority.

## Forward-jump rule

Large forward movement can be a fault/attack too.

A forward jump may safely expire authority earlier, but it can also corrupt freshness/recovery assumptions. Policy therefore defines maximum tolerated discontinuity/uncertainty for accepting a new continuity observation.

An out-of-profile jump causes uncertainty/freeze rather than automatic destructive action or automatic trust in the new time.

## Source holdover

A deployment may use local oscillator/monotonic holdover during loss of external synchronization, but only within a verified policy-defined uncertainty growth model.

As holdover uncertainty expands, the verified interval widens. Once the interval no longer proves required temporal predicates, new positive authority freezes.

Outage never extends an existing grant/evidence expiry.

## Reboot/restart continuity

Production time continuity across reboot must be established using deployment-appropriate durable/trusted evidence, for example a monotonic hardware/security state, authenticated checkpoint, multiple external references, or another reviewed mechanism.

No particular hardware mechanism is constitutionally required. What is required is that restart cannot erase temporal anti-rollback state silently.

## Time capability is not a wire format

Like other `Verified*` capabilities, trusted time is a verifier-produced process capability, not trusted by direct deserialization.

Durable storage records raw observations, accepted continuity/checkpoint evidence, and provenance. On restart, verification reconstructs the capability.

## Relationship to authorization

A valid time capability is one predicate among many. It cannot cancel:

- quarantine/revocation;
- fork/non-operational state;
- stale cursor/head;
- capability/budget restrictions;
- invalid grant/quorum/policy;
- containment drift;
- unadmitted build/runtime identity.

## External architectural references

This design is informed by, but not certified against:

- IETF RFC 8915 Network Time Security (NTS), which adds authentication/security to NTP exchanges;
- NIST resilient timing work emphasizing diverse time references, monitoring, redundancy, and resilience;
- NIST work on timing uncertainty and secure/resilient timing infrastructure.

These sources support authenticated/resilient time distribution; RSK adds its own conservative authority-interval and continuity semantics.

## Threat coverage

Primary threats:

- T11 revoked/expired signer replay where time affects lifecycle;
- T12 clock rollback/pre-evaluation commit;
- T13 forward jump/time-source disagreement;
- T18 rollback of durable continuity state;
- T23 outage extends validity;
- T29 time-service denial-of-service.

## Evidence discipline

This ADR is design evidence only. It does not close #1669.

Production promotion requires executable evidence for rollback, restart, source disagreement, uncertainty growth/holdover, stale observation, forward jumps, expiry boundaries, current-negative dominance, and exact runtime identity.

## Consequences

### Positive

- uncertainty is explicit instead of hidden in a scalar timestamp;
- all temporal ambiguity moves authority toward denial, never toward extension;
- authenticated source identity is separated from accuracy/continuity trust;
- restart and rollback become first-class authority concerns;
- multi-source designs cannot choose the convenient clock;
- holdover degrades gracefully as uncertainty widens.

### Cost

- deployments must define time-source/uncertainty/failure-domain profiles;
- time arithmetic and continuity persistence become part of the production TCB;
- outages/disagreement may freeze replication authority earlier than permissive systems;
- exact source/fusion implementations require deployment-specific validation.

## Related work

- #1335 production-admission umbrella
- #1669 trusted time/continuity
- #1676 threat model
- #1726 temporal-monotonicity semantic floor
- #1766 verified evidence boundary
- #1772 recovery separation

**Production admission remains DENIED.**