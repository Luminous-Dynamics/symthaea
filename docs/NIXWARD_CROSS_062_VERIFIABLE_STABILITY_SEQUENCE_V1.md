# Nixward CROSS-062 — Verifiable Stability Sequence V1

## Purpose

CROSS-062 replaces the former scalar stability assertion with a verifiable sequence of independently captured observations.

A stability claim now requires evidence of the form:

```
sample 1 -> sample 2 -> ... -> final post-state observation
```

rather than:

```
sample_count >= 2
```

The distinction matters because a count does not prove that any particular observations existed, that they were ordered, or that they described the same service identity and state.

## Stability sample

Each `NixPostStateStabilitySampleV1` binds:

- operation;
- canonical service unit;
- exact systemd Unit object path;
- NixOS generation;
- systemd unit-definition identity digest;
- semantic service-state digest;
- systemd manager unique owner;
- invocation identity;
- systemd monotonic state-change timestamp;
- monotonic capture timestamp.

The semantic state digest commits operation, unit, ActiveState, SubState, UnitFileState, and InvocationID. Identity fields such as object path and manager owner are bound separately so a state collision cannot erase provenance.

## Sequence contract

A valid stability sequence must satisfy all of:

1. at least two samples;
2. a declared monotonic window whose duration is at least the required window;
3. every sample lies inside that window;
4. capture timestamps are strictly increasing;
5. every sample reports the same operation, canonical unit, Unit object path, generation, definition digest, semantic state digest, systemd manager owner, invocation ID, and StateChangeTimestampMonotonic;
6. every sample's systemd state-change timestamp is at or before the beginning of the stability window;
7. the deterministic sequence digest exactly matches the ordered sample list.

The maximum sequence length is bounded to keep the verification surface finite.

## Observer sealing

The raw stability structure is serializable data, not authority.

Production receipt construction accepts only `NixVerifiedPostStateStabilityEvidenceV1`, whose constructor is crate-visible and observer-only. The mechanical observation-boundary scanner rejects calls to that sealing factory from every protected authority module other than the independent read-only systemd observer.

This keeps "data that claims stability" separate from "evidence produced by the observation boundary."

## Live observer protocol

The systemd observer captures two real post-state observations separated by the requested monotonic interval.

The observer:

1. captures the systemd manager unique owner;
2. resolves the exact Unit object;
3. reads the required Unit and Service properties;
4. rechecks the systemd manager owner and rejects a manager-incarnation transition;
5. records the monotonic observation time;
6. waits for the requested interval;
7. performs a second independent observation;
8. constructs the two typed samples;
9. computes the canonical sequence digest;
10. seals the sequence before returning it.

The receipt builder then requires a final observer-sealed post-state observation at or after the sequence's end timestamp and verifies that its identity/state fields match the sequence.

## Why StateChangeTimestampMonotonic is still necessary

Repeated snapshots are stronger than a sample count because they prove actual observations and give the verifier a deterministic sequence to inspect.

The sequence also binds systemd's monotonic state-change timestamp. Therefore a state transition that updates that timestamp inside the declared stability window invalidates the stability claim even when the eventual state returns to the same value.

This remains an observation-based claim, not an atomic kernel guarantee.

## Claim ceiling

CROSS-062 does **not** claim:

- atomic CAS semantics against concurrent writers;
- cryptographic attestation of systemd or the local D-Bus;
- proof that a transient event invisible to the captured systemd properties did not occur;
- proof that a service definition's file contents remained unchanged when only FragmentPath/DropInPaths identities were unchanged.

CROSS-060 remains responsible for authorization-bound service-definition commitment. CROSS-063 carries systemd manager incarnation into durable Job evidence and receipts. CROSS-061 ensures the lifecycle JobRemoved watcher is armed before mutation dispatch.

## Qualification standard

The implementation is not considered qualified merely because it compiles locally, is mergeable, or has queued/pending Actions.

The required standard is exact-head GitHub Actions evidence for the checked-in qualification path, including the source-boundary fence and post-state tests.
