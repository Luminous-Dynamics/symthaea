# Nixward CROSS-071 — Observer-Sealed Service Pre-State V1

## Purpose

CROSS-071 closes the provenance-origin gap in service authorization.

Earlier layers could bind a pre-state digest into an authorization context, but an admission caller could still supply that digest as ordinary string data.

This tranche makes the pre-state itself an observer-produced capability.

## Trusted path

`systemd D-Bus observation -> NixServiceObservedStateV1 -> NixVerifiedServicePreStateV1 -> service-effect admission -> intent`

The sealed token contains:

- canonical unit;
- NixOS generation;
- exact service-state digest;
- canonical generation-bound pre-state identity.

The wrapper is non-serializable and non-cloneable.

## Native observation

The read-only systemd observer resolves the exact unit and reads the same strict state properties used by the existing Nixward service-state digest:

- Id;
- Names;
- LoadState;
- ActiveState;
- SubState;
- UnitFileState.

The observer rechecks the systemd manager unique owner across this observation and refuses an incarnation rollover.

The existing Nixward digest algorithm is reused rather than creating a second semantic definition of service state.

## Admission

`NixServiceEffectAdmissionV1` no longer accepts raw pre-state identity, generation, or pre-state digest inputs.

It receives `NixVerifiedServicePreStateV1`.

The admission layer derives the context directly from the sealed token and the independently observer-sealed definition-content commitment.

This means a caller cannot obtain an authority-valid service intent merely by knowing a digest format and fabricating matching text.

## Cross-record binding

The sealed pre-state is carried into:

`pre-state token -> effect context -> intent digest -> authorization context digest -> post-state expectation`

The context continues to bind the pre-state digest into the exact canonical service pre-state identity. The authority chain therefore still detects generation/unit/state divergence.

## Fail-closed behavior

The sealed pre-state constructor rejects:

- generation zero;
- malformed service state;
- non-canonical service-unit identity;
- invalid observed Unit identity/name set;
- invalid state vocabulary.

The admission layer rejects a sealed pre-state whose unit does not equal the observer-sealed definition commitment's unit.

## Claim ceiling

This does not provide an atomic snapshot against every concurrent actor.

A later service-state change can still occur after the observer snapshot and before mutation dispatch; the existing pre-dispatch revalidation and later transaction hardening remain necessary.

The token is an authenticated-in-process provenance capability, not a cryptographic signature from systemd.

## Relationship to the hardening chain

- CROSS-060 binds the service-effect context into authorization.
- CROSS-061 pre-arms JobRemoved.
- CROSS-062 verifies repeated stability observations.
- CROSS-063 binds systemd manager incarnation.
- CROSS-065 binds native dispatch to that incarnation.
- CROSS-066 makes the receipt semantically self-verifying.
- CROSS-067 makes source identity self-verifying.
- CROSS-068 adds byte-level source-content observation.
- CROSS-069 binds the content commitment into the authority identity.
- CROSS-070 fixes lifecycle primitive ordering.
- CROSS-071 makes the pre-state itself observer-originated.

## Qualification

Exact-head GitHub Actions remain the qualification authority. Queued, pending, cancelled, mergeable, or static-only results are not PASS evidence.