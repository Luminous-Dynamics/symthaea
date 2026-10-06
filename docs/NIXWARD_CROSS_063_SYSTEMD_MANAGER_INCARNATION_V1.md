# Nixward CROSS-063 — systemd Manager Incarnation Binding V1

## Purpose

CROSS-063 prevents durable Nixward evidence from collapsing distinct systemd manager lifetimes into one historical identity.

The systemd well-known name `org.freedesktop.systemd1` identifies the service, while the bus supplies a unique connection name for the current owning connection. D-Bus guarantees that a unique connection name remains bound to one connection for its lifetime and is never reused for another connection on that bus. citeturn117926search0turn117926search1

## Binding

For lifecycle jobs, the observer now carries the systemd manager's unique owner through the full evidence path:

```
systemd manager owner
    ↓
live Job handle
    ↓
JobRemoved sender check
    ↓
NixSystemdJobEvidenceV1
    ↓
NixPostStateReceiptV1
    ↓
receipt digest
```

The owner is deliberately distinct from Nixward's own process/daemon incarnation. They describe different failure domains.

## Live capture

The read-only observer resolves the owner of `org.freedesktop.systemd1` before/around the governed watcher transaction and validates it as a D-Bus unique name.

The live Job handle stores:

- Job.Id;
- JobType;
- canonical unit;
- Job object path;
- unit object path;
- systemd manager unique owner.

The handle is not constructible through the public API and validation rejects malformed owner identities.

## Completion correlation

The pre-armed watcher accepts `JobRemoved` only when:

- the message sender equals the captured systemd manager owner;
- the native `uoss` body is valid;
- Job.Id matches exactly;
- Job object path matches exactly;
- canonical unit matches exactly.

A manager-owner mismatch is therefore not silently interpreted as the same systemd epoch.

## Durable provenance

`NixSystemdJobEvidenceV1` now requires `manager_owner` whenever lifecycle job evidence exists.

`NixPostStateReceiptV1` carries the same value as `systemd_manager_owner`, and its deterministic receipt digest includes that value.

This makes the manager incarnation part of the durable provenance commitment rather than transient observer state.

The effect digest intentionally does **not** include the manager owner. The owner identifies the observed execution/observation epoch; it is not an authorization parameter. CROSS-060 remains responsible for authorization-bound service-definition identity.

## Fail-closed cases

The model rejects:

- empty or malformed manager owner;
- manager owner missing when Job evidence is present;
- manager owner present when no Job evidence exists;
- a Job handle with malformed manager owner;
- a `JobRemoved` message from another sender.

Owner changes are treated as a new observation epoch. They cannot be substituted with the Nixward daemon incarnation.

## Claim ceiling

This does not make D-Bus itself a cryptographic attestation channel. The bus's sender/header information is trusted according to the local message-bus security model, and the receipt still needs whatever external digest/signature mechanism is used to make durable evidence tamper-evident.

It also does not eliminate execution TOCTOU/CAS or prove that an unrelated actor could not alter service state between observations.

CROSS-062 remains responsible for replacing scalar stability counts with a verifiable repeated-observation sequence.
