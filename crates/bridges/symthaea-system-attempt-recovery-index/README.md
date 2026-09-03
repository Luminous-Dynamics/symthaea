# Symthaea System Attempt Recovery Index

Read-only crash-recovery discovery for `symthaea-system-attempt-evidence` v1.

This crate exists because durable evidence is only useful after a crash if a new
process can discover it without relying on an in-memory `AttemptEvidenceHandle`
or a caller-retained attempt key.

## Safety contract

The recovery index:

- opens the attempt SQLite database through a separate read-only connection;
- discovers all persisted attempt keys;
- reloads every record in each chain rather than trusting only the latest row;
- recomputes record commitments and validates sequence/predecessor links;
- requires the privacy-minimized `AttemptEvidenceContext` to remain identical
  throughout a chain;
- enforces the semantic v1 transition shape around `DispatchArmed`, terminal
  dispatch evidence, and `RecoveryCompleted`;
- can match a discovered attempt to a reservation already present in an
  authenticated Agency Kernel checkpoint using the v1 reservation-ID digest;
- treats multiple incomplete attempts for one reservation as containment.

The restart rule is intentionally monotone:

> A `Reserved` execution found after restart becomes `OutcomeUnknown` unless an
> independently trusted proof establishes that dispatch did not occur.

Local attempt evidence cannot by itself return grant capacity after a crash.
`ProvenNotDispatched` is therefore classified as a claim requiring independent
trust, not as permission to release the reservation.

## Why this asymmetry matters

The SQLite journal is hash-chained, but its current live head is cached in
process memory. After process loss, local storage alone is not an independent
anti-rollback anchor. If missing or rewritten local evidence could prove
"nothing happened", deletion of a `DispatchArmed` record could become a retry
oracle. The recovery index refuses that interpretation.

A missing attempt record therefore cannot increase authority. The durable
runtime reservation remains the conservative source of the charge.

## Non-claims

This crate does **not**:

- authenticate the current Agency Kernel `CheckpointHead`;
- prove the SQLite filesystem honored power-loss durability;
- sign or remotely anchor attempt-evidence heads;
- establish trustworthy wall-clock time;
- execute, retry, compensate, or roll back an external effect;
- mint, refresh, or reuse Xenia capability authority;
- release an uncertain reservation;
- prove that a locally stored `Applied` or `ProvenNotDispatched` claim is
  independently trustworthy.

Those are separate trust boundaries.

## Next composition

The intended crash-recovery coordinator is:

```text
external trusted checkpoint head
              |
              v
authenticate SQLite Agency Kernel frontier
              |
              v
read authenticated GrantAccountCheckpoint
              |
              +----> every stranded Reserved => OutcomeUnknown
              |
              v
read-only attempt recovery index
              |
              v
cross-check reservation / grant / plan / checkpoint lineage
              |
              v
CAS-persist conservative recovery normalization
              |
              v
independent effect observation
              |
              v
append recovery evidence
              |
              v
QUIESCENT / NO AUTHORITY
              |
              v
require a fresh Xenia capability for any new effect
```

A crash may reduce availability. It must not recreate authority, erase an
uncertain effect, or convert missing evidence into permission to retry.
