# ADR-027: RSK Xenia state-witness adapter

**Status**: proposed  
**Date**: 2026-09-13  
**Change Class**: A

## Context

RSK #2239 defines an external monotonic-observation boundary for the complete
schema-registry anti-rollback state. Xenia draft PR
`Luminous-Dynamics/xenia-peer#335` defines a generic independently witnessed
state commitment.

Both protocols are individually fail-closed, but a loose adapter could still
introduce ambiguity around target identity, epoch identity, sequence numbering,
predecessor history, trust context, or timestamp meaning.

In particular, RSK's first accepted registry snapshot is sequence 1 while
Xenia's generic commitment reserves counter 0 for genesis with an all-zero
predecessor.

## Decision

Adopt `RSK_XENIA_STATE_WITNESS_ADAPTER_V0_1.md` as the normative composition
contract before implementing runtime integration.

The adapter fixes:

- namespace `symthaea.rsk.schema-registry.v1`;
- deployment+registry-bound target identity;
- RSK-recovery-epoch-bound Xenia epoch identity;
- exact `xenia.counter == RSK highest_sequence` mapping;
- mandatory witnessed counter-zero empty-tracker genesis for each fresh epoch;
- exact raw-byte mapping of the RSK SHA-256 complete-tracker digest into Xenia's
  opaque state digest;
- exact predecessor commitment chaining using retained Xenia witness history;
- independently derived verified trust-context identity;
- Xenia timestamp as evidence metadata, never a substitute for RSK trusted time;
- exact local-state/witness equality after Xenia witness verification;
- fail-closed crash mismatch and governed fresh-epoch recovery.

## Authority boundary

A Xenia witness result is evidence continuity, not replication authority.

It cannot mint or widen grants, extend expiry, clear quarantine/revocation/fork
state, replace trusted time, or authorize recovery by itself.

RSK remains responsible for signer lifecycle, true failure-domain independence,
trusted time, local constitutional state, and every positive-authority decision.

## Consequences

Positive:

- cross-system semantics are fixed before code integration;
- sequence-1 bootstrap has no special-case predecessor bypass;
- deployment/registry identity cannot be confused across witness chains;
- missing witness history fails closed;
- trust/time responsibilities remain explicit.

Costs:

- every new RSK epoch requires a witnessed zero-counter genesis;
- predecessor witness evidence must be retained;
- trust-policy changes within an epoch require governed transition rather than
  silent reinterpretation;
- actual integration remains blocked on qualified Xenia/provider evidence.

## Evidence status

This ADR and adapter contract are design evidence only. Xenia #335 is currently a
draft and Symthaea/Xenia exact-head workflows have not yet established executed
qualification for this composition.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
