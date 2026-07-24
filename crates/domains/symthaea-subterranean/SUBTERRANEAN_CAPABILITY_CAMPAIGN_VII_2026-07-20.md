# Subterranean Capability Campaign VII

Date: 2026-07-20
Baseline: hardened v6 / patch 50
Theme: operator authority, safe updates, degraded operation, and restart integrity

## Campaign objective

Close the field-readiness gap between autonomous mission execution and external human/software authority without allowing remote commands, stale messages, update state, or restart state to weaken physical safety.

## Patch sets 51–63

### 51. Validated operator command trust boundary

Adds typed operator identities, roles, authentication assertions, command envelopes, epochs, sequences, proposal IDs, freshness, expiry, and minimum-authentication policy.

### 52. Replay-resistant authority and recovery quorum

Adds per-operator epoch/sequence memory, replay rejection, restrictive constraints, and two-person hardware-backed resume approval.

### 53. Safety-monotonic embodiment integration

Applies operator constraints before physical recovery planning. Hazards remain authoritative. Operator hold and return behavior is observable in fallback evidence.

### 54. Staged activation and rollback controller

Adds externally supplied artifact/configuration/rollback digests, safe staging preconditions, bounded health validation, explicit rollback, and update state evidence.

### 55. Pluggable operational integrity chain

Adds bounded previous-digest audit records and a digest-provider trait. Includes a deterministic non-cryptographic provider only for tests.

### 56. Degraded-operation watchdog supervisor

Adds operator-link grace, autonomous return, safe hold, watchdog/reboot/checkpoint recovery lock, and authorized healthy-dwell clearing.

### 57. Two-slot checkpoint recovery journal

Adds monotonic generations, alternating slots, digest validation, newest-valid selection, and fallback to the previous generation.

### 58. Live degraded authority integration

Makes degraded mode part of mission selection, safety floors, nominal command restriction, and fallback-stage reporting.

### 59. Operational checkpoint schema v2

Persists operator replay state, active constraints, degraded supervisor state, and update state. Supports migration from schema v1 defaults.

### 60. Authority evidence

Records operator constraints, accepted/rejected commands, proposal state, degraded mode, link-loss duration, and update lifecycle in each bounded evidence frame.

### 61. Authority acceptance gates

Adds deterministic release contracts for replay, quorum, hazard blocking, audit continuity, rollback, watchdog latching, and journal fallback.

### 62. Protocol and trust-boundary documentation

Documents authority ordering, external cryptographic responsibilities, recovery semantics, update non-claims, and checkpoint recovery.

### 63. Mechanical formatting normalization

Applies repository-wide rustfmt normalization without intentional behavior changes.

## Intended merge gates

```bash
cargo fmt --check -p symthaea-subterranean
cargo clippy -p symthaea-subterranean --all-targets -- -D warnings
cargo test -p symthaea-subterranean
```

The full workspace must additionally exercise real cryptographic adapters, transport replay behavior, secure update installation, hardware watchdogs, abrupt power-loss injection, and controlled-hardware timing.
