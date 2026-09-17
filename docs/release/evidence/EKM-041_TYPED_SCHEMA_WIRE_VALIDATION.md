# EKM-041 — Typed Schema-Wire Semantic Validation

## Purpose

EKM-040 proves bounded parsing and corruption detection for typed policy/decision sidecar bytes. EKM-041 independently checks the semantic invariants that can be established from that sidecar alone.

## Central invariant

**Checksum-valid typed schema bytes are still untrusted until receipt lineage, policy semantics, decision consistency, and policy-linked failure payloads are independently validated.**

## Validated invariants

The validator checks:

- supported wire version and encoding
- schema capture cycle equals linked EKM-031 capture cycle
- linked revision count equals actual record count
- next receipt ID equals `record_count + 1` with checked arithmetic
- contiguous receipt IDs from 1
- no receipt evaluation after capture
- finite bounded proposed deltas
- policy reconstruction succeeds
- uncertainty caps remain canonical and bounded
- supported decision snapshot version
- `eligible == failures.is_empty()`
- non-empty nested routing failures
- source/dimension routing failures describe an actually denied route
- positive/negative evidence failures match delta direction
- provenance-threshold failures match policy threshold and decision root count
- delta-limit failures match the exact policy maximum and `abs(delta)`
- calibration failures are compatible with calibration policy thresholds
- uncertainty failures are compatible with enabled uncertainty policy and configured caps
- contradiction failures require positive strengthening plus contradiction-blocking policy
- causal-intervention failures require positive strengthening plus the intervention policy

## Deliberate non-claims

This sidecar alone does not contain the base ledger, evidence basis, calibration snapshot, uncertainty assessment, or claim kind. Therefore EKM-041 does **not** claim to prove:

- that evidence polarity actually warranted the recorded result
- that a claim was causal
- that an intervention/replication was actually present or absent
- that calibration observations really had the recorded values
- that uncertainty freshness or values matched the live evidence epoch
- that provenance-root counts match the base ledger

Those are cross-component checks for a later restart-v2 bundle validator.

## Authority boundary

The validator returns only a report or typed error. It creates no persistence capsule, revision history, restart capsule/quarantine, authorization object, writable belief state, or activation handle. It performs no file/network I/O.

## Qualification boundary

Exact-head CI remains authoritative. Queued runs do not establish format, compile, Clippy, test, or runtime PASS.
