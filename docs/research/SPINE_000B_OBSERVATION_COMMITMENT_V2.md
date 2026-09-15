# SPINE-000B — Observation Commitment v2

**Status:** preregistered byte-level evidence contract; runtime instrumentation pending

**Authority:** measurement-only

**Issue:** #3381

C2 extends, but does not reinterpret, the frozen C1 v1 evidence protocol. C1 v1 remains historical. C2 exists because runtime qualification now depends on data that C1 v1 did not commit: `SubsystemOutput::_reserved`, guard-witness events, and observer-buffer completeness state.

## 1. Non-goals

C2 does not define subsystem truth, usefulness, causal load, action authority, epistemic authority, counterfactual execution, wall-clock timing identity, or production admission policy.

## 2. Primitive encoding

C2 inherits C1 primitive rules:

```text
u8   = 1 byte
u16  = 2 bytes little-endian
u32  = 4 bytes little-endian
u64  = 8 bytes little-endian
bool = u8 0x00 false / 0x01 true only
sha256 = exactly 32 raw bytes
```

Canonical strings use the same C1 ASCII/path rules. Unknown tags, malformed booleans, count overflow, duplicate identities/indices, or noncanonical strings fail closed.

## 3. Domain separation

```text
EXECUTION_V2_DOMAIN = "symthaea.spine.000b.execution-receipt.v2\0"
GUARD_EVENT_DOMAIN  = "symthaea.spine.000b.guard-witness-event.v1\0"
GUARD_BUNDLE_DOMAIN = "symthaea.spine.000b.guard-witness-bundle.v1\0"
OBS_CYCLE_DOMAIN    = "symthaea.spine.000b.observation-cycle.v1\0"
OBS_GENESIS_DOMAIN  = "symthaea.spine.000b.observation-genesis.v1\0"
OBS_CHAIN_DOMAIN    = "symthaea.spine.000b.observation-chain-link.v1\0"
```

C1 v1 chain roots and C2 observation roots are different lineages and must never be mixed.

## 4. ProposalBitsV2

C1 v1 proposal bytes are retained in the same order and extended by the real production reserved field:

```text
confidence_delta_bits  u64
lr_modulation_bits     u64
exploration_delta_bits u64
arousal_delta_bits     u32
valence_delta_bits     u32
flags                  u32
reserved               u32
```

`reserved` is exact emitted `SubsystemOutput::_reserved`. It is not padding and is never normalized.

This is required because current production `SubsystemOutput::is_neutral()` ignores `_reserved`, while N1 qualification requires `_reserved == 0`.

## 5. SubsystemExecutionReceiptV2

ExecutionOutcome and Urgency wire tags remain exactly those frozen by C1 v1.

Canonical bytes after `EXECUTION_V2_DOMAIN`:

```text
cycle_number          u64
subsystem_identity    canonical string <=128
source_path           canonical repo-relative path
schedule_interval     u32
urgency               u8
eligible_to_run       bool
execution_outcome     u8
emitted               bool
admitted              bool
proposal              Optional<ProposalBitsV2>
overlap_ref_count     u16
overlap_refs[]         canonical strings, bytewise sorted, unique
```

Current production truth table:

```text
SKIPPED_SCHEDULE         emitted=false admitted=false proposal=None
SKIPPED_HEALTH_DISABLED emitted=false admitted=false proposal=None
PANICKED_CAUGHT          emitted=false admitted=false proposal=None
FAILED_OTHER             emitted=false admitted=false proposal=None
EXECUTED_NEUTRAL         emitted=true  admitted=false proposal=Some(exact output)
EXECUTED_NON_NEUTRAL     emitted=true  admitted=true  proposal=Some(exact output)
```

`EXECUTED_NEUTRAL` is defined by current production neutrality semantics. Therefore a reserved-only nonzero output may be `EXECUTED_NEUTRAL` while N1 later marks it outside the qualified proposal domain. C2 preserves that discrepancy rather than rewriting production behavior.

Digest:

```text
SHA256(EXECUTION_V2_DOMAIN || execution_v2_bytes)
```

## 6. GuardWitnessEventV1

Canonical guard events represent only predicates that production actually evaluated.

Canonical bytes after `GUARD_EVENT_DOMAIN`:

```text
cycle_number   u64
witness_index  u16
predicate_id   u16
outcome        bool
```

`outcome=false` means an evaluated predicate was false. `outcome=true` means an evaluated predicate was true.

There is deliberately no canonical `NOT_EVALUATED` event. G1 derives `NOT_EVALUATED` or `UNRESOLVED` from the frozen predicate DAG, source projection, actual witness set, and observer-completeness state.

For v1:

- witness indices are contiguous from zero within a cycle;
- predicate IDs are from the exact G1 overlay;
- a predicate ID appears at most once per cycle;
- witness order preserves production evaluation order.

Digest:

```text
SHA256(GUARD_EVENT_DOMAIN || guard_event_bytes)
```

## 7. GuardWitnessBundleV1

Canonical bytes after `GUARD_BUNDLE_DOMAIN`:

```text
cycle_number             u64
guard_observer_overflow  bool
witness_count            u16
witness_entries[]        witness_index u16 || event_digest[32]
```

Entries are in contiguous witness-index order. Duplicate predicate IDs, index gaps/duplicates, mismatched cycle numbers, unknown predicate IDs, or witness counts outside the frozen observer capacity fail closed.

Digest:

```text
SHA256(GUARD_BUNDLE_DOMAIN || guard_bundle_bytes)
```

## 8. ObservationCycleV1

C2 binds the full observational surface needed by later A2 interpretation.

Canonical bytes after `OBS_CYCLE_DOMAIN`:

```text
cycle_number                    u64
execution_count                 u32
execution_entries[]             subsystem_identity || execution_v2_digest[32]
integration_digest              sha256
application_count               u32
application_entries[]           application_index u32 || application_digest[32]
guard_bundle_digest             sha256
manager_observer_overflow       bool
application_observer_overflow   bool
guard_observer_overflow         bool
observer_buffers_complete       bool
```

Rules:

- execution identities are bytewise sorted and unique;
- execution child cycle numbers match the envelope;
- integration digest is the qualified C1 CycleIntegrationReceipt digest;
- application indices are contiguous from zero and preserve production order;
- application digest is the qualified C1 StateApplicationReceipt digest;
- guard bundle cycle number matches the envelope;
- `guard_observer_overflow` must equal the value committed inside GuardWitnessBundleV1;
- `observer_buffers_complete` is true iff all three overflow booleans are false.

Overflow means evidence incompleteness, not cognitive failure. Production cognition continues; the cycle is simply ineligible for qualified runtime SPINE evidence.

Digest:

```text
SHA256(OBS_CYCLE_DOMAIN || observation_cycle_bytes)
```

## 9. Observation evidence chain

Given an externally qualified 32-byte subject-manifest digest:

```text
root_0 = SHA256(OBS_GENESIS_DOMAIN || subject_manifest_digest)
root_n = SHA256(OBS_CHAIN_DOMAIN || root_(n-1) || observation_cycle_digest_n)
```

Campaign order is frozen externally. Reordering or replaying cycles changes the root.

## 10. Runtime timing remains noncanonical

Wall-clock timestamps, elapsed durations, host/runner IDs, PID/TID, profiler data, serialization time, and observer-overhead measurements remain outside canonical C2 identity.

They may be reported alongside evidence for I1 but cannot change replay identity.

## 11. Required controls

Independent implementations must establish at least:

1. exact neutral vs reserved-only nonzero -> different ExecutionReceiptV2 bytes/digests;
2. one reserved bit mutation -> different execution digest;
3. C1 v1 historical proposal bytes remain unchanged by C2 tooling;
4. guard TRUE vs FALSE -> different event digest;
5. missing witness is not encoded as FALSE;
6. witness reorder/index mutation -> digest change or validation reject;
7. duplicate predicate ID -> reject;
8. unknown predicate ID -> reject;
9. any observer overflow bit -> observation digest changes and `observer_buffers_complete=false`;
10. inconsistent guard overflow between bundle/envelope -> reject;
11. changing any execution/integration/application/guard child digest -> observation digest changes;
12. execution input reorder -> canonical observation digest unchanged after identity sort;
13. application reordering -> digest changes or validation rejects;
14. chain position/replay change -> different root;
15. exact byte vectors reproduced independently in Python and Rust before runtime use.

## 12. Qualification boundary

A C2 format/oracle PASS establishes deterministic evidence identity for synthetic records only.

Full C2 qualification requires an independently implemented Rust encoder reproducing exact frozen bytes and digests. Until then report:

```text
FORMAT_FROZEN / RUST_EQUIVALENCE_PENDING
```

Even a full C2 PASS establishes no runtime observer completeness, guard truth beyond captured evaluated witnesses, counterfactual execution, subsystem causality, benefit, or authority.
