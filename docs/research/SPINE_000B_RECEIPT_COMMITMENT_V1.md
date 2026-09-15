# SPINE-000B — Canonical Runtime Receipt Commitment v1

**Status:** preregistered byte-level evidence-identity contract; runtime instrumentation pending

**Authority:** measurement-only

**Issues:** #3255, #3261

This contract freezes how canonical SPINE-000B runtime evidence is converted to bytes and committed with SHA-256 before production runtime receipts exist. A commitment proves identity of a measurement record; it grants no cognitive, causal, epistemic, safety, or action authority.

## 1. Non-goals

This protocol does not define whether a subsystem is load-bearing, whether a proposal is true/useful, whether an action is authorized, wall-clock performance telemetry, JSON serialization, Rust memory layout, or Rust enum discriminants.

## 2. Canonical primitive encoding

All multi-byte integers use **little-endian** fixed-width encoding:

```text
u8   = 1 byte
u16  = 2 bytes LE
u32  = 4 bytes LE
u64  = 8 bytes LE
bool = u8 0x00 false / 0x01 true only
f32  = IEEE-754 to_bits() as u32 LE
f64  = IEEE-754 to_bits() as u64 LE
sha256 = exactly 32 raw bytes
```

No `usize`, decimal float rendering, Rust `Debug`, JSON object order, host endianness, or pointer width enters canonical bytes. Signed zero and distinct NaN payloads remain distinct at this byte layer; qualification policy may separately reject non-finite values.

Optional values are `0x00` for None and `0x01 || encode(x)` for Some(x). Unknown presence tags fail closed.

Canonical strings are `u16 byte_length || exact ASCII bytes`. They are never trimmed or Unicode-normalized. `subsystem_identity`, `operation_id`, and `destination_id` use `[A-Za-z0-9_.:-]+` and are 1..=128 bytes. `source_path` is a 1..=512-byte repo-relative POSIX path using `[A-Za-z0-9_./-]+`; absolute paths, backslashes, empty segments, `.` and `..` segments are rejected.

## 3. Domain separation

Each digest is SHA-256 over a NUL-terminated ASCII domain followed by canonical bytes:

```text
EXECUTION_DOMAIN   = "symthaea.spine.000b.execution-receipt.v1\0"
INTEGRATION_DOMAIN = "symthaea.spine.000b.cycle-integration.v1\0"
APPLICATION_DOMAIN = "symthaea.spine.000b.state-application.v1\0"
CYCLE_DOMAIN       = "symthaea.spine.000b.cycle-evidence.v1\0"
GENESIS_DOMAIN     = "symthaea.spine.000b.evidence-genesis.v1\0"
CHAIN_DOMAIN       = "symthaea.spine.000b.evidence-chain-link.v1\0"
```

Changing a domain/version starts a new commitment lineage.

## 4. Frozen wire tags

Wire tags are protocol constants, never inferred from Rust discriminants.

### ExecutionOutcome

```text
0 SKIPPED_SCHEDULE
1 SKIPPED_HEALTH_DISABLED
2 EXECUTED_NEUTRAL
3 EXECUTED_NON_NEUTRAL
4 PANICKED_CAUGHT
5 FAILED_OTHER
```

### Urgency

```text
0 CRUISE
1 NORMAL
2 CRITICAL
```

### Application source

```text
0 CONFIDENCE_DELTA
1 LR_MODULATION
2 EXPLORATION_DELTA
3 AROUSAL_DELTA
4 VALENCE_DELTA
5 FLAG
```

### Application source condition

This tag is required because a Phase-C operation may be caused by a non-identity scalar, by a flag being set, or by a flag being clear.

```text
0 SCALAR_NON_IDENTITY
1 FLAG_SET
2 FLAG_CLEAR
```

For scalar source tags, `source_flag` MUST equal zero and `source_condition` MUST be `SCALAR_NON_IDENTITY`. For `FLAG`, `source_flag` MUST be nonzero and `source_condition` MUST be `FLAG_SET` or `FLAG_CLEAR`. This makes the `REQUEST_GEODESIC` clear path representable without pretending the flag was set.

### StateChangeStatus

```text
0 UNCHANGED
1 CHANGED
2 NOT_OBSERVED_AT_BOUNDARY
```

### CanonicalValue

```text
1 F64_BITS || u64
2 F32_BITS || u32
3 U64      || u64
4 U32      || u32
5 BOOL     || u8(0|1)
6 DIGEST32 || 32 raw bytes
```

Unknown tags fail closed.

## 5. ProposalBits and IntegratedBits

Proposal bytes are always:

```text
confidence_delta_bits  u64
lr_modulation_bits     u64
exploration_delta_bits u64
arousal_delta_bits     u32
valence_delta_bits     u32
flags                  u32
```

ABI padding/reserved fields are excluded. `IntegratedBits` appends `n_contributors u32`; any production `usize` count is range-checked before conversion.

## 6. SubsystemExecutionReceipt

Canonical bytes after `EXECUTION_DOMAIN`:

```text
cycle_number          u64
subsystem_identity    canonical string <=128
source_path           canonical source_path
schedule_interval     u32
urgency               u8
eligible_to_run       bool
execution_outcome     u8
emitted               bool
admitted              bool
proposal              Optional<ProposalBits>
overlap_ref_count     u16
overlap_refs[]         canonical strings, bytewise sorted, unique
```

Truth table:

```text
SKIPPED_SCHEDULE         emitted=false admitted=false proposal=None
SKIPPED_HEALTH_DISABLED emitted=false admitted=false proposal=None
PANICKED_CAUGHT          emitted=false admitted=false proposal=None
FAILED_OTHER             emitted=false admitted=false proposal=None
EXECUTED_NEUTRAL         emitted=true  admitted=false proposal=Some(exact neutral)
EXECUTED_NON_NEUTRAL     emitted=true  proposal=Some(exact output)
```

Digest: `SHA256(EXECUTION_DOMAIN || execution_bytes)`.

## 7. CycleIntegrationReceipt

Exactly one integration receipt exists per instrumented cycle, including an empty collector:

```text
cycle_number          u64
admitted_count        u32
integrated_all        IntegratedBits
subject_count         u32
subjects[]            bytewise sorted by subsystem_identity
```

Each admitted subject entry:

```text
subsystem_identity           canonical string <=128
integrated_without_subject   IntegratedBits
changed_channel_mask         u8
uniquely_contributed_flags   u32
integration_changed          bool
```

Mask bits 0..4 are confidence, LR, exploration, arousal, valence; bits 5..7 MUST be zero. Contributor count is metadata and deliberately **not** an influence channel. `integration_changed` is true iff the scalar mask is nonzero or unique flags are nonzero. Every admitted identity appears exactly once.

Digest: `SHA256(INTEGRATION_DOMAIN || integration_bytes)`.

## 8. StateApplicationReceipt

State application is **cycle-level evidence, not per-subsystem causal attribution**. The record binds the production operation, the observed destination, the upstream source/condition, and the **actual operand** that crossed the application boundary. The applied operand matters because current Phase C casts several integrated `f64` values to `f32` before invoking feedback helpers.

Canonical bytes after `APPLICATION_DOMAIN`:

```text
cycle_number          u64
application_index     u32
operation_id          canonical string <=128
destination_id        canonical string <=128
source_tag            u8
source_flag           u32
source_condition      u8
applied               bool
applied_argument      Optional<CanonicalValue>
state_change_status   u8
before                Optional<CanonicalValue>
after                 Optional<CanonicalValue>
```

Rules:

- `application_index` is assigned at the production application boundary and defines semantic order; indices in one cycle are unique and contiguous from 0.
- `operation_id` names the production operation separately from the observed destination.
- `source_condition` distinguishes non-identity scalar application, set-flag application, and clear-flag application.
- `applied_argument` is the exact post-cast/post-constant operand given to the operation. It is None only when no stable scalar/digest operand exists at that boundary.
- current helper-backed confidence/LR/exploration paths use `F32_BITS` applied arguments because Phase C casts their integrated `f64` to `f32`.
- `UNCHANGED`/`CHANGED` require compatible before/after value kinds.
- `NOT_OBSERVED_AT_BOUNDARY` may use None before/after for complex/external side effects.
- `applied=true` does not imply state changed.
- before/after are observed around the real operation, never recomputed from the argument.

Digest: `SHA256(APPLICATION_DOMAIN || application_bytes)`.

## 9. Cycle evidence commitment

Execution records form a set keyed by subsystem identity and are sorted by identity before aggregation. Applications are semantically ordered and retain `application_index` order.

Canonical bytes after `CYCLE_DOMAIN`:

```text
cycle_number             u64
execution_count          u32
execution_entries[]      subsystem_identity || execution_digest
integration_digest       32 bytes
application_count        u32
application_entries[]    application_index u32 || application_digest
```

All child cycle numbers must match the envelope. Execution identities must be unique. Application indices must be contiguous from zero. Reordering input execution records before canonical sorting MUST NOT change the cycle digest; changing semantic application indices MUST change it or fail validation.

Digest: `SHA256(CYCLE_DOMAIN || cycle_bytes)`.

## 10. Append-only evidence chain

Given an externally qualified 32-byte `subject_manifest_digest`:

```text
root_0 = SHA256(GENESIS_DOMAIN || subject_manifest_digest)
root_n = SHA256(CHAIN_DOMAIN || root_(n-1) || cycle_digest_n)
```

The evidence runner appends cycles in its frozen campaign order. Replaying a cycle in another position changes the root. This is evidence integrity only.

## 11. RuntimeTelemetryEnvelope is noncanonical

`duration_ns`, serialization duration, hostname/runner ID, thread ID, wall-clock timestamp, PID, allocator diagnostics, and profiler samples are excluded from every canonical digest. They may be stored alongside evidence but cannot affect replay identity.

## 12. Validation is fail-closed

Reject unknown tags, invalid booleans, duplicate subsystem identities, duplicate/noncontiguous application indices, count overflow, noncanonical strings/paths, reserved changed-channel bits, scalar/flag source-condition mismatches, cycle-number mismatches, malformed optional/value tags, and invalid flag-zero combinations. Never silently trim, normalize, reorder state applications, coerce values, or insert defaults.

## 13. Required golden-vector controls

Independent implementations must establish at least:

1. different JSON presentation of one semantic record -> same canonical bytes/digest;
2. one IEEE payload bit change -> different digest;
3. execution outcome change -> different digest;
4. subsystem identity change -> different digest;
5. runtime `duration_ns` change -> canonical digest unchanged;
6. execution input reorder -> cycle digest unchanged after canonical sort;
7. application semantic index/order change -> digest changes or validation rejects;
8. `operation_id` change -> application digest changes;
9. one bit of `applied_argument` -> application digest changes;
10. `FLAG_SET` vs `FLAG_CLEAR` -> application digest changes;
11. scalar/flag source-condition mismatch -> reject;
12. unknown tag -> reject;
13. invalid/overlong identity/path -> reject;
14. count overflow -> reject;
15. genesis/chain vectors reproduced independently;
16. exact byte vectors reproduced independently, not merely final SHA-256.

## 14. Qualification boundary

The format/oracle layer may establish only that the SPINE-000B receipt identity protocol is frozen and deterministically maps synthetic records to canonical bytes and SHA-256 commitments. Full C1 qualification additionally requires an independently implemented Rust encoder to reproduce the frozen byte vectors and digests exactly. Until an exact-head run passes both implementations, report `FORMAT_FROZEN / RUST_EQUIVALENCE_PENDING`. Runtime influence and causal load remain unclaimed.
