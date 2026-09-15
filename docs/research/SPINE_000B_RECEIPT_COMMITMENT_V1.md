# SPINE-000B — Canonical Runtime Receipt Commitment v1

**Status:** preregistered byte-level evidence-identity contract; runtime implementation pending

**Authority:** measurement-only

**Issue:** #3255

This contract freezes how canonical SPINE-000B runtime evidence is converted to bytes and committed with SHA-256. It is intentionally defined before production runtime receipts exist.

A commitment proves identity of a measurement record. It does **not** grant cognitive, causal, epistemic, safety, or action authority.

## 1. Non-goals

This format does not define:

- whether a subsystem is load-bearing;
- whether a proposal is true or useful;
- whether an action is authorized;
- wall-clock performance telemetry;
- JSON serialization;
- Rust memory layout or enum discriminants.

## 2. Canonical primitive encoding

All multi-byte integers use **little-endian** fixed-width encoding.

```text
u8   = 1 byte
u16  = 2 bytes LE
u32  = 4 bytes LE
u64  = 8 bytes LE
bool = u8 0x00 false / 0x01 true only
f32  = IEEE-754 to_bits() encoded as u32 LE
f64  = IEEE-754 to_bits() encoded as u64 LE
sha256 = exactly 32 raw bytes
```

No `usize`, decimal float rendering, Rust `Debug`, JSON object order, host endianness, or platform pointer width enters canonical bytes.

Signed zero and distinct NaN payloads are distinct at this encoding layer because exact IEEE payload bits are committed. Qualification policy may separately reject non-finite values.

### Optional values

```text
None    = 0x00
Some(x) = 0x01 || encode(x)
```

Any other presence tag is invalid.

### Canonical strings

A canonical string is:

```text
u16 byte_length || exact ASCII bytes
```

General rules:

- length 1..=512 unless a narrower field limit is specified;
- ASCII only;
- no NUL/control characters;
- no leading/trailing whitespace;
- no Unicode normalization step exists because non-ASCII is rejected.

`subsystem_identity` is limited to 1..=128 bytes and `[A-Za-z0-9_.:-]+`.

`source_path` is a repo-relative POSIX path limited to 1..=512 bytes using `[A-Za-z0-9_./-]+`; it must not start with `/`, contain `\\`, contain an empty path segment, or contain `.` / `..` segments.

`destination_id` is limited to 1..=128 bytes and `[A-Za-z0-9_.:-]+`.

## 3. Domain separation

Each digest is SHA-256 over a NUL-terminated ASCII domain tag followed by canonical bytes.

```text
EXECUTION_DOMAIN  = "symthaea.spine.000b.execution-receipt.v1\0"
INTEGRATION_DOMAIN = "symthaea.spine.000b.cycle-integration.v1\0"
APPLICATION_DOMAIN = "symthaea.spine.000b.state-application.v1\0"
CYCLE_DOMAIN       = "symthaea.spine.000b.cycle-evidence.v1\0"
GENESIS_DOMAIN     = "symthaea.spine.000b.evidence-genesis.v1\0"
CHAIN_DOMAIN       = "symthaea.spine.000b.evidence-chain-link.v1\0"
```

Changing any domain/version creates a new commitment lineage.

## 4. Frozen wire tags

Wire tags are protocol constants and MUST NOT be inferred from Rust enum discriminants.

### ExecutionOutcome

```text
0 = SKIPPED_SCHEDULE
1 = SKIPPED_HEALTH_DISABLED
2 = EXECUTED_NEUTRAL
3 = EXECUTED_NON_NEUTRAL
4 = PANICKED_CAUGHT
5 = FAILED_OTHER
```

### Urgency

```text
0 = CRUISE
1 = NORMAL
2 = CRITICAL
```

### Application source

```text
0 = CONFIDENCE_DELTA
1 = LR_MODULATION
2 = EXPLORATION_DELTA
3 = AROUSAL_DELTA
4 = VALENCE_DELTA
5 = FLAG
```

### StateChangeStatus

```text
0 = UNCHANGED
1 = CHANGED
2 = NOT_OBSERVED_AT_BOUNDARY
```

### CanonicalValue

```text
1 = F64_BITS  || u64
2 = F32_BITS  || u32
3 = U64       || u64
4 = U32       || u32
5 = BOOL      || u8(0|1)
6 = DIGEST32  || 32 bytes
```

Unknown tags fail closed.

## 5. ProposalBits

A canonical proposal is always encoded in this exact order:

```text
confidence_delta_bits  u64
lr_modulation_bits     u64
exploration_delta_bits u64
arousal_delta_bits     u32
valence_delta_bits     u32
flags                  u32
```

ABI padding/reserved fields are excluded because they have no proposal semantics.

## 6. IntegratedBits

A canonical integrated result is:

```text
confidence_delta_bits  u64
lr_modulation_bits     u64
exploration_delta_bits u64
arousal_delta_bits     u32
valence_delta_bits     u32
flags                  u32
n_contributors         u32
```

`usize` contributor counts must be range-checked before conversion to `u32`.

## 7. SubsystemExecutionReceipt commitment

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
overlap_refs[]         canonical identity-like string, sorted bytewise, unique
```

Rules:

- `SKIPPED_SCHEDULE`, `SKIPPED_HEALTH_DISABLED`, `PANICKED_CAUGHT`, and `FAILED_OTHER` have `emitted=false`, `admitted=false`, `proposal=None`.
- `EXECUTED_NEUTRAL` has `emitted=true`, `admitted=false`, `proposal=Some(exact neutral proposal)`.
- `EXECUTED_NON_NEUTRAL` has `emitted=true`, `proposal=Some(...)`; production admission determines `admitted`.
- overlap references are semantic metadata and therefore canonical; nondeterministic timing is not.

Digest:

```text
execution_digest = SHA256(EXECUTION_DOMAIN || execution_bytes)
```

## 8. CycleIntegrationReceipt commitment

There is exactly one integration receipt per instrumented cycle, including an empty collector.

Canonical bytes after `INTEGRATION_DOMAIN`:

```text
cycle_number          u64
admitted_count        u32
integrated_all        IntegratedBits
subject_count         u32
subjects[]            sorted bytewise by subsystem_identity
```

Each subject entry:

```text
subsystem_identity           canonical string <=128
integrated_without_subject   IntegratedBits
changed_channel_mask         u8
uniquely_contributed_flags   u32
integration_changed          bool
```

`changed_channel_mask` bits:

```text
bit0 confidence_delta
bit1 lr_modulation
bit2 exploration_delta
bit3 arousal_delta
bit4 valence_delta
bits5..7 MUST be zero
```

Contributor count is metadata and is deliberately **not** an influence channel. `integration_changed` is true iff scalar mask != 0 or `uniquely_contributed_flags != 0`.

Every admitted subsystem appears exactly once; duplicate identities fail closed.

Digest:

```text
integration_digest = SHA256(INTEGRATION_DOMAIN || integration_bytes)
```

## 9. StateApplicationReceipt commitment

State application is cycle-level evidence, not per-subsystem causal attribution.

Canonical bytes after `APPLICATION_DOMAIN`:

```text
cycle_number          u64
application_index     u32
destination_id        canonical string <=128
source_tag            u8
source_flag           u32   # zero unless source_tag=FLAG
applied                bool
state_change_status    u8
before                 Optional<CanonicalValue>
after                  Optional<CanonicalValue>
```

Rules:

- `application_index` is assigned at the production application boundary and defines semantic order.
- indices within one cycle must be unique and contiguous from 0.
- `source_flag` must be zero for scalar source tags.
- `UNCHANGED` / `CHANGED` require compatible `before` and `after` values.
- `NOT_OBSERVED_AT_BOUNDARY` may use `None` values for side effects that cannot be represented as a stable scalar/digest at that boundary.
- `applied=true` does not imply `state_change_status=CHANGED`.

Digest:

```text
application_digest = SHA256(APPLICATION_DOMAIN || application_bytes)
```

## 10. Cycle evidence commitment

Execution records are a set keyed by subsystem identity, so their digests are sorted by canonical identity before aggregation. State applications are semantically ordered and therefore retain `application_index` order.

Canonical bytes after `CYCLE_DOMAIN`:

```text
cycle_number             u64
execution_count          u32
execution_entries[]      subsystem_identity || execution_digest
integration_digest       32 bytes
application_count        u32
application_entries[]    application_index u32 || application_digest
```

Requirements:

- execution identities unique and bytewise sorted;
- application indices contiguous from 0;
- integration receipt cycle number matches the envelope;
- every child record cycle number matches the envelope.

Digest:

```text
cycle_digest = SHA256(CYCLE_DOMAIN || cycle_bytes)
```

Reordering input execution records before canonical sorting MUST NOT change `cycle_digest`. Reordering application indices MUST change it or fail validation.

## 11. Append-only evidence chain

Given an externally qualified 32-byte `subject_manifest_digest`:

```text
root_0 = SHA256(GENESIS_DOMAIN || subject_manifest_digest)
root_n = SHA256(CHAIN_DOMAIN || root_(n-1) || cycle_digest_n)
```

Cycles must be appended in strictly increasing campaign order as defined by the evidence runner. Replaying a cycle at another position changes the chain root.

This chain is evidence integrity only.

## 12. RuntimeTelemetryEnvelope is noncanonical

The following are explicitly excluded from all canonical receipt/cycle digests:

```text
duration_ns
serialization_duration_ns
host name / runner id
thread id
wall-clock timestamp
process id
allocator diagnostics
profiling samples
```

They may be stored alongside canonical evidence but must never affect replay identity.

## 13. Validation is fail-closed

Reject rather than normalize:

- unknown enum/value tags;
- invalid boolean bytes;
- duplicate subsystem identities;
- duplicate/noncontiguous application indices;
- count overflow;
- invalid/noncanonical strings or paths;
- changed-channel mask reserved bits;
- scalar source with nonzero `source_flag`;
- record/envelope cycle mismatch;
- malformed optional/value tags.

Do not silently trim strings, normalize paths, reorder state applications, coerce values, or replace invalid fields with defaults.

## 14. Required golden-vector controls

Before runtime evidence uses this format, independent implementations must establish at least:

1. same semantic record with differently ordered/formatted JSON -> same digest;
2. one IEEE payload bit changed -> different digest;
3. outcome tag changed -> different digest;
4. subsystem identity changed -> different digest;
5. `duration_ns` changed -> canonical digest unchanged;
6. execution input order changed -> cycle digest unchanged after canonical sort;
7. state-application order/index changed -> digest changes or validation rejects;
8. unknown tag rejected;
9. invalid/overlong identity/path rejected;
10. count overflow rejected;
11. genesis/chain vector reproduced independently;
12. exact byte vector reproduced independently, not only final SHA-256.

## 15. Qualification boundary

The contract/oracle tranche can establish only:

> The SPINE-000B receipt identity protocol is frozen and an independent oracle deterministically maps synthetic records to canonical bytes and SHA-256 commitments.

Full C1 qualification additionally requires an independent Rust implementation to reproduce the frozen canonical byte vectors and digests exactly. Until then, report `FORMAT_FROZEN / RUST_EQUIVALENCE_PENDING`.
