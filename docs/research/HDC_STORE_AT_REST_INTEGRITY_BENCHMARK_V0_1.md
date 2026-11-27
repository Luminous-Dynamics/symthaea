# HdcStore At-Rest Integrity Benchmark v0.1

**Status:** executable baseline seed; not production qualification or a standards claim.

**Tracking issue:** [#13829](https://github.com/Luminous-Dynamics/symthaea/issues/13829).

## Why this test exists

The 1 November 2026 individual Internet-Draft draft-khandelwal-bpm-agent-memory-integrity-00 defines ten controlled at-rest edits (T1–T10), three benchmark-validity controls (C1–C3), three verdicts (REJECTED, REPORTED, ACCEPTED), and a distinction between read-time and audit-time detection. It is a current research method, not an IETF standard or an endorsed IETF document.

Symthaea's symthaea-hdc-store is a concrete SUT: an mmap-backed persistent format with two independently CRC64-checksummed headers, local generations, tombstones, recovery semantics, and an LSH snapshot sidecar.

The store's CONTENT_CHECKSUMS.md explicitly limits its CRC64 checksum to accidental-corruption detection. It is not a cryptographic authenticity primitive. The benchmark tests the rollback boundary rather than interpreting "checksum exists" as malicious-integrity protection.

## SUT boundary and configuration

- SUT: symthaea-hdc-store 0.2.0 at the tested source commit.
- Store API: HdcStore::append, HdcStore::delete, HdcStoreReader::open, HdcStoreReader::get.
- Seed: five distinct BinaryHV records with IDs 11, 22, 33, 44, 55, written only through the SUT.
- Two isolated contexts: two independent HdcStore files with different seeded values (the file path is the context boundary).
- Each case starts from a newly seeded file and applicable data-bearing sidecars.
- The SUT is closed before every storage-layer edit.
- Read verdicts are obtained only through HdcStoreReader. HdcStore::inspect is an independent operator audit and is not conflated with a read-time verdict.
- T9/T10 snapshot and restore the canonical store file plus optional .lsh and .txn sidecars. The .lock sidecar is intentionally preserved, not replaced: it is the live path-stable synchronization inode, not memory state. This boundary must be stated in every report.

## Controls

C1 no-op reload: reopen and serve all five records unchanged with no integrity signal.

C2 genuine post-restart write: reopen, append one new record through the SUT API, close, reopen and serve it.

C3 edit landed: verify raw bytes before and after every edit. For one-byte and payload edits, verify only the intended range changed. For T5, verify the new valid-format record and updated checksummed headers. For T9/T10, verify the restored data and LSH files match the complete pre-edit snapshot exactly.

A failed edit is an error and is excluded from the denominator; it is never counted as pass or fail.

## Current mapping to T1–T10

| Draft case | HdcStore mapping |
|---|---|
| T1 content tamper | Flip one byte in a committed BinaryHV payload, preserving ID, status, and header |
| T2 tail truncation | Change final live marker to tombstone while leaving header untouched |
| T3 middle deletion | Change middle live marker to tombstone while leaving header untouched |
| T4 reordering | Not applicable: HdcStore exposes an unordered ID-keyed vector set, not an ordered history with semantic order keys |
| T5 forged insertion | Insert a valid-format entry and recompute local CRC64 header pages, without an SUT write |
| T6 cross-context replay | Copy a genuine donor payload from context A's store into a victim slot in context B; preserve B's local ID |
| T7 rollback replay | Copy an older genuine payload into the newest record's payload slot; preserve the latest ID and metadata |
| T8 metadata tamper | Not applicable to normative T8: entries have no owner, role, source, or timestamp fields. Numeric ID remapping is a separate extension |
| T9 whole-store rollback | Snapshot data + LSH + transaction sidecars, append one genuine record, then restore the pre-write family |
| T10 erasure resurrection | Snapshot data + sidecars, delete a record through HdcStore::delete, then restore the pre-erasure family |

The benchmark deliberately excludes two normative test families from the denominator rather than claiming coverage it cannot provide. A later adapter can add owner/role/source/timestamp-bearing memory records to exercise full T8.

## Verdict semantics

- REJECTED: read path refuses to open or serve the affected memory.
- REPORTED: read path serves memory but raises a documented integrity signal before or during serving.
- ACCEPTED: read path serves edited or restored memory with no signal.

Audit-only diagnostics are separate and are not promoted to read-time REPORTED. The current test emits one machine-readable JSON line per case and a campaign summary. A published result set should bind the source head, toolchain, benchmark test digest, canonical seed, SUT package version, backup file-set digest, and raw log digest.

## Why T9 and T10 matter

T9 restores a store older than a genuine write. T10 restores a store older than a genuine erasure. A high-water mark or monotonic counter outside the store can detect T9, while T10 requires a commitment to the current state or a durable deletion accumulator. A generation counter kept beside the data, or a CRC stored in the same file, is rolled back together with the data.

The next architectural experiment is an external monotonic witness:
- canonical commitment over store identity, schema, live record IDs/content, tombstone/deletion epoch, and generation;
- authenticated by a key outside the store adversary's reach;
- persisted in an independent witness or hardware-backed monotonic service;
- verified before any memory is exposed to a cognitive loop;
- unknown witness result or mismatch means quarantine and reconciliation, never silent fallback to old data.

A benchmark-only in-memory witness can demonstrate the distinction but does not prove deployment security. A production profile requires an independently administered anchor, key lifecycle, witness-availability policy, and replay-resistant evidence.

## Reproduction

    cargo test --locked -p symthaea-hdc-store --test at_rest_integrity -- --nocapture

The focused GitHub Actions workflow checks out the exact triggering revision and preserves the raw report together with source SHA, Rust toolchain, and test digest.

## References

- [Current Internet-Draft: draft-khandelwal-bpm-agent-memory-integrity-00](https://datatracker.ietf.org/doc/draft-khandelwal-bpm-agent-memory-integrity/) (published 1 November 2026; informational, individual draft).
- [IETF draft HTML](https://datatracker.ietf.org/doc/html/draft-khandelwal-bpm-agent-memory-integrity).
- [HdcStore content checksum limitations](../../crates/domains/symthaea-hdc-store/docs/CONTENT_CHECKSUMS.md).
- [HdcStore recovery contract](../../crates/domains/symthaea-hdc-store/docs/RECOVERY.md).
- [HdcStore crash boundary testing](../../crates/domains/symthaea-hdc-store/docs/CRASH_BOUNDARY_TESTING.md).
- [HdcStore format v2](../../crates/domains/symthaea-hdc-store/docs/FORMAT_V2.md).
