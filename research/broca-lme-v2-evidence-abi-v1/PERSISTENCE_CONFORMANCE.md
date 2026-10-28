# Broca evidence persistence and crash-consistency conformance

**Status:** protocol/specification added alongside the snapshot codec. The disk-backed coordinator, atomic publication and crash-injection suite are **not implemented**. This document is a release gate, not a persistence qualification result.

## Invariant

An HDC candidate ID is not evidence. A query may receive a payload only when the candidate index and evidence snapshot belong to the same committed generation and the resolved record passes scope, lifecycle, validity-window and content-digest checks.

The snapshot codec preserves tombstones and rebuilds ID maps. `EvidenceCommitManifest` binds exact serialized HDC-index bytes and exact evidence-snapshot bytes to a nonzero generation. Its SHA-256 values are unkeyed consistency checks, not writer authentication. A hostile-storage deployment must verify a signature or externally pinned digest before trusting the manifest.

## Required publish protocol

The persistence coordinator must treat these bytes as one logical commit:

1. Allocate monotonically increasing generation `g`; stage an HDC index and evidence snapshot in a private generation-specific location.
2. Flush the exact index and snapshot bytes to stable storage using same-filesystem temporary files, file sync, and atomic rename where supported.
3. Create the commit manifest for `g` from those exact bytes; read all three artifacts back and verify their digests and schema before publication.
4. Publish the manifest last as the commit point; sync the containing directory where the platform supports it. A mutating API must not report success before the commit point is durable under the declared platform contract.
5. On query, load only artifacts named by the committed generation; verify index digest, snapshot digest, snapshot decoding, and generation equality before calling `resolve`.
6. If a committed artifact is missing, corrupt, has an unknown schema, or does not match the manifest, return no context and surface an integrity failure. Do not silently fall back to an older generation: it may predate a deletion, revocation or supersession.
7. Retain prior generations only according to a declared recovery and retention policy. If an operator explicitly authorizes recovery to an older generation, mark it as a recovery event and re-apply an authoritative deletion/revocation log before serving queries.

The in-memory module does not implement these filesystem steps. Atomic rename alone is not a universal power-loss guarantee; the implementation must state and test its actual filesystem/platform assumptions.

## Mutation and rebuild semantics

- **Insert/update:** reserve unique HDC and evidence identities; source bytes are canonicalized by the adapter and their SHA-256 is recorded. The query path never reconstructs missing evidence from caller-provided text.
- **Revoke/delete:** update lifecycle state in the evidence snapshot and publish the matching vector index generation. Deleted IDs remain tombstoned in snapshots. Logical deletion is not physical secure erasure.
- **Index rebuild:** rebuild vectors from an authoritative source while preserving evidence IDs and tombstones. Never regenerate the payload map from vector IDs alone. Publish the rebuilt index and the preserved evidence snapshot under a new manifest generation.
- **Supersession:** old records remain as lineage but cannot resolve; replacement IDs must exist in the snapshot and stay within the same scope.
- **Corruption:** a missing mapping, digest mismatch or failed restore produces zero usable evidence for the affected query. No best-effort fallback may turn a candidate ID into a text answer.
- **Images:** only canonical regular files under the configured payload root may be emitted. Image content digests are verified when queried. This does not prove the image remains physically present after a successful query.
- **Provenance:** the snapshot preserves each source digest, but the source hash cannot be rechecked on restore unless canonical source bytes or a separately verified source artifact are available.

## Crash and adversarial matrix

The future disk-backed integration must inject a process crash or forced error at each stage and then restart from the resulting directory.

| Fixture | Required result |
|---|---|
| Crash before any staged write | Previous committed generation remains readable |
| Crash after vector index write but before evidence snapshot | Staged generation is ignored |
| Crash after snapshot write but before manifest publication | Staged generation is ignored |
| Manifest refers to old index + new snapshot (or the inverse) | Integrity failure; zero context |
| Corrupted index or snapshot bytes | Integrity failure; zero context |
| Unknown manifest/snapshot version | Fail closed; no silent downgrade |
| Delete then restart | Deleted payload never resolves |
| Revoke then restart | Revoked payload never resolves |
| Supersede then restart | Old payload never resolves; active same-scope replacement remains eligible |
| Rebuild index with a stale deleted vector ID | Tombstone still blocks payload return |
| Duplicate HDC ID or evidence ID in a crafted snapshot | Restore fails |
| Text content differs from its digest | Snapshot restore or resolution fails |
| Image symlink escapes payload root | Resolution fails |
| Query includes unknown HDC IDs or duplicate hits | Unknown IDs are skipped and output contains no duplicated evidence |
| Commit manifest has a matching file digest but invalid evidence snapshot | Verification fails; hash agreement alone is insufficient |

## Acceptance gate

Before claiming persistent-memory integrity:

- Run at least 1,000 deterministic, seeded fault-injection schedules spanning every commit stage and mutation class; retain seeds and exact artifact hashes.
- Require **zero** stale, revoked, superseded or deleted payloads to appear after restart/rebuild.
- Require every mismatched/corrupt/unknown-version committed pair to fail closed; no older-generation fallback without explicit recovery authority.
- Demonstrate that a committed successful mutation survives restart on each claimed filesystem/platform configuration.
- Test idempotent replay and interrupted replay; retries must not duplicate identities or roll lifecycle state backwards.
- Audit physical deletion separately, including snapshots, journals, backups and filesystem behavior.
- Measure p50/p95/p99 query latency and write/compaction cost only after correctness gates pass.

These are proposed acceptance targets, not observed results. The current change supplies snapshot and manifest primitives plus unit-test cases; it does not yet pass this integration gate.
