# Broca evidence persistence and crash-consistency conformance

**Status:** prototype disk-generation coordinator, a typed real-HDC binding path, and in-process failpoint tests are authored in `memory_evidence_persistence.rs`. The code is not compiled in this environment; the 1,000-schedule qualification gate and real power-loss/platform tests remain open. This document is a release gate, not a persistence qualification result.

## Invariant

An HDC candidate ID is not evidence. A query may receive a payload only when the candidate index and evidence snapshot belong to the same committed generation and the resolved record passes scope, lifecycle, validity-window and content-digest checks.

The snapshot codec preserves tombstones, binds schema + generation + records into its digest, and rebuilds ID maps. `EvidenceCommitManifest` binds exact serialized HDC-index bytes and exact evidence-snapshot bytes to a nonzero generation. The typed production path additionally records the real HDC store's internal generation and logical checksum, and requires exact parity between live HDC IDs and Active evidence IDs. The disk store stages immutable artifacts, syncs each file, renames the generation directory, reads back and verifies the artifacts through `HdcStoreReader`, then atomically replaces `CURRENT` last. Recovery checks `CURRENT`, manifest and artifact digests and never falls back to an older generation. Its SHA-256 values are unkeyed consistency checks, not writer authentication. A hostile-storage deployment must verify a signature or externally pinned digest before trusting the manifest.

## Implemented prototype surface

`EvidenceDiskStore` exposes `open(root)`, the typed production `commit_hdc_store(&mut HdcStore, &EvidenceIndex)`, `recover()`, and `query_committed_context(&BinaryHV, candidate_k, scope_id, now_ms, payload_root, max_items)`. The generation is assigned under a filesystem lock from the last fully verified commit. The opaque-byte `commit` is test-only. Staging lives under `generations/`; generation directory names bind the generation number and manifest digest. The `CURRENT` pointer is the publication point and names exactly one commit.

The implementation is intentionally conservative:

- A corrupt, missing-artifact, unknown-version, or symlinked `CURRENT` is an error, not an empty store or a fallback trigger.
- A generation containing missing, symlinked or digest-mismatched artifacts cannot be recovered.
- The commit manifest and snapshot generation must agree; records are revalidated during snapshot restoration.
- Multiple threads using one store instance serialize through a mutex, and cross-process writers serialize through an exclusive `.writer.lock` file.
- A crashed process can leave `.writer.lock`. The store refuses new commits until an operator confirms the writer is dead and reconciles/removes the stale lock. It does not guess that a lock is stale.
- The filesystem contract is currently Unix directory-fsync + same-filesystem rename. Other platforms fail closed; no cross-platform durability claim is made.
- The production `commit_hdc_store(&mut HdcStore, &EvidenceIndex)` path syncs the canonical HDC store, captures a generation-pinned view, records its internal generation + logical checksum, requires exact equality of live HDC IDs and Active evidence IDs, and copies the canonical bytes into an immutable generation.
- On recovery, typed commits are opened via `HdcStoreReader` with deterministic ANN-index rebuild; the internal generation, checksum version/value, live count and ID parity are checked against the evidence snapshot. The opaque-byte `commit` method exists only for module tests.
- `prune_history(retain_generations)` uses the same writer lock, requires a verified CURRENT commit, retains the explicit generation-number window, removes unpublished future generations and orphan `.staging-*` directories, and syncs the generation directory. It preserves all immutable variants for a retained historical generation because no authenticated predecessor chain identifies which variant was once current.
- The application still must mutate vector and evidence lifecycle state together: delete/revoke/supersede the HDC vector before committing the inactive evidence state. A failed parity check leaves the previous committed generation current.

The in-process fault hooks cover interruption after the HDC-index write, evidence snapshot write, manifest write, generation publication, before `CURRENT` publication, and after `CURRENT` rename. They are deterministic failure-injection tests, **not** process-kill or power-loss tests.

## Required publish protocol

The persistence coordinator treats these bytes as one logical commit. The implemented `commit_hdc_store` path binds the HDC identity; adapter/query wiring and broader crash qualification remain open:

1. Allocate monotonically increasing generation `g`; stage an HDC index and evidence snapshot in a private generation-specific location. **Implemented** by `EvidenceDiskStore` after verifying the current commit.
2. Flush the exact index and snapshot bytes to stable storage using same-filesystem temporary files, file sync, and atomic rename where supported. **Implemented for the declared Unix prototype assumptions**; not yet power-loss-qualified.
3. Create the commit manifest for `g` from those exact bytes; read all three artifacts back and verify their digests and schema before publication. **Implemented** before the pointer is changed.
4. Publish `CURRENT` last as the commit point, pointing at the immutable manifest and generation; sync the containing directory. **Implemented**. An error after pointer rename is an ambiguous durability outcome and requires `recover()` inspection.
5. On query, load only artifacts named by the committed generation; verify index digest, snapshot digest, snapshot decoding, and generation equality before resolving payloads. **Implemented at the storage API** by `query_committed_context`: it reads `CURRENT` once, verifies the typed manifest, opens the exact HDC index for ranking, rechecks the opened reader identity and live-ID parity, and resolves payloads against that same recovered evidence snapshot. `resolve_with_provenance` preserves the HDC/evidence ID and source trajectory/step for every emitted context item; the result can be projected to the official `type/value` ABI. Wiring the binary-query API to Broca's live `MemoryBridge`, selecting a raw-text query encoder, and implementing the official LongMemEval-V2 worker remain open.
6. If a committed artifact is missing, corrupt, has an unknown schema, or does not match the manifest, return an integrity error. The caller must surface zero context. **Implemented at the store API**; any consuming adapter must preserve this failure rather than converting it to an empty successful result.
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

The disk-store prototype has deterministic in-process failpoints for commit stages plus retention tests for current-generation preservation, stale-staging cleanup, unpublished future-generation pruning, and refusal to prune on a corrupt CURRENT pointer. Query tests additionally cover committed-generation identity, scope filtering, refusal to serve opaque-byte fixtures, and the distinction between an uninitialized store and a corrupt CURRENT pointer. Qualification must additionally kill a separate process at each stage, restart from the resulting directory, and on supported lab hardware perform power-cut trials; the current failpoints do not qualify those conditions.

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

These are proposed acceptance targets, not observed results. The current branch supplies snapshot, manifest and disk-store prototypes plus typed HDC identity-parity checks and authored test cases; the code has not yet passed a Rust compile/test run or the persistence qualification gate.
