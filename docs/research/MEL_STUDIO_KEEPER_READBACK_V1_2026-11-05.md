# Muse Studio keeper read-back integrity v1

**Date:** 5 November 2026  
**Implementation:** `crates/domains/symthaea-muse/src/bin/muse_studio.rs`  
**Parent:** [#10051 — MEL-013A verified Studio artifact transactions](https://github.com/Luminous-Dynamics/symthaea/issues/10051)  
**Scope:** local keeper publication and exact-byte read-back; not yet participant authentication or co-creation lineage.

## Defect found

The current keep path stages `audio.wav`, pretty-printed `recipe.json`, and optional `performance.mid` under a temporary directory, then renames the directory to its generated keeper key. It computes `score_sha256` for the in-memory `Score`, but the Score bytes are not persisted. It also computes `recipe_sha256` using compact JSON serialization while the actual `recipe.json` file is pretty-printed, so that digest is not the digest of the exact file stored. Finally, the route previously trusted the successful write/rename path and did not re-open the published files before it allocated genealogy metadata or published the JSONL keeper index.

## Change

The keep path now:

1. Serializes the Score into exact compact JSON bytes and persists `score.json` inside the staged bundle.
2. Computes the audio, recipe and score hashes over the exact bytes that are intended to be written.
3. Writes and syncs each required artifact in the staging directory; if MIDI export succeeds, it syncs and hashes the exact staged MIDI bytes.
4. Atomically renames the staging directory to `data/taste/audio/<audio_key>`; on Unix, it syncs the parent directory after the rename.
5. Re-opens the published directory using its final path, reads every required file (and MIDI when present), re-computes SHA-256 and byte lengths, and rejects any digest mismatch, missing required file or unexpected MIDI file.
6. Creates a `KeeperArtifactReadbackReceiptV1` with the exact keeper key, artifact layout, digest scope and sorted file evidence.
7. Hashes the serialized read-back receipt and stores both receipt and receipt digest in the keeper JSONL entry and the keep response.
8. Only after successful read-back does it allocate the genealogy root and append the keeper index. A failed read-back removes the published bundle before it becomes discoverable. If the index append fails later, the existing rollback path removes the bundle and deletes any allocated genealogy manifest.

The receipt declares `sha256-exact-persisted-file-bytes-v1`. For new keepers, `recipe_sha256` is now the digest of the exact persisted pretty-printed `recipe.json` bytes, not the earlier compact serialization of the in-memory Recipe object. `score_sha256` remains the digest of the compact Score JSON bytes, which are now also stored as `score.json`. Existing keeper records are not rewritten; consumers must use the receipt's digest scope to distinguish new exact-file evidence from older records.

## Tests and workflow

Three targeted tests cover:

- receipt content, byte lengths, sorted artifact names and exact digests for audio, recipe, score and MIDI;
- rejection when the committed Score bytes differ from their expected digest;
- rejection of a missing Score file and an unexpected MIDI file when the keep record says MIDI is unavailable.

A new exact-head workflow pins Rust 1.96.0, checks formatting of the Studio binary source, and runs only the `keeper_readback_tests` module with `--locked` under the `studio` feature.

## What this proves—and does not

When the production keep handler succeeds, its receipt means that the server reopened the final keeper directory after the rename and observed the exact expected bytes at those paths. This is stronger than hashing only the buffers passed into the write routine. It does not establish an external hardware/media read, protection against a malicious process with write access, secure-boot or verifier-binary attestation, signature verification, authenticated actor identity, consent scope, copyright ownership, or musician-valued quality.

The current genealogy ledger still allocates a root manifest and still stores metadata pointers rather than artifact bytes. The keeper directory is unique-keyed, not a content-addressable object store. No multi-parent contribution DAG, trusted receipt-signing provider, participant-purpose consent verifier, or cross-session RMI session transaction is introduced here. Those remain MEL-013A work and must build on this existing storage path rather than create a parallel artifact registry.

## Next MEL-013A integration

- Bind this read-back receipt to a versioned Studio transaction ID and idempotency key.
- Add an explicit trusted authority verifier for actor, artifact, action and purpose scope; unavailable/invalid evidence fails closed for consent-sensitive uses.
- Record multi-parent lineage edges and source-event influence only when actual source artifacts and actor receipts exist.
- Add restart/crash-phase tests for bundle publish, read-back, genealogy write and keeper-index replacement.
- Connect RMI-Bench trial input/output event identities to the verified committed keeper bytes, while retaining its separate causal and musician-evaluation metrics.
