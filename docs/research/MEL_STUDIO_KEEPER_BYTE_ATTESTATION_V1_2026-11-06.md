# Muse Studio keeper byte attestation v1

**Date:** 6 November 2026  
**Implementation:** \`crates/domains/symthaea-muse/src/bin/muse_studio.rs\`  
**Parent:** [#10051 — MEL-013A verified Studio artifact transactions](https://github.com/Luminous-Dynamics/symthaea/issues/10051)  
**Related benchmark:** [#10088 — RMI-Bench](https://github.com/Luminous-Dynamics/symthaea/issues/10088)

## Research question

Does a successful Studio keep commit a bundle whose exact persisted bytes can be reopened and independently checked before the keeper index and genealogy metadata say that the artifacts exist?

A hash computed from an in-memory value is not necessarily the hash of the bytes saved to disk. In particular, the previous keep path wrote pretty-printed \`recipe.json\` but hashed a separate compact serialization of the recipe; it hashed audio from the in-memory WAV buffer; it did not persist \`score.json\`, despite recording a score digest. This tranche closes that mismatch for new keeper bundles.

## Keeper bundle v2

New keeps now publish the following directory contents:

| File | Requirement | Meaning |
|---|---|---|
| \`audio.wav\` | Required | Exact rendered audio bytes |
| \`recipe.json\` | Required | Exact persisted composition/renderer recipe |
| \`score.json\` | Required | Exact compact JSON serialization of the symbolic Score |
| \`performance.mid\` | Optional | MIDI export; absence is explicitly recorded |
| \`bundle-manifest.json\` | Required | Versioned artifact descriptors and transaction digest |

Each artifact descriptor records its fixed relative file name, media type, SHA-256 over exact staged bytes and byte length. The manifest binds the audio key, schema version, MIDI-availability flag and ordered artifact descriptors. A transaction digest commits to the serialized manifest body. The persisted manifest itself also receives a SHA-256, which is surfaced in the keeper response and JSONL index.

No caller-provided path or artifact filename is accepted by the verifier. The allowed artifact set and order are fixed by schema version. Untracked files, symbolic links, missing required files, wrong media types, empty required files, malformed hashes, wrong byte lengths, changed bytes, a mismatched keeper key or a changed transaction digest all fail verification.

## Commit order

The new keep path performs these operations in order:

1. Serialize the recipe and Score into the exact byte arrays intended for persistence.
2. Write audio, recipe, Score and optional MIDI into a fresh staging directory. Each file is synced.
3. Read the staged files back to construct the artifact descriptors and write the transaction manifest.
4. Sync the staging directory on Unix, then atomically rename the complete directory into its keeper key.
5. Sync the keeper parent directory on Unix.
6. Re-open the published directory. Recompute every artifact SHA-256 and byte length from the bytes at the published paths; validate the transaction manifest and exact directory contents.
7. Only after successful read-back, allocate the genealogy root using the verified recipe, Score and audio hashes.
8. Append the keeper index entry with the artifact and manifest commitments.

If read-back or the directory publication sync fails, the bundle is removed and the keep fails before genealogy/index metadata is committed. If keeper-index append fails—including a poisoned keeper-index mutex—the bundle is removed and any allocated genealogy root is rolled back. Cleanup failures are reported instead of silently hidden.

MIDI remains optional because the existing export API can fail. A missing MIDI file is not replaced with an invented digest; the manifest declares \`midi_available=false\` and has exactly the three required content artifacts.

## Compatibility

- Existing v1 keeper directories and flat-file compatibility paths are not rewritten.
- New directories add \`score.json\` and \`bundle-manifest.json\`; existing \`audio.wav\`, \`recipe.json\` and \`performance.mid\` route paths remain unchanged.
- The keeper JSONL entry now says \`keeper-directory-v2\` and carries the artifact and manifest digests. Existing consumer fields remain present.
- The keep response adds optional \`audio_sha256\` and \`artifact_manifest_sha256\` fields. The already-kept/deduplicated response reports these as absent because it does not resolve a specific prior keeper entry.

## Tests added

Seven new keeper-bundle integrity tests cover:
- exact persisted-byte hashes and successful re-open/read-back;
- mutation of audio, recipe, Score and MIDI bytes;
- missing required and untracked files;
- altered transaction digest and wrong keeper-key replay;
- manifest symlink rejection (Unix);
- explicit optional-MIDI absence;
- zero-length required artifact rejection.

The binary’s existing tests remain in place. These are authored tests; the dedicated repository CI result is the authority for whether they compile and pass on this PR head.

## Evidence and trust boundary

This change establishes **local byte-integrity evidence at the Studio keep boundary**. It does not establish:

- cryptographic signatures or authenticated identity of the actor;
- purpose-specific consent, collaborator ownership or copyright;
- a durable independent verifier service or remote replication;
- tamper resistance against a privileged local user who can rewrite files and manifests;
- long-term storage immutability after the successful read-back;
- multi-parent co-creation lineage (the live genealogy allocator still creates root entries);
- that a score or recording is musically good, valid or preferred.

SHA-256 proves equality to the committed bytes under this manifest contract; it does not prove who created them or whether they were authorized. The current directory read-back is a second filesystem read in the same Studio process, not an independent external verifier. Unix directory-entry fsync is used where supported; non-Unix platforms retain per-file sync but do not claim a portable directory fsync.

## Next integration seam

The next MEL-013A step is an independently callable read-back/verifier boundary that resolves the committed keeper key from Studio-owned storage, validates the manifest and hashes again, and binds the verified artifact receipt to a session contribution event. Only after the trusted identity/consent adapter verifies the actor and purpose-specific grant should the event be accepted into RMI-Bench or persistent multi-contributor lineage.

Keep the layers distinct:

- artifact digest/read-back = byte integrity;
- signed transaction/actor receipt = authenticity and authorization;
- event/DAG edge = contribution provenance;
- RMI contrast = measured causal adaptation;
- musician study = human musical relevance and collaborator-valued outcomes.

No one layer substitutes for another.
