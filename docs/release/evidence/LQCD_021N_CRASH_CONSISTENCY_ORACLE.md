# LQCD-021N — crash-consistent checkpoint publication oracle

Independent standard-library execution for the abstract crash-consistency semantics required by #2818, stacked on the exact checkpoint-commitment oracle in #2823.

Exact executed subject SHA-256:

`fede77443b94cbc1703def99f2f94724999bae126121c31a542df615b86a95b5`

Canonical result SHA-256:

`1a91f8962035524b9eafa7555cb81418fc23b332008f5dc1bdc72f758a020111`

## Qualified abstract persistence profile

Stable profile ID: `posix_local_atomic_rename_fsync_v1`.

The subject models this authority sequence:

```text
write complete temp bytes
-> fsync temp file
-> verify exact temp bytes
-> rename to immutable final path
-> fsync containing directory
-> verify exact final bytes
-> write generation-checked head record
-> fsync head record
```

A checkpoint becomes published only after the final-path rename is made durable by the declared directory-sync step. It becomes authorized only after the exact head record is durably advanced.

## Crash-point theorem

The oracle injects a crash before/after every protocol boundary. Recovery classifications are frozen as:

- steps 0–4: `NoPublishedCheckpoint`;
- steps 5–7: `PublishedButNotAuthorized`;
- step 8: `AuthorizedCheckpoint`.

Additional negative states prove:

- a mid-write temp file does not become published;
- a partial/un-fsynced head update leaves the durable old head authoritative;
- one-bit corruption of a durable final checkpoint yields `CorruptPersistenceState`;
- a durable new head without a durable exact final checkpoint yields `CorruptPersistenceState`;
- a second writer racing from generation 0 cannot silently overwrite the generation-1 authorized head;
- a distinct sibling child from the same predecessor is classified as a detectable fork candidate.

The immutable fixture path is content-bound:

`checkpoint-00000001-84d2328883ae397e179697f364db000180975946aa4b0f38a88d6a6e65b3a6b8.bin`

The synthetic checkpoint bytes have SHA-256:

`07ff7df8c1a43aa32b2aca30a76f2ee37f915d32f916928f1afced8c16fb497c`

The exact new-head record has SHA-256:

`b384f6b8f3e1d1f11aba05296db75b2697dd576e66165a82bce657554dd9e8e0`

## Scientific boundary

This establishes an **abstract authority/recovery state machine only**. It does not prove that a real Linux filesystem, ext4/XFS/NFS/object store, Windows volume, container overlay, virtual disk, or hardware cache obeys these durability semantics. Production qualification must execute the protocol on an explicitly identified OS/filesystem/storage profile with injected process/power-loss tests before claiming real crash durability or portability.
