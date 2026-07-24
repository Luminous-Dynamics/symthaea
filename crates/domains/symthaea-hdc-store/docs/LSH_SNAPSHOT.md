# LSH signature snapshot format

The canonical `.hdc` file remains complete and independently recoverable. An
optional sibling file ending in `.hdc.lsh` stores deterministic per-band LSH
signatures so reopening a large store does not require recomputing every random-
hyperplane projection.

## Trust boundary

A snapshot is an acceleration artifact, not canonical state. Readers accept it
only when all of the following match the opened store:

- header generation and committed entry counts;
- LSH bands, rows, and the deterministic hyperplane seed;
- a content fingerprint supplied by the store integration layer;
- exact sidecar length and entry count;
- strictly increasing, duplicate-free IDs;
- one bounded hash per configured band;
- CRC64-ECMA checksums for both header and payload.

A missing sidecar is normal. A present but invalid sidecar is reported as
`HdcStoreError::InvalidIndexSnapshot`; higher-level open policy decides whether
to rebuild or fail closed.

## Layout

The sidecar starts with a fixed 96-byte little-endian header followed by
ID-sorted records. Each record contains:

- `id: u64`;
- `lsh_bands` little-endian `u32` bucket hashes.

The header records the source generation, vector/live/tombstone counts, LSH
configuration, seed, live-entry count, store fingerprint, payload checksum, and
header checksum.

## Atomic publication

Writers create a unique file in the destination directory, stream the payload,
rewrite the sealed header, synchronize the complete file, rename it over the
previous sidecar, and synchronize the parent directory on Unix. Failed staging
files are removed by a guard.

## Store lifecycle integration

`HdcStore::open` uses `IndexOpenPolicy::PreferSnapshot`: a missing, stale, or
corrupt sidecar is recorded in `IndexStatus` and the index is rebuilt from the
canonical vectors. `IndexOpenPolicy::Rebuild` ignores sidecars for reproducible
validation, while `IndexOpenPolicy::RequireSnapshot` fails closed when the
artifact cannot be trusted.

`checkpoint_lsh` synchronizes the canonical store before publishing a snapshot.
Successful append, delete, header repair, or recovery-generation changes mark
the previous snapshot stale. Checkpointing is explicit so a sidecar write error
is never hidden in `Drop` and can be handled by the caller.
