# Portable live-set archives

Portable archives are compact, deterministic exports of the logical live vector
set. They are intended for backup, transfer, and format-independent validation;
they are not raw copies of the mmap file.

## Format

Version 1 contains a 128-byte checksummed header followed by ascending-ID records:

- 8-byte little-endian ID;
- 2,048 raw `BinaryHV` bytes.

The header binds:

- source generation;
- record count;
- LSH dimensions;
- record size and format version;
- logical content checksum;
- complete payload CRC64;
- header CRC64.

Tombstones, unused capacity, header-page history, journals, and LSH sidecars are
not exported.

## Publication guarantees

`export_portable_archive` writes and synchronizes a unique same-directory staging
file and publishes it with an atomic no-clobber hard link. Existing destinations
are never replaced.

`restore_portable_archive` validates the complete archive before creating a
staging store. It reconstructs the store, verifies the logical checksum, syncs
the staging file, publishes without overwrite, reopens the destination under the
path-stable coordination lock, and verifies the checksum again.

CRC64 detects accidental damage but does not authenticate an archive. Sign or
cryptographically digest archive files in the surrounding security layer when
provenance or hostile modification matters.

## Resource limits

Default inspection, export, and restore calls enforce `PortableArchiveLimits`:

- at most 100,000,000 records;
- at most 256 GiB of archive bytes.

The limit checks occur before payload traversal and before restore capacity is
allocated. Deployments with intentionally larger stores must use the
`*_with_limits` APIs and opt into explicit bounds appropriate to their storage
and operational budgets.

## Restore batching

Restore publishes vectors into the private staging store in journaled batches of
1,024 records. This bounds transaction-journal size and reduces durable header
commits from one per vector to approximately one per thousand vectors. The
staging path remains unpublished until all batches, the final sync, and the
logical checksum verification succeed.

Failure cleanup removes the staging data file plus any transaction journal, LSH
sidecar, or coordination-lock companion associated with that temporary path.
