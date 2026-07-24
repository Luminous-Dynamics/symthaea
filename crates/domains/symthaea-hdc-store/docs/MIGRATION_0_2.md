# Migration to symthaea-hdc-store 0.2

Version 0.2 tightens the mutation/search APIs and introduces on-disk format
version 2. Existing format-v1 files are never interpreted using v2 offsets.

## Migrate format-v1 files explicitly

A normal `HdcStore::open` reports `VersionMismatch` for a v1 file. Migrate it
with either the associated or free function:

```rust
let (store, report) = HdcStore::migrate_v1(path)?;
// Equivalent: symthaea_hdc_store::migrate_v1(path)?;
```

Migration validates the complete v1 committed region, rejects malformed or
duplicate live entries, copies live vectors into a synchronized same-directory
v2 temporary file, preserves file permissions, and only then replaces the
source path. Tombstones are discarded, so migration also compacts the store.
`MigrationReport` records source counts and the installed v2 generation.

Do not change the version bytes manually. Format v2 moves the data region from
offset 128 to offset 8192 and adds two checksummed header pages.

## Deletion now reports storage failures

Before:

```rust
let deleted: bool = store.delete(id);
```

After:

```rust
let deleted: bool = store.delete(id)?;
```

`Ok(false)` means the ID was not live. An I/O or synchronization failure is
returned instead of being silently converted into success.

## Creation no longer truncates implicitly

`HdcStore::create` now uses create-new semantics and fails when the destination
exists. Code that intentionally replaces an existing store must call
`HdcStore::create_or_replace`. Replacement truncates the old entry region before
allocating the new file, preventing stale committed-looking entries.

## Mutable opens are exclusive

A mutable store holds an advisory exclusive file lock. `HdcStore` is no longer
`Clone`; open another store only after the first handle is dropped. This closes
the stale-header overwrite race and protects references returned by zero-copy
`get` from concurrent mutable mappings created through this crate.

## Strict and recovering opens are separate

`HdcStore::open` validates without writing. `HdcStore::open_recovering` may
repair only entry-count disagreement inside the committed range and damaged
header redundancy. It returns a `RecoveryReport` and never promotes trailing
entries automatically. See `RECOVERY.md`.

## LSH construction validates dimensions

`LshIndex::new` now returns `Result<LshIndex, HdcStoreError>`. Bands must fit the
serialized `u16` bucket key, rows must be in `1..=32`, and total hyperplanes are
bounded before allocation.

## Exact and approximate search are separate

`scan_similar` now guarantees an exact full scan. Use `scan_similar_approx` when
LSH acceleration is acceptable. The approximate call returns `SearchOutcome`
with `examined`, `total_live`, and `exact` diagnostics.

New stores default to 32 bands by 8 rows. Migrated stores retain the dimensions
serialized in their v1 headers.
