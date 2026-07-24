# Read-only inspection

`inspect_store` and `HdcStore::inspect` provide an fsck-like structural report
without opening a mutable mapping or changing metadata.

Inspection acquires a shared advisory lock, validates both v2 header pages,
applies the same generation-selection rules as normal open, scans the committed
entry range, and reports:

- header checksum/semantic validity for each slot;
- selected slot and generation;
- declared and scanned live/tombstone counts;
- committed-region truncation;
- invalid status bytes and duplicate live IDs;
- committed-looking trailing entries beyond `vector_count`;
- legacy files that require explicit migration.

```rust
let report = HdcStore::inspect(path)?;
if !report.is_clean() {
    for issue in &report.issues {
        eprintln!("{issue:?}");
    }
}
```

`metadata_recovery_may_help` is true only for degraded redundancy or safe count
mismatch—the two classes handled by `open_recovering`. It does not imply that
trailing entries, duplicate IDs, invalid statuses, or two invalid headers can be
repaired automatically.
