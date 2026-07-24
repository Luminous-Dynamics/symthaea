# Journaled Write Batches

`HdcStore::apply_batch` publishes a validated set of appends and deletions with
one header generation.

The protocol is:

1. Validate every ID and all count/offset arithmetic without changing the store.
2. Grow the file if required.
3. Atomically install and synchronize a checksummed `.txn` intent journal.
4. Write and flush every affected entry.
5. Commit one alternate header page with the complete target counts.
6. Publish the process-local ID and LSH indexes.
7. Remove the journal and synchronize the parent directory.

A normal open fails with `PendingBatchTransaction` while a journal exists. This
prevents a partially completed transaction from being interpreted through the
ordinary metadata-repair path.

`HdcStore::open_recovering` resolves the journal idempotently:

- When the selected header is still the base generation, appended entries are
  cleared and deleted entries are restored to live status. The batch is rolled
  back completely.
- When the selected header is the target generation, every recorded ID, index,
  and status is validated. The transaction is already committed, so recovery
  removes only the stale journal.
- Any other generation or count combination fails closed.

The journal contains IDs and entry indexes, not vector payloads. Canonical
vector bytes are written only to the store file. The journal is protected by
independent header and payload CRC64 checksums and is never treated as a source
of vector truth.
