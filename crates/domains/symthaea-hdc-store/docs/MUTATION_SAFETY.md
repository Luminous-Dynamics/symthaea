# Mutation Safety and Poisoned Handles

`HdcStore` uses a fail-stop mutable handle.

Before mapped bytes are changed, validation, arithmetic, capacity growth, and
header-count calculations may fail without affecting the handle. After an
entry status or payload has been written, a later flush or header-commit error
can no longer be treated as an ordinary retryable failure: the operating
system may have persisted some, all, or none of the modified pages.

When that uncertainty occurs, the handle transitions from `StoreHealth::Healthy`
to `StoreHealth::Poisoned`. The original operation returns its original error,
and every subsequent mutation returns `HdcStoreError::StorePoisoned`.

A poisoned handle may still be inspected in memory for diagnostics, but its
contents must not be treated as a durable statement. Drop it and reopen the
path. Use `HdcStore::open` when the committed metadata remains consistent, or
`HdcStore::open_recovering` when the entry scan and header counts disagree.
Recovery never promotes trailing appended entries beyond `vector_count`.

This is intentionally stricter than continuing after an I/O error. Continuing
would allow later header generations or index snapshots to be built on top of
an uncertain mutation boundary.
