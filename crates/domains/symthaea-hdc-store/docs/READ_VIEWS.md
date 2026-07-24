# Generation-Pinned Read Views

`HdcStore::read_view` creates an `HdcReadView` that immutably borrows the open
store and captures its current generation plus a sorted ID-to-entry table.

While the view exists, Rust's borrow rules prevent the owner from calling any
method that requires `&mut HdcStore`, including append, delete, write batches,
header repair, checkpointing, and compaction. This provides a process-local
consistent-read boundary without copying 2 KiB vector payloads.

The view provides:

- zero-copy `get` by ID;
- deterministic ascending-ID iteration;
- exact nearest-neighbor search with deterministic tie ordering;
- pinned generation and count metadata.

This is not a filesystem snapshot and it does not permit another process to
write concurrently. Its purpose is to make multi-step reads, exports, and
validation passes internally coherent while preserving mmap performance.
