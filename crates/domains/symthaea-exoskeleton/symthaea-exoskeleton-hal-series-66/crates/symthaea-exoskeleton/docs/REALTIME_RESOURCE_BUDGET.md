# Real-Time Resource Budget

The actuation cycle has explicit budgets for execution time, lock wait, stack
high-water, heap allocation, and blocking I/O. The production profile permits
no heap allocation or blocking I/O on the fast path.

Instrumentation must be provided by the real-time runtime or allocator shim;
self-reported zero values are not qualification evidence. Repeated deadline
misses or any forbidden blocking behavior latch zero authority until a resource
audit and restart ceremony are completed.
