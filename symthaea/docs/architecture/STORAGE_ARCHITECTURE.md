# Symthaea Storage Architecture

Symthaea does not have one database. Storage is split by latency and query shape.

## Rules

- The cognitive cycle must not require a database read to complete.
- Hot state stays in memory: ring buffers, working memory, arenas, caches, and snapshot channels.
- Persistent backends are fed before or after cognitive cycles.
- Cognitive code depends on storage traits, not concrete engines.
- Code should depend on the narrowest storage trait that can satisfy the job.

## Current Backends

| Role | Current implementation | Status |
| --- | --- | --- |
| Hot cognitive state | In-memory cognitive loop structures | Default |
| Durable local memory and metadata | `SqliteMemory` via `ConsciousnessDatabase` and `StorageRuntimeHandle` | Default |
| Persistent vector/HDC memory | `LanceMemory` behind `lancedb-backend` | Optional |
| Zero-copy local BinaryHV store | `HdcStoreDatabase` behind `hdc-store` | Optional |
| Telemetry analytics | DuckDB behind `epistemic_auditor` | Optional |
| Distributed governance/history | Mycelix/Holochain bridges | Optional |

## Storage Contracts

`ConsciousnessDatabase` remains as the compatibility facade for existing code, but new code should prefer narrower contracts:

| Trait | Use |
| --- | --- |
| `MemoryRecordStore` | Store, get, delete, count, and export memory records |
| `BinaryHvSearchBackend` | BinaryHV/HDC similarity retrieval |
| `StorageHealthBackend` | Health checks and backend statistics |
| `CausalEventStore` | Durable causal-link/event storage |

This keeps the cognitive layer from depending on database-specific behavior and avoids forcing every backend to act like every other backend.

## Async Boundary

`databases::storage_runtime` provides a bounded write-behind worker for durable memory writes:

- hot paths can use `try_store_memory` / `try_delete_memory`;
- batch episodic flushes can use `try_store_memory_batch_guarded`;
- non-hot paths can await queue capacity;
- tests and shutdown paths can call `flush`;
- persistent backend errors are logged by the worker and do not stall the cognitive cycle.

When `memory_db_path` is configured, `EpisodicPersistenceManager` now attaches
SQLite through a thread-backed storage runtime. This keeps the synchronous
`CognitiveLoopService::new` constructor usable outside Tokio while preserving
non-blocking, batched write-behind persistence for periodic episode flushes.
If no runtime is attached, the older background-thread fallback remains as a
compatibility path.

## Backend Roles

### SQLite

SQLite is the default durable local truth store for memory records, causal links, curricula, and small metadata. It is portable, ACID, and always available in the main crate.

### LanceDB

LanceDB is a persistent vector-memory backend, not the Symthaea database. It is useful for columnar memory storage, metadata predicate pushdown, and future continuous-vector or embedding tables.

Current `LanceMemory` stores `BinaryHV` values as fixed-size Arrow binary and computes Hamming similarity in process. This preserves BinaryHV semantics while keeping LanceDB optional.

### HDC Store

`symthaea-hdc-store` is the local zero-copy BinaryHV organ. It stores vectors in an mmap file and the `HdcStoreDatabase` adapter stores `MemoryRecord` metadata in a JSON sidecar. It is appropriate for local high-throughput BinaryHV experiments, not as a replacement for SQLite metadata or LanceDB research tables.

### DuckDB

DuckDB is for telemetry, benchmarks, Φ traces, and offline analysis. It should not be placed on the hot cognitive path.

## Near-Term Stack

The best near-term default stack is:

- `SqliteMemory`: durable local metadata and ordinary memory records.
- `LanceMemory`: optional persistent vector/HDC memory for larger research datasets.
- `HdcStoreDatabase`: optional zero-copy BinaryHV experiments.
- DuckDB/Parquet: telemetry and scientific observability.

Qdrant, Tantivy, redb, and Postgres/pgvector remain valid future backends, but should be added only when a concrete query shape needs them.

## Next Implementation Steps

1. Keep reads on in-memory caches/snapshots, with explicit refresh points.
2. Add a DuckDB/Parquet telemetry sink behind a `TelemetrySink` trait.
3. Add backend benchmarks for SQLite, LanceDB, and HDC store under the same contract workload.
4. Only then evaluate Qdrant/Tantivy/Postgres for concrete production query shapes.
