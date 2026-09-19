# MATH-RET-RUNTIME-001A — Qualified Retrieval Execution Seam

Status: isolated executable candidate for issue #4245.

This package is intentionally outside the monorepo workspace for its first qualification tranche. The root workspace discovers `crates/core/*` automatically, so introducing a new core package there would also mutate the shared lock/resolution surface. This standalone package lets the execution seam be reviewed and tested without changing legacy cognitive-loop code, mathematical memory, or the root lockfile.

It has **zero external dependencies**.

## Purpose

The seam turns an already-qualified retrieval request into one deterministic execution and one atomic evidence transaction:

```text
QualifiedRetrievalRequest
        |
        v
QualifiedRetrievalBackend          one decision point
        |
        | source-object IDs only
        v
deterministic ranking / fusion
        |
        v
CanonicalSourceMaterializer        one shared source-fetch boundary
        |
        | canonical UTF-8 payload bytes
        v
canonical prefix packer
        |
        +--> RetrievalTraceSink
        +--> PayloadAuditSink
        |
        v
AtomicEvidenceSink::commit
```

The package deliberately does not know about:

- `MathMemory`;
- `MathService.memory`;
- HDC similarity thresholds;
- Phi;
- theorem truth;
- evidence score;
- result/search-memory retention.

Those omissions are architectural constraints, not missing features.

## Critical representation/payload separation

A representation-specific backend is allowed to decide only:

```text
WHICH source-object identities are ranked
```

It cannot supply downstream mathematical payload bytes.

Those bytes are obtained only from `CanonicalSourceMaterializer`, whose production implementation must correspond to the graph's exact:

```text
source_object_contract_sha256
source_fetch_policy_sha256
payload_serialization_sha256
```

This prevents lexical/sparse/HDC/normal-form arms from receiving different downstream context merely because their retrieval implementations serialize candidates differently.

## Main types

- `QualifiedRetrievalRequest`
- `RetrievalBudget`
- `QualifiedRetrievalBackend`
- `CanonicalSourceMaterializer`
- `BackendExecution`
- `SingleIndexExecution`
- `ChannelExecution`
- `FusionExecution`
- `RetrievalTraceSink`
- `PayloadAuditSink`
- `AtomicEvidenceSink`
- `RetrievalTraceReceipt`
- `PayloadAuditBatch`
- `QualifiedRetrievalOutcome`

`normalized_compute_microunits` is fixed-point (`1 unit = 1e-6 normalized compute units`) so the runtime boundary does not introduce floating-point accounting drift.

## Deterministic fusion

The seam implements the already-frozen MATH-RET-001B2 mechanics directly.

### Reciprocal Rank Fusion

```text
score(x) = sum_channel 1 / (rrf_k + one_based_rank(x))
```

The implementation uses integer rational numerators/denominators, never `f32`/`f64`, and ties by source-object digest ascending.

### Deterministic interleave

- frozen starting channel;
- strict alternation while both channels remain;
- duplicate consumes the turn and produces no backfill;
- after one channel is exhausted, only the already-returned remainder of the other list may be consumed;
- no deeper retrieval occurs during fusion.

Fusion input item/byte opportunity is checked against the same request budget before materialization.

## Packing

Only source identities from the immutable final ranking are materialized.

The packer:

1. preserves retrieved/fused rank;
2. materializes through the single shared source boundary;
3. requires non-empty valid UTF-8 payload bytes;
4. emits whole items only;
5. stops at the first non-fitting payload;
6. never skips a large ranked item to fish for smaller later items;
7. derives payload-audit targets from the exact same materialized bytes used by packing.

`source_object_sha256` remains distinct from any future payload-byte digest. MATH-RET-PAYLOAD-001A remains responsible for writing/hashing the materialized bytes and independently reconstructing byte counts.

## Query/resource canaries

The executor checks:

- query source identity cannot occur in a ranking;
- one single-index execution reports exactly one retrieval query;
- one fusion execution reports exactly two retrieval queries;
- returned ranking length cannot exceed requested `k`;
- fusion channel opportunity cannot exceed the shared item/byte budget;
- wall time, query count and normalized compute cannot exceed frozen ceilings;
- controls either carry both seed+artifact or neither, depending on the frozen transform.

## Atomic evidence rule

`AtomicEvidenceSink` implementations must guarantee:

```text
commit = all trace + payload evidence visible
      OR no staged evidence visible
```

If trace staging, payload staging, or commit fails, `RetrievalExecutor::execute` returns `Err` and calls `abort`. An experiment must therefore become non-qualifying rather than silently continuing without provenance.

## Qualification command

```bash
cargo test \
  --manifest-path tools/math-retrieval-runtime-seam/Cargo.toml \
  --locked
```

The package has no registry dependencies; `Cargo.lock` contains only this package.

## Mechanical canaries

The unit tests freeze:

1. one backend execution call per qualified request;
2. one shared materializer determines the payload bytes used by packing and evidence;
3. query-source self leakage is rejected before materialization/evidence commit;
4. evidence-stage failure aborts instead of succeeding without provenance;
5. interleave duplicate turns are consumed without backfill;
6. exact RRF ranks a source present in both channels correctly;
7. incorrect retrieval-query accounting is non-qualifying;
8. representation-specific backends cannot supply or mutate downstream payload bytes.

## Deliberate nonclaims

This tranche does not establish that:

- it has been connected to the cognitive loop;
- a real index implementation obeys this seam yet;
- this package has passed `cargo test` on a runner yet;
- the Python MATH-RET-TRACE/PAYLOAD validators have consumed its typed output yet;
- the materializer implementation correctly realizes the frozen source-fetch policy yet;
- retrieved sources are mathematically relevant;
- HDC, normalization, or fusion helps;
- any theorem is true or novel.

The next gate after Rust qualification is a **contract adapter** that serializes these typed receipts to the frozen MATH-RET-TRACE-001A and MATH-RET-PAYLOAD-001A JSON shapes and validates a deterministic fixture end-to-end. Only after that should a real syntax/HDC/normal-form backend implement `QualifiedRetrievalBackend`.
