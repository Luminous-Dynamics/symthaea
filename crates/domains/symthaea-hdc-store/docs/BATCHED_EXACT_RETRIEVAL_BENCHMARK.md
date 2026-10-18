# Corpus-major exact HDC retrieval benchmark

Tracking: [HDC-STORE-BATCH-EXACT-001 #8007](https://github.com/Luminous-Dynamics/symthaea/issues/8007).

## Run

```sh
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
RUSTC_VERSION="$(rustc --version)" \
HOST_PROFILE="$(uname -m) / <CPU model>" \
STORAGE_PROFILE="<filesystem and physical device model>" \
cargo run -p symthaea-hdc-store --release --example batched_exact_scan_bench -- \
  --vectors 16384 --queries 32 --top-k 10 --batch-sizes 1,2,4,8,16 \
  --rounds 30 --seed 7 --csv /tmp/hdc-batched-exact.csv
```

Choose a fresh CSV path for each run. The output sidecar is derived with the extension `.meta.txt`; the benchmark refuses to overwrite either pre-existing output. Start at 4,096 vectors, then scale deliberately: every vector is 2,048 bytes and the fused score-tuple memory is bounded by `--vectors * max(batch_sizes) <= 4,194,304` tuples.

## What it compares

- **Repeated exact scans:** calls the existing public `HdcStoreReader::scan_similar` separately for each query in the batch.
- **Corpus-major batch:** traverses `reader.iter()` once per query batch; for each mapped vector, computes similarities against every query in that batch using the same `BinaryHV::similarity` dispatch path, then sorts and truncates every query's score list with the exact existing ordering (score descending, ID ascending).

The program uses deterministic generated corpus/query vectors, existing public store APIs, fixed IDs, the same top-k and same exact score function. Each measured group checks the complete ordered top-k IDs **and exact f32 bit patterns** against the reference before recording the sample. Any mismatch aborts the run.

This is deliberate: initially it isolates traversal order and query batching from approximate retrieval and from a different top-k algorithm. A bounded heap, SIMD layout rewrite, or explicit prefetch path should be a separate benchmark axis rather than bundled into this one.

## Why batch at all?

Faiss's documented binary flat index uses popcount-based Hamming distance and explicitly applies batching on the query and database sides to avoid cache misses. It provides a public batched binary-search interface: [Faiss Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-Indexes). This is evidence that batching is a credible optimization class, not proof that this store will gain from it.

Elastic's June 8, 2026 report on SIMD vector throughput describes batch prefetching and structural loop changes among optimizations that produced throughput gains in its own workload: [Elasticsearch SIMD vector throughput](https://www.elastic.co/search-labs/blog/elasticsearch-simdvec-vector-throughput). Its reported metrics are not directly transferable to 16,384-bit HDC vectors or memory-mapped storage; this experiment must establish the local result.

## Measurement protocol

- Use at least 30 rounds per configuration for reportable findings.
- Batch sizes default to 1, 2, 4, 8 and 16. Each group is measured repeatedly; algorithm order alternates across rounds/groups to reduce order bias.
- Record raw batch samples, query count, corpus size, top-k, result checksum, source SHA, compiler, host/storage labels and build context.
- Report queries/s and p50/p95/p99 **batch** latency. Amortized time per query can be calculated as batch elapsed time / actual query count.
- Cache state is uncontrolled, and the reference phase reads the corpus before timings. Treat this first benchmark as warm/repeated mapped-store behavior; don't claim cold-start performance or device I/O.
- Include a corpus smaller than last-level cache and one larger than LLC when that cache capacity is known. Label unknown hardware details as unknown rather than infer them.
- Compare every supported CPU target separately; the current similarity API already dispatches across architecture-dependent SIMD/POPCNT paths.

## Proposed promotion gates (not achieved)

1. 100% exact top-k agreement, IDs, f32 scores and tie ordering against the existing single-query implementation; include top-k=0 and tie-heavy test fixtures in a follow-up test.
2. On a named host and corpus larger than LLC, target at least 1.5x exact queries/s at batch size 8 with no single-query regression >5%. The target is a decision threshold, not a forecast.
3. Report p50/p95/p99, sample dispersion, queries/s, allocations or peak RSS where measured, CPU model/features, OS/kernel, page size, and corpus/query sizes.
4. If the fused path is slower, identify whether sorting, memory budget, SIMD execution, query working set or cache behavior dominates, and keep the current implementation.
5. Do not replace the public production method on this benchmark alone. A follow-up production change needs code review, repeatable measurements, normal unit tests and exact-head runtime/CI evidence.

No approximation, format migration, additional dependency, or production search-path modification is included by this example.
