# HDC-TILED-BATCH-SEARCH-DSE-002 — cache-blocked exact retrieval

Status: benchmark experiment authored; no Rust compilation or timing has been performed from this environment.

## Hypothesis

A single traversal order may not be optimal for every batch and corpus size. Query-major traversal keeps one query active but revisits the corpus for each query. Corpus-major traversal visits candidates sequentially and reuses each candidate across a batch, but can keep too many query vectors active at once. A two-dimensional tiled traversal can constrain the active query block while retaining a small candidate block for reuse.

For 16,384-bit BinaryHV, each vector is 2,048 bytes. The initial candidate-block settings of 1, 4 and 16 represent 2, 8 and 32 KiB of candidate payload; query tiles of 1 and 4 represent 2 and 8 KiB of query payload. Those payload sizes are not total cache-footprint estimates: result state, code, prefetching, associativity, competing threads and cache hierarchy also matter. No target cache size is assumed.

## New benchmark

File: crates/core/symthaea-core/benches/hdc_tiled_batch_search.rs

Run on an exact source revision:

    cargo bench -p symthaea-core --bench hdc_tiled_batch_search

Compared variants:
- query-major exact top-1;
- corpus-major exact top-1;
- tiled corpus-major with query tile sizes 1 and 4 and candidate tile sizes 1, 4 and 16.

Axes:
- batch sizes 4 and 16;
- corpora 256, 4,096 and 16,384 candidates;
- independent and correlated query batches;
- canonical similarity score and candidate-ID equality required before timed iterations.

The correctness contract additionally uses a 257-candidate deterministic corpus and duplicate exact matches in a five-candidate fixture. All paths preserve lowest-candidate-index tie-breaking. The benchmark reports host wall time only; it does not claim physical cache misses, DRAM bytes, energy or accelerator results.

## What would count as a useful result

Capture exact HEAD, rustc/Cargo versions, CPU model and ISA features, compiler flags, OS/kernel, thermal/power profile, raw Criterion output and repeated-run variability. Report per-query latency, batch throughput, and the cost of preparing any packed representation separately. For each corpus/query regime, compare tile choices against both baselines; reject a tile that only wins in one easy synthetic profile while regressing materially elsewhere.

Use hardware performance counters when available to explain a result (cycles, instructions, LLC misses, cache references and memory bandwidth), but do not substitute generic estimated counters for measured values. Run on at least one mobile/laptop-class CPU and one server/workstation CPU before making portability claims. For an accelerator comparison, add submission/transfer and result-return overhead to the same end-to-end boundary.

## External baseline context

FAISS's Binary Index guide describes IndexBinaryFlat as an exhaustive Hamming scan optimized with CPU popcount and says batching is applied on both the query and database sides to avoid cache misses. That is the baseline discipline this experiment tries to extend—not evidence that Symthaea already improves on FAISS:
https://github.com/facebookresearch/faiss/wiki/Binary-Indexes

FAISS's general documentation supports batched queries and explicitly frames approximate search as a speed/precision trade-off. Keep exact top-1 results separate from approximate recall/latency Pareto curves:
https://faiss.ai/

MLPerf Tiny v1.4 uses fixed workloads and quality targets and makes latency and measured energy first-class metrics. For future phone/edge comparisons, emulate the fixed-workload and calibrated-energy discipline rather than infer joules from timing or TDP:
https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/

Vortex v3.0 is relevant as an open architecture research platform for memory staging, asynchronous transfer/scheduling, simulation and synthesis. It is not a directly comparable HDC product baseline:
https://github.com/vortexgpgpu/vortex/releases

## Predeclared decision gates

1. Exact candidate ID and score agree with the canonical reference on every test fixture and workload.
2. Compare matched data and build configuration; do not compare kernel-only time against end-to-end time.
3. No universal winner is declared from one CPU or synthetic query distribution.
4. Do not claim hardware superiority until qualified RTL/FPGA measurements include transfer, orchestration and the same exact result contract.
5. Keep this PR draft until exact-head build and benchmark checks have actually run successfully.

This is a search-space reduction experiment, not a claim of performance improvement.