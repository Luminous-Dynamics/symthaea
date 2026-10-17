# HDC Store Workload Benchmark

This document covers the synthetic workload trace example `storage_workload_bench`. The run measures behavior of the existing `symthaea-hdc-store` API over deterministic packed BinaryHV data. It does **not** isolate SSD latency or physical I/O bytes: mmap, OS page cache, filesystem, index construction, and runtime all contribute to observed times.

## Run

From the repository root, record and inject source/tool identity:

```sh
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
RUSTC_VERSION="$(rustc --version)" \
cargo run -p symthaea-hdc-store --release --example storage_workload_bench -- \
  --vectors 1024 --queries 32 --batch-size 128 --seed 7 \
  --csv /tmp/hdc-store-samples.csv
```

The example uses a temporary directory by default and removes the test store at exit. To put a new benchmark store on a chosen filesystem/device, use `--store /path/on/target-filesystem/new-store.hdc`; the path must not already exist and the benchmark never overwrites an existing store. The explicit store is kept after the run. Use a fresh, dedicated path on each run. The CSV output gets a matching `.meta.txt` sidecar.

Parameters:
- `--vectors N`: number of synthetic 16,384-bit BinaryHV records (1 through 1,048,576).
- `--queries N`: fixed-seed query samples; queries come from stored vectors so exact top-1 has a known exact match.
- `--batch-size N`: appends per durable `WriteBatch` commit.
- `--seed N`: deterministic corpus/query-selection seed.
- `--csv PATH`: raw per-operation CSV and run-metadata sidecar.
- `--store PATH`: explicit, previously nonexistent store path on the target filesystem.

The maximum vector count is intentionally bounded because the benchmark holds the synthetic corpus in memory and the store itself uses a memory mapping. Very large runs can require several gigabytes of working set and should be attempted only on a suitable host.

## Workload profiles

| Profile | What it includes | Important caveat |
|---|---|---|
| `corpus_generate` | Deterministic BinaryHV generation | Not included in write or search timing |
| `store_create` | File/header setup and capacity preallocation | File length does not equal physical media bytes written |
| `batch_append_commit` | One journaled WriteBatch commit per batch | Raw logical payload bytes exclude metadata/journal amplification |
| `sync_all` | Explicit final mapping/file synchronization | Not a controlled power-loss durability experiment |
| `mutable_reopen` | Mutable open and index load/rebuild | Combines storage access and in-memory index construction |
| `exact_scan_top1` | Exact top-1 full scan | Correctness baseline; per-query logical scan size is reported |
| `approx_scan_top1` | Approximate LSH search against exact top-1 IDs | Reported recall is top-1 ID agreement with the deterministic exact result; fallbacks are separate |
| `mutable_random_point_read` | Random ID lookup through the mutable store's in-memory ID map | OS page cache may already be warm after open |
| `readonly_reader_open` | Validated, read-only reader open | Includes validation/mapping, not only file-open syscall time |
| `random_point_read` | Random ID lookup through the read-only reader | Includes reader ID lookup cost; logical bytes are not physical page reads |
| `sequential_full_scan` | Full iteration over all live vectors | Counts logical vector payload; operating-system caching is not controlled |
| `clean_recovery_open` | Recovery-capable open of an uncorrupted store | Tests normal clean-open overhead, not a crash-injection or power-loss recovery path |

Samples report latency in nanoseconds. Summary p50/p95/p99 are percentiles over recorded operation samples; for profiles with a single sample, percentiles are not statistically meaningful. Throughput is calculated from the corresponding operation count and summed sample time. Warm-up, ambient load, CPU governor, temperature, filesystem, storage device, OS cache state, index snapshot state and build mode must be recorded when comparing machines.

## Measurement limitations

- The benchmark does not flush the operating system page cache and does not claim media-cold latency.
- `logical_bytes_estimate_not_physical_io` is an accounting estimate based on the work requested, not bytes observed at the physical device.
- No power meter is used; energy and joules/query are not inferred from runtime or nominal device TDP.
- The non-cryptographic FNV-1a fingerprint binds the generated vector bytes for repeatability but is not a security integrity primitive.
- Synthetic random BinaryHV vectors provide repeatable capacity, exactness and lookup-workload tests. They do not substitute for task-realistic HDC corpora, adaptive mutation traces or long-running service traffic.
- A passing run requires the example to compile and execute at an exact source SHA with exit code 0 and preserved stdout/stderr/raw CSV/metadata. This change has not been run in an environment with Cargo/Rust installed; source presence is not a benchmark PASS.

## Next comparisons

Use the same corpus / IDs / query plan to compare:
1. several batch sizes for write throughput and commit latency;
2. mutable hash-map point access against read-only sorted-ID lookup;
3. exact full scan vs approximate LSH with top-1 agreement and candidate/fallback counts;
4. vector counts spanning local hot data to larger working sets;
5. the same scenarios over a real target filesystem and then a declared FPGA external-memory interface.

For published results, also run real, workload-specific traces. MLCommons Storage v3.0's Vector Database benchmark is a helpful reference because it covers on-disk ANN queries with small random reads and checks recall against exact brute-force ground truth. Symthaea's 16,384-bit BinaryHV workload and result contract differ from the MLPerf workload; running this example does not imply MLPerf compliance.
