# HDC Store Payload and Metadata Page-Locality Experiment v0.1

**Status:** exact virtual-page arithmetic + experimental layout proposal; no runtime or hardware performance results.  
**Tracking:** [HDC-STORE-LAYOUT-001 #7934](https://github.com/Luminous-Dynamics/symthaea/issues/7934), parent [MEMORY-SUBSTRATE-001 #7919](https://github.com/Luminous-Dynamics/symthaea/issues/7919).  
**Machine-readable model:** `crates/domains/symthaea-hdc-store/schemas/page_locality_model_v0_1.json`.

## 1. What we found in the current format

The current format-v2 `symthaea-hdc-store` fixed-record layout defines two 4,096-byte header pages, data offset 8,192 bytes, 2,080-byte entries, and a 2,048-byte `BinaryHV` payload beginning 32 bytes into each entry.

- [Header constants](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-hdc-store/src/header.rs)
- [Store implementation](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-hdc-store/src/store.rs)
- [Read-only reader](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-hdc-store/src/reader.rs)
- [BinaryHV definition](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-core/src/hdc/binary_hv.rs)

The vector-payload start for record index (i) is:

`offset(i) = 8192 + 32 + i * 2080`

A 2,048-byte payload touches:

`floor((offset + 2048 - 1) / page_size) - floor(offset / page_size) + 1`

virtual pages. This is a mapping-page calculation, not a physical SSD-read calculation.

For a 4 KiB base page, the current start phase repeats every 128 entries because `gcd(2080, 4096) = 32`. Enumerating the cycle gives 65 payloads wholly within one page and 63 that span two pages. The average is 1.4921875 virtual-page intersections per full-cycle payload access.

A hypothetical page-aligned 2,048-byte payload array has one intersection per vector: it lowers the modeled per-payload page intersections by about 33%. This is a useful random-access hypothesis, but it is not the largest potential gain.

## 2. The larger opportunity: keep payload pages cold during metadata open

The read-only reader's `scan_live_entries` walks every record and reads the status byte; for live entries it also reads the eight-byte ID. In format v2, these nine metadata bytes sit inside the same 2,080-byte-stride file extent as the 2,048-byte payload. Because each metadata probe is less than one 4 KiB page apart, the scan touches the payload-containing pages through the whole file, even though the code is only asking for status and ID.

A stronger candidate is a **single-file structure-of-arrays (SoA) layout**:

1. dual headers and versioned file metadata;
2. a compact, contiguous metadata array—provisionally 16 bytes per record for status, ID and reserved fields;
3. alignment padding to a declared page boundary;
4. a contiguous payload array with 2,048-byte stride.

The metadata scan can then validate status/ID without dereferencing the payload region. Exact similarity searches still need to read the vector payloads; they simply do not need to page them in just to open the store. This could improve large-store open behavior and reduce cache pollution before queries begin, even if sequential vector-scan bandwidth barely changes.

The current format must remain unchanged until a candidate format has explicit versioning, checksum semantics, crash/recovery tests, export/import/migration behavior and measured results.

### Exact 4 KiB model for selected store sizes

| Vectors | Current v2 metadata scan: unique virtual pages | SoA candidate metadata scan: unique virtual pages | Candidate / v2 | Full payload scan: v2 pages → SoA pages |
|---:|---:|---:|---:|---:|
| 64 | 33 | 1 | 33× fewer | 33 → 32 |
| 1,024 | 520 | 4 | 130× fewer | 520 → 512 |
| 16,384 | 8,320 | 64 | 130× fewer | 8,320 → 8,192 |
| 131,072 | 66,560 | 512 | 130× fewer | 66,560 → 65,536 |

These are counts of virtual pages intersected under declared address-layout equations. They are **not** page-fault counts, physical media reads, I/O bytes, latency or speedup. The actual storage/OS may satisfy reads from cache, use readahead, or incur other effects. The SoA candidate also needs a status/ID index and padding; all process memory and complete file extents must be measured.

The model calculates alignment padding and total file extent for each candidate rather than assuming the metadata split makes the file smaller. At 64 vectors, 4 KiB alignment padding makes the candidate extent 2 KiB larger than v2. At 1,024 vectors, 16-byte metadata plus aligned payload yields a model extent 16 KiB smaller. These are format-size arithmetic only.

## 3. Distinguish the metrics

The machine-readable model includes three separate concepts:

1. **Per-payload intersections**: the pages one vector payload touches; this may count the same page across multiple accesses.
2. **Unique payload pages in a complete sequential scan**: distinct virtual pages containing payload bytes.
3. **Unique pages touched by opening/scanning metadata**: current v2 metadata probes versus the compact SoA metadata region.

The first is relevant to random payload access; the second to full scans; the third to startup/page-cache pollution. Do not combine them into one "I/O amplification" or "speedup" number.

## 4. Alternative page-size profiles

The JSON enumerates 4 KiB, 64 KiB and theoretical 2 MiB page-size profiles. The actual OS base page size must be measured on the target host. The 64 KiB case applies only on systems with that page-size profile.

The 2 MiB case is a mathematical profile, **not** an assertion that a regular file-backed `mmap` uses transparent huge pages. Linux kernel documentation currently states that THP support applies to anonymous memory mappings and tmpfs/shmem, with extension to other filesystems described as future work. Huge-page/TLB experiments must prove that the exact host/kernel/filesystem supports the relevant mapping.

Reference: [Linux transparent huge-page documentation](https://docs.kernel.org/admin-guide/mm/transhuge.html).

## 5. Experimental design

Keep format v2 untouched as the baseline. Compare:
- current interleaved v2;
- single-file SoA with 16-byte metadata then page-aligned, packed payload array;
- optionally, compact 9-byte metadata as a lower-bound model only, unless a versioned format review justifies it;
- optional segmented immutable payload extents, only if access traces demonstrate value.

Freeze the same corpus, stable IDs, query set, index configuration and task semantics. Test separately:
- random IDs;
- skewed/hot-set IDs with published seed and distribution;
- clustered and nearby IDs;
- cold-start reader open and warm repeated open;
- exact top-k retrieval and approximate LSH with exact ground-truth comparison;
- sequential full scan;
- batched mutations, compaction, export/import and recovering open.

Record p50/p95/p99 latency, throughput, total RSS, index/metadata memory, file extents, cache state, page-fault counts where host tools can measure them, and bytes observed at the actual block layer where available. Do not label logical bytes or virtual page intersections as physical media traffic.

On Linux, use a pinned run and record process-scoped page-fault counters (for example via `getrusage`, `perf stat` or `/usr/bin/time -v`) and note which operations each counter includes. Page faults are still not equivalent to physical device reads: cache, readahead and filesystem policy intervene.

## 6. Proposed performance and safety gates (not achieved)

- Exact-mode results identical to the current v2/canonical software oracle for all fixtures; logical content checksum unchanged across export/import or migration.
- No invalid/truncated record accepted; interrupted writes recover to a validated generation or explicit error/incomplete state.
- Target at least 10% better end-to-end p95 random-lookup latency **or** 15% lower measured process page faults / page intersections under a precisely defined host profile.
- For cold-start/open workloads, measure whether metadata-only validation reduces page faults and cache pollution without changing corruption detection or open-time semantics.
- Sequential full-scan throughput must not regress by more than 5% for the declared target workload unless a preregistered Pareto trade-off is justified.
- Do not claim NAND read/write amplification unless actual device/block-layer telemetry supports it.
- Use at least 30 measured batches per configuration where practical; retain raw samples, dispersion, source SHA, OS/kernel, page size, filesystem, device identity, compiler and build profile.
- Durability/recovery tests are independent gates and must pass before a new persistent format can advance.

## 7. Paradigm shift

Persistent layout is part of the compute architecture. The format should be optimized not only for record mutation but also for how an opener discovers IDs, how much inactive payload it pulls into cache, how a query traverses memory, and how a recovery tool validates a generation. Metadata-first layout is a hypothesis for making the data plane cold until needed, while payload alignment can improve random-access page geometry.

The correct outcome may still be that v2 wins overall because its simpler mapping or mutation path outweighs cold-open costs. This experiment decides rather than preordains a rewrite.

## 8. Explicit nonclaims

This model has no measured benchmark, page-fault observation, physical-device I/O observation, energy measurement or alternative layout implementation. It does not establish that v2 is slow or that the candidate is faster. All numbers are exact counts from declared address-layout equations only.
