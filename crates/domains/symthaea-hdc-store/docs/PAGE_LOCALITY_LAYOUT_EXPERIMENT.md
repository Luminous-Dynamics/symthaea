# HDC Store Payload Page-Locality Experiment v0.1

**Status:** exact virtual-page arithmetic + experimental layout proposal; no runtime or hardware performance results.  
**Tracking:** [HDC-STORE-LAYOUT-001 #7934](https://github.com/Luminous-Dynamics/symthaea/issues/7934), parent [MEMORY-SUBSTRATE-001 #7919](https://github.com/Luminous-Dynamics/symthaea/issues/7919).  
**Machine-readable model:** `crates/domains/symthaea-hdc-store/schemas/page_locality_model_v0_1.json`.

## 1. What we found in the current format

The version-2 `symthaea-hdc-store` fixed-record layout defines two 4,096-byte header pages, data offset 8,192 bytes, 2,080-byte entries, and a 2,048-byte `BinaryHV` payload starting 32 bytes into each entry.

- [Header constants](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-hdc-store/src/header.rs)
- [Store implementation](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-hdc-store/src/store.rs)
- [BinaryHV definition](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-core/src/hdc/binary_hv.rs)

The payload start for record index (i) is:

`offset(i) = 8192 + 32 + i * 2080`

A payload spans:

`floor((offset + 2048 - 1) / page_size) - floor(offset / page_size) + 1`

pages, where these are virtual mapping pages and not physical SSD reads.

For a 4 KiB page, the current start phase repeats every 128 entries because `gcd(2080, 4096) = 32). Exact enumeration of this cycle yields 65 payloads touching one page, and 63 touching two pages. That is an average of 1.4921875 virtual-page intersections per random payload access in the uniform full-cycle model.

The alternative considered here is a page-aligned packed payload array: a page-aligned data base, 2,048-byte stride and a separate ID/metadata mapping. With a 4 KiB page, each 2 KiB payload is wholly within one page and two payloads occupy each page. This reduces the **modeled page intersections per payload** from 1.4922 to 1.0 (about 33.0%).

**This does not imply a 33% latency or physical-I/O improvement.** Page caching can make a requested page already resident; multiple payloads share a page; metadata mapping incurs cost; storage devices read internal blocks, not necessarily virtual pages; the layout may change index locality, write patterns and recovery behavior.

## 2. The distinction that prevents a misleading claim

The machine-readable model reports two different counts:

1. **Per-payload intersections**: the pages a single vector payload range touches; this can count the same page for multiple vector accesses.
2. **Unique payload pages during one full sequential scan**: the distinct virtual pages that contain at least one byte from any payload in that store generation.

For a 1,024-vector store and 4 KiB page profile, the model calculates:
- 1,528 per-payload page intersections in current v2 vs 1,024 in the packed layout;
- 520 unique payload pages for a full current-v2 scan vs 512 for the packed array.

The first number illustrates a possible random-access locality issue. The second suggests only a 1.54% reduction in unique pages for a sequential full scan. These are different access patterns; neither is a page-fault, storage block-I/O or performance result.

Also, the page-aligned candidate doesn't magically delete metadata. Metadata/IDs still need a versioned layout and mapping; the 2,048-byte payload stride and metadata split change write/read locality and indirection overhead. Any comparison must record total process RSS and total bytes across all files rather than comparing only payload-file size.

## 3. Alternative page-size profiles

The JSON enumerates 4 KiB, 64 KiB and theoretical 2 MiB mapping-page profiles. The actual OS base page size must be measured on the target host. The 64 KiB case only applies where that is the actual host page size.

The 2 MiB case is a mathematical profile, **not** an assertion that the store's regular file-backed `mmap` uses transparent huge pages. Linux kernel documentation currently states that THP support applies to anonymous memory mappings and tmpfs/shmem, with extension to other filesystems described as future work. Any huge-page/TLB experiment must prove the actual mapping behavior on its OS/kernel/filesystem rather than assuming it.

Reference: [Linux transparent huge-page documentation](https://docs.kernel.org/admin-guide/mm/transhuge.html).

## 4. Experimental design

Keep format-v2 unchanged as the baseline. Do not silently reinterpret v2 offsets or introduce migration in the first experiment.

Compare:
- current interleaved v2 format;
- page-aligned packed payload prototype (separate experimental profile/format);
- optional segmented immutable payloads, only if the first two profiles establish a meaningful access-pattern reason.

Freeze the same corpus, stable IDs, query list, index configuration and task semantics. Access patterns:
- uniform random IDs;
- Zipfian/hotset IDs with a published fixed seed and skew parameter;
- clustered/nearby IDs;
- sequential full scan;
- exact top-k retrieval;
- approximate LSH with exact-result quality check;
- batch updates, compaction and clean/recovery open.

Record each separately. The benchmark should include p50/p95/p99 latency, throughput, total RSS, index/metadata memory, file extents, cache state, page-fault counts when measured through host-appropriate instrumentation, and bytes observed at the actual block layer when available. Never label logical bytes or virtual page intersections as physical media traffic.

On Linux, use a pinned run and record `getrusage` page fault counters or an external, process-scoped measurement (for example `perf stat` / `/usr/bin/time -v`); report which phases the instrumentation includes. Page faults are still not the same as device reads if pages are served by cache/readahead.

## 5. Proposed performance and safety gates (not achieved)

- Exact-mode output identical to current v2/canonical software oracle for all fixtures; logical live-content checksum unchanged across format export/import.
- No invalid/truncated entry accepted; every injected interruption recovers to a validated generation or explicit error/incomplete state.
- For random lookup at equal task semantics, target at least 10% better end-to-end p95 latency **or** 15% lower measured per-process page faults/intersections under an explicitly defined, repeatable host profile.
- Sequential full-scan throughput must not regress by more than 5% for the designated target workload unless a predeclared Pareto trade-off is justified.
- Do not claim physical NAND read/write amplification unless actual device or block-layer telemetry supports it.
- Include at least 30 measured batches per configuration when practical; preserve raw samples, dispersion, source SHA, OS/kernel, page size, filesystem, device identity, compiler and build profile.
- All durability/crash tests must remain independent of benchmark timing and must pass before a new persistent format could advance.

## 6. Paradigm shift

Storage layout is part of the compute design, especially for fixed-width, 2 KiB hypervectors. A data structure chosen only for convenient record updates can be suboptimal for a memory-mapped similarity engine. The research opportunity is to co-design:
- payload alignment with the host and storage access units;
- index/metadata residency and ID lookup;
- HDC scan/retrieval batching;
- memory-page/TLB locality;
- persistent write/compaction behavior;
- exactness, integrity, migration and recovery.

The correct result may be that v2 is already good enough or wins on the whole workload. This experiment exists to decide, not to preordain a format rewrite.

## 7. Explicit nonclaims

This model has no measured benchmark, no page-fault observation, no physical-device I/O observation, no energy measurement and no storage-layout implementation. It does not establish that v2 is slow or that the alternative is faster. All numbers are exact counts from the declared arithmetic model only.
