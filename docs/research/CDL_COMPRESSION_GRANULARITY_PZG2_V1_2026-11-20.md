# CDL-Bench: compression-context granularity and selective telemetry reads (PZG2)

**Date:** 2026-11-20  
**Tracking:** [CDL-Bench-014 #12964](https://github.com/Luminous-Dynamics/symthaea/issues/12964), [CDL-Bench-011 #12935](https://github.com/Luminous-Dynamics/symthaea/issues/12935)  
**Status:** local C++20 reference experiment; synthetic fixtures; not production, not Rust validation, not a SOTA claim

## 1. Research question

PZC1 made each `(entity, series)` child an independent Zstandard frame. This enables selective decompression, but each child starts a fresh compression context. The whole-frame PRN1 baseline preserves cross-series redundancy but requires decoding the whole frame for a single-series read.

PZG2 tests an intermediate granularity: concatenate a bounded number of independently encoded per-series PRN1 payloads into one group, compress the group as one Zstandard frame, and keep a compressed directory that maps a key to the child's byte range within its group.

The intended trade-off is explicit:

- smaller groups: better single-key read selectivity and smaller corruption domains, but more frame resets and metadata;
- larger groups: better shared compression history, but more unrelated series are decompressed and a group corruption affects more keys;
- whole-frame compression: lower storage cost on many fixtures, but no independent group decode.

PZG2 is an experiment in this design space, not a claim that grouped frames or seekable compression are new. Zstandard's seekable format already defines collections of independent frames and a jump table; this work asks how the data layout and query semantics should be co-designed for typed multi-series telemetry.

## 2. PZG2 reference format

The C++20 reference and reproduction script are supplied in the companion artifact bundle `CDL_PZG2_reference_bundle_2026-11-20.zip`. They were executed locally, but **the source and shell script have not yet been imported into this repository branch**. The benchmark source is therefore not currently available at the two repository paths named in the earlier draft of this report. This is a research artifact, not a checked-in implementation claim.

PZG2 contains:

1. A fixed header with magic/version discriminator (`PZG2`), total row and key counts, configured group size, schema ID, Zstandard level, group count, compressed/uncompressed directory lengths, directory checksums, and header CRC.
2. A separately Zstandard-compressed directory. Per-group entries contain group row-payload length, compressed length and CRCs. Per-child entries contain `(entity, series)`, row count, offset, child length and child CRC.
3. One independently compressed Zstandard payload per group. Each group has compressed-payload CRC and uncompressed-payload CRC; child slices have their own CRCs.
4. A selective read path that validates and decompresses the directory, then decompresses only the selected group, checks group and child checksums, and decodes the selected series.
5. A full decoder that reads each group, validates every child and restores global event order with a min-heap merge.

The independently compressed directory avoids carrying a large uncompressed per-series directory in tiny/idle workloads. Group checksums are intentionally local: corruption of one payload group does not make a different intact group unreadable, provided the header and directory remain valid. Every key in the damaged group is affected because the compression frame is shared by that group.

PZG2 v1 deliberately accepts only globally strictly increasing unique timestamps. Duplicate timestamps or regressions fail closed; the PRN1 tie-template and explicit-order-schedule fallback modes are not yet integrated. The current group membership is first-seen key order, not query affinity or learned similarity.

## 3. Reproduction

Requirements: C++20 compiler, `pkg-config`, libzstd headers/library. After extracting the companion artifact bundle, run the included shell script from a Git worktree that has the required system tools. The script expects the source at `research/cdl-bench/reference/compression_granularity_group_reference.cpp`; until the files are imported, use the exact source and script from the bundle rather than assuming those paths exist in the checkout.

The checked local run used GCC 14.2.0, libzstd 1.5.7, `-O2 -std=c++20 -Wall -Wextra -Wconversion -Wno-misleading-indentation -Werror`, and Zstandard level 19. A fresh local rerun on 2026-11-20 reproduced the same level-19 complete-frame sizes exactly and re-ran the test suite. A level-3 smoke run also passed. The warning suppression is limited to misleading-indentation warnings arising from compact reference formatting; all other enabled warnings remain fatal.

The fixtures are deterministic synthetic workloads with 65,536 records, 64 keys and eight typed fields, using the same generator and seed as the existing PRN1 comparison harness (`0x7e1e0001`). Workloads: sparse updates, bursty updates, mostly idle and noisy/all-field updates. These are controlled stress fixtures, not production observations.

The baseline is the complete PRN1 row-state frame compressed as one Zstandard frame through the same libzstd API and at the same level. The PZG2 number includes its header, separately compressed directory, group metadata and compressed group frames. Thus the table compares complete serialized frames under the same library/API/level. It does not compare only the child payloads.

## 4. Results: size versus read amplification

**Complete frame bytes, Zstandard 1.5.7 level 19**

| Workload | Whole PRN1 | PZG2 g=1 | g=2 | g=4 | g=8 | g=16 | g=32 | g=64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse | 61,860 | 77,382 | 73,763 | 70,470 | 67,825 | 65,528 | 63,851 | 62,278 |
| Bursty | 204,917 | 237,018 | 227,979 | 221,153 | 215,740 | 211,460 | 208,120 | 205,332 |
| Mostly idle | 4,780 | 7,766 | 6,976 | 6,357 | 5,946 | 5,639 | 5,444 | 5,239 |
| Noisy | 2,252,906 | 2,375,091 | 2,339,721 | 2,315,636 | 2,299,308 | 2,282,601 | 2,268,290 | 2,253,280 |

**PZG2 complete-frame size overhead versus whole PRN1**

| Keys per group | Sparse | Bursty | Mostly idle | Noisy | Max unrelated keys decoded for one-key read |
|---:|---:|---:|---:|---:|---:|
| 1 | +25.09% | +15.67% | +62.47% | +5.42% | 0 |
| 2 | +19.24% | +11.25% | +45.94% | +3.85% | 1 |
| 4 | +13.92% | +7.92% | +32.99% | +2.78% | 3 |
| 8 | +9.64% | +5.28% | +24.39% | +2.06% | 7 |
| 16 | +5.93% | +3.19% | +17.97% | +1.32% | 15 |
| 32 | +3.22% | +1.56% | +13.89% | +0.68% | 31 |
| 64 | +0.68% | +0.20% | +9.60% | +0.02% | 63 |

The single-key read amplification column is the number of *other* keys whose payload shares the group and therefore must be decompressed; it is derived from the group-size policy, not from a measured disk-read or latency benchmark. A group of 8 decompresses at most eight series in total for a one-key lookup (the selected series plus up to seven unrelated series).

### Interpretation

- Compression cost falls monotonically as groups grow, as expected from shared Zstandard history. At group size 64, PZG2 is within 0.02–9.60% of the whole PRN1 frame; mostly-idle streams remain disproportionately sensitive to fixed metadata/frame overhead.
- Group size 8 costs 2.06–9.64% on sparse, bursty and noisy, but 24.39% on mostly-idle data. It gives a bounded eight-series decompression domain.
- Group size 16 reduces the overhead to 1.32–5.93% for sparse, bursty and noisy, at the cost of decoding up to 16 series per one-key query; mostly-idle overhead is still 17.97%.
- There is no universally optimal group size across workloads. A deployable design should choose from a query-and-storage Pareto frontier, not hard-code one group size based on a synthetic compression ratio.
- PZG2 is larger than whole-frame PRN1 for every fixture at every grouping level. It has not demonstrated a storage win. The result is evidence about how quickly the selective-decode penalty approaches the whole-frame baseline as compression contexts are shared.

## 5. Validation actually completed

The strict-warning local C++20 run reported:

```text
group_roundtrip_tests=PASS empty=PASS zero_group_rejected=PASS duplicate_time_rejected=PASS regression_rejected=PASS group_sizes=1,2,4,8,16,32,64 selective_key_decode=PASS group_corruption_isolation=PASS header_corruption=REJECTED
```

For each group size, the harness performed a full exact record round-trip, a selective one-key read, and corruption checks. It damaged one group payload, confirmed a query from a different intact group still succeeded, and confirmed the damaged group's read failed. It also checked empty streams, invalid group size, duplicate timestamps, timestamp regressions and a header mutation.

This validates the local PZG2 reference implementation and its declared test cases only. It does not validate the checked-in Rust crate, CI, a production storage engine, random-access latency or a full public-data benchmark.

## 6. What is still missing

- No true file-I/O benchmark: no random disk reads, cold/warm cache separation, p50/p95/p99 latency, bytes read or decompression CPU measured.
- No bounded-memory or streaming encoder/decoder; all records, child streams, group payloads and directory bytes are materialized in memory.
- No integration with PCL1 field-columnar children or PZC1 per-series post-Zstd layout selection.
- No shared trained Zstandard dictionaries, dictionary training cost, dictionary versioning or dictionary-size accounting.
- No adaptive group assignment; first-seen key order can be a poor compression or access-affinity grouping.
- No timestamp-range index/checkpoints inside groups, and no partial replay within a child.
- No duplicate/out-of-order timestamp fallback.
- No external codec comparisons (Parquet tuned Zstd, ALP, Pco, tparquet), no public production dataset, no Rust compilation, no CI evidence.
- The C API `ZSTD_compress` measurements differ by a few bytes from earlier CLI-wrapped PRN1 results because the wrapper/frame settings differ. This table uses the same libzstd API, version and level for both sides; it should not be mixed with the CLI table as if byte-identical.

## 7. Next experiments that could make this a field-level contribution

### 7.1 Group by affinity, not first-seen order

Use a frozen training window to derive key affinity from co-access, temporal alignment, shared field innovations and service topology. Partition keys into groups under explicit constraints. Evaluate:

- query-affinity grouping (keys requested together);
- compression-affinity grouping (keys with correlated residuals);
- topology-aware grouping (service/dependency neighborhoods);
- random grouping and numeric-order grouping as negative controls.

The grouping model and its metadata must be included in frame size. Fit only on training windows; evaluate on later unseen windows and changed topology. Do not leak evaluation query patterns into the group assignment.

### 7.2 Jointly select group size, child layout and dictionary scope

For each candidate partition, compare PRN1 row-state and PCL1 field-columnar children, then compare independent-frame, group-frame, shared-dictionary and shared-context options. Count all directory, dictionary, selector and checkpoint bytes. Start with a small deterministic candidate set and exact complete-frame comparison. An offline oracle can set an upper bound, but a streaming/online policy must be evaluated separately with its compute budget and regret.

### 7.3 Add query semantics to the benchmark

Generate and then freeze representative query traces: one key, a small key cohort, a complete service, a time range, and a service plus time range. Report bytes fetched/decompressed, keys touched, groups touched, rows replayed, CPU time and p50/p95/p99 latency under cold and warm cache. Include scan queries so an index cannot win by optimizing only selective reads.

### 7.4 Bounded replay and corruption domains

Add checkpoints inside long per-key children, directory-level and group-level checksums, and recovery bounds. Test single-bit mutations in header, directory, each group, child slice and checkpoint. Verify that unrelated group reads remain valid and that corrupt metadata fails closed.

## 8. Industry comparison and claim threshold

PZG2 should be compared as a storage/container design, not as a replacement for telemetry transport. Mandatory storage baselines remain:

- Zstandard whole-frame PRN1/PCL1 and raw/normalized payloads;
- Apache Parquet with tuned Zstandard and ALP where numeric fields qualify;
- Pco/Pcodec and tparquet on exactly matched columns and complete file metadata;
- Gorilla/Chimp/Chimp128/Elf where their exact-bit semantics and data domain match;
- PZC1 independent per-series compressed children and PZG2 grouped children.

OpenTelemetry Arrow/OTAP is a distinct transport/dataflow comparison: compare ingest throughput, CPU/core, memory, batching, backpressure, loss and network bytes, not just persisted file size. ALP is now available in the Parquet/Arrow ecosystem and is not optional in a serious numeric compression evaluation. Zstandard's seekable format is prior art for independently decompressible frames and frame indexes; PZG2 must differentiate on telemetry-aware grouping, query behaviour, corruption semantics or measured end-to-end trade-offs—not on the existence of grouped frames.

A credible superiority claim requires at least two independent production-derived telemetry families; identical rows, ordering, schema and lossless semantics; all metadata/index/dictionary bytes counted; reproducible versions and commands; exact decoded output hashes; paired intervals across datasets/seeds; and a non-dominated size/throughput/RSS/query-latency result. A target worth pursuing is at least 5% lower complete bytes than the strongest reproduced matched baseline on two independent families with paired 95% intervals excluding zero, while preserving exactness and a declared query-latency/resource budget. This is a target, not an achieved result.

## 9. Current conclusion

PZG2 establishes a working intermediate-granularity reference and a measurable compression/selectivity trade-off. It does **not** beat whole-frame PRN1 on these synthetic fixtures. The strongest next step is not more codec complexity in isolation: it is **query-conditioned and correlation-aware key grouping with measured partial-read behaviour**, combined with PCL1/PRN1 child choices and bounded replay. That is the path to a differentiated result beyond known columnar codecs and seekable compression primitives.