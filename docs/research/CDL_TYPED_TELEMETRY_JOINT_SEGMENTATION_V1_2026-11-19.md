# CDL-Bench: typed telemetry records and joint segmentation research note

**Date:** 2026-11-19  
**Status:** Research design and benchmark specification; no codec result is claimed  
**Scope:** Proposed CDL-Bench-009. This note does not change the existing F64C/F64S format or production Symthaea behavior.

## Executive finding

The current exact-f64 track is a useful bit-preserving transform laboratory, but it is not yet a production-telemetry benchmark. It represents one ordered vector of binary64 values. Its RLE transform groups adjacent identical scalar u64 patterns and then XOR-codes one value per run.

That must not be described as reproducing record-level RLE. The Dynatrace SIGMOD 2026 Industry Track study analyzes application-monitoring telemetry whose records often remain unchanged for long periods; it combines record-level RLE with XOR codecs and a block-level selector. The authors report approximately 23% compression-ratio improvement and 34% speed improvement over standalone methods, and more than 13% improvement over the best single RLE-enhanced compressor for their selector. These are the paper's measurements, not CDL-Bench results.

The next research increment should therefore move from a scalar stream to a typed, multi-field record stream. It should test repetition at three distinct levels: identical scalar values, repeated field vectors within one series, and repeated complete records across time. It should also make segmentation a first-class variable rather than fixing chunk lengths to 64/256/1024.

## What is already implemented—and what it does not establish

The current PR #12828 branch contains four exact-bit f64 transforms (raw, XOR-varint, reusable XOR window, scalar RLE+XOR), two byte entropy models, and fixed-chunk eight-candidate selectors (offline best-of-eight, full historical and sparse-probe). All frame headers, checksums and child frames are counted. These are meaningful controls.

They do not yet establish:
- record-level RLE across typed multi-field telemetry rows;
- preservation and compression of series/entity identity, timestamps, nullability and schema;
- cross-field or cross-series redundancy;
- variable-length segmentation jointly selected with a codec;
- matched performance against a reproduced ALP implementation;
- results on the production telemetry distribution studied by Dynatrace.

The offline best-of-eight selector remains an upper oracle over its fixed chunk boundaries, not an online or adaptive-segmentation result.

## New hypothesis: typed records expose a different redundancy surface

Represent a telemetry observation as a canonical record with a stable schema, for example:

`(timestamp, entity_id, series_id, cpu_f64, memory_f64, request_rate_f64, status_code, missing_bitmap)`

The exact schema is dataset-specific. The format must preserve:
- integer timestamps exactly, including irregular deltas and duplicate timestamps;
- entity and series identity (dictionary IDs are permitted only when dictionary bytes and ordering are encoded);
- field types, nulls, NaN payloads, signed zero and all IEEE-754 bits;
- original record order, or an explicit reversible permutation if sorting is tested;
- missingness separately from NaN and any application-specific sentinel values.

Do not silently assume that records are regularly sampled, that equal values mean equal records, or that records from different series are interchangeable.

### Three distinct run models

1. **Scalar run:** consecutive identical values in one typed field. For f64, equality is by exact u64 bit pattern, not floating-point equality.
2. **Field-vector run:** a run in which the selected value fields for one entity/series repeat, while timestamp and perhaps counters continue to advance. The repeat grammar must explicitly encode which fields are repeated and how timestamps advance; irregular timestamps cannot be synthesized from an assumed cadence.
3. **Whole-record run:** consecutive canonical records are identical under an explicitly declared equivalence relation. If timestamp is excluded from the repeated payload, its delta stream is still encoded and charged.

Run-length coding is not universally beneficial. Every run form needs a literal escape and a break-even rule that includes run length, type/tag overhead, timestamp cost and checksums. Short runs should remain literals when that is cheaper.

## New hypothesis: jointly choose segmentation and codec

The current fixed chunks make boundary placement exogenous. Production data can contain long idle periods, sharp bursts and mixed regimes that do not align with 64/256/1024 boundaries.

Add two distinct policies:

### A. Offline joint-segmentation oracle (research upper bound)

For each possible segment start, consider a declared set of legal segment lengths and the available transform/codec candidates. Let (C(i,j,k)) be the complete encoded size for rows ([i,j)) under candidate (k), including child frame, headers, dictionaries, checksums and any model side information. Dynamic programming can find the minimum total:

`DP[j] = min_{i,k}(DP[i] + C(i,j,k) + boundary_cost)`

Constrain maximum lookahead/segment length and publish the candidate set. The oracle must not be presented as streaming. Its purpose is to quantify how much fixed boundaries leave on the table and to set a transparent upper bound for a declared candidate family.

### B. Bounded-compute decoder-synchronous policy

Build a deployable policy that makes decisions using either:
- statistics of the already-decoded prefix and past candidate costs; or
- a cheap, deterministic feature vector computed from the current block, with the selected child codec/length written into its self-describing header.

The second is a block-local selector, not a causal or prequential predictor. Its selection features, configuration/model bytes and child metadata must be counted. After decoding, the decoder can recompute the same features from the reconstructed block and verify the declared choice. The policy must use integer/fixed-point decisions where cross-platform determinism matters.

Candidate features worth ablation:
- scalar and whole-record run-length histograms;
- fraction of zero XOR residuals;
- leading/trailing-zero span distribution;
- delta and delta-of-delta magnitude histograms;
- decimal-roundtrip / ALP eligibility rate;
- per-field entropy and sparsity;
- missing/null density;
- cross-field repetition and same-entity persistence;
- block-local regime-change statistics.

Do not add all features at once. Begin with a cheap feature-only classifier and compare it with full candidate probing on the same blocks. Evaluate both ratio and decision compute.

## Public data track

Use at least two public production-derived sources before any claim about production monitoring:

1. **Latest Alibaba microservice time/call-rate/replica dataset (preferred compact fixture).** Zenodo record 14245634 (v3, published 2024-11-29) describes a filtered subset of Alibaba runtime traces. The documented columns include two timestamps, microservice ID, container ID, replica count, microservice time and call rate. The record contains an 8.2 MB CSV, and documents that ten of the original 24 parts were extracted and filtered. Freeze the downloaded file hash and parser/schema version before using it: https://zenodo.org/records/14245634
2. **Earlier Alibaba microservice time/call-rate artifact (legacy fixture).** The older Zenodo record 8310376 v1 describes a different 180,000-row file with microservice times and call rates over a 30-second interval. The record itself points to a newer version, so do not treat its CSV as the latest v3 dataset or merge the two artifacts without a versioned manifest: https://zenodo.org/records/8310376
2. **Alibaba production microservice trace (broader service population).** The official clusterdata repository describes 2021 traces covering more than 20,000 microservices over 12 hours, including runtime metrics and call dependencies: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021

A scale-oriented third source is Google's 2019 Borg cluster trace. Its official documentation describes eight cells for May 2019 and CPU-usage histograms per five-minute period; the full release is roughly 2.4 TiB compressed and is accessed through BigQuery, so use a documented sampled query before attempting full-scale runs: https://github.com/google/cluster-data/blob/master/ClusterData2019.md

The Dynatrace application-monitoring dataset described in the SIGMOD 2026 Industry Track paper is the most directly relevant target, but the public publication page does not provide an open dataset download. Record it as a non-reproduced production reference unless the authors provide access. Do not fabricate a public replica.

For every imported dataset, freeze a manifest with URL/DOI, license, version, source hash, download date, schema, parser version, raw row count, filtered row count, dtypes, missing-value semantics, timestamp units/order and exact train/validation/test split. Preserve raw files and publish normalized-output hashes. No random row shuffling.

## Matched baseline matrix

### Mandatory floating-point / numeric baselines
- ALP (SIGMOD 2024) through its official implementation and/or Apache Parquet 2.14.0 / Arrow Rust 60.0.0 support.
- Gorilla, Chimp and Chimp128; Elf/ElfPlus and Patas where buildable.
- ACTF and Rabbit (2026) for the adaptive floating-point flag/significant-bit family.
- Pco/Pcodec for typed numerical sequences.
- Zstandard on raw bytes and, where fair, on the same normalized column/record representation.
- Current CDL-Bench scalar F64C/F64S transforms and selectors.

ALP is not a niche paper baseline anymore: Parquet format 2.14.0 includes ALP, and Arrow Rust 60.0.0 announced an ALP encoder/decoder. ALP should be run on decimal-like sensor and monitoring columns, with its exceptions and page metadata counted. Its own design is not expected to dominate arbitrary full-range IEEE-754 bit patterns.

Rabbit's 2026 paper reports an average 4.15% improvement over ACTF on 28 datasets; reproduce its implementation and data settings before adopting that number as a comparison. WZS (September 2026) reports adaptive segmentation and selection results across 43 datasets, but treat abstract-level numbers as literature claims until code/configuration/data can be reproduced.

### Required fairness rules
- Same raw records, schema, row/column order, block limits and lossless semantics.
- No dropping timestamps, identifiers, null bitmaps, exceptions or indexes from the numerator.
- Separate value-payload-only size from end-to-end file size; use end-to-end size for superiority claims.
- Report cold-start model/code/config costs and amortized costs at several stream lengths.
- Give all methods the same hardware, core/thread budget and memory ceiling; publish versions, flags and CPU details.
- Report random access and partial-decode costs; smaller whole-file output can be worse for query workloads.
- Keep exact-lossless and bounded-error tracks separate.

## Metrics and gates

### Correctness (hard gate)
- 100% byte/bit-exact reconstruction of all values, timestamps, IDs, nulls and schema.
- Corruption/truncation/overflow/noncanonical encoding rejection.
- Deterministic output for a frozen configuration across supported targets.
- Decoder output hash equals the normalized source hash.
- No labels, hidden change boundaries or test-derived parameters enter the bitstream/predictor.

### Compression and compute
Report per-dataset and macro aggregates:
- bits/value, bits/record and complete bytes/record;
- end-to-end ratio including schema/dictionaries/indexes/metadata;
- encode/decode throughput and p50/p95/p99 block latency;
- peak RSS, allocations, bytes copied and energy where measurable;
- segment count, run-length distribution, candidate switches and selection regret;
- number of candidate encodes and decoder re-encodes;
- cold-start and amortized break-even length;
- online adaptation observations/bits after regime changes.

### Suggested targets (not achieved results)
- First gate: complete-frame size within 2% of the offline joint-segmentation oracle at no more than two extra candidate encodes per emitted segment on the declared mixed telemetry suite.
- Research win: at least 5% lower end-to-end size than the strongest reproduced matched baseline on at least two independent dataset families, with a paired 95% interval excluding zero and no exactness failures.
- Stability: no more than 1% median size regression against the best fixed policy on stationary/incompressible controls.
- Operational trade-off: publish the full size/throughput/memory Pareto frontier; do not hide an expensive oracle behind a headline ratio.
- Production-readiness: include bounded memory, streaming behavior, corruption recovery, version negotiation and independent decoder validation.

These are research targets, not results or promises.

## Paradigm-shift opportunity and novelty boundary

Do not claim novelty for RLE+XOR, ALP, adaptive block choice, online expert mixing or adaptive segmentation individually; these are all prior art. The credible research question is whether a **schema-aware, record-structured codec portfolio with a decoder-verifiable, compute-budgeted joint segmentation policy** can outperform specialized per-column codecs on real telemetry without sacrificing exactness, streaming constraints or queryability.

A stronger follow-on is mechanism-aware multi-series prediction: encode reusable cross-field/service mechanisms and their innovations, but retain per-field noncausal baselines and interventions as negative controls. A mechanism that only wins observationally but fails after service restarts, autoscaling or configuration changes is not a useful compressor.

## References

- Dynatrace SIGMOD 2026 Industry Track: https://www.dynatrace.com/research/publications/analyzing-and-improving-floating-point-compression-for-application-monitoring-time-series/ and https://doi.org/10.1145/3788853.3803097
- ALP SIGMOD 2024: https://github.com/cwida/ALP and https://2024.sigmod.org/toc-1-4.html
- Apache Parquet ALP adoption (2026-09-22): https://parquet.apache.org/blog/2026/09/22/alp-adaptive-lossless-floating-point-encoding-in-apache-parquet/
- Arrow Rust 60.0.0 ALP support (2026-09-29): https://arrow.apache.org/blog/2026/09/29/arrow-rs-60.0.0/
- Rabbit (Symmetry 2026, 18(4), 558): https://doi.org/10.3390/sym18040558
- WZS (Information Processing & Management, September 2026): https://doi.org/10.1016/j.ipm.2026.104697
- Alibaba microservice dataset: https://zenodo.org/records/8310376
- Alibaba production microservice trace: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021
- Google Borg 2019 trace: https://github.com/google/cluster-data/blob/master/ClusterData2019.md
