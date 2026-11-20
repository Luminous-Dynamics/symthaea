# CDL-Bench: public telemetry datasets and query-aware evaluation (v1)

**Date:** 2026-11-20  
**Tracking:** [CDL-Bench-015 #12970](https://github.com/Luminous-Dynamics/symthaea/issues/12970), [CDL-Bench-014 #12964](https://github.com/Luminous-Dynamics/symthaea/issues/12964)  
**Status:** dataset source metadata verified; no external archive downloaded in this environment; acquisition and codec evaluation pending

## Why this changes the benchmark

The current PRN1/PCL1/PZC1/PZG2 results establish useful codec mechanics on deterministic synthetic traces. They do not establish that the architecture is better for actual observability workloads. The next milestone is a repeatable public-data pipeline with explicit schema, run-level holdouts, query traces and all-cost accounting.

The benchmark question changes from "which frame is smallest?" to:

> Which exact, verifiable telemetry container gives the best storage, ingest, selective-read, scan, recovery and resource trade-off under a frozen workload?

This avoids optimizing the byte ratio while ignoring the work users actually ask a telemetry platform to perform.

## Public dataset candidates

### LO2v2: primary multi-run microservice observability target

Source: https://zenodo.org/records/18937117  
DOI: https://doi.org/10.5281/zenodo.18937117  
Dataset licence: CC-BY-4.0.  
Version: v2, published 2026-03-10; data collection executed in February 2025.

The record describes 115 runs, each containing 54 tests. Each test contains per-container and Locust log files; cAdvisor and NodeExporter host/container metrics (raw JSON, with CSV exports available); and per-container traces in CSV. Test order and durations are randomized, durations range from 20 to 180 seconds, and pauses from 1 to 5 seconds. These properties make the dataset useful for regime changes, anomaly cohorts, temporal drift and cross-modal query workloads.

Available sizes (as published by Zenodo):
- Full raw LO2v2: 65.6 GB compressed, approximately 640 GB uncompressed.
- Metrics-only: 1.3 GB compressed, approximately 6.3 GB uncompressed.
- Example-run metrics package: 77.6 MB compressed.
- Run index: 31.0 MB; describes run/test order and times plus log/trace/metric counts and sizes.
- Log package: approximately 2.9 GB compressed.
- Replication source package: approximately 19.7 MB.

**Practical acquisition order:** first the 77.6 MB example-run metrics archive for parser and pipeline smoke; then the 1.3 GB metrics-only archive for across-run evaluation; then logs/traces only when the query workload needs cross-modal joins. The small example is a bring-up fixture, not evidence of run-level generalization.

### Alibaba microservice time / call-rate / replica trace: compact companion

Source: https://zenodo.org/records/14245634  
DOI: https://doi.org/10.5281/zenodo.14245634  
Version: v3, published 2024-11-29.

Zenodo lists an 8.2 MB CSV and published MD5 `148f0b2c9fc14bf65a25ae8d7ef3786b`. The dataset describes a filtered extraction from ten of the original 24 Alibaba microservice trace parts over a 30-second interval, with timestamps, microservice/container identity, replica count, microservice time and call rate.

The prose description has ambiguous column numbering (it says the first two columns are timestamps while also referring to a microservice ID in column 2). Do not freeze a parser schema from prose alone. Inspect the downloaded header and rows, compare with the DOI metadata, record the exact parser version, and fail closed on unexpected column counts/types. The dataset should be labelled as a filtered trace, not the full Alibaba production trace.

## Acquisition and data integrity protocol

The execution container could resolve neither Zenodo host DNS nor download the archive during this research pass. The Zenodo records and published checksum metadata were inspected, but no data bytes were acquired, and no local checksum is claimed. Do not turn this into an implicit "dataset downloaded" claim.

The future acquisition command must:
1. Download to a temporary path with HTTPS and a bounded retry/timeout policy.
2. Verify published MD5 when supplied, then calculate and freeze SHA-256 locally.
3. Preserve the exact source archive; never normalize in place.
4. Record URL, DOI, version, licence, bytes, source hashes, acquisition timestamp, host/tool versions, extraction command and normalized output hashes.
5. Validate archive member paths against traversal, reject symlinks/special files and enforce decompressed-size limits before extraction.
6. Emit a machine-readable manifest; if any required digest or schema check fails, stop before benchmark execution.

## Canonical normalized event model

Keep source relations distinct. A normalized row should preserve at least:
- event timestamp as an integer with an explicit unit and source clock domain;
- source/run/test/file identifiers and original row ordinal;
- service, container, host and process identity where available;
- metric name or stable metric ID, typed numeric value, unit, and null/presence state;
- trace/span identifiers and parent relationships in a separate relation;
- log timestamp, severity, body hash/body reference and trace/span linkage in a separate relation.

Do not flatten logs, metrics and traces into one denormalized row merely to help the codec. The benchmark should distinguish metric-series compression from event-log and trace-span compression, and should include a joined query as a workload, not as the only storage schema.

For exact floating-point tracks, preserve IEEE-754 bit patterns, including signed zero and NaN payloads where present. For normalized integer counters, preserve the original integer domain and units. Any semantic conversion (for example, scaling or timestamp normalization) belongs in a versioned preprocessing manifest and must be reversible or separately reported.

## Split and holdout design

- **Bring-up:** the one-example-run LO2 package; parser correctness only.
- **Development:** selected complete runs; tune schemas and candidate policies.
- **Validation:** disjoint complete runs and test cohorts; freeze policy parameters.
- **Blind test:** held-out complete runs plus unseen error-task/test cohorts where possible.
- **Temporal shift:** later runs, changed test order, duration/pause variation and changed service/container population.
- **Cross-source replication:** the compact Alibaba trace as an independent data family.

Never random-shuffle rows. Keep each run's chronology and event ordinal. Freeze the full split manifest and hashes before model/policy training. Do not use held-out query traces to learn group membership or tune group size.

## Query benchmark: make the workload first-class

Freeze a query trace with at least these families:

| Query family | Example semantics | Main cost to report |
|---|---|---|
| Single series | one container / metric for a time interval | bytes fetched and decompressed; point lookup latency |
| Service cohort | all container metrics for one service | groups touched; unrelated series decoded |
| Host cohort | all services on one host | fan-out, bytes, p95 latency |
| Time window | all metrics in a bounded interval | index pruning, rows replayed |
| Service + time | one service over one interval | intersected index selectivity |
| Anomaly drill-down | pre/post window around a known test/error | checkpoint replay distance |
| Cross-service slice | service, dependent services and aligned window | topology grouping and join cost |
| Full scan | all metrics in order | throughput and CPU |
| Ingest with late arrivals | append plus bounded-late event | reorder/spill cost and correctness |

Report cold-cache and warm-cache separately. For every query, record requested series/rows, groups touched, unrelated keys decoded, bytes read, bytes decompressed, rows replayed, index bytes, CPU time, wall-clock p50/p95/p99 and peak RSS. For scans, report records/s and MB/s. For ingestion, report records/s under backpressure and under bounded memory.

## Comparator matrix and industry boundary

### Storage and numerical codecs
- Raw typed source and Zstandard over the same canonical bytes.
- PRN1 and PCL1 whole-frame Zstandard; PAX1 raw-size selection; PZC1 independent child frames; PZG2 grouped frames.
- Apache Parquet with tuned Zstandard and ALP where the actual numeric domain qualifies. Parquet's 2026 ALP adoption makes this a mandatory numeric baseline; Arrow Rust 60.0.0 includes ALP encoder/decoder support.
- Pco/Pcodec on matched numeric columns; its public benchmark reports real-world numeric dataset comparisons, including taxi and air-quality data. Reproduce in the same host/hardware/threads.
- Gorilla, Chimp/Chimp128 and Elf only where their exact value domain and bit-preservation contract matches; explicitly count exceptions.
- tparquet/PCo where buildable and schema-aligned.

### Transport/dataflow
OpenTelemetry Arrow (OTAP) is not a persisted-file codec competitor. It is the correct comparator for ingest/transport/dataflow overhead. Reproduce its batch-size, backpressure, filtering, saturation and multi-core scaling workloads. Its current public benchmark description includes approximately 100k logs/records per second under several one-core scenarios and targets more than 0.90 scaling efficiency up to 16 cores. CDL-Bench should report corresponding ingress/output, CPU, RAM, network bytes and dropped records, but must not equate transport throughput with compressed-file size.

### Seekability prior art
Zstandard's official seekable format already combines independent frames with a jump table. Therefore, the contribution cannot be "we have chunks and an index." The differentiated question is whether typed telemetry semantics, query-affinity and topology-aware grouping, per-series projection, bounded replay and localized integrity produce a measurably better workload Pareto frontier.

## Targets and claim threshold

These are targets, not achieved numbers:
- Exact decoded hash and 100% bitwise preservation for the exact track.
- >=5% reduction in complete storage bytes against the strongest reproduced matched baseline on two independent telemetry data families, with paired 95% intervals excluding zero.
- No more than 5% p95 regression on the declared service/time queries relative to the best queryable baseline, unless a clearly quantified storage or operational benefit justifies the trade.
- >=100k records/s/core is a useful transport-oriented target because it matches the scale of current OTAP benchmark scenarios; publish the real measured result and hardware, do not assume the target.
- Report p50/p95/p99, RSS and CPU; a size win that exceeds the memory budget or makes drill-down unacceptably slow is not a win.
- For segmented/grouped designs, publish the full Pareto surface over group size, grouping policy, compression level, and query family.

## Paradigm shift: optimize the workload graph, not isolated series

The next novel direction is **query-conditioned, topology-aware storage layout**:
1. Model access as a weighted graph whose vertices are typed series and whose edges are observed co-access, temporal alignment or service dependency.
2. Learn grouping only from an earlier training window; keep the grouping plan deterministic and include its metadata/model bytes.
3. Compare first-seen, numeric, random, query-affinity, compression-affinity and service-topology groupings.
4. Evaluate on future queries, unseen error cohorts and changed service topology.
5. Jointly tune group size, child layout (row or field-columnar), checkpoint interval and dictionary scope against an explicit storage/read/CPU/RSS objective.
6. Retain a simple policy fallback and report regret against a bounded offline oracle. The oracle is an upper bound, not a deployable result.

This shifts the unit of optimization from a scalar time series to a verified collection of related observations and the queries that consume them. It should be treated as a research hypothesis until validated on the public datasets.

## References

- LO2v2 dataset (Zenodo v2): https://zenodo.org/records/18937117
- Alibaba microservice trace (Zenodo v3): https://zenodo.org/records/14245634
- Apache Parquet ALP adoption: https://parquet.apache.org/blog/2026/09/22/alp-adaptive-lossless-floating-point-encoding-in-apache-parquet/
- Arrow Rust 60.0.0 ALP support: https://arrow.apache.org/blog/2026/09/29/arrow-rs-60.0.0/
- Pco/Pcodec public benchmark results: https://github.com/pcodec/pcodec/blob/main/docs/benchmark_results.md
- OpenTelemetry Arrow benchmark protocol/results: https://github.com/open-telemetry/otel-arrow/blob/main/docs/benchmarks.md
- OpenTelemetry API performance benchmark guidance: https://opentelemetry.io/docs/specs/otel/performance-benchmark/
- Zstandard seekable format: https://github.com/facebook/zstd/blob/dev/contrib/seekable_format/README.md
