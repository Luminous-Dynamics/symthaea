# HARDSEC-010 — Streaming Graph Accounting and Trigger-Path Retention

**Status:** Rust streaming audit prototype and pre-registered graph-integrity protocol  
**Scope:** exact edge accounting and retention of caller-declared reference paths  
**No claim:** this tool is not a Verilog parser, Trojan detector, proof of RTL semantics, or industrial-scale performance result.

## 1. Why graph scale needs a different benchmark

A DATE 2026 paper, “Automated Hardware Trojan Insertion in Industrial-Scale Designs” (DOI 10.23919/DATE69613.2026.11539640), describes gate-level designs up to approximately 1.78 million nodes and 9.68 million edges. It reports a peak parser/graph-build memory footprint around 140.5 GB for its largest design and describes failure modes in representative graph-learning detectors on generated industrial-scale challenge instances.

The study highlights a basic mismatch: academic graph benchmarks often evaluate classification on relatively small designs, while large designs expose memory blow-up, long-range dependencies, reconvergence, and partition-boundary effects. Sampling or partitioning can sever trigger-to-payload or source-to-observation paths; once severed, a downstream model may mistake missing context for absent logic.

These published resource numbers are the motivation for a stress target, not measurements from this repository.

## 2. Paradigm shift: path preservation is an independent property

The graph pipeline should not be evaluated only on node/edge counts or downstream model F1. For every declared source-to-observation reference path, it must do one of two things:

- preserve every exact edge and allow the path to be reconstructed; or
- explicitly report one or more edges as excluded, with a reason, so the path becomes a visible coverage limitation.

A missing edge that is neither represented nor listed in an exclusion ledger is a hard failure. An embedding or sampled graph must never silently convert unknown connectivity into a negative security fact.

## 3. First Rust artifact

Adds `examples/hardsec_graph_path_retention_audit.rs`, registered behind Cargo feature `hardsec-graph-audit`.

It performs a streaming three-way merge over:
1. the original source edge stream;
2. the reduced/partitioned represented edge stream;
3. an explicit exclusion ledger.

The declared reference paths are held in memory; the full edge streams are not. The algorithm requires each stream to be sorted lexicographically by edge ID, enabling deterministic merge-join without materializing the entire adjacency graph.

### Input TSV contracts

**Source and represented edge files** have this exact header and five columns:

```text
edge_id<TAB>src<TAB>dst<TAB>relation<TAB>partition_id
```

- Edge IDs must be unique and strictly increasing.
- Source and represented files must use stable source edge IDs.
- The source endpoints and relation of every retained edge must exactly match its represented counterpart.
- The represented partition ID may differ from the source partition ID; it records the partition in the reduced representation.
- This generic auditor does not derive or validate pin roles/Verilog semantics. The caller supplies the relation field from a pinned parser or fixture.

**Exclusion ledger** has this exact header:

```text
edge_id<TAB>reason
```

Exclusion IDs must be unique and strictly increasing. Every excluded edge must exist in the source stream and have a non-empty reason. An edge cannot be both represented and excluded.

**Reference path file** has this exact header:

```text
path_id<TAB>step_index<TAB>edge_id<TAB>src<TAB>dst
```

- Path IDs are strictly increasing by path group.
- Step indices start at 1 and are contiguous within each path.
- The path must be endpoint-contiguous.
- Every path edge must exist in the source graph with exactly matching endpoints.

### Audit outcomes

For the complete source graph:
- every source edge must appear exactly once in the represented graph or the exclusion ledger;
- represented edges absent from the source are rejected;
- exclusions absent from the source are rejected;
- silent drops are rejected;
- endpoint or relation changes are rejected.

For each declared path:
- `retained`: every edge was preserved with an exact source identity and endpoint match;
- `cut_and_reported`: at least one edge was explicitly excluded with a recorded reason.

The receipt reports the path endpoints, edge counts, retained/excluded counts, partition crossings, and every individual path step with edge ID, source/destination, disposition, represented partition, and exclusion reason. It also records each input path locator alongside its SHA-256 digest. It is unsigned and has an explicit claim ceiling.

## 4. Implementation and reproduction

Cargo feature and example:
- `hardsec-graph-audit`
- `hardsec_graph_path_retention_audit`

Run the isolated test suite:

```bash
cargo test --locked \
  --manifest-path tools/hardsec-graph-audit-harness/Cargo.toml \
  --features hardsec-graph-audit \
  --example hardsec_graph_path_retention_audit
```

Run a real audit:

```bash
cargo run --locked \
  --manifest-path tools/hardsec-graph-audit-harness/Cargo.toml \
  --features hardsec-graph-audit \
  --example hardsec_graph_path_retention_audit -- \
  --source-edges /data/source-edges.tsv \
  --represented-edges /data/partitioned-edges.tsv \
  --exclusions /data/exclusions.tsv \
  --reference-paths /data/reference-paths.tsv \
  --receipt /data-receipts/graph-retention.json
```

The tool hashes each input from a stable open file handle, re-hashes after parsing to detect input mutation between the hash and audit passes, rejects symlinks, and writes a new receipt outside all input directories without overwriting.

The first version has 23 authored unit tests for full retention, explicit cuts, silent drops, orphan/duplicate IDs, edge mutation, path discontinuity/absence, sorted stream rules, partition crossings, shared path edges, empty exclusion reasons, file-hash receipts, symlink inputs and receipt-path safety. Compilation/CI results are tracked separately; authored tests must not be described as passing until the targeted job confirms them.

## 5. Scale benchmark plan

After the correctness oracle passes, measure deterministic size tiers:

| Tier | Nodes | Edges | Status |
|---|---:|---:|---|
| S0 | 10,000 | 50,000 | smoke / unit-scale |
| S1 | 100,000 | 500,000 | medium |
| S2 | 500,000 | 2,500,000 | large |
| S3 | 1,000,000 | 5,000,000 | primary scale target |
| S4 | 2,000,000 | 10,000,000 | stretch |
| DATE-sized | 1,780,000 | 9,680,000 | literature-motivated stress target |

Measure:
- peak resident set size (RSS);
- input scan / merge throughput in edges per second;
- total wall time and per-file digest time;
- reference-path metadata memory;
- partition crossings and path-retention rate;
- bytes read/written and temporary disk usage.

**Proposed engineering gates, not results:** 100% of reference paths retained or explicitly cut-and-reported; 100% source edges accounted for; zero false reachability or silent loss on exact fixtures; and a 10x RSS reduction against a full in-memory baseline on a matched large graph. A 1M-node / 5M-edge run under 16 GiB RSS is a proposed target, not an established capability.

Do not claim performance at million-node scale until a run records exact graph counts, hardware, RSS, timing, parser/configuration digests and audit receipt.

## 6. Next research step: hierarchical sparse relations

Once this integrity layer is validated:
1. add a streaming index from net/cell identity to typed local edges;
2. add a compact cross-partition relation table preserving all cut edges and provenance;
3. compare full adjacency, external-memory CSR, hierarchical component/local neighborhoods and bounded sampling;
4. query the exact path fixtures through each representation;
5. report any abstraction that cannot reconstruct a path as unknown/abstain;
6. only then compare RF/XGBoost, graph models and Symthaea HDC representations on the same retained-path fixtures and lineage-held-out splits.

HDC is not presumed to solve the scaling problem. The falsifiable hypothesis is that stable region/relation bindings and sparse cross-partition retrieval can retain useful long-range context at lower memory cost without sacrificing path-query correctness.

## 7. Claim ceiling

A passing receipt establishes exact accounting of the supplied edge IDs and retention/exclusion of caller-declared reference paths. It does not prove the reference path set is complete, validate Verilog/netlist semantics, establish all possible trigger-to-payload paths, prove Trojan absence, measure detector quality, or certify an industrial chip.