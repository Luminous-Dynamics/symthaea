# HARDSEC-010 — Industrial-Scale Netlist Analysis with Path-Preserving Hierarchy

**Status:** first executable witness-path audit; no industrial netlist or model-scale benchmark run yet  
**Scope:** defensive graph partition assurance and scale evaluation; no Trojan payload construction or security certification  
**Related work:** [HARDSEC-005 lineage/source auditor](https://github.com/Luminous-Dynamics/symthaea/pull/9091), [HARDSEC-007 evidence packet and sequential witness validation](https://github.com/Luminous-Dynamics/symthaea/pull/9108), [HARDSEC-010 issue](https://github.com/Luminous-Dynamics/symthaea/issues/9245)

## 1. Why graph scale is a security property

Popryho, Pal and Partin-Vaisband, “Automated Hardware Trojan Insertion in Industrial-Scale Designs,” DATE 2026, DOI [10.23919/DATE69613.2026.11539640](https://doi.org/10.23919/DATE69613.2026.11539640), evaluate seven designs from about 30K nodes / 184K edges to 1.78M nodes / 9.68M edges. The paper reports a largest-case parsing time of 64.7 minutes and 140.5 GB AST peak memory. Its GNN4TJ configuration processes the smaller D1-D3 graphs but fails on D4-D7; Netwise and NetVGE scale further, although rare-tail performance and overall accuracy remain limitations.

A particularly important failure is structural: partitioning or local-neighborhood sampling can remove long-range trigger-to-payload relationships. A graph pipeline may then report a low-risk local region because the evidence connecting trigger, state, and observable effect was discarded during preprocessing.

**Claim boundary:** the paper's generated patterns are function-preserving Trojan-like stress patterns and explicitly are not full malicious payload effects. They are valuable for scale, rarity, and structural stress, but are not proof of real-world malicious behavior or silicon transfer.

## 2. Paradigm shift: hierarchical, path-preserving analysis

The research hypothesis is that large netlists can be analyzed with bounded local memory if the global representation preserves dependency paths across partitions.

Proposed architecture, to be tested rather than assumed to win:

1. **Streaming ingest:** parse node/edge/cell attributes and calculate local summaries without materializing a fully expanded tensor.
2. **Global testability index:** SCOAP CC0/CC1/CO or source-supported approximations, fanout, reconvergence, sequential distance, and interface observability.
3. **Hierarchy with explicit boundaries:** partition by design hierarchy or bounded cones while retaining named boundary ports and cross-partition dependency edges.
4. **Long-range path ledger:** preserve selected witness paths as compact ordered edge references crossing partitions.
5. **Adaptive refinement:** score coarsely first, then expand only candidate regions and their boundary context.
6. **Independent evidence lane:** validate reachability, activation, and payload observation separately from detector scores.

The critical invariant: no path may be reported as preserved unless all its supplied edges exist in the partitioned graph and every cross-partition edge is represented by an explicit bridge.

## 3. Initial Rust implementation

Adds `examples/hardsec_partition_path_audit.rs`, registered behind Cargo feature `hardsec-design-audit`. It is source-independent and can be run on safe fixtures without a proprietary SoC.

The four strict TSV input formats are:

```text
original-edges.tsv:
edge_id    src_node_id    dst_node_id

partitioned-edges.tsv:
edge_id    src_node_id    dst_node_id    representation

node-partitions.tsv:
node_id    partition_id

witness-paths.tsv:
path_id    sequence_index    edge_id    src_node_id    dst_node_id
```

Fields are tab-separated, headers and field counts are exact, IDs are non-empty, witness sequence indices begin at zero and are contiguous, and each path is an ordered connected edge sequence. The partitioned edge's representation must be `internal` when both endpoints share a partition and `bridge` when they cross partitions.

The audit checks:
- all original graph edge endpoints have partition assignments;
- all partitioned edges exist in the original graph with identical endpoints;
- internal/bridge classification matches the node partition map;
- every supplied witness edge exists in the original graph and retains its endpoints;
- each supplied witness edge remains in the partitioned representation;
- per-path and aggregate preserved/lost edge counts, bridge counts, partition counts, and lost edge IDs;
- SHA-256 for all four input files.

If a supplied path is severed, the tool writes a deterministic receipt with `result=fail` and exits non-zero. The receipt is unsigned and has claim scope `input_integrity_and_supplied_witness_path_preservation_only`.

It does **not** discover missing witness paths, prove all possible paths survive, infer a trigger/payload relationship, assess detector performance, or certify a design. A zero lost-edge count means only that the paths supplied to this tool were preserved.

Run it using the isolated harness:

```bash
cargo test --locked \
  --manifest-path tools/hardsec-design-audit-harness/Cargo.toml \
  --features hardsec-design-audit \
  --example hardsec_partition_path_audit
```

Nine unit tests are authored for a preserved cross-partition path, lost witness edge with fail receipt, incorrect bridge classification, edge endpoint mutation, disconnected witness, non-contiguous path index, duplicate partition-edge ID, missing node partition assignment, and receipt overwrite. They are not yet reported as passing.

## 4. Benchmark matrix and target gates

Scale ladder, where authorized design inputs exist:

| Graph size | Primary question |
|---:|---|
| 0.2M edges | ingest and receipt overhead |
| 0.5M edges | baseline local-memory scaling |
| 1M edges | first million-edge regime |
| 2.5M edges | partition and bridge growth |
| 5M edges | adaptive refinement cost |
| 10M edges | declared industrial-scale target |
| 25M edges | out-of-core operation |
| 100M edges | stress-test envelope, if source and hardware allow |

Record input nodes/edges, fanout and reconvergence distributions, largest component, partition count, bridge count, supplied witness count and maximum path span. Measure p50/p95 parse, index, partition, coarse-scoring and refinement time; peak RSS; temporary-disk footprint; bytes per node/edge; CPU/GPU utilization; and energy only when measured using a calibrated instrument.

Report detector metrics separately: rare-tail recall at frozen false-positive budgets, exact gate/cone localization where gold labels exist, path preservation, and worst-stratum results. Compare HW2VEC/GNN4TJ, Netwise, NetVGE, a streaming SCOAP-only baseline, the path-preserving hierarchy, and Symthaea HDC/multiresolution candidates on matched inputs and compute conditions.

Proposed project gates (not achieved results or industry standards):
- process a 10M-edge graph within 32 GB peak RSS on a declared machine without silent sampling;
- provide an out-of-core path beyond the in-memory envelope, recording every retained/discarded item;
- preserve 100% of supplied positive and negative witness-path edges or fail the preservation gate;
- improve rare-tail recall/localization at a fixed false-positive budget without breaking path preservation;
- publish p50/p95 and worst-stratum scaling across at least five graph sizes and three independent design families before claiming broad scalability;
- do not claim physical or industrial transfer from function-preserving synthetic patterns alone.

## 5. Next work

1. Pass the focused formatter and regression tests for both the lineage auditor and this path auditor.
2. Add a deterministic streaming edge-list ingest/index benchmark with input/output hashes and bounded-memory measurements.
3. Build larger synthetic *structural graph fixtures* that are not deployable Trojan payloads; measure memory and path preservation as nodes, edges and partition boundaries grow.
4. Obtain rights-qualified DATE challenge inputs and exact generation/parser code before reproducing paper timings or detector metrics.
5. Add witness-path retention to an adaptive-refinement prototype, then compare with the same local GNN/KGE baselines and the same split groups.

## 6. Industry comparison and claim ceiling

The competitive deliverable is not just a larger graph model. It is a reproducible graph-ingest, partition, path-ledger and evidence pipeline whose memory/latency costs are measured and whose dropped dependencies are visible. That supports industry-style design review and verification work; it does not replace formal verification, silicon validation, or a platform-specific security assessment.