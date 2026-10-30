# HARDSEC-009 — Scalable SCOAP Rarity and Long-Range Cone Preservation

**Status:** initial Rust audit utility and known-answer fixture; graph partition/path-retention evaluator remains the next stage  
**Scope:** source-bound SCOAP metric ranking and reproducibility; no HDL parser, gate-level detector, million-node benchmark or path-preservation experiment is claimed  
**Issue:** [HARDSEC-009 #9273](https://github.com/Luminous-Dynamics/symthaea/issues/9273)

## 1. Why this workstream exists

Popryho, Pal and Partin-Vaisband, “Automated Hardware Trojan Insertion in Industrial-Scale Designs,” DATE 2026 (DOI [10.23919/DATE69613.2026.11539640](https://doi.org/10.23919/DATE69613.2026.11539640); [paper PDF](https://past.date-conference.com/proceedings-archive/2026/DATA/1294.pdf); [arXiv](https://arxiv.org/abs/2511.08703)) report seven graph designs spanning roughly 30,000 to 1.78 million nodes and 184,000 to 9.68 million edges. Their largest reported graph-generation case approaches 64.8 minutes and 140.5 GB peak memory.

The paper reports that HW2VEC/GNN4TJ parsed its smaller D1-D3 designs but failed on larger D4-D7 designs. Netwise and NetVGE scaled farther, but reported detection performance remained modest and exact trigger localization was difficult. The authors identify a key structural problem: partitioning and neighborhood sampling can sever long-range trigger-to-payload dependencies.

The work uses SCOAP CC0/CC1/CO metrics to prioritize rare trigger candidates and then analyzes reconvergent cones. Its own scope limits matter: sequential cuts do not capture all temporal reachability; equivalence checks constrain triggers inactive; the inserted payload templates are pass-through analogues rather than full malicious effects; and open designs are not production silicon. We use this paper as a scalability/methodology comparator, not as proof of production-chip coverage.

## 2. Research hypothesis

A graph pipeline should be judged on more than whether it can load a large netlist or rank rare nodes.

It must jointly preserve:
1. rare-region prioritization;
2. long-range connectivity between a trigger candidate and a downstream payload cone;
3. design/net/cone localization;
4. memory, latency and throughput within a declared resource envelope.

A million-node graph that loads successfully but loses the trigger-to-payload dependency during partitioning is not a successful industrial-scale detector.

## 3. SCOAP rarity audit: implemented first stage

The branch adds `examples/hardsec_scoap_rarity_audit.rs`, an opt-in, Rust-first audit utility. It consumes supplied per-node SCOAP metrics rather than attempting to parse Verilog or recalculate SCOAP.

The composite score is:

`R(n) = max(CC0(n), CC1(n)) + alpha * CO(n)`

The CLI uses fixed-point arithmetic (`score_milli`) and a manifest-declared `alpha_milli` from 500 to 1000. Its default comparison fixture uses alpha=1.0. This fixed-point representation avoids floating-point tie drift when sorting and emitting reproducible ranks.

### Input contract

The JSONL manifest consists of:
- one header record binding dataset identity, analyzer name/version/config digest, sequential-cut policy, split protocol, alpha and the design registry;
- one record per design node, with node ID, explicit metric status, CC0/CC1/CO when measured, and post-ranking evaluation label.

The header requires `record_order=design_id_then_node_id_ascending`. Records are streamed and processed one design at a time. That bounds working memory to the largest individual design rather than retaining the source JSONL text and all designs together.

Each design is ranked independently. There is no pooled SCOAP scale across different designs. Equal scores receive the same empirical midrank percentile. An entirely tied design therefore has a midpoint percentile, not a false 100th-percentile result.

### Data-integrity behavior

- Missing/unknown SCOAP metrics are represented explicitly as unknown and are not imputed.
- A measured all-zero SCOAP tuple remains a valid measured tuple; it is not conflated with unknown.
- Duplicate node IDs and out-of-order design/node records are rejected.
- Identical netlist hashes cannot cross train/validation/test partitions.
- Leave-one-base-family-out mode rejects a host family split across partitions.
- Optional `--evidence-root` verifies exact netlist bytes against SHA-256, rejects path traversal/symlinks, and binds each verified artifact into the receipt.
- Ranking output contains no labels. Labels are used only for post-ranking aggregate recall/precision summaries.
- Receipt is unsigned and its claim scope is limited to the supplied SCOAP metric integrity and per-design ranking audit.

### Reproduction command

```bash
cargo test --locked \
  --manifest-path tools/hardsec-scoap-harness/Cargo.toml \
  --features hardsec-scoap-audit \
  --example hardsec_scoap_rarity_audit

cargo run --locked \
  --manifest-path tools/hardsec-scoap-harness/Cargo.toml \
  --features hardsec-scoap-audit \
  --example hardsec_scoap_rarity_audit -- \
  --manifest tests/fixtures/hardsec/scoap-rarity-known-answer.jsonl \
  --ranking-output /tmp/scoap-ranking.tsv \
  --receipt /tmp/scoap-audit.json
```

The known-answer fixture contains three designs across train/validation/test. It includes a trigger candidate, payload label, benign nodes, an unknown-label node, an unknown SCOAP row, and a measured all-zero SCOAP row. The fixture lets us verify that:
- alpha=1 produces score_milli=12000 for max(CC0,CC1)=2 and CO=10;
- percentiles are local to each design, not pooled by absolute SCOAP score;
- ties have equal midrank percentiles;
- unknown metric rows remain unknown;
- measured zero scores remain measured;
- the output ranking omits labels.

## 4. Next stage: long-range path-retention benchmark

The initial utility is not a graph partitioner. The next implementation should operate on source-bound graph artifacts and add independent path-retention controls.

Compare full-graph reference, fixed-radius GNN, neighbor sampling, hard partitions, hierarchical COI summaries with explicit ports, and path/reachability-preserving summaries. For each transformation, preserve:
- source net IDs and node identity;
- cross-partition edges and boundary port identity;
- trigger and payload endpoint identity;
- path length and reconvergence metadata;
- sequential cut/state-dependency information;
- dropped/approximated/unknown edges.

Required measurements:
- edge retention;
- trigger-to-payload path recall by path-length bucket;
- boundary-closure errors;
- cone localization precision/recall/IoU;
- rare-tail candidate recall at the 99th, 95th, 90th and 80th percentiles;
- peak RSS/VRAM, graph build time, ranking/inference/localization latency;
- nodes/s, edges/s and memory per million edges.

Path retention must be checked against exact edge/path fixtures. Node overlap alone is not proof of semantic path preservation.

## 5. Proposed performance gates

These are research targets, not achieved results or standards:
- process a 1M-node / 5M-edge profile without OOM on a declared workstation budget;
- retain at least 95% of labeled trigger-to-payload paths under a predeclared memory cap on held-out fixtures;
- improve rare-tail trigger recall over the strongest matched baseline without exceeding a predeclared false-positive/analyst-review budget;
- publish exact net/cone localization, not only node saliency;
- report generalization by held-out host, host family, trigger template family and graph scale;
- use independent designs for uncertainty, never the node count as the number of independent trials.

The DATE paper's largest graph and memory figures are comparator data, not measurements of Symthaea. We have not run a million-node workload or established that the local GPU/host can process it.

## 6. Industry comparison

OpenTitan is a useful assurance-practice comparator: its DV methodology combines simulation, formal property verification, coverage closure and explicit security countermeasure checks. Symthaea should complement this with scalable ranking, path-preservation and reviewable evidence—not substitute for formal verification or qualified silicon validation.

References:
- OpenTitan security docs: https://opentitan.org/book/doc/security/
- OpenTitan DV methodology: https://opentitan.org/book/doc/contributing/dv/methodology/index.html
- NIST IR 8615 (2026): https://doi.org/10.6028/NIST.IR.8615

## 7. Current validation boundary

The Rust utility has 15 authored unit tests in the current branch. A focused workflow has been added to run rustfmt, compile and execute the utility in a nested Cargo workspace, validate the JSONL fixture/receipt schemas, and run the deterministic known-answer fixture.

Those tests are authored, not yet reported as passing. No external SCOAP analyzer was run, no DATE source netlists were downloaded, no graph partitioner was implemented, and no path-retention result has been measured.

The next meaningful result is a passing isolated Rust/schema/known-answer run. Only then should we build the graph-path fixtures and compare partition strategies.
