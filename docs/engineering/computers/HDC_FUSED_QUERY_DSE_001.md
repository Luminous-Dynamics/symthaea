# HDC-FUSED-QUERY-DSE-001 — exact search, fusion, and data-movement frontier

Status: research protocol plus software benchmark draft. No performance result has been collected from this change.

## Decision to make

For the canonical 16,384-bit BinaryHV, does a fused exact similarity-to-argmax path improve complete query cost over Symthaea's best available software path, after preprocessing, memory residency, launch/transfer overhead, and result materialization are included?

Do not widen the custom RTL just because wider words look attractive. First identify whether the real bottleneck is bitwise compute, score reduction, allocation, representation conversion, memory traffic, or accelerator orchestration.

Parents: [SOC-DESIGN-001 #8459](https://github.com/Luminous-Dynamics/symthaea/issues/8459), [HDC RTL equivalence #5808](https://github.com/Luminous-Dynamics/symthaea/issues/5808), workload baselines #5805, and [oracle-driven RTL sweep #8463](https://github.com/Luminous-Dynamics/symthaea/pull/8463).

## Current code baseline and new software slice

The existing `crates/core/symthaea-core/benches/hdc_simd_compare.rs` already compares SIMD, scalar, packed, binding, similarity, Hamming distance, bundling, and search. Its search corpus sizes are 64, 256, and 1,024 and the timed result is the maximum score, not an explicitly checked winning ID.

The companion `crates/core/symthaea-core/benches/hdc_fused_argmax.rs` adds:
- candidate counts 1, 16, 64, 256, and 1,024;
- exact top-1 results carrying both ID and score or distance;
- deterministic lowest-index tie resolution across all implementations;
- pre-timing correctness checks on seeded vectors, duplicate winners and zero/one vectors;
- one-pass fused score reduction versus a deliberately materialized score vector;
- SIMD similarity, Hamming-distance minimization, packed search with persisted encodings, and packed preparation-plus-search;
- separate packed query/corpus preparation timings;
- compare historical per-bit bipolar expansion with direct `BinaryHV` → little-endian `u64` packing via `PackedBipolar::from_binary_hv`, both persisted and setup-inclusive.

Run after checking out the exact source revision:

```sh
cargo bench -p symthaea-core --bench hdc_fused_argmax
```

This microbenchmark measures software execution time only. It does not measure joules, physical memory traffic, FPGA behavior, or the custom RTL. The packed conversion routine is a comparator for the current public conversion path, not a claim that all packed encoders must use that implementation.

## Paradigm shift: fuse the semantic endpoint

For exact top-1 retrieval, the output is an ID and score, not a 16,384-bit intermediate vector. A candidate architecture should therefore evaluate:

```text
candidate words -> XOR -> popcount/Hamming -> exact score -> argmax reducer
                                                        -> winning ID + score
```

Keep the query resident in local memory/registers where possible, stream candidate words, and return only the winner. Make query residency explicit; CPU caches and accelerator SRAM/DMA have different costs.

One BinaryHV occupies 2,048 bytes. A search over N candidates has at least 2,048 × N bytes of logical candidate payload; a non-resident query adds another 2,048 logical bytes per query. The hardware protocol should not write an intermediate 2,048-byte XOR vector per candidate if the only requested output is a scalar score and winner ID.

Important counter-hypothesis: a good software implementation may already fuse XOR/popcount in registers and keep the query cache-hot. That is why we must measure against the current SIMD baseline instead of assuming the software path writes temporary vectors. The likely hardware opportunity is the whole data path—query reuse, streaming, score reduction, batching and launch/transfer amortization—not bit width in isolation.

## Measurement phases

### Phase A — software truth and break-even

Use the new Criterion benchmark to compare identical frozen query/candidate bytes. Preserve the full terminal output, Criterion estimates, repository HEAD, source diff, rustc/Cargo versions, target CPU and enabled ISA features, build profile, OS/kernel, thermal/power policy, and whether candidate representations are already packed.

Measure kernel-only and preparation-inclusive paths independently. Report latency per query, candidates/s, candidate count, exact winner ID, exact score/distance, corpus representation bytes, and setup cost. Run multiple fixed seeds and retain raw results. Do not report energy from runtime or infer power from TDP.

The software comparison is informative before the HDL tools become available. It must not claim an accelerator win.

### Phase A2 — exact early-abandonment as an algorithmic baseline

Before paying for a custom accelerator, test an algorithmic alternative that keeps exact top-1 semantics. Hamming distance is a sum of non-negative bit mismatches. While processing a candidate in fixed byte chunks, the partial distance is a lower bound on its final distance. Since candidates are visited in index order and ties preserve the earlier ID, a candidate can be abandoned as soon as its partial distance is greater than or equal to the best fully evaluated distance.

The new benchmark includes 16-byte and 64-byte chunk policies. Correctness is checked against the full exact scan. Two stress conditions are designed:
- a random-only deterministic corpus, where early abandonment may be weak because nearest candidates are not especially close;
- a deterministic near-match at the corpus midpoint, with approximately 10% of query bits flipped in a distributed pattern, exposing how an improved incumbent can make later candidates cheap to reject.

It scales through 4,096, 16,384 and 65,536 candidates (8, 32 and 128 MiB of logical candidate payload, respectively). These are host-memory workloads—not claims about a particular CPU cache, cold DRAM or physical I/O. Cache behaviour depends on the target machine and must be recorded. Untimed diagnostics count candidate bytes visited; timed variants compile out this counter path.

The exact pruning bound can lose against a straight SIMD/vectorized scan because branches, prefix traversal and early-exit divergence cost time. If it wins only on a planted easy-match corpus but regresses on random or realistic traces, treat it as workload-specialized rather than a universal replacement. A later extension should randomize near-match position over fixed seeds and add domain-derived corpus traces. This remains software-only and does not qualify the RTL.

Independent mathematical reference check (not Rust execution): 12 deterministic model seeds × random/planted conditions × 16/64/128-byte chunks matched the full-scan winner in all 72 cases. In a separate 4,096-candidate / 8 MiB traversal-count pilot, the random-only corpus visited 97.70% of candidate bytes with 16-byte chunks and 99.06% with 64-byte chunks. With one approximately 10%-distance target planted at the midpoint, the model visited 59.08% and 60.51%, respectively. This is a count of modeled bytes visited—not execution time, energy, a compiled benchmark, or a Rust-source test. A separate 4,096-candidate correlated-query model seeded the incumbent from the prior query's candidate: cold pruning visited 59.38% / 60.82% of candidate bytes, while the hinted exact search visited 20.40% / 21.89% (16-byte / 64-byte chunks), approximately 65.6% / 64.0% fewer modeled candidate-byte visits in that specific scenario. The exact top-1 was the midpoint candidate at distance 1,638 bits. This is encouraging algorithmic evidence but not a wall-clock, cache-traffic, or energy result; the gain depends on the prior result remaining a good candidate. Random-only inputs offer little work reduction. These figures should be corroborated by exact-head Rust execution and realistic query traces.

### Phase A3 — temporal-incumbent exact search

A useful cognitive workload often arrives as a sequence of related queries rather than independent random vectors. If the previous query's nearest candidate ID remains a strong candidate for the new query, it can initialize the best-known complete Hamming distance. The hint is not accepted as the answer: the algorithm still checks every other candidate fully or rejects it using its monotone partial-distance lower bound.

The new `hinted_early_abandon_hamming_argmin` benchmark path receives that prior candidate ID. It uses tie-safe pruning:
- partial distance strictly greater than the incumbent distance can always be pruned;
- equality can be pruned only when the candidate index is greater than the incumbent index;
- a fully evaluated equal-distance candidate replaces the incumbent only if its index is smaller.

That subtlety matters because changing candidate evaluation order must not change the public lowest-index tie rule. It also means this optimization is exact, unlike approximate nearest-neighbor shortlists.

The streaming test models temporal locality by storing a prior query at the midpoint of the candidate corpus, then making the current query a deterministic ~10%-bit perturbation of that prior vector. It compares a cold scan against the hinted scan, with winner-ID equality against the exact full scan. A random-only scenario is retained as the control. This experiment asks whether a temporally valid incumbent substantially tightens the bound; it does not presume that all production query streams are correlated.

Independent Python mathematical model check (not Rust execution): 2,100 cases across 64–2,048-bit synthetic vectors, 1–64 candidates, randomized hint IDs, duplicates, and multiple equal exact matches all returned the same lexicographic minimum `(Hamming distance, candidate index)` as exhaustive exact search. This checks the algorithm's tie/bound logic on small synthetic dimensions; it does not compile, test or benchmark the Rust implementation.

### Phase A4 — batched exact queries and corpus-major traversal

A single-query latency benchmark does not establish the best architecture for bursts of queries. FAISS documents that batch queries improve efficiency by amortizing index-memory access, and that CPU/GPU crossovers depend on index size, batch size and whether the index/query is already resident. Reference: [FAISS CPU/GPU comparison](https://github.com/facebookresearch/faiss/wiki/Comparing-GPU-vs-CPU) and [FAISS performance guidance](https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster).

Run the batch-order sweep independently of the fused/streaming search bench:

```sh
cargo bench -p symthaea-core --bench hdc_batch_search
```

The new independent `hdc_batch_search` Criterion target compares:
- query-major traversal (finish the entire corpus scan for query 0, then query 1, etc.);
- corpus-major traversal (visit each corpus candidate, update every query's incumbent before advancing);
- batch sizes 1, 4, 8 and 16;
- corpus sizes 256, 1,024, 4,096 and 16,384 (0.5, 2, 8 and 32 MiB logical corpus payload);
- independent random query batches and deterministic correlated-query batches;
- strict exact top-1 winner ID and score equality, including duplicate-candidate tie controls.

For B queries and N candidates, both software paths still compute B×N exact similarities. The key hypothesis is not fewer mathematical comparisons: it is better locality and reduced memory-system/launch overhead when a corpus is scanned for multiple queries. Do not infer physical DRAM traffic from logical bytes or source-level loop order. The actual target's cache/memory counters or device telemetry are required to make that claim.

The next system-level extension is a measured adaptive policy that chooses direct CPU execution vs batched corpus-major execution vs an accelerator based on corpus size, query batch size, data residency, latency SLO and transfer cost. Train/tune this policy only on calibration traces, and evaluate the selected rule on held-out corpus sizes, query correlations and device classes. Do not assume GPU wins: FAISS's published guidance explicitly notes that a small index / small batch frequently favors CPU, whereas large indices and batches are more likely to amortize GPU overhead.

### Phase A5 — candidate broadcast across query lanes (conditional hardware branch)

If Phase A4 shows a repeatable batch-size benefit, the next architecture question is *query lanes versus candidate lanes*. A compute tile could keep B query hypervectors resident, stream each candidate vector once into a broadcast path, and update B independent exact Hamming/argmax states. The arithmetic remains B×N comparisons, but a hardware dataflow could amortize external-corpus reads across B queries; this is not automatically true for the CPU loop-order benchmark, which still performs B×N software similarities.

Proposed staged extension after the single-query RTL is formally/simulator qualified:
1. Select one best-supported datapath width from Phase B's area/timing/cycle Pareto frontier instead of sweeping width again.
2. Sweep query lanes B ∈ {1, 2, 4, 8} while holding candidate lanes at 1. Preserve a single shared candidate input port / broadcast boundary where feasible.
3. Keep candidate lanes fixed initially, then test candidate lanes {1, 2} only if query broadcasting is not the dominant area/fan-out/timing cost.
4. Compare query-resident and streaming-query modes, corpus transfer bytes per batch, per-query energy, end-to-end batch throughput and p95 latency at batch sizes {1, 4, 8, 16}.
5. Require exact top-1 equality for every query, independent argmax state, deterministic tie policy, and no cross-query state leakage. Formally assert candidate handshake accounting, valid/ready backpressure, lane-ID isolation and that every accepted query produces exactly one result.

Do not jump to a 180-configuration factorial sweep. Use Phase A4 and Phase B evidence to select widths and narrow the query-lane matrix. Vortex v3.0's documented global-to-local transfer acceleration and asynchronous scheduler are reminders that transfer/control interfaces are architecture features in their own right; Symthaea's RTL must measure them, not borrow Vortex's performance claims.

Decision gate: if batch-aware data reuse does not improve directly measured bytes read per query or joules/query at equal results, retain the simple single-query tile and focus on CPU SIMD / memory-layout improvements.

### Phase B — qualified RTL architecture sweep

Prerequisite: the software-oracle generator and all 12 cases in #8463 compile and simulate at its exact head. Once that gate is met, add a distinct fused similarity-to-argmax primitive; do not treat word-level XOR/popcount success as qualification of the full search datapath.

Initial sweep: 4 word widths × 3 candidate-lane counts × 3 corpus sizes = 36 configurations.

| Dimension | Values | Why |
|---|---|---|
| Word datapath | 64, 128, 256, 512 bits | 256, 128, 64, or 32 chunks per 16,384-bit vector for a single word lane |
| Candidate lanes | 1, 2, 4 | Trade area / broadcast fanout / memory pressure for concurrency |
| Candidates per query | 16, 256, 1,024 | Expose small-query overhead and throughput scaling |
| Held fixed initially | Query resident; exact XOR → popcount → argmax; lowest-index ties | Avoid a factorial sweep before a first Pareto frontier exists |

Only after a first trustworthy frontier should we ablate query residency, batch size, non-resident transfer, score materialization, and command/DMA overhead. If the software baseline already wins on realistic low-batch tasks, stop or redesign instead of adding lanes by default.

### Phase C — synthesis, board, and energy

For every RTL candidate, preserve source/config hash, simulator and synthesis tool versions, constraints, clock target, resource or cell counts, critical path, and receipt. Distinguish simulated cycles, synthesis timing estimates, FPGA observations, and physical-device measurements. Open-PDK results do not predict advanced-node mobile silicon without a validated scaling model.

Only an identified physical target and calibrated measurement setup can establish joules per query. Measure complete query energy with explicit idle/baseline subtraction, repetition and uncertainty. Report energy separately for transfer, compute and orchestration where instruments permit; do not add estimates that share overlapping boundaries.

## Representation optimization

The historical packed path expands 16,384 bits into a temporary vector of 16,384 signed bytes, then repacks those values into 256 words. The new `PackedBipolar::from_binary_hv` path translates each eight-byte little-endian block directly into a u64, avoiding the intermediate i8 vector and per-bit re-encoding loop. This is a representation-conversion optimization, not a new HDC algorithm. It should reduce conversion overhead and temporary working set, but the gain must be measured. A unit test compares direct packing with the established bipolar encoder on zero, all-one and seeded random vectors; Rust execution remains pending.

## Correctness contract

- The winner index must be identical to canonical BinaryHV exact top-1.
- The score/distance must agree with the canonical result; approximate search is a separate experiment.
- Ties choose the lowest candidate index, independently of lane scheduling or completion order.
- Test empty input policy explicitly in a production API; this Criterion fixture simply avoids empty corpora.
- Include zero and all-one vectors, identical candidates, exact match at multiple indices, random deterministic corpora and non-power-of-two corpus counts in the next fixture expansion.
- An incorrect result is a failed candidate, not a throughput result.

## Do not reinvent a standard population-count instruction

Bit population count is already available in several general-purpose ISA families. The ratified RISC-V vector bit-manipulation extension `Zvbb` specifies `vcpop.v`; selected x86 CPUs expose AVX-512 `VPOPCNTD/Q` or `VPOPCNTB/W` forms. These are not present on every SKU or target, so feature detection and exact target identity matter. References: [RISC-V vector bit-manipulation specification](https://docs.riscv.org/reference/isa/unpriv/vector-crypto.html) and [Intel instruction-set manual](https://cdrdv2-public.intel.com/671110/325383-sdm-vol-2abcd.pdf).

Therefore, "we built a popcount RTL block" is not itself a competitive claim. Use the best supported CPU path on the same machine as the first baseline. A credible custom accelerator opportunity must come from improved end-to-end dataflow—fused XOR/popcount/reduction, query reuse, transfer amortization, predictable latency/energy, or capabilities not reached efficiently by the target ISA. The baseline record must include detected CPU features and actual compiled target flags; never assume AVX-512 or RISC-V vector instructions are available on a phone.

## External exact-search baselines: do not benchmark in a vacuum

FAISS is the most useful first external comparator because it has explicit binary-vector indexes using packed byte arrays and Hamming distance. Its documented choices include:
- `IndexBinaryFlat`: exhaustive exact scan optimized for CPU popcount;
- `IndexBinaryIVF`: coarse inverted-file retrieval;
- `IndexBinaryHNSW`: graph-based approximate search;
- `IndexBinaryHash` / `IndexBinaryMultiHash`: hash-bucket families, including a multi-index hashing approach related to exact Hamming-neighbor search.

Reference: [FAISS binary index guide](https://github.com/facebookresearch/faiss/wiki/Binary-Indexes) and [FAISS IndexBinaryFlat API](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexBinaryFlat.html). The original Multi-Index Hashing work showed exact sublinear search for compact 64/128/256-bit codes at billion-item scale, under its specific code distributions and search-radius conditions: [Norouzi, Punjani & Fleet](https://arxiv.org/abs/1307.2982). That result does not establish the same behavior on Symthaea's 16,384-bit vectors.

For an honest comparison, first run FAISS `IndexBinaryFlat` over identical fixed bytes and candidate IDs, with the same exact top-1/tie semantics. Then compare MIH and HNSW separately: MIH's radius/substring work can grow sharply at high dimension or large Hamming radius, while HNSW is an approximate candidate-generation path unless exactness is independently established. Sweep recall/latency only as a separate approximate-search frontier; never compare an approximate result to exact throughput without reporting recall and quality.

Proposed external baseline matrix:
1. Symthaea canonical SIMD flat scan.
2. FAISS `IndexBinaryFlat` exact flat scan.
3. Symthaea early-abandon and temporal-incumbent exact scan.
4. FAISS multi-index hash / MIH on a declared radius or exact top-k workload.
5. FAISS `IndexBinaryHNSW` at published recall targets.
6. Only then, Symthaea fused RTL / FPGA with identical results and explicit transfer boundaries.

Keep vector dimension, bit order, corpus bytes, query order, top-k value, tie policy, cache and setup boundary matched. Report build cost, bytes/vector, index overhead, p50/p95/p99, QPS, exact recall, and measured joules when available. A research improvement can win on 16K HDC and lose on the 256-bit compact-code workload; report each regime honestly rather than claiming a universal SOTA.

## Industry reference and how we compete

- **MLPerf Client** provides a useful end-to-end discipline for local AI and agentic work: quality gates and complete task duration rather than theoretical peak numbers. It is not a directly comparable HDC score. https://mlcommons.org/benchmarks/client/
- **MLPerf Tiny v1.4 (July 2026)** demonstrates the edge-device comparison discipline we should emulate: fixed workloads and quality targets, measured latency and calibrated energy, plus attention to always-on duty cycle. Its published analysis reports substantial accelerations from dedicated accelerators on specific matched tasks; those numbers are not transferable to HDC without a matched workload. https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/
- **MLPerf Tiny** offers fixed-reference workloads, accuracy conditions and optional energy evaluation for constrained edge devices. Use its matched-workload/evidence discipline when designing later phone-class power tests. https://mlcommons.org/benchmarks/inference-tiny/
- **Vortex GPU 3.0** is a relevant open architecture-research reference for memory staging, local-memory use, scheduling, simulation and synthesis flow. It is not a drop-in mobile GPU or a ready-made HDC accelerator. https://github.com/vortexgpgpu/vortex/releases

First beat the best reproducible software baseline at the *same exact task*. Then compare a qualified RTL or FPGA implementation with matching input, output, data residency, and timing boundaries. Only then compare against commercial devices with matched workload and quality.

## Proposed decision gates — not results

A custom block advances only if it achieves either:
- at least 1.5× higher end-to-end exact-query throughput, or
- at least 30% fewer directly measured joules per successful exact query,

while returning identical results and showing no material regression in p95/p99 latency, sustained thermal behavior, or system correctness. The thresholds are predeclared go/no-go targets, not claimed achievements.

Publish a Pareto frontier over exact queries/s, joules/query when measured, p95 latency, area/resources, clock, data movement, preparation cost and error rate. Reject candidates that win only on a narrow kernel timing while losing after preparation or transfers.

## Current validation boundary

The benchmark and protocol have been authored but not compiled or executed in this authoring environment. The existing environment lacks Cargo/Rust, Nix and Verilator; the new source's API/type correctness, Criterion execution and benchmark numbers remain unverified until an exact-head Rust run. No RTL simulation, synthesis, FPGA, energy, PPA, SOTA or product-superiority claim is made here.
