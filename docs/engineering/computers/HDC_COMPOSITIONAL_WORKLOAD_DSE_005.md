# HDC-COMPOSITIONAL-WORKLOAD-DSE-005 — retrieval over bound-and-bundled records

Status: workload benchmark authored; Rust compilation and Criterion execution have not been performed in this environment.

## Why a new workload is needed

The current exact-search experiments mostly use independent pseudo-random BinaryHV vectors, plus deliberately correlated or localized-tail controls. Those are useful for isolating memory traversal and pruning, but they do not represent a common HDC task: construct a memory record by binding role/value pairs, bundle those facts into one record vector, and retrieve the record from one known constituent cue.

A kernel can return the exact nearest vector and still retrieve the wrong record for the intended task. This experiment records both outcomes separately:

- **Exact nearest-neighbor correctness:** the optimized scan agrees with an independent byte-XOR/popcount oracle, including lowest-ID tie behavior.
- **Semantic retrieval hit@1:** the nearest record is the record from which the query atom was drawn.
- **Target rank and margin:** rank of the intended record and its similarity margin over the strongest impostor.
- **Query latency:** full exact scan against a resident corpus.
- **Encoding/build cost:** separately measured record construction from role-bound facts.

A semantic miss is not an exact-search bug. It can indicate that the representation, bundle capacity, dimension, or corpus scale does not preserve the desired association strongly enough.

## Workload implementation

File: **crates/core/symthaea-core/benches/hdc_compositional_workload.rs**

Cargo target: **hdc_compositional_workload**

Each record contains K independently generated facts:

    fact(slot, record) = role(slot) bind value(slot, record)

The facts use BinaryHV XOR binding, and the record is the majority bundle of the facts. The query cue is the first role-bound fact, which is guaranteed to have contributed to its target record. K is odd so majority bundling has no bit-level tie. Values and roles are generated deterministically with distinct seed ranges. This is a reproducible synthetic HDC workload, not a claim that it substitutes for production Symthaea memory traces.

The sweep covers:

- bundle arity K = 3, 7, 15;
- corpus size N = 256, 1,024, 4,096, 16,384;
- 16,384-bit BinaryHV records (2,048 bytes/record);
- deterministic query samples spread across each corpus;
- a fixed corruption sweep of 0%, 1%, 2%, 5% and 10% exact query-bit flips, reported as task-level retrieval robustness;
- exact optimized top-1 search against the resident record corpus;
- record-build benchmarks for the smaller 256 / 1,024 / 4,096 corpus sizes;
- an independent byte-wise XOR/popcount top-1 oracle on smaller non-power-of-two and ordinary corpora;
- a duplicated exact-match fixture that enforces lowest-candidate-ID tie behavior;
- optimized-versus-scalar top-1 oracle checks at every query-noise rate on 17- and 257-record fixtures, plus an assertion that each noise fixture flips exactly the requested number of bits.

Run on an exact revision:

    cargo test -p symthaea-core test_binary_hv_direct_pack_matches_reference_encoders
    cargo bench -p symthaea-core --bench hdc_compositional_workload

Criterion output measures host execution time for the clean representative query only. Separate diagnostic lines print semantic_hit_at_1, sampled median target rank, and the minimum observed target-versus-impostor similarity margin for the fixed eight-query sample at every corruption level. The noise schedule flips exactly floor(DIM × rate) bits using a deterministic coprime-stride pattern. Eight queries are a pilot sample, not a statistical claim about population-level recall; increase and hold out query seeds before reporting a stable quality estimate.

The corpus and query cues are fully resident during query timing. The build group measures deterministic role/value encoding and record bundling, not cold storage, file I/O, OS page faults, or an accelerator transfer. Query latency and build cost must remain separate; do not call setup free.

## Independent model and evidence boundary

A separate NumPy-based Python pilot (not Rust execution) tested N = 256, eight sampled query/target pairs, and the same role-bind / majority-bundle workload semantics. The sampled semantic hit@1 was 8/8 for each arity. The minimum target-versus-strongest-impostor similarity margin across those eight queries was 0.23383 for K = 3, 0.13812 for K = 7, and 0.08746 for K = 15. The margin declines as more facts are bundled, consistent with stronger superposition interference. This is a small synthetic pilot only: it does not estimate population-level recall, does not cover the 1,024–16,384 corpus sizes, and uses NumPy-generated data rather than the Rust branch's BLAKE3-based seed generator. It is not a Rust source test or a performance measurement.

No performance result is claimed by authoring this benchmark. The benchmark source has not been compiled or executed here because the authoring environment lacks Cargo/Rust. A successful source fetch and manifest registration establish only that the files exist at the named revision. Exact nearest-neighbor oracle checks are encoded in the target but remain unexecuted until the Rust target is run.

## How this changes the research program

### 1. Co-design the representation and retrieval algorithm

The earlier early-abandon and pivot-bound probes ask how to scan a given corpus faster. This workload adds the question that must precede optimization: does the representation itself produce a robust target margin at the required memory capacity? If the intended record's rank degrades as arity or corpus size grows, changing CPU tiles or popcount width will not solve the semantic problem. Test bundle arity, role reuse, dimension, sparse vs. dense encoding, and memory consolidation as separate representation variables before optimizing the kernel.

### 2. Test robustness, not just one lucky sample

The benchmark now includes a deterministic query-cue corruption sweep at 0%, 1%, 2%, 5%, and 10% bit flips. Next extensions should use multiple declared seeds and held-out query sets, then stress:
- partial or missing role/value facts;
- role reuse and record overlap;
- correlated records with shared components;
- stale or updated memories;
- open-set queries whose source record is not stored;
- corpus growth beyond the point where false nearest neighbors dominate.

Report target hit@1 / hit@k, target rank, score margin, and query-time p50/p95/p99. Include the exact top-1 oracle result separately from semantic target rank. For approximate search, measure recall at the same semantic task and report it as approximate; do not silently mix exact and approximate results.

### 3. Make an industry comparison meaningful

First export identical packed bytes and candidate IDs and compare against FAISS IndexBinaryFlat, which is an exhaustive Hamming search with popcount and batch-oriented locality optimizations. Its binary index guide describes exact flat search and separately lists inverted-file, HNSW and hash-based alternatives: https://github.com/facebookresearch/faiss/wiki/Binary-Indexes

Use the same vector dimension, byte order, corpus, query order, result count, exactness requirement, tie policy, cache residency, and setup boundary. Compare:
1. Symthaea's runtime-dispatched exact SIMD scan;
2. FAISS IndexBinaryFlat;
3. Symthaea's exact early-abandon, pivot-bound, tiled, and workload-aware alternatives;
4. FAISS approximate/indexed baselines on a separate recall-versus-latency frontier;
5. a hardware accelerator only after its software baseline and data-transfer boundary are measured.

For every method, record query latency distribution, query/s, build/index time, bytes per record including index metadata, semantic hit@1 / hit@k, exactness/recall, and hardware/CPU-feature identity. An accelerated method that returns wrong target records is not better just because it is fast.

### 4. Adopt the edge-benchmark discipline

MLPerf Tiny explicitly targets low-power platforms and defines workloads plus latency and energy measurement rules; its current working group highlights fixed reference tasks and energy measurement as part of the benchmark design: https://mlcommons.org/working-groups/benchmarks/tiny/

Symthaea should follow that discipline without pretending its own workload is an official MLPerf score:
- freeze workload seeds and release the generated corpus or its source digest;
- state cold vs. warm cache, single-query vs. batch mode, and memory residency;
- report p50/p95/p99 and sustained thermal conditions;
- measure energy on an identified device with calibrated equipment and a declared system boundary;
- report both joules per query and joules per successful semantic retrieval;
- retain raw output, toolchain versions, CPU feature flags, source SHA, and exact corpus/query digests.

Runtime, TDP, logical bytes visited, and modeled popcount counts are not energy measurements. The user's mobile/laptop target needs actual target hardware runs before any battery or energy-efficiency claim.

## Go/no-go gates

- The optimized top-1 ID agrees with the independent scalar Hamming oracle on every correctness fixture.
- Semantic hit@1 and target-rank measurements are reproducible across fixed seeds; report uncertainty when comparing approaches.
- Keep a search optimization only when it improves measured query latency on the compositional workload without a material regression on random controls.
- Keep a representation variant only when it improves the task-level quality/capacity or memory trade-off at an acceptable build/runtime cost.
- Call a method an industry improvement only after a same-byte, same-workload baseline comparison.
- Call it energy-superior only after direct device measurement with the same successful retrieval definition.

This is a workload-definition and benchmark addition, not a claim of speedup, model quality, SOTA, or energy savings.
