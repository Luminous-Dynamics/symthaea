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
- open-set rejection diagnostics: calibrate a nearest-similarity acceptance threshold on clean known cues and absent-target cues, freeze that threshold, then report held-out false-reject rates for noisy known cues and false-accept rates for absent-target cues;
- optimized-versus-scalar top-1 oracle checks at every query-noise rate on 17- and 257-record fixtures, plus an assertion that each noise fixture flips exactly the requested number of bits.

Run on an exact revision:

    cargo test -p symthaea-core test_binary_hv_direct_pack_matches_reference_encoders
    cargo bench -p symthaea-core --bench hdc_compositional_workload

Criterion output measures host execution time for the clean representative query only. Separate diagnostic lines print semantic_hit_at_1, sampled median target rank, and the minimum observed target-versus-impostor similarity margin for the fixed eight-query sample at every corruption level. The noise schedule flips exactly floor(DIM × rate) bits using a deterministic coprime-stride pattern. The open-set diagnostic calibrates a maximum-similarity acceptance threshold using eight clean known-record cues and eight absent-target cues, then tests eight distinct known record IDs and eight new unknown-value cues. It does not retune the threshold at each noise level. These small in-benchmark counts are pilots, not a population-level error-rate guarantee; increase calibration and held-out sets before reporting stable false-accept/false-reject rates.

The corpus and query cues are fully resident during query timing. The build group measures deterministic role/value encoding and record bundling, not cold storage, file I/O, OS page faults, or an accelerator transfer. Query latency and build cost must remain separate; do not call setup free.

## Open-set acceptance: a crucial distinction from exact top-1

Nearest-neighbour search always returns a winner when the corpus is non-empty. That does **not** imply the query belongs to any stored record. The Rust benchmark now produces role-bound query atoms whose value was not used in any corpus record, calibrates an acceptance threshold from clean known and absent-target query sets, and reports:
- calibration false-reject rate for known cues;
- calibration false-accept rate for absent-target cues;
- held-out false-reject rate when known cues have 0%, 1%, 2%, 5% or 10% bit corruption;
- held-out false-accept rate on absent-target cues, without threshold retuning.

The threshold operates on the nearest BinaryHV similarity score and accepts when the score is at least the threshold. Calibration and test examples have distinct target IDs / unknown value seeds. The in-benchmark pilot uses eight known and eight unknown calibration examples and eight held-out examples of each class; these counts are deliberately labeled too small for a stable population-level guarantee.

Independent Python model pilot (not Rust execution and not the branch's BLAKE3 vector generator): for each of N = 256 and 1,024 and bundle arities K = 3, 7, 15, I calibrated the threshold on 32 clean known cues and 32 absent-target cues, then evaluated 48 held-out known cues and 48 held-out absent-target cues. The clean calibration false-reject and false-accept rates were both zero in these synthetic trials. At 5% corruption, known-cue false-reject rates were 54.2% / 64.6% for K=3 at N=256 / 1,024; 4.2% / 8.3% for K=7; and 8.3% / 2.1% for K=15. At 10% corruption, rates were 93.8% / 100% for K=3, 20.8% / 41.7% for K=7, and 29.2% / 18.8% for K=15. No absent-target query was falsely accepted in these 48-query-per-setting synthetic tests. Those zeros do not prove zero real-world false accepts: both query classes are generated from the same clean synthetic distribution, and the query counts are limited. The key observation is that a threshold calibrated on clean inputs can reject many genuinely related but corrupted cues, even when the top-1 search algorithm remains exact. This calls for an explicit quality/abstention trade-off, not a looser exact-search contract.

## Independent model and evidence boundary

A separate NumPy-based Python pilot (not Rust execution) tested N = 256, eight sampled query/target pairs, the same role-bind / majority-bundle semantics, and the same deterministic bit-flip schedule. In every arity/noise cell, sampled semantic hit@1 was 8/8. The minimum target-versus-strongest-impostor similarity margins were:

| Bundle arity K | 0% noise | 1% noise | 2% noise | 5% noise | 10% noise |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.23383 | 0.2287 | 0.2237 | 0.2094 | 0.1825 |
| 7 | 0.13812 | 0.1334 | 0.1286 | 0.1193 | 0.1060 |
| 15 | 0.08746 | 0.0851 | 0.0822 | 0.0773 | 0.0684 |

The margin declines both as more facts are bundled and as cues become noisier, even though all eight sampled targets remain top-1 in this small fixture. This illustrates why a single hit@1 number can conceal shrinking robustness margins. It does not estimate population-level recall, cover the 1,024–16,384 corpus sizes, or reproduce the Rust branch's BLAKE3-based seed generator. It is not a Rust source test or a performance measurement.

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
- open-set distribution shift, including unknown values with partial overlap to stored components rather than only independent unseen values;
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
- - Open-set acceptance is reported separately from top-1 exactness, and its threshold is frozen before held-out/noisy testing; never call a nearest-neighbour result “accepted” solely because it is the nearest candidate.
- Call a method an industry improvement only after a same-byte, same-workload baseline comparison.
- Call it energy-superior only after direct device measurement with the same successful retrieval definition.

This is a workload-definition and benchmark addition, not a claim of speedup, model quality, SOTA, or energy savings.
