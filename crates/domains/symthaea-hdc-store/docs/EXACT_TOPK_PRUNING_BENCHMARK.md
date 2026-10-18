# Exact top-k selection and Hamming early-abandon benchmark

Tracking: [HDC-STORE-TOPK-EXACT-001 #8009](https://github.com/Luminous-Dynamics/symthaea/issues/8009).

## Run

\`\`\`sh
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \\
RUSTC_VERSION="$(rustc --version)" \\
HOST_PROFILE="$(uname -m) / <CPU model>" \\
STORAGE_PROFILE="<filesystem and physical device model>" \\
cargo run -p symthaea-hdc-store --release --example exact_topk_pruning_bench -- \\
  --vectors 16384 --queries 16 --top-ks 1,10,100 --corpora random,clustered,mixed \\
  --chunk-sizes 32,64,128,256 --rounds 30 --seed 7 \\
  --csv /tmp/exact-topk-pruning.csv
\`\`\`

The benchmark uses only existing Rust dependencies and creates its synthetic stores in a temporary directory. Choose a fresh CSV path for each run; the program refuses to overwrite the CSV or \`.meta.txt\` sidecar. Begin with smaller corpora and scale deliberately.

## Compared algorithms

1. **Canonical full sort:** the public \`HdcStoreReader::scan_similar\` implementation, the correctness oracle.
2. **Bounded heap, full distance:** compute exact Hamming distance with the existing architecture-dispatched \`BinaryHV::hamming_distance\` kernel, retain only k candidates, then sort those candidates by distance ascending and ID ascending.
3. **Bounded heap with chunked early abandonment:** compute Hamming distance in 32/64/128/256-byte chunks. Since partial Hamming distance cannot decrease, abandon a candidate when its lower bound exceeds the current worst accepted distance. If the lower bound equals the threshold, abandon only if the candidate's ID cannot win the ascending-ID tie-break. The reader iterator's ascending-ID contract is checked on this path.

This is exact candidate pruning, not approximate retrieval. The integer ranking order is equivalent to the store's similarity ordering for fixed-width BinaryHV values, and the benchmark validates ordered result IDs and exact score bit patterns against the canonical method for every timed sample.

## Corpus distributions

- **random:** independent deterministic BinaryHV vectors and independent queries. A critical negative control; random distances concentrate around half the 16,384-bit space, so pruning may not save much work.
- **clustered:** vectors are deterministic small Hamming perturbations around four fixed centers, with queries at those centers.
- **mixed:** a mix of clustered vectors and random distractors, queried at cluster centers.

These synthetic modes are diagnostic, not a claim to represent the deployment distribution. Add captured or privacy-safe real traces before making product decisions.

## Measurements

The raw CSV records each query/round/algorithm, elapsed time, corpus size, k, number of payload bytes **algorithmically examined**, vectors pruned and a checksum of exact results. The summary reports queries/s and p50/p95/p99 per-query latency.

- Algorithmic bytes examined count only corpus payload bytes touched by the candidate loop. They are not memory-controller traffic, page faults or physical SSD/NAND reads.
- The canonical reference searches are run before timings, so this first comparison measures repeated/warm mapped-store behavior; cache state is not controlled.
- Record compiler, CPU feature set/model, OS/kernel, filesystem/device, page size, power mode and source SHA. Compare corpora that fit and exceed LLC when the capacity is known.
- Each candidate uses a full-size heap only for k entries. The purpose is to compare O(k) retained selection with the baseline's O(N) score list and full sort; allocations/RSS remain worthwhile follow-up measurements.
- This PR deliberately does not bundle query batching, prefetch instructions, or a new SIMD kernel. Those are separate dimensions.

## Research grounding

Faiss's binary flat index uses exhaustive Hamming distance with popcount, and documents query/database batching to avoid cache misses: [Faiss Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-Indexes). This reinforces the value of benchmarking data movement as well as arithmetic, but does not validate this exact top-k implementation or predict its performance.

## Proposed gates (not achieved)

- 100% ordered top-k ID and exact f32 score-bit agreement with the canonical oracle for random, clustered and mixed workloads; include duplicate vectors, equal-distance ties, k=0 and k=N.
- Bounded heap maintains no more than k score candidates per query.
- At k=10, target at least 1.5x exact queries/s on at least one named workload/host without more than 5% regressions for small k. This is a falsifiable target, not a forecast.
- Early abandonment must reduce algorithmic payload bytes substantially on clustered/mixed corpora and is expected to deliver much less on the random control. If not, retain full distance.
- At least 30 samples per configuration for reportable comparisons, with raw output retained.
- Before production adoption, add unit/property tests to the library path, run exact-head tests and validate behavior on supported SIMD dispatch variants. CI status skipped is not a pass.

No production algorithm or persistent format is changed by this benchmark example.
