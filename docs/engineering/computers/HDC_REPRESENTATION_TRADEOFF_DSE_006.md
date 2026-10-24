# HDC-REPRESENTATION-TRADEOFF-DSE-006 — prototypes versus component memory

Status: benchmark authored and registered; Rust compilation and Criterion timing remain unverified in this environment.

## Research question

A single bound-and-bundled BinaryHV is a compact record prototype, but a majority bundle can erase some information about its constituent facts. Keeping each role-bound fact increases payload storage but allows a query to match a constituent directly. This is a representation-level trade-off, not merely a different loop order.

The new target compares three retrieval paths over the same deterministic role/value workload:

1. Bundled prototype scan: one 16,384-bit (2,048-byte) majority-bundle per record, exhaustive exact top-1 by Hamming similarity.
2. Component scan: retain K role-bound facts for each record and rank a record by its most similar fact.
3. Hybrid exact-then-fuzzy: hash the 2,048-byte cue with a BLAKE3-derived 64-bit fingerprint, inspect its fingerprint bucket and verify full-vector byte equality. An exact hit returns the lowest matching record ID. A miss falls back to component scan, so noisy cues are still handled by fuzzy retrieval.

File: crates/core/symthaea-core/benches/hdc_representation_tradeoff.rs

Cargo target: hdc_representation_tradeoff

The fingerprint is only an index-routing hint: the implementation verifies the complete 2,048-byte vector before accepting an exact match. A truncated digest collision therefore increases bucket work but cannot itself create a false identity hit. The index is an in-memory benchmark structure, not a serialized cross-version format or a cryptographic authentication mechanism.

## Axes and reported measures

- bundle arity K = 3, 7, 15;
- record count N = 256, 1,024, 4,096;
- query corruption = 0%, 1%, 5%, 10% exact bit flips;
- clean-cue exact-index coverage and target hit@1;
- latency for bundled scan, component scan and exact-index-with-fuzzy-fallback;
- index build time and representation build time (the separate full representation build sweep is bounded at N = 256 / 1,024 to keep repeated Criterion allocations practical);
- logical payload and a lower-bound estimate for fingerprint/flat-ID metadata;
- independent scalar Hamming oracles for both scan strategies, a duplicate-ID tie contract, a forced fingerprint-bucket collision contract, and a contract for hybrid fallback equivalence.

Run on an exact branch revision:

    cargo bench --no-run -p symthaea-core --bench hdc_representation_tradeoff
    cargo bench -p symthaea-core --bench hdc_representation_tradeoff

The query benchmark reports candidates/records processed by strategy, not physical DRAM transactions or energy. The component scan performs K comparisons per record; interpret wall-time and throughput with that additional work in mind. The hybrid's clean path hashes one query and checks candidate-vector equality; its noisy path falls back to K-component scan. The complete index-build cost is measured separately.

## Payload arithmetic

A BinaryHV contains 2,048 bytes.

| Records N | Bundled prototype payload | Component payload K=3 | K=7 | K=15 |
|---:|---:|---:|---:|---:|
| 256 | 0.5 MiB | 1.5 MiB | 3.5 MiB | 7.5 MiB |
| 1,024 | 2 MiB | 6 MiB | 14 MiB | 30 MiB |
| 4,096 | 8 MiB | 24 MiB | 56 MiB | 120 MiB |

Component payload is K times the bundled prototype payload. These figures exclude allocation metadata and any exact index. A hybrid that retains both prototypes and facts stores both payloads, plus its index; do not advertise it as the component-only footprint. The emitted fingerprint/ID floor is a lower-bound model, not a total HashMap resident-size estimate.

## What this model tests—and what it cannot prove

The component representation retains the exact role-bound cue used to construct each record. Therefore a clean cue can have similarity 1.0 with its constituent while the bundled prototype offers a weaker match. The hash path handles such exact cues without scanning every record; the fuzzy path preserves retrieval when a cue is corrupted and no longer matches an exact fact.

A separate NumPy-based pilot (not Rust execution; independently generated random vectors) compared bundled-prototype and component scans for N = 256 / 1,024, K = 3 / 7 / 15 and 0 / 1 / 5 / 10% deterministic query corruption, with eight sampled queries per condition. Both strategies achieved 8/8 target hit@1 in every tested cell. This is useful negative evidence: the simple synthetic fixture is too easy to demonstrate a hit@1 advantage at these sizes. It does not establish equal robustness at realistic scale, nor any latency advantage. The Rust target should measure time and bytes explicitly; next quality extensions should add overlapping records, shared constituents, open-set queries and larger arities/corpora without letting test-generation shortcuts leak target identity.

## Industry comparison and research direction

FAISS IndexBinaryFlat stores packed binary vectors and performs exhaustive Hamming search, with popcount optimizations and batch-oriented locality handling. Its supported binary index family also includes indexed/approximate alternatives. The external comparison must therefore use identical query bytes, record bytes, candidate IDs, top-1/tie policy and warm/cold residency, then report quality and costs together.

Reference: https://github.com/facebookresearch/faiss/wiki/Binary-Indexes

HDC/VSA representation capacity is itself an active theoretical research area. The IEEE CIS HDC/VSA task force identifies compositional representation, sketching, associative memory and hardware-aware strategies as active themes. IBM Research's 2026 capacity analysis explicitly connects VSA representation capacity to set-membership, intersection estimation and sketching-style guarantees. These references motivate testing capacity and representation cost, rather than treating vector scan throughput as the whole problem.

References:
- https://cis.taskforce.ieee.org/hdcvsa/
- https://research.ibm.com/publications/capacity-analysis-of-vector-symbolic-architectures

The system-level hypothesis is a capacity-aware retrieval policy:
- use the compact bundled representation for low-ambiguity, memory-constrained workloads;
- use exact component indexing for clean identity-bearing cues when the memory budget permits;
- use component scan for uncertain/noisy cues;
- abstain or require provenance/structured evidence when the query is semantically underdetermined.

This policy must be measured against all single-strategy baselines. Do not count the exact-hash fast path as an improvement unless end-to-end query throughput, index build cost, component payload and workload quality are reported together. Do not infer joules from runtime, payload bytes or processor TDP.

## Go/no-go requirements

- Both scan strategies match their own independent scalar-Hamming oracle, including lowest-ID ties.
- The hybrid result equals component-scan top-1; exact hash matches are full-vector verified and a forced collision cannot create a false match.
- Clean-hit index coverage and noisy fallback behavior are measured separately.
- Query latency and index/representation build costs are reported separately and amortized against a declared query count.
- The comparison includes full payload and index overhead; no hidden setup costs.
- Industry/SOTA claims require same-byte baselines on the same hardware and compiler/CPU-feature settings.
- Energy claims require calibrated device-level measurement with a declared boundary.

No speedup, quality gain, industry superiority or energy improvement is claimed by authoring this target. It has not yet compiled or run in this environment.
