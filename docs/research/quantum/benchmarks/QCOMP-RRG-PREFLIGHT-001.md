# QCOMP-RRG-PREFLIGHT-001: bounded generator smoke sample

Status: **140-graph independent JavaScript preflight; not Rust execution or distributional certification**

## Canonical implementation handoff

The only Rust implementation is now centralized in [PR #8324](https://github.com/Luminous-Dynamics/symthaea/pull/8324): module `crates/domains/symthaea-quantum-comp/src/mis_random_regular.rs`, inspected commit `38ea70144397a7b2a864dac42c9ec473b50a369e`, blob `e2ad816606b774ce0b5e97170d6a8f7255f666a8`. Its CLI is `qcomp_rrg_generate`. It enforces 100,000 vertices, 1,000,000 stubs and 100,000 default pairing attempts. The duplicate `random_regular_mis.rs` implementation was removed from the cohort PR so there is one canonical API.

The Rust source is aligned to this preflight's seed-stream semantics: XorShift64*, rejection-sampled Fisher–Yates draws and in-place re-shuffling of stubs between rejected pairings. This is an implementation alignment, not proof of parity: the Rust tests and CLI have not yet been compiled or executed, and the 140 serialized edge-list fingerprints in the JSON were produced by the independent JavaScript translation.

## What was exercised

An in-tool JavaScript translation of the proposed configuration-pair-rejection generator created the first 20 seeds at each of the seven paper-shaped sizes (N = 50, 100, 200, 500, 1,000, 2,000, 5,000) with degree 3, using the cohort manifest's derived seed formula. Each attempt used the specified XorShift64* 64-bit stream, rejection-sampled bounded integers, Fisher–Yates stub shuffle, explicit self-loop/repeated-edge rejection, sorted canonical edges and exact degree/edge-count validation.

The companion JSON records each workload seed, attempt count, edge count, connected-component count, triangle count, and a non-cryptographic FNV-1a fingerprint of its canonical edge list. Duplicate detection used exact edge-list string equality within each size, not the FNV fingerprint.

## Observed smoke-sample results

- 140 graphs generated successfully under the 100,000-attempt cap.
- Every graph had the exact requested degree 3 and exactly 3N/2 edges.
- All 140 graphs were connected.
- All 140 canonical edge lists were unique within their size group.
- Attempt counts ranged from 1 to 31; mean attempts across the 140 cases was 6.686.

| N | Samples | Edges per graph | Max attempts observed | Connected | Unique |
|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 75 | 31 | 20/20 | 20/20 |
| 100 | 20 | 150 | 21 | 20/20 | 20/20 |
| 200 | 20 | 300 | 12 | 20/20 | 20/20 |
| 500 | 20 | 750 | 21 | 20/20 | 20/20 |
| 1,000 | 20 | 1,500 | 19 | 20/20 | 20/20 |
| 2,000 | 20 | 3,000 | 19 | 20/20 | 20/20 |
| 5,000 | 20 | 7,500 | 23 | 20/20 | 20/20 |

## Small-N distributional sanity check

The JSON companion now contains a second, separate experiment with 100,000 generated six-vertex, degree-three graphs. Exhaustive enumeration over all 2^15 = 32,768 possible labelled simple graphs on six vertices independently retained exactly 70 degree-three graphs: 10 labelled K3,3 graphs without triangles and 60 labelled triangular-prism graphs with triangles. Each simple graph corresponds to (3!)^6 stub pairings, so a uniform configuration-model pairing conditioned on simplicity predicts fractions 1/7 and 6/7.

An independent JavaScript translation observed 14,277 K3,3 graphs and 85,723 prism graphs. The K3,3 frequency was 0.14277 (theoretical 0.142857); the reported standardized deviation was -0.079, and mean attempts to reach a simple graph were 10.593 versus an exact acceptance probability of about 0.09478. This is a useful regression diagnostic against gross small-N sampling bias. It is **not** certification of the Rust generator or of distributional correctness at larger sizes. The PR now also authors a deterministic Rust test using 5,000 seeds and a coarse ±0.04 frequency tolerance; that test has not been compiled or run in the current environment.

## Interpretation and limitations

This shows that the specified seeds and algorithm semantics produce simple, connected, exact 3-regular graphs for a small smoke slice, and that the chosen retry cap is generous for these samples. It does **not** prove that the pseudorandom generator is high quality, that rejection sampling has no practical bias under the finite PRNG, or that the sample matches the distribution or exact graph instances in the published paper. Twenty graphs per size are far fewer than the paper's reported 200-per-size simulation cohort and 100,000-per-size SOTA-comparison campaign.

This is not the Rust binary and does not establish byte-for-byte parity with the Rust implementation. The FNV-1a fingerprint is non-cryptographic; a Rust materialization should record SHA-256 of every exact DIMACS byte stream, toolchain/generator commit and attempts used. No MIS solver, QAOA expectation values, circuit simulator, or physical QPU was run.
