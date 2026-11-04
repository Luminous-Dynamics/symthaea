# ETH-029 — Skolem membership layout microbench v1

**Claim ceiling:** synthetic CPython representation experiment. This is not a SCPOG benchmark, not a CakeML measurement, and not a formal verification result.

## Question

SCPOG reverse-implication checking needs repeated membership checks over Skolem-node quantified-literal sets. Which storage representation offers a useful memory/lookup trade-off?

## Experiment

- Universe: 65,536 literal IDs.
- 896 synthetic Skolem nodes: 128 each with degrees 1, 4, 16, 64, 256, 1,024 and 4,096.
- 96 identical-pattern membership queries per node; 86,016 queries total.
- Seed: 0xE7029.
- Compared Python hash sets, sorted packed uint32 arrays, dense bitsets and a hybrid that uses a dense bitset at degree >=2,048 and a sorted array below it.
- All representations returned the same 43,384 positive memberships.

## Results

| Representation | Storage | Lookup time | ns/query |
|---|---:|---:|---:|
| Python hash set | 40.105 MiB | 9.410 ms | 109.4 |
| Sorted uint32 arrays | 2.742 MiB | 27.465 ms | 319.3 |
| Dense bitsets | 7.056 MiB | 22.804 ms | 265.1 |
| Hybrid sparse/dense | 1.787 MiB | 23.864 ms | 277.4 |

The hybrid used about 22.4x less storage than Python hash sets, at about 2.38x the lookup time in CPython. It used less storage than the always-sorted or always-dense alternatives on this fixture.

## Interpretation

This supports testing an adaptive sparse/dense index in the CakeML implementation. It does not establish that the same threshold or time trade-off holds in CakeML: Python object layouts, array, bytearray access and set membership have very different costs from CakeML's verified data structures.

The initial crossover estimate is the equality of packed 32-bit sparse storage and a dense bitset:

4 * degree ~= universe / 8, hence degree ~= universe / 32.

The experiment used that threshold (2,048 for a 65,536-literal universe). A formal implementation must include metadata overhead, integer range constraints, duplicate handling, and a proof that every membership answer is identical.

## Next

Port the three representations to a CakeML/HOL4 microbenchmark using the actual SCPOG Skolem literal access pattern. Measure verified heap allocations and runtime on the published 172-input cohort. Select the representation by measured degree/density and include the data-structure invariant in the refinement proof.


## Exact uploaded-artifact execution

The uploaded script was fetched from the public branch and reconstructed byte-for-byte locally. The reconstructed file's Git blob SHA-1 matched the remote file before execution.

- Git blob SHA-1: 4bf3dad70ac8a06e7f68fb4c05d6464b43084d12
- Script SHA-256: a85fa5baae40159ba60cdcf1eae23d02a6080d003ee1545cb6ea87f3cc125a9a
- Runtime: CPython 3.13.5, Linux
- Exact run: 86,016 queries, 43,384 identical positive memberships across all four layouts.

Lookup times are a single process run and remain indicative only; do not infer CakeML performance from them.


## Prior-art correction: CakeML already has verified array and bitset optimizations

The CakeML cake_lpr checker already uses native byte arrays for a compact bitset-like assignment, array-based clause storage, a lazily maintained active-clause index, per-literal first-occurrence lookup arrays, verified hashtables for compositional inclusion checks, and buffered proof streaming. The paper reports proof files from 300 MB to 4 GB being streamed line by line. These are established CakeML techniques, not new Symthaea contributions.

This v1 microbenchmark therefore does not justify proposing "use arrays/bitsets in CakeML" as a novelty. Its narrower question is how to represent the quantified-literal sets attached to SCPOG Skolem nodes, and how to evaluate membership under the actual clause-by-node query pattern. V2 models that clause-intersection operation more directly. Any CakeML implementation must reuse the existing verified optimization/refinement pattern and show a POG-specific benefit.

Source: Tan, Heule and Myreen, *Verified Propagation Redundancy and Compositional UNSAT Checking in CakeML*, 2023, Section 5.3.1–5.3.4.
