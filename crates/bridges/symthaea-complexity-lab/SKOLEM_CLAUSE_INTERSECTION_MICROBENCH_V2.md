# ETH-029 — Skolem clause-intersection layout microbench v2

**Claim ceiling:** synthetic CPython query-shape experiment. Not a CakeML benchmark, not an SCPOG benchmark, and not a formal verification result.

## Query shape

The SCPOG paper describes reverse checking each input clause by traversing the POG and, at a Skolem node, testing whether any literal in the input clause belongs to that node. V1 measured random single-literal lookups. V2 models the actual clause-intersection operation with early exit.

- 65,536 quantified variables; 131,072 signed literal IDs.
- 672 synthetic Skolem nodes, 96 each at degrees 1, 4, 16, 64, 256, 1,024 and 4,096.
- 128 input-clause queries per node; clause lengths cycle through 3, 8 and 16.
- Half the queries contain one Skolem literal, half contain none.
- 86,016 clause-intersection queries and 774,144 candidate literal positions.
- Identical generated query stream for all layouts; seed 0xE7029.

Representations: Python hash set, sorted packed uint32 IDs with binary search, dense 2-bit variable state (absent/positive/negative), and a sparse/dense hybrid.

## Results

| Representation | Storage | ns/clause | Positive clauses |
|---|---:|---:|---:|
| Python hash set | 30.107 MiB | 722.1 | 43,008 |
| Sorted uint32 | 2.056 MiB | 2,052.3 | 43,008 |
| Dense 2-bit | 10.542 MiB | 2,014.4 | 43,008 |
| Hybrid | 2.090 MiB | 2,096.5 | 43,008 |

All layouts agreed on the 43,008 positive clause intersections and on the 599,519 literal probes after identical short-circuit behavior.

## Interpretation

This query shape is closer to SCPOG than v1's random single-literal query. The sparse representation is compact; hash sets and dense 2-bit states are faster in CPython. The hybrid chooses dense only at degree 4,096 and gains no storage advantage over sorted arrays in this fixture.

This is a negative/qualifying result: a simple density-threshold hybrid is not automatically superior. More importantly, CakeML's verified LPR checker already has established optimized representations: native byte arrays for a bitset-like assignment, array-based clause storage, a lazy active-clause index, per-literal occurrence lookup arrays, verified hash tables and buffered proof streaming. These are prior art, not new Symthaea contributions.

The next investigation should profile real Skolem-node degree distributions and the exact SCPOG reverse-check membership pattern. Then compare a Skolem-specific sparse index against reuse of CakeML's established array/byte-array techniques, with the refinement proof carried into HOL4. Do not select an implementation from this Python benchmark alone.

Source for CakeML prior art: Tan, Heule and Myreen, *Verified Propagation Redundancy and Compositional UNSAT Checking in CakeML*, 2023, Section 5.3.1–5.3.4.


## Exact uploaded-script execution

The uploaded V2 script was reconstructed byte-for-byte and its Git blob SHA-1 matched the fetched GitHub file before execution.

- Git blob SHA-1: c18d1a9397d4d47e528a6941dded432cc7213df1
- Script SHA-256: 711869eee2f6b99f5bf0f2268534b2d0bb92b01d2d911b75164a8c63ee72a315
- Runtime: CPython 3.13.5, Linux
- 86,016 clause queries; 769,440 candidate literal positions; 599,519 actual probes after short-circuit; 43,008 clause hits.
- All four representations returned identical hit and probe counts.

A single run is indicative only. No CakeML or SCPOG memory/time claim follows.
