# Logarithmic-time projective e-closure index

**Status:** research prototype; draft stacked on the countable/global-projective e-closure work in PR #9695. This is an exact decision-auditing accelerator, not a replacement for the rejection-set optimizer.

## Exact reduction

For the raw affine separator in #9695, let r=|R|>0 and

$$
b_i = \alpha w_i(M_i-1),\qquad
Q(R)=\sum_i\max\left(0,\frac{\mathbf 1\{i\in R\}}r-b_i\right).
$$

Define the all-hypothesis baseline C = sum_i max(0,-b_i). For a selected coordinate, its increment over the unselected contribution is

$$
\phi_r(b_i)=\max(0,1/r-b_i)-\max(0,-b_i)
=\begin{cases}
1/r,&b_i<0,\\
1/r-b_i,&0\le b_i<1/r,\\
0,&b_i\ge 1/r.
\end{cases}
$$

Therefore,

$$
Q(R)=C+
\frac{\#\{i\in R:b_i<1/r\}}r
-\sum_{i\in R,\ 0\le b_i<1/r}b_i.
$$

The final two quantities are count-prefix and sum-prefix queries over the selected b_i values. An augmented AVL tree stores subtree cardinality and exact sum; C is maintained separately. With one-coordinate updates, evidence changes, or rejection toggles, update and score operations are worst-case O(log n). A batch of k sparse updates costs O(k log n).

Inactive hypotheses must be represented with M_i=1, hence b_i=0, until activation. The identity is unchanged by dormant-one padding.

## Important limits

- This accelerates feasibility auditing of a proposed rejection set. It does not solve the arbitrary-utility/capacity MILP; that optimization remains combinatorial.
- When every M_i changes every round, n AVL updates may be slower than one O(n) scan. This is intended for sparse/asynchronous evidence or decision updates.
- The prototype uses exact Fraction inputs. Since real e-processes often produce floating values, a production implementation must use outward-rounded intervals or certified high precision. A non-certified floating comparison must never turn an uncertain score into an accepted decision.
- Keep the existing linear-time separator as an independent verifier.

## Prototype evidence (local, not CI)

The algebraic identity was checked against the original sum-of-positive-parts expression on 50,000 randomized exact-rational instances (dimensions 1–10, random rejection sets); every comparison was exact. The AVL prototype then passed 10,000 randomized evidence-update / selection-toggle operations with exact differential checks and periodic verification of height, balance, subtree count and subtree sum.

A local Python exact-rational event benchmark compared k evidence updates plus one closure query against k updates plus a full O(n) scan. Representative measurements:

| n | k changed | AVL update + query | full scan | observed speedup |
|---:|---:|---:|---:|---:|
| 1,000 | 1 | 0.167 ms | 2.197 ms | 13.2x |
| 1,000 | 16 | 1.083 ms | 2.383 ms | 2.2x |
| 1,000 | 100 | 3.813 ms | 2.318 ms | 0.61x |
| 10,000 | 1 | 0.078 ms | 22.712 ms | 289.7x |
| 10,000 | 64 | 3.200 ms | 23.800 ms | 7.4x |
| 10,000 | 1,000 | 47.716 ms | 21.893 ms | 0.46x |
| 100,000 | 1 | 0.340 ms | 227.746 ms | 669.5x |
| 100,000 | 64 | 3.200 ms | 220.809 ms | 69.0x |
| 100,000 | 10,000 | 472.544 ms | 221.260 ms | 0.47x |

These are single-environment prototype timings, not Rust production measurements or a general performance claim. They show the regime boundary: large savings when only a few coordinates change, and a loss when 10% of the family changes.

## Files

- validation/projective_closure_index.py: augmented AVL and exact query.
- validation/test_projective_closure_index.py: rational differential and structural tests.

## Next research

1. Integrate the index as a candidate-feasibility accelerator in the #9695 optimizer without removing the independent scan.
2. Build a Rust implementation with worst-case AVL/B-tree guarantees and benchmark it against the scan and MILP candidate validation.
3. Add authenticated subtree aggregates and evidence-root-bound query receipts.
4. Define an interval-valued key ordering for floating evidence, with a fail-closed fallback when intervals straddle 0 or 1/r.
5. Benchmark event rates, update density, tree rebuild thresholds, cache locality, and memory cost—not just query complexity.
