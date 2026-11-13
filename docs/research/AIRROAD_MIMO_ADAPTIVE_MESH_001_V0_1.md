# AIRROAD-MIMO-ADAPTIVE-MESH-001 — proof-guided frequency-knot refinement

**Status:** V0.1 synthetic exact-rational Bernstein-guided adaptive mesh  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11923  
**Composes with:** #11918/#11919 generator remainder, #11914 cross-frequency UQ, #11868 active excitation, ROB-DOE #4817, ROB-EXP #4800, #11801 robust MIMO stability

## 1. Objective

Choose the next frequency knot using a certified interval remainder upper bound. The algorithm does not rank intervals by sampled maxima. It:
1. constructs the exact interpolation residual on every interval;
2. computes exact Bernstein coefficients on the interval;
3. uses the convex-hull property to bound the full residual range;
4. splits the interval with the largest certified upper radius;
5. replays the new endpoint/interpolation certificates;
6. stops only when the global radius meets its target or the split budget is exhausted.

This is an AIRROAD-specific refinement policy and benchmark, not a generic experiment optimizer or hardware executor.

## 2. Bernstein upper-bound rule

For interval [a,b], map omega=a+(b-a)t for t∈[0,1]. Express the residual polynomial in the degree-four Bernstein basis:

    e(t) = sum_(i=0..4) beta_i * C(4,i)t^i(1-t)^(4-i)

All basis functions are nonnegative and sum to one. Therefore:

    min_i beta_i <= e(t) <= max_i beta_i
    |e(t)| <= max_i |beta_i|

Because the matrix factor P has Frobenius norm one, the same bound applies to the matrix interpolation residual.

This produces a valid, replayable upper bound. It can be conservative; it must never be replaced by the maximum of a finite sample grid.

## 3. Exact adaptive trajectory

Reuse d(omega)=16*omega*(1-omega)*(omega-1/2)^2 and P from #11918. Initial knots are 0, 1/2 and 1, all with zero matrix values.

Initial interval bounds:
- [0,1/2]: Bernstein coefficients [0,1/2,1/6,0,0], radius 1/2;
- [1/2,1]: [0,0,1/6,1/2,0], radius 1/2.

Tie-break by the lowest lower endpoint. The first split inserts omega=1/4, where d=3/16:
- [0,1/4] radius 13/64;
- [1/4,1/2] radius 5/96;
- [1/2,1] remains 1/2.

The second split selects the still-largest interval [1/2,1], inserting omega=3/4, d=3/16:
- [1/2,3/4] radius 5/96;
- [3/4,1] radius 13/64.

The final global certified radius is 13/64 after two splits, compared with the initial 1/2. The exact original curve supremum was 1/4; the refined linear interpolant now tracks additional points, and the remaining interpolation error bound is smaller.

## 4. Policy and evidence identity

The policy binds model, topology, operating point, norm, frequency unit, calibration/acquisition identity, endpoint values, full interval partition, candidate measurement profile, split budget and deterministic tie-break.

A midpoint observation is only a prospective measurement protocol in physical use. The verifier uses an exact synthetic polynomial value and zero hard observation error. Real measurement uncertainty and common calibration dependencies must propagate into the new endpoint generator set; they cannot be omitted or assumed independent.

## 5. Benchmark comparisons

Compare:
- uniform midpoint subdivision;
- largest Bernstein-upper-bound refinement;
- largest sampled-residual heuristic (diagnostic only);
- exact global polynomial extremum;
- interval-Horner and Bernstein hybrid enclosures.

Report certified radius per experiment, maximum true residual on known-answer cases, bound conservatism, complete interval coverage, proof cost, and remaining unsupported frequency regions separately.

## 6. Negative controls

The verifier rejects sampled maxima used as universal bounds, reversed tie-breaks, skipped largest-bound intervals, missing/gapped/overlapping partitions, duplicate knot IDs, stale model/calibration/unit identities, unapproved midpoint proposals, statistical-to-deterministic promotion, and claiming the target met when the budget ends above the target.

## 7. Claim ceiling

V0.1 is a synthetic exact-polynomial adaptive mesh reference. It does not establish physical excitation safety, actual FRF regularity, robust stability, IQC synthesis, aircraft safety or airworthiness/certification.
