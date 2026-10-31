# Exact Pareto dynamic programming for capacity-aware raw projective e-closure

**Status:** exact-rational research prototype. The all-intersection certificate is O(N); the utility optimizer uses an exact Pareto frontier whose worst-case size is exponential. Guarantee scope is simultaneous stopped-FDR, not SupFDR.

## 1. From all intersections to two additive resource budgets

For the raw projective merger

\[
E_t^S=1+\sum_{i\in S}w_i(M_{i,t}-1),
\quad w_i\geq0,\quad\sum_iw_i\leq1,
\]

the dynamic e-closure condition for a nonempty rejection set \(R\), \(k=|R|\), is

\[
\max_{S\subseteq I_t}
\left\{
\frac{|S\cap R|}{k}
-\alpha\sum_{i\in S}w_i(M_{i,t}-1)
\right\}
\leq\alpha.
\]

For each active hypothesis define

\[
h_i=\alpha w_i(M_{i,t}-1).
\]

The maximum over all subsets is the sum of positive coefficients. Equivalently,

\[
L(R)=
\underbrace{\sum_{i\in I_t}[-h_i]_+}_{B_t}
+
\sum_{i\in R}
\underbrace{\left([1/k-h_i]_+-[-h_i]_+\right)}_{d_i(k)}
\leq\alpha.
\]

Because

\[
d_i(k)=\min\{1/k,[1/k-h_i]_+\}\geq0,
\]

the closure load is a baseline plus nonnegative per-selected-item charges. For fixed \(k\), the statistical constraint is now one additive budget.

The operational capacity is another additive budget:

\[
\sum_{i\in R}c_i\leq C_t.
\]

Thus, for every fixed cardinality \(k\), the decision problem is an exact-cardinality, two-resource selection problem:

\[
\max_{R}\sum_{i\in R}v_i
\]

subject to

\[
|R|=k,\quad
\sum_{i\in R}d_i(k)\leq\alpha-B_t,\quad
\sum_{i\in R}c_i\leq C_t,
\]

plus any previously committed rejections that must remain selected.

This is the useful structural reduction: we no longer enumerate every possible rejection set and independently check every intersection.

## 2. Exact Pareto dynamic programme

The implementation processes optional hypotheses one at a time. For each partial cardinality it retains nondominated states

\[
(\text{closure charge},\text{operational cost},\text{utility},\text{selected indices}).
\]

A state dominates another if it has no greater risk charge, no greater cost and no lower utility. The deterministic objective tie order is:
1. maximize utility;
2. minimize cost;
3. minimize closure load;
4. lexicographically minimize selected indices.

The fast frontier pruner sorts states by cost/risk and uses a Fenwick prefix-maximum tree over compressed exact rational risk charges. It prunes a state whenever an earlier state has no greater cost/risk and at least as much utility. A separate quadratic Pareto-pruning implementation replays the optimizer on the same exact rational inputs for independent optimality verification.

No floating-point rounding is used in the resource or risk budgets. Frontier growth is explicitly capped; the solver fails closed rather than silently approximating or dropping a potentially optimal state.

### Complexity

For a single cardinality, the DP is proportional to the number of optional items times the size of its Pareto frontiers, with a logarithmic Fenwick pruning factor. It is exact, but **not polynomial-time in the worst case**: multiobjective 0/1 knapsack can have exponentially many nondominated states.

Across all cardinalities the reference implementation runs one DP per candidate rejection count. Cheap optimistic bounds on minimum risk charge and minimum resource cost skip impossible cardinalities before constructing a frontier.

This is a practical exact solver for instances with manageable Pareto frontiers, not a universal scalability result.

## 3. New integration: capacity-aware Anchored e-BH

Ramdas (2026), arXiv:2610.11279, proves that Anchored e-BH strictly improves ordinary e-BH under arbitrary dependence while remaining incomparable to mean closed e-BH. Its local collection is

    E_S = (sum_{i in S} e_i + m - |S|) / m
        = 1 + sum_{i in S} (e_i - 1) / m.

The ordinary e-BH rejection set B is always certified by this collection. Therefore the capacity optimizer can use uniform global weights w_i = 1/m and initialize its mandatory set with B. It then selects a utility-maximizing certified superset while preserving every baseline e-BH discovery.

### Maximum-cardinality Anchored e-BH baseline

The paper gives an O(m log m)-time, O(m)-memory step-up algorithm for the largest-cardinality certified superset. The branch now implements that baseline explicitly in `anchored_e_bh_max_cardinality`.

For h_i = alpha * (e_i - 1) / m, the closure load for a top-r candidate is

    L_r = sum_i max(-h_i, 0)
          + sum_{i=1}^r max(1/r - max(h_i, 0), 0).

Sorting e-values in descending order also sorts h_i. For each r, the positive terms form a suffix, so one binary search plus a prefix sum evaluates L_r in O(log m). Scanning r = 1,...,m gives O(m log m) after sorting. The candidate top-r set retains B whenever r >= |B|; an exact O(m) separator independently checks the selected certificate.

### Capacity-aware objective

The new capacity-aware wrapper retains all baseline e-BH discoveries and optimizes a predeclared utility under operational capacity. Its output now includes the published maximum-cardinality Anchored e-BH result as an explicit comparator. The capacity-aware choice may differ from the maximum-cardinality result because it optimizes a different objective.

Two exact fixtures:
- e = (30,14,6), alpha = 0.1: ordinary e-BH returns {1}; maximum-cardinality Anchored e-BH returns {1,2}. With utilities (1,100,1), unit costs and capacity 2, the capacity-aware optimizer also returns {1,2}.
- e = (29,1,0), alpha = 0.1: ordinary e-BH returns the empty set, while Anchored e-BH certifies {1}.
- The paper's strict example e = (39.5,1), alpha = 0.05: ordinary e-BH is empty, while maximum-cardinality Anchored e-BH returns {1}.

The guarantee is fixed-family FDR under arbitrary dependence, conditional on each input e_i being a valid marginal e-value. It does not extend to sequential evidence or SupFDR by itself. Capacity must be sufficient to retain the full baseline e-BH set.

This is a capacity-aware operational extension of established Anchored e-BH, not an invention of the anchoring principle. Required comparisons include ordinary e-BH, maximum-cardinality Anchored e-BH, capacity-aware Anchored e-BH, and closed e-BH/RC-eBH. The paper shows Anchored e-BH and mean closed e-BH are pointwise incomparable, so neither should be described as a universal replacement for the other.

## 4. Irrevocability and non-persistence warning

The raw intersection e-processes are future-extension coherent but are not necessarily time-monotone. Their certified menus need not be nested.

If a previously selected set becomes uncertified after new evidence decreases, an irrevocable nested extension may cease to exist. The solver fails closed with an explicit error instead of silently retracting decisions or claiming a current certificate that is no longer true.

A larger new set can sometimes restore certification because its larger cardinality reduces the per-item \(1/k\) charge. The test suite includes both:
- a one-hypothesis case where no nested certified extension exists; and
- a two-hypothesis case where the old singleton fails at the next time but adding a sufficiently strong new hypothesis restores a certified superset.

For genuinely persistent/irrevocable decisions, use the separate time-monotone adjusted SupFDR construction in #9538, or define a different commitment contract. The raw branch retains simultaneous stopped-FDR scope only.

## 5. Independent verification and negative controls

The source module validation/stopped_eclosure_capacity_pareto.py provides:
- exact rational charge decomposition;
- Fenwick-based Pareto frontier optimization;
- independent quadratic Pareto replay;
- O(N) final closure verification;
- nestedness, cumulative capacity and prefix-digest checks;
- fail-closed frontier and irrevocability behavior.

The test module validation/test_stopped_eclosure_capacity_pareto.py covers:
- charge decomposition against the O(N) positive-part separator;
- exact DP versus exhaustive selection on randomized small instances;
- independent verifier rejection of a valid-but-suboptimal decision;
- future evidence restoring a nested certificate;
- non-persistence / no-feasible-irrevocable-extension failure;
- a 100-hypothesis sparse-strong-signal fixture.

Run from the repository root:

    PYTHONPATH=validation python -m unittest -v test_stopped_eclosure_capacity_pareto.py
    PYTHONPATH=validation python -m unittest -v test_anchored_capacity_e_bh.py

The committed suite has not been executed in this environment. A separate private exact-rational prototype matched exhaustive subset search on 800 randomized instances with N=1..8. A 100-hypothesis sparse-strong-signal prototype completed with a maximum frontier of 48 states in approximately 0.6 seconds. Those are development measurements, not CI or a general runtime guarantee.

## 6. Benchmarks and industry comparison

Compare against:
- exhaustive exact subset enumeration (small-N oracle);
- the previous O(N)-membership / O(2^N N)-candidate reference;
- HiGHS MILP / branch-and-bound formulations of the two-budget cardinality problem;
- meet-in-the-middle for moderate N;
- safe approximation with outward-rounded risk charges and a separately verified certificate;
- unconstrained top-value ranking and top-value-per-cost ranking;
- the adjusted SupFDR dynamic closure (#9538) where the stronger time-uniform guarantee is required.

Benchmark N=20, 50, 100, 250, 1,000, and 10,000 where frontier sizes permit. Record:
- number of cardinalities pruned by optimistic bounds;
- Pareto states generated, retained and dominated;
- maximum/mean frontier width;
- p50/p95 runtime and peak memory;
- exact verifier runtime;
- utility regret versus the small-N exhaustive oracle;
- capacity utilization and closure slack;
- failure rate due to explicit frontier caps.

For security alert triage or scientific candidate selection, compare analyst-hours per useful finding, precision@B, recall@B, wasted follow-up capacity and decision latency. Statistical validity, operational value and truth-labelled utility remain separate metrics. An ordinary severity score is not a valid e-value; upstream evidence generation must satisfy its own conditional e-process contract.

## 7. Guarantee boundary

The raw globally weighted projective e-collection is future-extension coherent. Under valid coordinate e-processes in a common global filtration, it gives simultaneous stopped-FDR under the dynamic e-closure theorem. It is not time-monotone by default, so this PR makes no SupFDR claim.

## References

- Ramdas (2026), [Anchored multiple testing: a transparent use of e-closure to improve FDR procedures](https://arxiv.org/abs/2610.11279).
- de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence](https://arxiv.org/abs/2608.09927).
- Xu, Fischer & Ramdas (2026), [Online analogs of e-closure and compound e-values](https://proceedings.mlr.press/v337/xu26a.html).
- [PR #9599 — linear-time raw projective e-closure separator](https://github.com/Luminous-Dynamics/symthaea/pull/9599).
- [PR #9538 — persistent dynamic e-closure / SupFDR path](https://github.com/Luminous-Dynamics/symthaea/pull/9538).
