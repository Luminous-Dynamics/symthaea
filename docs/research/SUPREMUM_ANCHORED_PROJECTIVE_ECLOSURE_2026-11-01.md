# Supremum-anchored projective e-closure

**Status:** finite-family exact-rational reference checker and candidate theorem. This is an extension hypothesis inspired by Ramdas's October 2026 anchored-multiple-testing preprint; it is not yet a proved improvement over published dynamic e-closure or SupFDR procedures.

## 1. Pathwise budget construction

Let B_t be a baseline rejection-set process. For each coordinate i, define the projective ever-rejected indicator

    J_i,t = 1{there exists u <= t with i in B_u}.

Assume the baseline comes with **ex-ante coordinatewise risk budgets** q_i >= 0 satisfying

    P_Hi(i is ever rejected by the baseline) <= alpha*q_i,
    sum_i q_i <= 1.

For any intersection S, define

    c_S = sum_{i in S} q_i
    A_S,t = (1/alpha) * sum_{i in S} J_i,t.

Then A_S,t is adapted and nondecreasing, and

    E_HS[ sup_t A_S,t ] <= c_S.

This follows from linearity of expectation and the coordinatewise ever-rejection bounds. The bound must hold under every distribution in H_S; it is sufficient that it holds under each marginal null H_i because H_S implies every H_i for i in S.

The ever-rejected indicator is deliberately more conservative than the denominator-sensitive baseline contribution. It is projective when the baseline decision for coordinate i depends only on its own evidence stream and fixed ex-ante q_i. An arbitrary global baseline may make the event "i is ever rejected" depend on unrelated coordinates; such a baseline needs a separate extension-invariance proof before this construction can be called projective.

Now let F_S,t be an auxiliary **anytime-adjusted** local evidence process with

    E[ sup_t F_S,t | H_S ] <= 1.

Define the anchored local process

    E_S,t = A_S,t + (1 - c_S) F_S,t.

By subadditivity of the supremum,

    E[ sup_t E_S,t | H_S ]
      <= E[ sup_t A_S,t | H_S ]
         + (1-c_S) E[ sup_t F_S,t | H_S ]
      <= c_S + (1-c_S) = 1.

Thus the running-supremum budget of each local e-collection is at most one. No independence between baseline and auxiliary evidence is required.

## 2. Baseline retention and pathwise SupFDR

For every t and every S,

    FDP_S(B_t)
      = |B_t intersect S| / (|B_t| vee 1)
      <= |B_t intersect S|
      <= sum_{i in S} J_i,t
      = alpha A_S,t
      <= alpha E_S,t.

Therefore every baseline rejection set B_t belongs to the time-t e-closure certified by E_S,t. The baseline is never removed by augmentation.

Suppose an output process R_t is selected so that, for every t and every nonempty intersection S,

    FDP_S(R_t) <= alpha E_S,t.

For the actual true-null set I_0, taking the supremum over t yields

    sup_t FDP_{I_0}(R_t) <= alpha sup_t E_{I_0,t}.

If the intersection I_0 itself has a valid local path budget/evidence process satisfying the assumption above, then

    E[ sup_t FDP_{I_0}(R_t) ] <= alpha.

This is a **SupFDR pathwise certificate**. It is not automatic stopped-FDR/e-process validity: the auxiliary F_S,t must already be adjusted so its expected running supremum is at most one. A raw e-process is not enough.

## 3. Concrete coordinate-budget baseline

A simple baseline is per-hypothesis anytime e-testing with weights q_i summing to at most one. Reject i once its valid e-process crosses 1/(alpha q_i), and retain that rejection thereafter. Ville's inequality gives

    P_{H_i}(i is ever rejected) <= alpha q_i.

Since sup_t a_i,t <= 1{i is ever rejected}, the coordinate path-budget condition follows. This is conservative, but supplies a mathematically clean reference baseline.

A better baseline could provide the sharper coordinate contribution bound directly:

    E_{H_i}[ sup_t 1{i in B_t}/(|B_t| vee 1) ] <= alpha q_i.

This is the research opportunity: construct powerful online/SupFDR procedures with small ex-ante q_i, so substantial residual budget 1-c_S remains for auxiliary projective intersection evidence.

A global SupFDR statement alone need not provide useful c_S<1 for arbitrary S. If only the global bound is known, the safe generic assignment may be c_S=1, leaving zero augmentation budget. Do not invent per-coordinate slack.

## 4. A concrete auxiliary path from admissible running-maximum adjusters

The auxiliary path F_S,t need not remain an abstract assumption when a valid raw intersection e-process E_S,t is available. Let

    M_S,t = max_{u<=t} E_S,u

be its running maximum. Tavyrikov, Goeman and de Heide (2026) show that A(M_S,t) is again an e-process whenever A is an admissible adjuster: increasing/right-continuous, diverging at infinity and satisfying

    integral_1^infinity A(x)/x^2 dx = 1.

Because A(M_S,t) is nondecreasing, its expected path supremum is at most one by monotone convergence. Thus it is a concrete choice for the auxiliary F_S,t.

A normalized family is

    A_beta(x) = ((1-beta)/beta) * (x^beta - 1),  0 < beta < 1,

with beta=0 continuous limit A_0(x)=log(x). Its normalization is exact:

    integral_1^infinity A_beta(x)/x^2 dx
      = ((1-beta)/beta) * (1/(1-beta)-1) = 1.

At beta=1/2, A(x)=sqrt(x)-1. A fixed ex-ante mixture over beta values is also an e-process; selecting the pointwise best beta after observing the same data is not. This makes it possible to attach a power-adaptive but valid auxiliary path to each #9695 projective intersection e-process or to a #9712/#9758 intersection process under its stated model assumptions.

The deterministic implementation supports both ordinary values and log-domain paths to avoid overflow for large capital. See SUPREMUM_EPROCESS_ADJUSTERS_2026-11-01.md. The theorem does not supply the coordinate baseline budgets q_i, prove the raw intersection processes valid, or establish the countable-family composition.

Source: Tavyrikov, Goeman & de Heide, Carefree multiple testing with e-processes, EJS 20(2), 2026, https://doi.org/10.1214/26-EJS2546.

## 5. Countable dynamic hypothesis families

For a countable universe, the construction requires:
- a fixed ex-ante sequence q_i >= 0 with sum_i q_i <= 1;
- a predictable, extension-stable coordinate-local ever-rejection process J_i,t;
- an auxiliary local path-adjuster F_S,t for every relevant finite or countable intersection S;
- a validity result for the actual true-null intersection I_0, including its countably infinite case;
- extension-coherent local certificates that do not change when future hypotheses are appended.

The finite Python checker below enumerates all nonempty subsets. It does not establish the countably infinite theorem or provide a scalable separation oracle. The global-projective affine construction (#9695) and persistent adjuster still require a careful composition proof.

## 6. Exact-rational reference checker

Files:
- validation/supremum_anchored_eclosure.py computes running coordinate contributions and local path certificates.
- validation/test_supremum_anchored_eclosure.py tests exact baseline retention, residual-budget arithmetic, pathwise closure, deterministic largest certified supersets and fail-closed budget/input checks.

This checker verifies deterministic inequalities only. It cannot establish the stochastic q_i or auxiliary running-supremum assumptions. Inputs called "adjusted auxiliary" must already be supported by a theorem/adjuster and source receipts.

## 7. Additive auxiliary special case: exact e-BH reduction

There is a faster restricted construction when the auxiliary evidence is allocated coordinate by coordinate. Choose fixed nonnegative auxiliary budgets v_i such that

    sum_i (q_i + v_i) <= 1,

and coordinate auxiliary paths F_i,t satisfying

    E_Hi[sup_t F_i,t] <= 1.

Define each coordinate path

    e_i,t = J_i,t/alpha + v_i F_i,t.

For any intersection S,

    E_S,t = sum_{i in S} e_i,t
          = (1/alpha) sum_{i in S} J_i,t
            + sum_{i in S} v_i F_i,t.

The expected running supremum is at most

    sum_{i in S}(q_i+v_i) <= 1.

This is exactly the anchored construction with residual auxiliary term

    F_S,t = [sum_{i in S} v_i F_i,t] / (1-c_S),

when c_S<1. The global budget condition ensures the numerator has expected running supremum at most 1-c_S. If c_S=1, it forces all v_i inside S to be zero.

### Why full e-closure reduces to e-BH here

For a proposed rejection set R of size r>0, the closure inequality for every nonempty S is

    FDP_S(R)/alpha <= sum_{i in S} e_i,t.

Since

    FDP_S(R)/alpha
      = sum_{i in S} 1{i in R}/(alpha*r),

the condition is equivalent to

    sum_{i in S} [1{i in R}/(alpha*r) - e_i,t] <= 0
    for every nonempty S.

A finite family satisfies all subset-sum inequalities if and only if every singleton term is nonpositive. Therefore closure feasibility is exactly

    e_i,t >= 1/(alpha*r)  for every i in R.

Selecting the largest r satisfying the sorted e-value threshold is the e-BH rule. Sorting gives O(n log n) time and O(n) storage, rather than enumerating 2^n intersections. The baseline is retained: every ever-rejected coordinate has J_i,t=1 and hence e_i,t>=1/alpha; the e-BH cutoff for r>=1 is no larger than 1/alpha. If multiple coordinates already qualify, the maximal e-BH rank expands to include all of them.

This is an exact computational special case, not a universal replacement for intersection-specific evidence. It can be conservative because all auxiliary mass is coordinate-separable and because the safe baseline uses ever-rejected indicators. The richer #9695/#9684 intersection processes may certify sets that this additive e-BH restriction cannot.

### Fixed-prior portfolio over auxiliary budget allocations

A single vector v_i can be brittle when alternatives are heterogeneous. The additive construction now accepts a finite set of auxiliary allocations v_i^(j), each paired with coordinate auxiliary paths F_i,t^(j), and a fixed ex-ante prior pi_j. Every strategy must independently satisfy

    sum_i(q_i + v_i^(j)) <= 1,
    E_Hi[sup_t F_i,t^(j)] <= 1.

The combined coordinate path is

    e_i,t = J_i,t/alpha
            + sum_j pi_j v_i^(j) F_i,t^(j).

Because the prior is fixed and every component is valid,

    sum_i(q_i + sum_j pi_j v_i^(j))
      = sum_j pi_j sum_i(q_i+v_i^(j)) <= 1.

Thus the portfolio remains valid and the intersection-local values remain additive. e-BH still gives the O(n log n) selector. This lets us hedge allocations for sparse, dense, clustered, or value-weighted alternatives without choosing the best allocation after seeing the confirmatory stream.

The helper is build_additive_anchored_portfolio_e_paths in validation/supremum_anchored_eclosure.py. Tests verify exact convex-mixture arithmetic, baseline retention, closure feasibility, and rejection of invalid/post-hoc weight vectors.

### Scaling benchmark

Run the exact-rational selector benchmark:

```bash
python validation/benchmark_supremum_anchored_ebh.py \
  --sizes 8 12 16 100 1000 10000 100000 \
  --repeats 7 --output anchored-ebh-results.json
```

For each n it reports e-BH selector latency, an independent O(n) feasibility-verifier latency, traced peak allocation, baseline anchor size, additional discoveries and whether the anchor was retained. At n<=16 it also compares the selected set with the exhaustive all-intersections selector. Larger instances are verified using the independent additive-closure singleton criterion. This is a computational benchmark only; it does not test stochastic q_i/v_i assumptions or establish FDR/SupFDR calibration.

### Files and finite verification

The reference checker now includes:
- `build_additive_anchored_auxiliary`, which constructs each normalized intersection auxiliary process F_S,t;
- `build_additive_anchored_coordinate_e_paths`, which produces the per-coordinate e_i,t paths;
- `select_additive_anchored_e_bh`, a deterministic O(n log n) selector with stable ID tie-breaking and an explicit anchor-retention check.
- `validation/benchmark_supremum_anchored_ebh.py` and `validation/test_benchmark_supremum_anchored_ebh.py`, an O(n log n) scaling scorecard and exhaustive small-family parity controls.

Tests compare the constructed intersection paths with the sums of coordinate paths, compare e-BH against the exhaustive finite closure selector, verify every resulting closure inequality, and exercise residual-zero/invalid-budget/empty-horizon cases. These tests are authored but have not been executed from a repository checkout; no CI pass is claimed.

## 8. Benchmark targets

Compare separately:
1. weighted e-Bonferroni baseline (known coordinate budgets);
2. onlineFDR / e-LOND baselines only where an appropriate coordinate path budget is proved;
3. dynamic affine projective closure #9695;
4. countable prior mixture #9684;
5. scalar/vector/preconditioned conditional-mgf experts #9712/#9751/#9758, only after valid SupFDR adjustment;
6. static anchored BH/BY/e-BH from Ramdas (2026) under its own assumptions.

Report:
- baseline discoveries retained (must be 100%);
- additional discoveries over baseline;
- SupFDR and FDR with confidence intervals;
- pathwise FDP quantiles and maximum;
- residual budget distribution c_S and 1-c_S;
- utility/regret, capacity, latency, memory and certificate size.

No SOTA claim is justified until the theorem, countable extension and matched confirmatory benchmark are complete.

## 9. Prior art

- Ramdas, Anchored multiple testing (October 2026): https://arxiv.org/abs/2610.11279
- Tavyrikov, Goeman & de Heide, Carefree multiple testing with e-processes, EJS 20(2), 2026: https://doi.org/10.1214/26-EJS2546
- Xu, Fischer & Ramdas, Improving online FDR procedures via online analogs of e-closure and compound e-values, UAI 2026: https://proceedings.mlr.press/v337/xu26a.html
