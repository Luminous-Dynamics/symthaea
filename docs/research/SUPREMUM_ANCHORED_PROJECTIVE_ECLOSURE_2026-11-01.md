# Supremum-anchored projective e-closure

**Status:** finite-family exact-rational reference checker and candidate theorem. This is an extension hypothesis inspired by Ramdas's October 2026 anchored-multiple-testing preprint; it is not yet a proved improvement over published dynamic e-closure or SupFDR procedures.

## 1. Pathwise budget construction

Let B_t be a baseline rejection-set process. For each coordinate i, define

    a_i,t = 1{i in B_t} / (|B_t| vee 1)
    G_i,t = max_{u <= t} a_i,u

Assume the baseline comes with **ex-ante coordinatewise path budgets** q_i >= 0 satisfying

    E[ sup_t a_i,t / alpha | H_i ] <= q_i,
    sum_i q_i <= 1.

For any intersection S, define

    c_S = sum_{i in S} q_i
    A_S,t = (1/alpha) * sum_{i in S} G_i,t.

Then A_S,t is adapted and nondecreasing, and

    E[ sup_t A_S,t | H_S ] <= c_S.

This follows from sup_t sum_i a_i,t <= sum_i sup_t a_i,t and linearity of expectation. It requires the coordinatewise path budgets to hold under every distribution in H_S. The easy sufficient condition is a bound under each H_i separately, because H_S implies H_i for each i in S.

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
      = sum_{i in S} a_i,t
      <= sum_{i in S} G_i,t
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

## 4. Countable dynamic hypothesis families

For a countable universe, the construction requires:
- a fixed ex-ante sequence q_i >= 0 with sum_i q_i <= 1;
- a predictable baseline path and coordinate contributions;
- an auxiliary local path-adjuster F_S,t for every relevant finite or countable intersection S;
- a validity result for the actual true-null intersection I_0, including its countably infinite case;
- extension-coherent local certificates that do not change when future hypotheses are appended.

The finite Python checker below enumerates all nonempty subsets. It does not establish the countably infinite theorem or provide a scalable separation oracle. The global-projective affine construction (#9695) and persistent adjuster still require a careful composition proof.

## 5. Exact-rational reference checker

Files:
- validation/supremum_anchored_eclosure.py computes running coordinate contributions and local path certificates.
- validation/test_supremum_anchored_eclosure.py tests exact baseline retention, residual-budget arithmetic, pathwise closure, deterministic largest certified supersets and fail-closed budget/input checks.

This checker verifies deterministic inequalities only. It cannot establish the stochastic q_i or auxiliary running-supremum assumptions. Inputs called "adjusted auxiliary" must already be supported by a theorem/adjuster and source receipts.

## 6. Benchmark targets

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

## 7. Prior art

- Ramdas, Anchored multiple testing (October 2026): https://arxiv.org/abs/2610.11279
- Tavyrikov, Goeman & de Heide, Carefree multiple testing with e-processes, EJS 20(2), 2026: https://doi.org/10.1214/26-EJS2546
- Xu, Fischer & Ramdas, Improving online FDR procedures via online analogs of e-closure and compound e-values, UAI 2026: https://proceedings.mlr.press/v337/xu26a.html
