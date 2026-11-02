# Research contract and benchmark plan

## Status and claim ceiling

This crate is a dependency-free, fixed-point reference prototype for Issue
[#9894](https://github.com/Luminous-Dynamics/symthaea/issues/9894), with a
finite staircase-calibrator implementation for Issue
[#9916](https://github.com/Luminous-Dynamics/symthaea/issues/9916).

It does not calculate or validate the input e-processes, establish the global
filtration, authenticate arrival timestamps, run a discovery optimizer, or
certify FDR/SupFDR in production. It is intentionally not a workspace member.
No compile, test, CI, or performance result is claimed here.

The prototype uses a deliberately restricted routing rule: each donor's
recipient ID is committed in its candidate specification. Adaptive recipient
selection is not implemented. This avoids allowing a longer hypothesis
horizon to rewrite the destination of an earlier escrow token.

## 1. Mathematical objects

Let the candidate universe have a fixed global order. Each candidate has a
precommitted weight gamma_i >= 0, with sum_i gamma_i <= 1. Let X_i,t be a
nonnegative, adapted, pathwise nondecreasing e-process under its own null,
with E[X_i,infinity] <= 1. The base process is already time-monotone: this
crate does not infer that property from arbitrary inputs.

For a donor with fixed rational reserve rate a_i / b_i, the reference
implementation applies a fraction of its own new weighted evidence increment
to escrow. It also enforces the donor cap

    O_i,t <= gamma_i min(X_i,t, 1).

No incoming amount can finance a new reservation.

Each token has a source ID and a precommitted recipient ID greater than the
source. If that recipient is absent from the current finite horizon or is
dormant, the token remains in escrow. A token may be credited in pieces, but
the total credited amount cannot exceed its original amount.

Define incoming and outgoing weighted capital I_i,t and O_i,t. The compound
score is

    C_i,t = X_i,t + (I_i,t - O_i,t) / gamma_i

for positive weights. The implementation stores the numerator

    gamma_i C_i,t = gamma_i X_i,t + I_i,t - O_i,t

so there is no floating-point division.

## 2. Ledger invariants

The required invariants are:

1. O_i,t and I_i,t are nonnegative and nondecreasing.
2. O_i,t <= gamma_i min(X_i,t,1).
3. Each reserve increment is no larger than the donor's own new weighted
   evidence increment: Delta O_i,t <= gamma_i Delta X_i,t.
4. Credits travel only to a precommitted later recipient; no token is credited
   beyond its remaining balance.
5. Every finite prefix obeys
   sum_{i<=m}(I_i,t - O_i,t) <= 0.
6. A dormant candidate has zero evidence and zero incoming/outgoing capital.
7. Existing candidate scores do not change when a later, dormant candidate is
   appended with new weight rather than renormalizing existing weights.
8. Escrow conservation is exact:
   uncredited token value = total outgoing - total incoming.

The fixed-point reserve computation carries a rational remainder. If the
weighted evidence increment is q, then

    floor((a_i q + r_i) / b_i) <= q

because 0 <= a_i <= b_i and 0 <= r_i < b_i. This prevents a rounding
remainder from borrowing against future evidence.

The current reference implementation scans all candidates and tokens after
each event to validate invariants. Its present update cost is therefore
O(n + T), not O(log n). A production optimization should use an augmented
order-statistic tree or segment tree whose nodes store subtree balance and
maximum prefix balance. A donor reserve decreases one point balance; a
recipient credit increases one point balance; the root maximum-prefix query
checks all prefix budgets in O(log n). The independent verifier should remain
a linear scan, rather than reusing that optimized tree.

## 3. Compound null-mass bound

Let N(P) be the fixed true-null set under data-generating law P. For any
finite prefix [m], write A(P)=[m]\\N(P) for its alternatives. Since the
total prefix balance is nonpositive,

    sum_{i in N(P) intersect [m]} (I_i,t - O_i,t)
    <= sum_{i in A(P)} (O_i,t - I_i,t)
    <= sum_{i in A(P)} O_i,t
    <= sum_{i in A(P)} gamma_i min(X_i,t,1).

Consequently,

    sum_{i in N(P) intersect [m]} gamma_i C_i,t
    <= sum_{i in N(P) intersect [m]} gamma_i X_i,t
       + sum_{i in A(P)} gamma_i min(X_i,t,1).

Taking expectations uses only null validity of each base process and the
pointwise bound min(X,1) <= 1 for alternatives:

    E_P[sum_{i in N(P) intersect [m]} gamma_i C_i,t]
    <= sum_{i<=m} gamma_i
    <= 1.

Taking increasing limits in the finite prefix and time horizon gives the
terminal null-mass bound, provided the candidate universe is fixed and the
usual measurability/monotone-convergence conditions hold:

    E_P[sum_{i in N(P)} gamma_i C_i,infinity] <= 1.

This proof does not make each C_i an individual e-process. It establishes a
compound weighted null-mass bound.

## 4. SupFDR consequence

Suppose every finite rejection set R_t is measurable with respect to the
global history and satisfies, for each rejected candidate,

    gamma_i C_i,t >= 1/(delta |R_t|).

Because C_i,t is nondecreasing and bounded above by C_i,infinity, pathwise,

    FDP_t
    = |R_t intersect N(P)| / |R_t|
    <= delta sum_{i in N(P)} gamma_i C_i,infinity

for every time t. Hence E_P[sup_t FDP_t] <= delta.

This is the main proof obligation for the overall method. The algebra is
closely related to online compound e-values built by donations (Xu, Fischer
and Ramdas, UAI 2026); the project is not claiming a new general SupFDR
inequality. The potential contribution is an auditable, process-valued
realization with monotone interim scores, fixed horizon-independent routing,
and exact event replay. The result still depends on the stated e-process,
global-filtration, fixed-universe, finite-rejection-set and arrival assumptions.

## 5. Staircase calibration

For a base e-process E_t, set S_t = sup_{s<=t} E_s. For thresholds K_j >= 1
and fixed prior masses pi_j >= 0 with sum_j pi_j <= 1, define

    A_pi(x) = sum_j pi_j K_j 1{x >= K_j}.

Ville's inequality gives P(S_infinity >= K_j) <= 1/K_j, so

    E[A_pi(S_infinity)]
    = sum_j pi_j K_j P(S_infinity >= K_j)
    <= sum_j pi_j
    <= 1.

The resulting process is pathwise nondecreasing and is therefore a
time-monotone e-process. The implementation supports finite grids, exact
integer thresholds, inclusive boundary ties and conservative downward
rounding. It does not optimize the grid. Threshold selection must be
precommitted or trained on data independent of the confirmatory streams under
a justified conditional-validity argument.

Dynamic e-closure (de Heide, 2026, Proposition 2.9 and Appendix B.6) gives a
broader characterization of time-monotone e-processes and shows adjusted
running maxima are not exhaustive. The staircase family is a discrete,
auditable design space within that broader class; the formula itself follows
from Ville's inequality and is not claimed as a novel theorem.

## 6. Prior art and software baselines

The benchmark must keep guarantee classes separate.

- Donation e-LOND: Xu, Fischer and Ramdas, UAI 2026. Published one-e-value-
  per-arriving-hypothesis online compound e-values and an O(log t) decision
  implementation. This is the essential donation baseline; do not silently
  replace a past e-value by a later snapshot and reuse the one-shot theorem.
- Online e-closure: compare to its finite-family and online variants only
  with matched arrival and information timing.
- Persistent multiple testing: Tavyrikov, Goeman and de Heide, Electronic
  Journal of Statistics 20(2), 2026. Use valid adjuster-based persistent
  e-values; raw running maxima with ordinary e-BH are not a valid baseline.
- Dynamic e-closure: de Heide, arXiv:2608.09927 (August 2026). Compare
  future-extension coherence, stopping and time-monotone persistence under
  its global-filtration assumptions.
- Adaptive sequential mergers: Vovk and Wang (2024), Merging sequential
  e-values via martingales. Include as a matched baseline only when the
  order in which same-round coordinate observations are revealed preserves
  the required conditional validity.
- P-value software: Bioconductor onlineFDR provides LOND, LORD, SAFFRON,
  ADDIS and supLORD, among other procedures. A Python online-fdr package also
  exposes sequential/batch p-value and e-value methods and optional parity
  tests against the R package. These are useful API, parity and systems
  baselines; their dependence assumptions and exact guarantee classes must
  be carried through rather than treated as interchangeable.

## 7. Confirmatory benchmark protocol

### Matched scenarios

Use a shared manifest binding:
- base evidence generator and source SHA;
- global filtration and within-round reveal semantics;
- fixed candidate universe, weights, arrival policy and delays;
- null configuration and alternative family;
- target level, stopping/reporting schedule and rejection rule;
- adjuster grid/prior, reserve policy and fixed destination mapping;
- hardware, OS, Rust compiler, thread count and benchmark binary digest.

### Statistical matrix

- Family sizes: 10, 100, 1,000, 10,000, 100,000 and 1,000,000.
- Horizons: 100, 1,000 and 10,000 evidence rounds.
- Dependence: independent; block-correlated; shared-control/common-shock;
  time-varying; and adversarial coupling where every coordinate still meets
  its conditional null budget in the declared global filtration.
- Alternatives: global null; one strong; sparse; dense weak; heterogeneous;
  clustered; switching; late-arriving; delayed and bursty arrivals.
- Evidence increments: bounded likelihood-ratio factors, Rademacher factors,
  Gaussian factors with exact conditional normalization, and heavy-tailed
  nonnegative factors whose conditional expectation is verified not to exceed
  one. Do not label an arbitrary heavy-tailed simulation valid without this
  check.

### Metrics

Report separately:
- fixed-time FDR, stopped-FDR, SupFDR, FDX and probability of any false
  discovery, with confidence intervals;
- power, true discoveries, precision/recall, time-to-first and time-to-useful
  discovery;
- donation amount, stranded escrow, recipient concentration and policy
  regret;
- p50/p95 update and decision latency, verifier latency, peak memory,
  serialized receipt size, allocation count and throughput;
- null-budget utilization and conservatism introduced by calibration.

Use at least 50,000 frozen streams per confirmatory statistical cell; target
200,000 or more for low-tail diagnostics. Preregister seeds and generators
before looking at confirmatory results. Monte Carlo intervals are calibration
diagnostics, not substitutes for a theorem.

### Systems targets

The current reference path deliberately performs a full invariant scan after
each update. It is a correctness oracle, not a performance implementation.
The optimized target is:
- O(log n) amortized reserve/credit and prefix-invariant update using an
  augmented balanced tree;
- O(n + e) independent receipt verification, where e is the event count;
- exact agreement with the reference implementation on bounded generated
  traces;
- stable memory use and no floating-point threshold-boundary differences.

Do not claim SOTA from these targets. A superiority claim requires exact
same-machine comparison with donation e-LOND's published O(log t) method,
persistent adjuster baselines, and maintained R/Python implementations, under
matched guarantees and data assumptions.

## 8. Required next steps

1. Compile and run this standalone crate in a connected Rust checkout; record
   exact source SHA, toolchain, test command and result.
2. Add a separately implemented verifier using a different data structure,
   not just replay through the same transition engine.
3. Add negative controls for missing/duplicate reservation rows, wrong
   recipient, repeated/overdrawn credit, evidence rollback, round rollback,
   invalid weights/rates, overflow and altered final snapshots.
4. Add authenticated source-arrival receipts by integrating with #9809.
5. Add a Merkle commitment over canonical event rows, while keeping the
   verifier independent of the writer.
6. Replace the full-scan prefix check with an augmented balanced tree and
   retain the linear scan as an oracle.
7. Run the matched confirmatory protocol before any statistical superiority
   statement.
