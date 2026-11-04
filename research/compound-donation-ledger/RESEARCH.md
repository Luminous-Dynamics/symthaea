# Research contract and benchmark plan

## Status and claim ceiling

This crate is a dependency-free, fixed-point reference prototype for Issue
[#9894](https://github.com/Luminous-Dynamics/symthaea/issues/9894), with a
finite staircase-calibrator implementation for Issue
[#9916](https://github.com/Luminous-Dynamics/symthaea/issues/9916).

It does not calculate or validate the input e-processes, establish the global
filtration, authenticate arrival timestamps, run a discovery optimizer, or
certify FDR/SupFDR in production. It is intentionally not a workspace member.
No compile, test, CI, or performance result is claimed here. The independent
verifier source and its two differential/tamper tests are authored but also
not compiled or executed in this environment.

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

### Alternative reserve policy: capped weighted overshoot

The ledger retains the fixed rational-rate baseline and adds a separate
SCORE-informed reserve policy. For a fixed global candidate weight
\(\gamma_i\in[0,1]\) and an already time-monotone evidence process \(X_{i,t}\),
define
\[
D_{i,t}
=\min\{(\gamma_iX_{i,t}-1)_+,\gamma_i\}.
\]
This is not the uncapped SCORE refund: it is a bounded transfer reserve
designed to satisfy the compound-donation donor budget.

For all \(x\ge0\) and \(\gamma\in[0,1]\),
\[
0\le \min\{(\gamma x-1)_+,\gamma\}
\le\gamma\min(x,1).
\]
If \(x\le1\), then \(\gamma x\le1\) and the left side is zero. If \(x>1\),
the left side is at most \(\gamma=\gamma\min(x,1)\).

The map \(f_\gamma(x)=\min\{(\gamma x-1)_+,\gamma\}\) is nondecreasing and
\(\gamma\)-Lipschitz. Therefore for nondecreasing evidence,
\[
0\le \Delta D_{i,t}\le\gamma_i\Delta X_{i,t}.
\]
The reserve satisfies the existing donor cap and own-increment cap, so the
prefix-budget and compound null-mass/SupFDR arguments in Sections 2–4 are
unchanged. No additional statistical assumption is introduced beyond the
validity and time monotonicity of \(X_i\), and the original fixed-destination
routing and predictable arrival requirements.

The bounded-refund tail argument at level \(a=\gamma_i\) also gives, under
the null,
\[
\mathbb E[D_{i,\infty}]
\le\gamma_i\log(1+\gamma_i)
\le\gamma_i^2.
\]
This is a tighter expected null debit than the worst-case deterministic
donor cap. It does **not** prove improved power. It suggests a policy that
preserves all evidence until the donor crosses its own weighted boundary,
then withdraws at most one unit of weighted mass.

The Rust writer implements this as the explicit
DonationReservePolicy::CappedOvershoot mode. The independent verifier
replays the same policy using its own BTreeMap state model; the rate policy
remains the default reference. Candidate specifications use zero reserve
rate in overshoot mode, and a missing destination disables donation for that
candidate. Both modes retain fixed destination IDs and exact receipt replay.

### Exploratory reserve-policy comparison

The new runner compares three methods on the same streams:
1. no donation;
2. fixed-rate donation at \(1/20\) of each donor's own weighted evidence
   increment, subject to the existing donor cap;
3. capped weighted overshoot, \(D_{i,t}=\min\{(\gamma_iX_{i,t}-1)_+,\gamma_i\}\).

The design uses \(m=100\), \(T=100\), \(\alpha=0.05\), 3,000 independent
simulation streams per cell, and the fixed global weights
\[
\gamma_0=0.50,\quad \gamma_1=0.01,\quad
\gamma_2=\cdots=\gamma_{99}=0.005,
\qquad \sum_i\gamma_i=1.
\]
Only candidate 0 has a precommitted destination (candidate 1); the other
candidates have no destination and therefore cannot donate. The donor is
strong (Bernoulli \(p=0.90\)) in the "strong-to-many-weak" scenario, and
candidates 1–40 are weaker alternatives (Bernoulli \(p=0.71\)); remaining
candidates are null (\(p=0.5\)). Other cells include all-null, weak-only and
strong-donor-only alternatives. Null likelihood-ratio factors are 1.5 on
success and 0.5 on failure, so each null coordinate has conditional mean-one
increments. The dependent generators share a uniform shock across null
coordinates or use a randomized balanced count of null successes while
preserving each coordinate's conditional success probability.

| Dependence | Alternative pattern | No-donation power | Fixed-rate power | Capped-overshoot power | No-donation SupFDR | Capped SupFDR |
|---|---|---:|---:|---:|---:|---:|
| Independent | Strong donor + weak alternatives | 43.34% | 43.77% | 43.77% | 0.0061% | 0.0061% |
| Independent | Weak alternatives only | 42.35% | 42.71% | 42.70% | 0.0047% | 0.0050% |
| Shared-uniform null | Strong donor + weak alternatives | 43.33% | 43.74% | 43.74% | 0.0555% | 0.0547% |
| Shared-uniform null | Weak alternatives only | 42.50% | 42.89% | 42.89% | 0.0000% | 0.0000% |
| Balanced-negative null | Strong donor + weak alternatives | 43.12% | 43.56% | 43.56% | 0.0033% | 0.0033% |
| Balanced-negative null | Weak alternatives only | 42.73% | 43.07% | 43.06% | 0.0151% | 0.0151% |

These results suggest a small, regime-dependent power gain of about 0.3–0.4
percentage points in the weak-signal cells when a high-weight donor can
transfer capital to a low-weight recipient. The fixed-rate and capped-
overshoot methods tie in the strong-donor cases because both reach the
donor's cumulative cap by the terminal look. In the weak-only cells, the
capped policy's extra donation is small and does not consistently improve on
the fixed-rate policy. In all-null cells the observed SupFDR is small, but
that is not evidence of a theorem or calibration at the nominal boundary.

At \(n=3{,}000\), the one-sided 95% Hoeffding half-width for each bounded
FDP-mean estimand is \(0.02234\), or 2.234 percentage points. The resulting
bounds are much wider than the observed point estimates. This experiment is
exploratory and only tests one weight topology, one alpha, one LR model and
three dependence constructions. It is **not** confirmatory evidence and it
does not justify superiority over fixed-rate donation, SCORE, e-LOND or
industry methods.

The committed runner and seed-by-cell output are
benchmarks/overshoot_donation_policy_comparison.py and
benchmarks/overshoot_donation_policy_comparison_2026-11-02.json.
The recorded run was executed by a private Python reimplementation matching
the committed harness logic; independently run the committed script before
using the numbers as confirmatory evidence.

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

A threshold-grid example explains why this is worth testing. With family size
m=1,000, target alpha=0.05, the weighted e-BH rank-r boundary is
h_r=m/(alpha r). For r in {1,10,100,1,000}, h_r is respectively
{20,000, 2,000, 200, 20}. A four-point equal-prior staircase can assign
mass pi=1/4 to each of those target e-values, placing its raw crossing
threshold at K_r=h_r/pi=4h_r. The square-root adjuster A(x)=sqrt(x)-1
reaches the same adjusted boundary h_r only at S=(h_r+1)^2.

| Rejection rank r | Required adjusted value h_r | Square-root raw supremum | Four-point staircase raw supremum |
|---:|---:|---:|---:|
| 1 | 20,000 | 400,040,001 | 80,000 |
| 10 | 2,000 | 4,004,001 | 8,000 |
| 100 | 200 | 40,401 | 800 |
| 1,000 | 20 | 441 | 80 |

This is a comparison of threshold geometry, not a power experiment or a claim
of dominance. The staircase prior allocates its entire mass to these four
rank thresholds and may be worse when the true discovery rank or family size
falls outside the grid. It motivates held-out threshold-prior optimization
and explicit out-of-grid stress tests. The calibrator's validity is retained
because the total prior mass is one; the power benefit, if any, must be
measured on matched independent confirmatory streams.

### Universal dyadic threshold prior

A useful non-trained, cross-horizon prior is
\[
K_j=2^j,\qquad \pi_j=\frac{1}{j(j+1)},\qquad j=1,2,\ldots.
\]
The masses sum to one exactly by telescoping. The contribution from threshold
\(K_j\) alone is
\[
\pi_jK_j=\frac{2^j}{j(j+1)}.
\]
For any target adjusted threshold \(h\geq1\), choose the smallest \(j_h\)
with \(2^{j_h}/[j_h(j_h+1)]\geq h\). Then the staircase calibrator reaches
at least \(h\) by raw running supremum \(2^{j_h}\). For \(j_h>1\), minimality
implies
\[
2^{j_h}<2h(j_h-1)j_h.
\]
Since \(j_h=\log_2 h+O(\log\log h)\), the raw-threshold overhead is
\(O((\log h)^2)\) relative to \(h\), rather than the \(O(h)\) multiplicative
overhead of the square-root adjuster, whose raw threshold is \((h+1)^2\).


For the implemented 64-point prefix, with each mass rounded down to
\(10^{-9}\), the exact integer staircase reaches the following target values
at these first dyadic raw thresholds:

| Target \(h\) | Dyadic raw supremum | Square-root raw supremum \((h+1)^2\) | Ratio |
|---:|---:|---:|---:|
| 1 | 2 | 4 | 0.500 |
| 20 | 1,024 | 441 | 2.322 |
| 200 | 32,768 | 40,401 | 0.811 |
| 2,000 | 524,288 | 4,004,001 | 0.131 |
| 20,000 | 8,388,608 | 400,040,001 | 0.021 |
| 1,000,000 | 536,870,912 | 1,000,002,000,001 | 0.000537 |
| 1,000,000,000 | 1,099,511,627,776 | 1,000,000,002,000,000,001 | 0.00000110 |

These are exact fixed-point threshold calculations for the finite staircase
formula (using the sum of all activated dyadic terms), not statistical
power measurements. The small-threshold penalty is real; at larger adjusted
thresholds the universal grid becomes much more efficient in raw-evidence
scale. A fixed-prior mixture with the square-root adjuster is an obvious
robustness candidate, but its threshold cost must be computed and its power
measured rather than inferred from this table.

This is an asymptotic threshold-geometry result, not a claim of empirical
power dominance. At moderate thresholds, the constants can make the dyadic
prior less efficient than the square-root adjuster; at other scales it may
be substantially better. A fixed mixture of the dyadic prior and smooth
adjusters is valid and can hedge the regimes, at the cost of allocating less
prior mass to each component. The finite Rust implementation includes a
truncated dyadic prefix with each rational mass rounded down, so its total
mass remains at most one. A finite prefix only covers a finite threshold
range; thresholds beyond its largest grid point are unsupported until the
prior is extended under a precommitted horizon-independent rule.

### e-BH rank-geometric staircase

For a fixed finite family-size envelope \(m\) and target level \(\alpha=a/b\), define the
geometric rank grid
\[
\mathcal R_m=\{1,2,4,\ldots,2^{\lfloor\log_2m\rfloor}\}\cup\{m\},
\qquad Q=|\mathcal R_m|.
\]
Give each grid point the fixed prior mass
\[
\pi_r=\frac{\lfloor P/Q\rfloor}{P},\qquad P=10^9,
\]
leaving any quantization remainder unspent. For each grid rank r, let
\[
h_r=\frac{m}{\alpha r},\qquad
H_r=\lceil P_E h_r\rceil,\qquad
K_r=\left\lceil\frac{H_r}{\pi_r}\right\rceil,
\]
where \(P_E=10^6\) is the evidence fixed-point scale. The displayed formula
uses equal weights \(\gamma_i=1/m\). For a candidate with an arbitrary
precommitted rational weight \(\gamma_i=u_i/v_i\), replace the boundary by
\(h_{i,r}=v_i/(\alpha r u_i)=1/(\alpha r\gamma_i)\) and compute
\(H_{i,r}=\lceil P_E h_{i,r}\rceil\). Each candidate receives its own
fixed calibrator; the ledger's global weight budget still requires
\(\sum_i\gamma_i\leq1\). The staircase
\[
A_m(x)=\sum_{r\in\mathcal R_m}\pi_rK_r\,\mathbf1\{x\ge K_r\}
\]
is valid by the same Ville budget proof as any other fixed staircase prior.
The implementation computes the ceilings from integer numerators and
denominators, so every supported grid rank has a single term whose
fixed-point contribution is at least its weighted e-BH boundary \(H_r\).

For any actual e-BH rejection rank \(k\), choose the largest power of two
\(r\le k\), unless \(k=m\) is the separately included terminal rank. Then
\(k/2<r\le k\), and therefore \(h_r<2h_k\). Since
\(Q=O(\log m)\) and \(\pi_r\) is within a fixed-point rounding factor of
\(1/Q\), the raw supremum threshold supporting rank k is bounded by
\[
K_r/P_E
\le
\frac{2Q}{1-Q/P}\,h_k+O(Q/P_E).
\]
Thus the raw-evidence penalty relative to the e-BH boundary is
\(O(\log m)\), while the square-root adjuster requires raw supremum
\((h_k+1)^2\). This is a family-size-aware, decision-targeted construction.
The formula follows from the staircase/Ville argument; it is not a claim of
a new general e-process or FDR theorem.

**Important horizon condition:** this calibrator is for a fixed precommitted
family size m (or a precommitted finite envelope \(m_{\max}\)). Recomputing a
different rank prior at every growing active-family size can change old
calibrated scores and break projectivity. For random arrivals inside a fixed
candidate universe, use the fixed envelope and retain the same calibrator;
this may be conservative when the active roster is much smaller. A countably
unbounded family requires a separate universal prior or fixed mixture over
predeclared family-size designs.

The Rust reference adds e_bh_rank_geometric(m, alpha_numerator,
alpha_denominator) for equal weights and weighted_e_bh_rank_geometric(m,
weight_numerator, weight_denominator, alpha_numerator, alpha_denominator)
for arbitrary precommitted candidate weights. Both use O(log m) prior points,
exact rational ceiling arithmetic, merge coincident integer thresholds, and
precompute prefix numerators so evaluation is O(log log m) (binary search
over O(log m) points). The new projective_weighted_e_bh_rank_prior constructor
instead uses a fixed 64-point rank prior, does not take family size as input,
and is unchanged when higher-ID hypotheses arrive. The independent verifier
continues to use a linear scan.

### Exploratory rank-calibration experiment

Fixed seed 20261102; 3,000 independent Monte Carlo streams per cell;
m=100; T=100; alpha=0.05; null observations Bernoulli(0.5); alternative
observations Bernoulli(0.75); per-round likelihood-ratio factor is
\(1.5\) on a success and \(0.5\) on a failure. The same LR process is used
for every hypothesis, so each null coordinate is a test martingale.
Compared three time-monotone calibrators: square-root, the 25-point universal
dyadic prior, and the e-BH rank-geometric prior. At every round, ordinary
e-BH is applied to the calibrated values. SupFDR is the Monte Carlo mean of
the per-stream maximum FDP across all 100 rounds; power is the terminal
fraction of alternatives rejected. A second run uses balanced negative
dependence among null coordinates: each round has a random subset with an
exactly randomized floor/ceiling number of successes so every null coordinate
still has conditional success probability 0.5 given the preceding global
history.

| Null dependence | Alternative fraction | Square-root power | Dyadic power | Rank-geometric power | Rank-geometric SupFDR | Rank-geometric P(any false) |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 5% | 58.73% | 70.42% | 90.44% | 0.843% | 3.37% |
| Independent | 20% | 82.93% | 84.16% | 95.48% | 0.898% | 11.13% |
| Independent | 50% | 91.52% | 89.23% | 96.62% | 0.552% | 14.47% |
| Balanced negative | 0% | 0% | 0% | 0% | 0.333% | 0.33% |
| Balanced negative | 5% | 58.42% | 70.25% | 90.48% | 0.627% | 2.67% |
| Balanced negative | 20% | 82.54% | 83.81% | 95.55% | 0.948% | 11.53% |
| Balanced negative | 50% | 91.63% | 89.35% | 96.65% | 0.553% | 15.17% |

These cells suggest a potentially large power gain for the rank-aligned prior
in this fixed-family Bernoulli likelihood-ratio model. They do not establish
superiority: the design is specialized to m=100 and alpha=0.05, there are only
3,000 streams per cell, the alternatives are homogeneous and strong/moderate,
and the dependence experiment covers one balanced negative-dependence
construction rather than arbitrary copulas. SupFDR is a mean of a bounded
per-stream maximum-FDP variable; the one-sided 95% Hoeffding half-width at
n=3,000 is 0.02235. P(any false) is a separate Bernoulli/FWER diagnostic,
not the FDR estimand. These results are exploratory and require frozen
confirmatory seeds, heterogeneous and switching alternatives, positive and
negative dependence, and family-size/alpha transfer tests.

### Horizon-independent projective rank prior

The fixed-family rank-geometric prior above allocates equal mass over ranks
for one precommitted family-size envelope. It can be powerful, but recomputing
it as the active family grows changes old candidate calibrators and violates
projectivity. A different construction removes family size entirely.

For candidate \(i\), fix its global summable weight \(\gamma_i>0\) and target
level \(\alpha\). For every dyadic rank \(r_j=2^j\), \(j=0,1,\ldots\), allocate
\[
\pi_j=\frac{1}{(j+1)(j+2)},\qquad \sum_{j=0}^{\infty}\pi_j=1.
\]
Let \(S_{i,t}=\sup_{s\le t}E_{i,s}\) be the running supremum of its base
e-process, and define
\[
K_{i,j}=
\max\left\{1,\left\lceil
\frac{1}{\alpha\gamma_i\,2^j\,\pi_j}
\right\rceil\right\},
\qquad
V_{i,t}=\sum_{j=0}^{\infty}
\pi_jK_{i,j}\mathbf1\{S_{i,t}\ge K_{i,j}\}.
\]
The clamp at one is valid because a running supremum starting at one is
always at least one; when the un-clamped threshold is below one, replacing it
by one only decreases the term's contribution relative to the intended
threshold condition. More directly, if the e-BH boundary is below one, the
baseline crossing already occurs at \(S=1\).

By Ville's inequality,
\[
\mathbb E[V_{i,\infty}]
\le\sum_j\pi_j\le1.
\]
It is adapted and pathwise nondecreasing, so it is a valid time-monotone
e-process. Crucially, it depends on the candidate's fixed \(\gamma_i\) and
\(\alpha\), but **not on the number of currently active hypotheses**.

For any actual e-BH rejection rank \(k\), choose
\(j=\lfloor\log_2 k\rfloor\), giving \(2^j\le k<2^{j+1}\). The single
corresponding staircase term satisfies
\[
\pi_jK_{i,j}
\ge\frac{1}{\alpha\gamma_i2^j}
\ge\frac{1}{\alpha\gamma_i k}.
\]
Thus it reaches the candidate's weighted e-BH rank-k boundary. Conversely,
because \(k<2^{j+1}\), its raw supremum threshold is bounded, up to fixed-
point rounding and the lower clamp at one, by
\[
K_{i,j}
\lesssim
\max\left\{1,\,
\frac{2(j+1)(j+2)}{\alpha\gamma_i k}
\right\}.
\]
Since \(j=O(\log k)\), the rank overhead is \(O((\log k)^2)\), while a
square-root adjuster requires raw supremum proportional to the square of
the adjusted boundary. This is the price paid for covering every possible
future rank with one precommitted prior.

The finite Rust constructor stores j=0..63. The masses sum to 64/65, leaving
the tail unspent, and cover every rejection rank representable in a u64
finite family (maximum rank below 2^64). Each candidate's rational global
weight must itself be precommitted and must not be renormalized when new
hypotheses arrive. This makes the calibrator itself projective under
initial-segment horizon extension.

This is a direct construction from Ville's inequality plus a dyadic rank
cover. The possible contribution is its use as a fixed, projective,
decision-targeted adjuster for weighted SupFDR, along with a careful empirical
comparison against horizon-specific calibration. It is not claimed as a new
general e-process theorem.

### Constant-factor finite-u64 rank prior

The pure telescoping prior is more power-efficient at early ranks but assigns
mass of order \(1/j^2\) to rank \(2^j\), which gives a worst-case factor of
\(O((\log k)^2)\). Because this implementation already uses u64 candidate
indices, we can explicitly cover all 64 dyadic rank scales with a fixed
uniform component.

Define
\[
\widetilde\pi_j=
\frac12\frac{1}{(j+1)(j+2)}+\frac{1}{128},
\qquad j=0,\ldots,63.
\]
The first component is half of the telescoping prior; the second is half of
a uniform prior over the 64 representable dyadic scales. Its total mass is
\[
\sum_{j=0}^{63}\widetilde\pi_j
=\frac12\frac{64}{65}+\frac12
=\frac{129}{130}<1.
\]
The unspent mass is \(1/130\). In fixed-point arithmetic, each mass is
rounded down, preserving validity.

Since \(\widetilde\pi_j\ge1/128\), the raw threshold for rank \(2^j\) is no
larger than \(128/(\alpha\gamma_i2^j)\), up to fixed-point rounding. For any
actual rank \(k\), choose \(2^j\le k<2^{j+1}\), so
\[
K_{i,j}\lesssim
\max\left\{1,\frac{256}{\alpha\gamma_i k}\right\}.
\]
This is a **uniform constant-factor (256) overhead** over the weighted e-BH
boundary for every u64 rank, independent of the active family size. The
finite-u64 cap is part of the theorem statement; it is not an infinite-rank
result. The pure telescoping prior remains available for smaller early-rank
overhead, and a precommitted fixed-prior mixture can trade early-rank power
against the worst-case constant.

The Rust constructor is
\`projective_weighted_e_bh_rank_prior_hybrid_u64\`. It uses exact integer
prior masses, threshold ceilings and the same binary-search evaluator. The
independent verifier does not share the writer's staircase lookup structure.

### Exploratory projective-rank experiment

We extended the Bernoulli likelihood-ratio screen with the horizon-independent
projective rank prior. Fixed seed 20261102; 3,000 independent streams per
cell; m=100; T=100; alpha=0.05; null p0=0.5; alternative p1=0.75; per-round
LR factors 1.5 and 0.5. Compared square-root, 25-point universal dyadic,
fixed-family equal-prior rank-geometric, projective telescoping rank, and
projective hybrid (half telescoping, half uniform-u64) calibrators. SupFDR is the mean over streams of max_t FDP_t; power is the
terminal fraction of alternatives rejected. Balanced negative dependence
uses a random floor/ceiling count of null successes with a uniformly selected
subset, preserving conditional marginal success probability 0.5 for every
null coordinate.

| Null dependence | Alternative fraction | Square-root power | Universal dyadic power | Fixed-family rank power | Projective rank power | Projective SupFDR |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 5% | 58.73% | 70.42% | 90.44% | 88.84% | 0.573% |
| Independent | 20% | 82.93% | 84.16% | 95.48% | 90.69% | 0.314% |
| Independent | 50% | 91.52% | 89.23% | 96.62% | 92.38% | 0.121% |
| Balanced negative | 5% | 58.42% | 70.25% | 90.48% | 89.12% | 0.453% |
| Balanced negative | 20% | 82.54% | 83.81% | 95.55% | 90.48% | 0.306% |
| Balanced negative | 50% | 91.63% | 89.35% | 96.65% | 92.49% | 0.115% |

The projective rank prior gives up some power relative to the finite-family
specialist, as expected, but remains materially ahead of square-root and
universal dyadic in this one model while keeping the same candidate
calibrator as the family grows. Its global-null SupFDR estimates were 1.667%
(independent) and 1.767% (balanced negative); in mixed-alternative cells the
largest projective SupFDR estimate was 0.573% (independent) and 0.453%
(balanced negative). At n=3,000 the one-sided 95% Hoeffding half-width is
0.02235. Thus the largest projective SupFDR point estimate has a conservative
upper bound below 4.01% in these cells.

This is an exploratory result, not a guarantee or SOTA claim. It is one
family size, one alpha, one likelihood-ratio model, homogeneous signals and
one negative-dependence design. The committed run manifest and script must
be independently reproduced before confirmatory use. The next experiment
must test family-size transfer with a fixed global weight sequence, multiple
alpha values, heterogeneous and switching alternatives, time-varying
dependence, delayed/predictable arrivals and 50,000+ frozen streams.

### Fixed-weight family-size transfer screen

A second exploratory screen checks the projective property under active-family
growth. It uses one global weight sequence with \(\gamma_i=1/1000\) for every
candidate, and compares initial prefixes of size \(m\in\{10,100,1000\}\).
The weights are not renormalized at the smaller horizons. This is the correct
way to test a projective weighted procedure; replacing \(\gamma_i\) by \(1/m\)
at each prefix would change old candidates' calibrators and confound the test.

The run used 300 streams per cell, 80 looks, alpha 0.05, null Bernoulli
probability 0.5 and alternative probability 0.75. The data generator is the
same likelihood-ratio test-martingale model as the previous screen. Power
is terminal alternative rejection fraction.

| Active prefix m | Alternative fraction | Square-root | Universal dyadic | Fixed-family rank | Projective rank | Hybrid projective rank |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10% | 0.0% | 8.7% | 43.0% | 48.7% | 45.7% |
| 10 | 50% | 1.8% | 18.4% | 60.8% | 55.5% | 49.9% |
| 100 | 10% | 2.0% | 19.9% | 64.9% | 55.9% | 51.4% |
| 100 | 50% | 14.4% | 42.4% | 78.6% | 62.6% | 61.8% |
| 1,000 | 10% | 37.1% | 55.7% | 82.4% | 68.9% | 69.1% |
| 1,000 | 50% | 80.3% | 75.7% | 90.9% | 75.4% | 79.3% |

This is intentionally not presented as a win. The fixed-family specialist often
has the greatest power, because it spends its whole prior budget on the
currently known rank range. The projective prior beats the square-root and
universal-dyadic calibrators in several sparse/short-prefix cells, but is
worse in other cells. The hybrid is a little more robust at the highest ranks
but loses some early-rank power. This is the expected price of preserving
calibration under horizon extension.

There were only 300 streams per cell. The one-sided Hoeffding half-width for
a bounded SupFDR mean is about 7.06 percentage points, so the SupFDR point
estimates in the JSON artifact are descriptive only. This screen is a
projectivity/power-transfer experiment, not confirmatory error-rate evidence.
The recorded output is
\`benchmarks/projective_rank_horizon_transfer_2026-11-02.json\`, and the
committed runner is
\`benchmarks/projective_rank_horizon_transfer.py\`.

## Persistent SCORE-style overshoot refund primitive

This is a separate alpha-wealth accounting experiment, not a change to the
source-tagged evidence-capital ledger. It is motivated by SCORE (Kuang, Gang
and Xia, ICML 2026), whose one-shot online procedures refund the overshoot
\((\alpha e-1)_+\) and whose SCORE+ retroactive variant uses conditional
positive quadrant dependence (CPQD). The repeated-evidence extension must
not silently inherit a one-e-value-per-arrival theorem or the CPQD-specific
SCORE+ result.

Let \(V_{i,t}\) be an already-valid, nonnegative, adapted, pathwise
nondecreasing e-process under the null, with a **known, precommitted initial
value** \(V_{i,0}=v_{i,0}\ge0\), and assume its conditional terminal budget
at activation satisfies
\[
\mathbb E[V_{i,\infty}\mid\mathcal F_{i,0}]\le1.
\]
The common normalized case is \(v_{i,0}=1\), but adjusted processes such as
the staircase calibrator may legitimately start at zero. The initial value
must be supplied to both the writer and independent verifier; it is not
inferred from the first observed increment.
Fix a predictable rational test level \(a_i\in(0,1)\) before observing the
process. Define
\[
O_{i,t}=\min\{(a_iV_{i,t}-1)_+,a_i\},\qquad
C_{i,t}=a_iV_{i,t}-O_{i,t}.
\]
The exact piecewise form of the cost is
\[
C_{i,t}=
\begin{cases}
a_iV_{i,t},&a_iV_{i,t}<1,\\
1,&1\le a_iV_{i,t}\le1+a_i,\\
a_iV_{i,t}-a_i,&a_iV_{i,t}>1+a_i.
\end{cases}
\]
Thus \(O_{i,t}\) and \(C_{i,t}\) are both nonnegative and pathwise
nondecreasing, \(0\le O_{i,t}\le a_i\), and
\[
\mathbf1\{a_iV_{i,t}\ge1\}\le C_{i,t}\le a_iV_{i,t}.
\]
Under the stated conditional e-process budget,
\[
\mathbb E[C_{i,\infty}\mid\mathcal F_{i,0}]\le a_i.
\]

The refund itself has a sharper conditional bound. For \(0\le u<a_i\),
\[
\Pr(O_{i,\infty}>u\mid\mathcal F_{i,0})
\le
\Pr\!\left(V_{i,\infty}>\frac{1+u}{a_i}\mid\mathcal F_{i,0}\right)
\le\frac{a_i}{1+u}.
\]
Integrating the tail,
\[
\mathbb E[O_{i,\infty}\mid\mathcal F_{i,0}]
\le a_i\log(1+a_i)
\le a_i^2.
\]
This is a conditional expected refund bound for one null process; it does
not establish that the refund can be spent without limit or that an entire
online FDR procedure is valid.

### Uncapped SCORE refund versus bounded operational refund

The first implementation uses a capped refund to bound individual exposure.
The published SCORE algebra itself uses the **uncapped** overshoot
\[
O^{\mathrm{SCORE}}_{i,t}=(a_iV_{i,t}-1)_+,
\qquad
C^{\mathrm{SCORE}}_{i,t}
=a_iV_{i,t}-O^{\mathrm{SCORE}}_{i,t}
=\min\{a_iV_{i,t},1\}.
\]
This residual cost is nonnegative, pathwise nondecreasing when V is, and
\[
\mathbf1\{a_iV_{i,t}\ge1\}\le C^{\mathrm{SCORE}}_{i,t}\le a_iV_{i,t}.
\]
For a null e-process,
\[
\mathbb E[O^{\mathrm{SCORE}}_{i,\infty}\mid\mathcal F_{i,0}]
\le
a_i\mathbb E[V_{i,\infty}\mid\mathcal F_{i,0}]
\le a_i.
\]
The uncapped refund is not bounded pathwise, but its conditional expectation
is. The residual cost is capped at one, exactly matching the one-shot SCORE
identity.

The original capped design remains useful as an exposure-controlled variant:
\[
O^{\mathrm{cap}}_{i,t}
=\min\{(a_iV_{i,t}-1)_+,a_i\},
\qquad
C^{\mathrm{cap}}_{i,t}=a_iV_{i,t}-O^{\mathrm{cap}}_{i,t}.
\]
Its refund is pathwise at most a_i and has the sharper expectation bound
\[
\mathbb E[O^{\mathrm{cap}}_{i,\infty}\mid\mathcal F_{i,0}]
\le a_i\log(1+a_i)\le a_i^2,
\]
while its cost may exceed one for very large V. The uncapped variant offers
more refund potential but larger realized tail exposure; the capped variant
trades that potential for a deterministic exposure ceiling. Both are
exactly-once processes, and neither is yet a complete global wealth-spending
algorithm.

The Rust reference now implements both variants separately:
- \`ProcessOvershootRefundUncapped\`: full SCORE-style overshoot, with
  residual cost min(aV,1);
- \`ProcessOvershootRefund\`: capped overshoot, with residual cost aV-O_cap.

The verifier has separate transition and sequence replay functions for the
uncapped process, so the two definitions cannot be confused by a receipt
consumer.

### Exactly-once fixed-point implementation

The Rust reference exposes `ProcessOvershootRefund::new` for the normalized
initial value one and `with_initial_evidence_value` for a precommitted
initial value such as zero. The independent sequence verifier has matching
normalized and explicit-initial entry points. It checks the previous
cumulative refund against the previous evidence value at every transition,
not merely that the cumulative amount has not decreased. The Rust reference
stores the refund as the exact numerator
\[
R_{i,t}=
\min\!\left\{
(a_{i,\mathrm{num}}V_{i,t,\mathrm{ticks}}
-a_{i,\mathrm{den}}P_E)_+,\,
a_{i,\mathrm{num}}P_E
\right\}
\]
over denominator \(a_{i,\mathrm{den}}P_E\), where \(P_E=10^6\).
The update receipt contains:
- the newly observed monotone evidence value;
- the cumulative capped refund numerator;
- the newly released increment \(R_{i,t}-R_{i,t-1}\);
- the effective cost numerator \(a_{i,\mathrm{num}}V_{i,t,\mathrm{ticks}}-R_{i,t}\);
- the exact rejection-boundary bit.

The state rejects evidence rollback and round rollback. Updates are staged and
committed atomically. A separate verifier recomputes every field from the
policy, previous state and new evidence, and rejects repeated or forged
refund increments. It does not call the writer transition logic.

### Critical theorem still open

The algebra above only says a persistent test process has a bounded cumulative
overshoot account and a cost that upper-bounds its rejection indicator. It is
not yet an online FDR algorithm. In particular, \(C_{i,t}\) increases as old
evidence accumulates. A future spending policy must reserve for that
continuing increase and must not spend the same refund twice. Each
\(\Delta O_{i,t}\) becomes available only after observation t and may fund
only later predictable tests. The required global alpha-wealth/FDP estimator
must remain valid at every stopping time under the stated dependence
conditions.

Do not merge the alpha-refund units with the weighted evidence-capital units
in the compound donation ledger. They are different accounting systems until
a joint theorem explicitly connects them. The candidate direction is to use
the process refund as a second, source-tagged alpha-wealth ledger, with its own
independent verifier, before attempting composition with #9894.

The writer and verifier unit tests are authored but not compiled or executed
in this environment. No production, FDR, SupFDR, or SOTA claim is made.

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
- SCORE: Kuang, Gang and Xia, ICML 2026, *SCORE: A Unified Framework for
  Overshoot Refund in Online FDR Control*. It introduces overshoot refunds
  for e-LOND/e-LORD/e-SAFFRON by using
  \(\mathbf1\{\alpha_j e_j\ge1\}\le\alpha_j e_j-
  (\alpha_j e_j-1)_+\). The basic SCORE methods retain finite-sample FDR
  control; SCORE+ allows retroactive past-cost updates under conditional
  positive quadrant dependence (CPQD). Include the published method and
  author code, if available, as a mandatory one-shot baseline. Do not assign
  the CPQD assumption to ordinary SCORE, and do not reuse the one-shot theorem
  for repeated evidence by refreshing old e-values. The PMLR proceedings page
  links the paper PDF and OpenReview but does not expose an implementation
  repository; record any independently located author code separately rather
  than inventing a code baseline.
  Paper: https://proceedings.mlr.press/v306/kuang26c.html
  Preprint: https://arxiv.org/abs/2601.20386
- e-GAI: Zhang, Wei, Ren and Zou, ICML 2025. The framework introduces
  e-LORD and e-SAFFRON, with dynamic test-level allocation and strategies
  against alpha-death. Include their paper and author reference implementation
  as serious online e-value investing comparators; inspect exact theorem
  assumptions before placing a method in an arbitrary-dependence lane.
  Paper: https://proceedings.mlr.press/v267/zhang25cd.html
  Code: https://github.com/zijianwei01/e-GAI
- Adaptive sequential mergers: Vovk and Wang (2024), Merging sequential
  e-values via martingales. Include as a matched baseline only when the
  order in which same-round coordinate observations are revealed preserves
  the required conditional validity.
- P-value/e-value software: Bioconductor onlineFDR stable 2.20.0
  (Bioconductor 3.23) and development 2.21.2 (Bioconductor 3.24) both matter.
  The package supports online FDR/FWER procedures for sequential hypothesis
  arrivals. Pin the exact release/devel source commit, R version, package
  version and method defaults. Do not benchmark only the stable release when
  the devel line has advanced.
- Python online-fdr: its maintained package exposes sequential/batch
  procedures and e-process utilities; its parity tests have historically
  targeted an older onlineFDR release. Pin its exact commit and run parity
  against both stable 2.20.0 and devel 2.21.2. A passing older parity suite is
  not evidence of parity with current Bioconductor versions.
- e-GAI reference software: the paper authors publish an R repository with
  e-LORD, e-SAFFRON, pL-RAI and pS-RAI implementations and simulation drivers:
  https://github.com/zijianwei01/e-GAI . Include this repository as a
  reproducibility baseline, and preserve its theorem assumptions and exact
  version/commit. The repository includes e-LORD/e-SAFFRON and pL-RAI/pS-RAI
  functions plus simulation and real-data drivers; it is a useful direct
  algorithm implementation, not a substitute for matching the paper's
  dependence assumptions.

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

Use at least 50,000 frozen independent streams per confirmatory statistical
cell; target 200,000 or more for low-tail diagnostics. Preregister seeds and
generators before looking at confirmatory results.

Interval types must match the estimand. Fixed-time FDR, stopped-FDR and
SupFDR are means of per-stream FDP-valued variables in [0,1]; use a preregistered
bounded-mean interval (e.g. empirical Bernstein/Hoeffding) or an independent-
stream bootstrap. Wilson/Clopper-Pearson intervals apply to Bernoulli
estimands such as probability of any false discovery and FDX exceedance.
Do not require a finite-sample upper interval strictly below alpha when the
theorem allows equality at alpha. Predeclare a Monte Carlo tolerance epsilon,
compare a one-sided upper bound with alpha+epsilon, and include conservative
interior levels to check implementation slack. Monte Carlo intervals diagnose
implementation/calibration; they do not substitute for a theorem.

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
2. Compile and run the separate linear-scan verifier crate; add cross-implementation
   differential tests against the writer and verify that its code never invokes
   DonationLedger or writer transition helpers.
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
