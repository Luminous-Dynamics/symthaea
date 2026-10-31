# Exact finite-exchangeability CRC recertification prototype

**Status:** exact-rational finite-support research oracle. It is not a continuous-observation conformal implementation and makes no empirical SOTA claim.

## Why this is a separate high-value direction

Bai & Ramdas (2026), arXiv:2610.35821, develop recertification for conformal risk control (CRC) as well as randomized e-BH. The CRC construction is especially useful for our proof-carrying statistics architecture: it turns unused risk budget on every hypothetical augmented sample into a pointwise less conservative controller while retaining validity under every finite exchangeable law.

This prototype implements the finite-support case exactly. It deliberately does not approximate a continuous observation space by a grid and call that a proof.

## Mathematical setup

Let the observation support have \(m\) types. A calibration sample of size \(n\) is represented by a count vector \(D\in\mathbb N^m\), \(|D|=n\). An augmented sample \(S\) has size \(n+1\).

A controller \(A\) maps every calibration multiset to an action \(A(D)\in[0,1]\). We use an affine loss family

\[
\ell(j,a)=b_j+d_j(1-a),\qquad b_j,d_j\geq0.
\]

The action is interpreted so larger \(a\) is more conservative: the loss is nonincreasing and convex (affine) in \(a\).

For an augmented multiset \(S\), define the total unused budget

\[
\Delta_A(S)=(n+1)\alpha-
\sum_{j=1}^{m} S_j\,\ell\!\left(j,A(S-e_j)\right),
\]

where terms with \(S_j=0\) are zero and \(e_j\) is the unit count vector for type \(j\).

For a finite-support symmetric controller, validity under every finite exchangeable law is equivalent to

\[
\Delta_A(S)\geq0
\quad\text{for every augmented multiset }S,\ |S|=n+1.
\]

Why exhaustive multiset checking is exact: every finite exchangeable distribution is a mixture over permutation orbits, and each orbit is identified by one count vector. The extremal risk checks are therefore the finite set of augmented multisets, not an IID Monte Carlo sample.

The recertified controller chooses the smallest action satisfying, for every possible held-out type \(j\),

\[
\ell(j,R A(D))
\leq \ell(j,A(D)) + \frac{\Delta_A(D+e_j)}{n+1}.
\]

For the affine loss family this has the exact rational solution

\[
R A(D)=
\max\left\{
0,\;
A(D)-\min_{j:d_j>0}
\frac{\Delta_A(D+e_j)}{(n+1)d_j}
\right\},
\]

with the minimum over an empty set interpreted so the action can be zero when all slopes are zero. The implementation computes the equivalent maximum of per-type lower bounds directly.

Summing the held-out inequalities over the \(n+1\) positions gives

\[
\sum_{i=1}^{n+1}\ell(z_i,R A(S_{-i}))
\leq
\sum_{i=1}^{n+1}\ell(z_i,A(S_{-i}))+\Delta_A(S)
=(n+1)\alpha.
\]

Thus the finite-state verifier can check the exact risk statement and the pointwise no-more-conservative property independently of the recertification routine.

## Implementation

The module validation/crc_finite_recertification.py provides:

- count-vector enumeration for every calibration and augmented multiset;
- exact rational arithmetic using Fraction;
- finite-exchangeable baseline verification with a worst augmented-multiset witness;
- exact unused-budget computation;
- one-step recertification with fail-closed rejection of an invalid baseline;
- independent checks of all held-out budget inequalities, pointwise action dominance and every augmented-sample risk;
- iterative recertification with exact history and controller digests.

No IID assumption is used in the finite-state certificate. The verifier checks the entire finite support and all multiset compositions.

## Exact development results

### Two observation types, n=2

Parameters:

- \(\alpha=1/2\);
- baseline action \(A(D)=1\) for every calibration multiset;
- intercept losses \(b=(1/2,1/4)\);
- slopes \(d=(3/10,3/10)\).

The baseline is valid on all 4 augmented multisets, with minimum slack exactly zero. One recertification gives:

| Calibration counts \(D\) | Baseline action | Recertified action |
|---|---:|---:|
| (0,2) | 1 | \(4/9\) |
| (1,1) | 1 | \(13/18\) |
| (2,0) | 1 | 1 |

The independent step verifier checks every held-out type, pointwise dominance, and the full augmented state space. The recertified controller remains valid with minimum slack exactly zero.

Eight successive exact-rational iterations also remained valid and nonincreasing. In this deliberately small affine fixture, successive sup-action changes shrink by a factor of \(1/3\); each observed ratio is below the paper's general geometric-rate benchmark \(n/(n+1)=2/3\). This is one finite example, not an independent proof of the general convergence theorem.

### Three observation types, n=50

This directly exercises the finite-state enumeration target in issue #9357.

\[
\#\mathcal D_n={n+m-1\choose m-1}={52\choose2}=1326,
\qquad
\#\mathcal S_{n+1}={n+m\choose m-1}={53\choose2}=1378.
\]

Parameters:

- \(\alpha=9/20\);
- baseline action 1 on every calibration multiset;
- intercept losses \(b=(9/20,7/20,1/4)\);
- slopes \(d=(1/20,1/20,1/20)\).

Results from the exact-rational development run:

| Check | Result |
|---|---:|
| Calibration controller states | 1,326 |
| Augmented multisets checked for baseline | 1,378 |
| Augmented multisets checked after recertification | 1,378 |
| Baseline minimum slack | 0 |
| Recertified minimum slack | 0 |
| Calibration states with a strictly smaller action after one step | 1,325 / 1,326 |
| Held-out inequalities checked | 3,978 |
| Arithmetic | Exact rational |

The single unchanged calibration state is the all-type-0 sample, whose all-type-0 augmentation has no unused risk budget. The verifier retains it rather than spending nonexistent slack.

## Tests

From the repository root:

    PYTHONPATH=validation python -m unittest -v test_crc_finite_recertification.py

Tests cover:

- exact two-type rational values;
- repeated recertification validity and monotonicity;
- complete three-type \(n=50\) enumeration;
- invalid-baseline fail-closed behavior;
- zero-slope losses.

## Scope boundaries

- Finite observation support only.
- Affine (therefore convex) loss in a scalar action on [0,1].
- Exact finite-exchangeability risk verification; no IID-only shortcut.
- No continuous-space validity claim.
- No claim to reproduce the paper's CIFAR-100 numbers; its original data, seeds and complete code path still need an independent reproduction.
- No equivalence claim with the separate ReCIRC approach; it is a related but different way to improve conditional/group risk while preserving marginal CRC.

## Next research gates

1. Reproduce the paper's exact two-type example and equations from the primary manuscript, documenting notation and action orientation.
2. Generalize from affine to arbitrary convex losses where the per-type inverse can be computed exactly.
3. Add a certified continuous-space mode using interval branch-and-bound or a globally valid lower bound on slack. A finite grid alone is never a validity proof.
4. Verify the general contraction theorem using exact rational fixtures and adversarial loss curves.
5. Add a reproducible hierarchical CIFAR-100 experiment with frozen model predictions, calibration split, action grid, preprocessing and seeds.
6. Integrate the resulting typed finite-exchangeability receipt into the active symthaea-statistics architecture from issue #9357; do not create another statistics crate.

## References

- Bai & Ramdas (2026), [Recertifying decisions via e-closure](https://arxiv.org/abs/2610.35821).
- Angelopoulos et al. (ICLR 2024), [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html).
- Resende et al. (2026), [ReCIRC: Rectified Conformal Risk Control](https://arxiv.org/abs/2609.38112).
- [Issue #9357 — proof-carrying risk decisions](https://github.com/Luminous-Dynamics/symthaea/issues/9357).
