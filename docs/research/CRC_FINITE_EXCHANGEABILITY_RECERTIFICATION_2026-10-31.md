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

### Exact reproduction of Bai & Ramdas, Section 3.4, Table 1

The implementation now reproduces the paper's two-type example exactly:

\[
\ell(H,\lambda)=1-\lambda,\qquad
\ell(L,\lambda)=\tfrac12(1-\lambda),\qquad
n=2,\quad\alpha=\tfrac12.
\]

The starting controller is the paper's envelope-CRC table, not an artificial constant baseline:

| Calibration multiset | Envelope CRC \(A^{(0)}\) | One equal-share recertification \(A^{(1)}\) |
|---|---:|---:|
| HH | \(1/2\) | \(1/2\) |
| HL | \(2/5\) | \(23/60\) |
| LL | \(1/4\) | \(1/5\) |

The baseline is valid on every augmented multiset, with exact minimum slack zero. The independent verifier checks all four augmented multisets and every held-out constraint, and the recertified controller remains valid with exact minimum slack zero.

Eight repeated exact-rational iterations are also checked against the paper's pointwise admissible limit \((1/2,3/8,1/8)\), using the theorem's error bound \(\rho^k\) with \(\rho=n/(n+1)=2/3\). This tests the actual convergence statement, rather than comparing consecutive step sizes.

### New allocation experiment: loss-proportional slack

Appendix B.2 of the paper proves that any permutation-equivariant nonnegative allocation summing to one across held-out positions preserves validity. It specifically identifies allocation proportional to baseline held-out losses as an alternative to equal sharing.

The prototype now supports both:
- equal sharing: \(w_i(S)=1/(n+1)\);
- loss-proportional sharing: \(w_i(S)=r_i^A(S)/T_A(S)\), with equal fallback if \(T_A(S)=0\).

For the exact two-type example, the one-step loss-proportional controller is:

| Calibration multiset | Equal sharing | Loss-proportional |
|---|---:|---:|
| HH | \(1/2\) | \(1/2\) |
| HL | \(23/60\) | \(11/29\) |
| LL | \(1/5\) | \(1/6\) |

The loss-proportional controller is pointwise no more conservative than the equal-share controller in this example. The exact step verifier independently checks its allocation inequalities and every augmented-multiset risk constraint. This is a finite example, not a universal dominance result for loss-proportional allocation.

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

### Nonconvex finite-grid completion: coordinate-sweep admissibility

The paper's Proposition 3.7 gives a different completion for nonconvex monotone losses when the observation space is finite. Instead of relying on convexity and repeated global recertification, lower one controller coordinate at a time. If calibration multiset \(D\) contains \(\nu_D(z)\) copies of type \(z\), lowering \(A(D)\) to \(a\) changes the held-out loss on \(D\cup\{z\}\) exactly \(\nu_D(z)+1\) times. The coordinate can be lowered precisely when

\[
(\nu_D(z)+1)\{\ell(z,a)-\ell(z,A(D))\}
\leq \Delta_A(D\cup\{z\})
\quad\forall z.
\]

The module validation/crc_discrete_admissibility.py implements this result for exact finite action grids and one or more monotone loss/risk tables. It uses a deterministic reverse-lexicographic count-vector sweep, independently checks every augmented multiset, and verifies that no single coordinate can be lowered while holding the others fixed. Under the paper's finite-support monotonicity argument, a coordinate that cannot be lowered at its turn cannot become reducible after later updates consume more shared slack.

For the paper's two-type example, using a 1/40 action grid and the envelope-CRC baseline (HH=1/2, HL=2/5, LL=1/4), the one-pass sweep returns the exact admissible controller (HH=1/2, HL=3/8, LL=1/8). All four augmented multisets are checked; the three mixed/high-loss constraints bind with exact slack zero, while LLL retains positive slack. The baseline is valid but the admissibility verifier returns a concrete lowerable-coordinate witness.

This finite-grid route is important for step-function losses such as miscoverage indicators, where the convex interval theorem is unavailable. It is still finite-support only; continuous outcomes require a certified continuum optimizer or a uniform approximation bound.

The module also accepts multiple loss tables sharing the same ordered action grid. Its current multi-risk regression test uses a proportional second risk as a consistency check; genuinely heterogeneous multi-risk benchmarks remain future work.

## Tests

From the repository root:

    PYTHONPATH=validation python -m unittest -v test_crc_finite_recertification.py
    PYTHONPATH=validation python -m unittest -v test_crc_discrete_admissibility.py

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
2. Generalize the finite-grid coordinate sweep to multiple independent loss families and benchmark order sensitivity / admissible controllers.
3. Generalize affine recertification to arbitrary convex losses where per-type inverses can be computed exactly.
4. Add a certified continuous-space mode using interval branch-and-bound or a globally valid lower bound on slack. A finite grid alone is never a validity proof.
4. Verify the general contraction theorem using exact rational fixtures and adversarial loss curves.
5. Add a reproducible hierarchical CIFAR-100 experiment with frozen model predictions, calibration split, action grid, preprocessing and seeds.
6. Integrate the resulting typed finite-exchangeability receipt into the active symthaea-statistics architecture from issue #9357; do not create another statistics crate.

## References

- Bai & Ramdas (2026), [Recertifying decisions via e-closure](https://arxiv.org/abs/2610.35821).
- Angelopoulos et al. (ICLR 2024), [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html).
- Resende et al. (2026), [ReCIRC: Rectified Conformal Risk Control](https://arxiv.org/abs/2609.38112).
- [Issue #9357 — proof-carrying risk decisions](https://github.com/Luminous-Dynamics/symthaea/issues/9357).
