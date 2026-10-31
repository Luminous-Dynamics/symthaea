# Finite-state conformal risk recertification

**Date:** 2026-10-31  
**Status:** exact two-type mathematical reproduction in a local reference prototype; committed implementation is a finite-state research engine. Not a CIFAR-100 replication and not a new theorem.

## Research pivot

Bai & Ramdas (2026), *Recertifying decisions via e-closure*, arXiv:2610.35821, has a second major application beyond randomized e-BH: it recertifies conformal risk control (CRC). The validity assumption is finite exchangeability, not IID alone and not arbitrary distribution shift.

For an augmented multiset \(S=(z_1,\ldots,z_{n+1})\), define
\[
r_i^A(S)=\ell(z_i,A(S_{-i})),\quad
T_A(S)=\sum_i r_i^A(S),\quad
\Delta_A(S)=(n+1)\alpha-T_A(S).
\]
For a symmetric controller, validity under every finite exchangeable law is equivalent to \(T_A(S)\le(n+1)\alpha\) for every deterministic augmented multiset.

Equal-share recertification uses
\[
(RA)(D)=\min\left\{\lambda\le A(D):
\ell(z,\lambda)-\ell(z,A(D))\le
\frac{\Delta_A(D\cup\{z\})}{n+1},\ \forall z\right\}.
\]
Each finite round remains valid and is pointwise no more conservative than its starting controller. Convex losses on an interval yield geometric convergence at rate \(n/(n+1)\). For finite observation spaces and merely monotone losses, a fixed-order coordinate sweep can reach a pointwise-admissible controller without convexity.

## Implementation

The research implementation uses count-vector multisets:
- calibration states: all \(K\)-type count vectors of total \(n\), \(\binom{n+K-1}{K-1}\);
- augmented states: all count vectors of total \(n+1\);
- held-out total: \(T_A(S)=\sum_z S_z\,\ell(z,A(S-e_z))\).

This avoids enumerating \((n+1)!\) permutations and counts duplicate observations with their multiplicity. The current engine supports multiple risks and a shared action for losses linear in a conservativeness parameter:
\[
\ell_r(z,\lambda)=g_{r,z}(1-\lambda),\quad \lambda\in[0,1].
\]
Envelope CRC and standard CRC are included. Equal-share and fixed-order coordinate-sweep updates use exact linear inversions. Every update verifies all augmented states and fails closed on negative slack or invalid output.

## Exact two-type reproduction

Use \(Z=\{H,L\}\), \(n=2\), \(\alpha=.5\), and
\[
\ell(H,\lambda)=1-\lambda,\qquad
\ell(L,\lambda)=\tfrac12(1-\lambda).
\]
The four augmented multisets are HHH, HHL, HLL and LLL. Exact finite-exchangeability validity requires each leave-one-out total to be at most 1.5.

| Calibration multiset | Standard CRC | Envelope CRC | Equal-share round 1 | Equal-share round 2 | Equal-share limit |
|---|---:|---:|---:|---:|---:|
| HH | 0.7500 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| HL | 0.6667 | 0.4000 | 0.3833 | 0.3778 | 0.3750 |
| LL | 0.5000 | 0.2500 | 0.2000 | 0.1722 | 0.1250 |

All four augmented-state constraints hold at every step. The equal-share sequence converges to \((0.50,0.375,0.125)\), matching the paper's exact example.

### A subtle non-lattice result

A different fixed-order coordinate sweep reaches another admissible controller \((0.50,0.40,0.10)\). Both controllers are pointwise admissible and neither dominates the other. Their pointwise minimum \((0.50,0.375,0.10)\) is **invalid**: its HLL augmented-sample held-out total is 1.525, exceeding the 1.5 budget.

This is important for system design: valid risk controllers are not generally closed under pointwise minima. One cannot combine multiple individually improved controllers by simply taking their most liberal action at every calibration state.

## Multiple risks

The engine supports a vector of losses and levels with one common action. It computes a separate exact slack for each risk and imposes all type constraints simultaneously. A 3-type, 2-risk example (coefficients [[1,.5],[.5,1]], levels [.5,.4]) remained valid after equal-share recertification and after the fixed-order coordinate sweep. This is a functional test, not a real-world multi-risk benchmark.

## Scaling implications

For \(K=3\) finite types and \(n=50\) calibration examples, there are:
- \(\binom{52}{2}=1,326\) calibration count states;
- \(\binom{53}{2}=1,378\) augmented states.

This is practical for table precomputation and reuse when the controller depends on data only through type counts. It does not solve the feature-adaptive setting: when each test feature has its own score and every possible label must be certified, state compression is harder. A finite sample of hypothetical test points is not a valid substitute for checking all labels.

## CIFAR-100 reproduction target

The paper evaluates 100 fine classes and 20 superclasses with a frozen ResNet-18, \(n=50\), 300 paired splits, 16 test images per split and risk levels \(\{.02,.05,.10,.15\}\).

At \(\alpha=.10\), for the confidence-adaptive rule it reports:
- envelope CRC: correct specificity 67.572%, mean prediction node size 19.40, risk 0.09052;
- 500 recertification rounds: correct specificity 68.235%, mean node size 17.03, risk 0.09880.

These are paper reference values, not locally reproduced results. The targeted search did not locate the exact cached ResNet-18 prediction artifact or split manifest cited by the paper. A CIFAR claim is blocked until those artifacts are available or independently regenerated with a frozen classifier and exact split manifest.

References:
- Paper: https://arxiv.org/abs/2610.35821
- PDF: https://arxiv.org/pdf/2610.35821

## Next gates

1. Add general monotone loss-table support, including step/indicator losses on finite action grids.
2. Add test-feature / possible-label recertification, checking every label at each observed feature.
3. Reproduce the exact 300-split CIFAR-100 protocol once prediction cache and split manifests are available.
4. Evaluate one, 10, 50 and 500 rounds; report risk, specificity, prediction-set size and runtime.
5. Stress calibration/test exchangeability, group shifts and non-exchangeable controls; distinguish expected-risk validity from conditional-risk guarantees.
6. Port the finite-state count-vector engine to Rust after the Python reference and exact-state tests stabilize.

Reproduction:
    PYTHONPATH=validation python -m unittest -v test_finite_state_crc_recertification.py
    PYTHONPATH=validation python validation/finite_state_crc_recertification.py --output /tmp/finite-state-crc.json
