# SDCB Gaussian-to-categorical experiment likelihood adapter v1

**Status:** bounded numerical likelihood adapter. Eight Rust tests are authored but not compiled or executed in this environment. Known-answer values are standard Gaussian probabilities; no physical measurement or policy tournament is claimed.

## 1. Why this adapter exists

The continuous temporal SDCB references #13903, #13937 and #13940 produce predictive Gaussian measurement distributions. Existing finite SDCB instrument-aware and common-cause decision oracles #13762, #13793 and #13804 consume categorical outcome likelihoods.

This adapter bridges those representations without creating another planner. It converts a predictive scalar Gaussian measurement into probabilities for a declared finite partition of the instrument output.

## 2. Declared outcome partition

For predictive observation:

\[
Y\mid\mathcal I,a\sim\mathcal N(\mu,v),
\]

and finite ordered thresholds \(e_1<\cdots<e_k\), define outcomes:

\[
(-\infty,e_1],(e_1,e_2],\ldots,(e_k,\infty).
\]

The likelihood row is:

\[
P_j=\Phi\!\left(\frac{e_j-\mu}{\sqrt v}\right)
-\Phi\!\left(\frac{e_{j-1}-\mu}{\sqrt v}\right)
\]

with the corresponding infinite-bound conventions.

The adapter uses deterministic 64-point Gauss–Legendre integration of the standard-normal density, evaluating tail intervals directly rather than subtracting two values close to one. The upper integration cutoff at 12 standard deviations leaves omitted absolute tail probability below approximately \(2\times10^{-33}\). This is a numerical method with finite floating-point error, not symbolic exact arithmetic.

For \(v=0\), the output is deterministic and uses the same half-open bin convention. Invalid, non-finite or non-monotonic thresholds are rejected. Probabilities must sum to one within the declared numerical tolerance.

## 3. Correct use with SDCB

The input \((\mu,v)\) should be produced by a declared predictive state/measurement model, for example:
- the interval observation operator in #13940;
- a hypothesis/common-cause/device-state conditional Gaussian prediction;
- an explicit measurement-system covariance scenario.

The thresholds must be one of:
1. actual ADC / instrument quantization thresholds; or
2. a benchmark outcome partition frozen before evaluation.

If the thresholds are coarser than the instrument's actual output, the induced information loss must be measured. Do not call a coarse partition an exact continuous-observation policy or hide its bin width.

The returned probabilities can populate the finite likelihood table used by #13762, #13793 and #13804. Those owners retain their own finite-state transitions, robust maximin policy semantics, hard pathwise budgets and abstention. This module does not choose experiments or update evidence ledgers.

The current adapter handles a single scalar Gaussian output. Multichannel joint bins require a multivariate Gaussian rectangle-probability method that preserves residual correlations; multiplying marginal bin probabilities would be invalid when outputs are correlated.

## 4. Numerical known answers

For \(Y\sim N(0,1)\):
- thresholds [0] -> [0.5, 0.5];
- thresholds [-1,1] -> [0.1586552539314571, 0.6826894921370859, 0.1586552539314571];
- threshold 8 -> upper-tail probability approximately \(6.22096\times10^{-16}\), which must not be rounded to zero by naive one-minus-CDF subtraction.

The frozen tail interval [8,9] is evaluated directly and has probability approximately \(6.21983198586573\times10^{-16}\). These are numerical probability references, not measured likelihood calibration.

## 5. Required integration tests

- Compare generated likelihood rows with #13762 on binary threshold outcomes.
- Reproduce #13793/#13804 Markov/common-cause controls using state-conditional Gaussian output distributions.
- Check bin refinement convergence against a continuous expected-utility/quadrature baseline.
- Verify that actual ADC thresholds preserve the actual outcome semantics.
- Include a correlation negative control for multichannel measurements; no product-of-marginals shortcuts.
- Measure policy regret due to partition coarsening and report it against the continuous numerical upper reference.

## 6. Limits

This v1 is a scalar-output likelihood adapter. It does not:
- produce multivariate Gaussian rectangle probabilities;
- preserve a full continuous posterior after each outcome bin;
- calculate policy value for correlated sequential Gaussian observations;
- infer Gaussian predictive moments;
- prove that a supplied outcome partition is physically meaningful;
- replace the exact finite policy or evidence owners.

It is exact only for the declared finite categorical protocol up to numerical quadrature error. If bins are merely a computational discretization, the discretization is an explicit modelling approximation.
