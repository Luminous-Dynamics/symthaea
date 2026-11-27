# SDCB full-covariance Gaussian design oracle v1

**Status:** exact bounded linear-Gaussian reference; twelve Rust tests authored, not compiled or executed here.  
**Implementation:** \`scripts/research/sdcb_full_covariance_gaussian.rs\`  
**Fixture:** \`SDCB_FULL_COVARIANCE_GAUSSIAN_V1.example.json\`  
**Expected result:** \`SDCB_FULL_COVARIANCE_GAUSSIAN_V1.expected.json\`

## 1. Why this extension matters

The diagonal-prior Gaussian oracle (#13848) assumes independent latent prior coordinates. Real scientific and instrument states may be correlated before a new measurement. The previous reference also treated all candidate residual errors as conditionally independent once explicit factors were represented.

Version 1 generalizes the exact small-instance reference to both:
- a full positive-definite prior covariance \(\Sigma_0\);
- a full positive-definite residual covariance \(R\) across candidate measurements.

This prevents two common overconfidence sources: discarding known prior dependence and counting correlated residual observations as independent evidence.

## 2. Model and posterior

Let the latent vector \(x\) contain the target scientific parameter and all registered continuous nuisance factors:

\[
x\sim\mathcal N(\mu_0,\Sigma_0).
\]

Candidate measurement \(a\) has linear loading \(h_a\):

\[
y_a=h_a^Tx+\epsilon_a.
\]

For the complete candidate registry, conditional residual errors have covariance:

\[
\epsilon\sim\mathcal N(0,R).
\]

For selected subset \(S\), let \(H_S\) contain selected loading rows and \(R_S\) the matching principal submatrix of \(R\). Then:

\[
\Sigma_S=
\left(
\Sigma_0^{-1}+H_S^TR_S^{-1}H_S
\right)^{-1}.
\]

The target-specific selection objective remains:

\[
S^*=\arg\min_{S:\sum_{a\in S}c_a\le B}(\Sigma_S)_{jj}.
\]

Joint information gain is separately reported:

\[
IG(S)=\frac12\log\frac{\det\Sigma_0}{\det\Sigma_S}.
\]

The selector does not optimize this joint information score. A nuisance factor can be learned without directly improving the target, so the scientific coordinate is the declared objective.

The Gaussian posterior mean depends on observed values, but posterior covariance under this fixed linear-Gaussian model does not. This is why the variance-minimizing subset oracle is non-adaptive. Outcome-adaptive decisions are handled by the separate categorical two-stage reference.

## 3. Numerical method and limits

The implementation:
- checks covariance dimensions, symmetry, finite values and positive definiteness;
- computes precision and covariance with Cholesky factorization and triangular solves;
- preserves cross-action residual covariance by extracting the selected principal submatrix;
- enumerates all subsets up to twelve actions;
- enforces the integer resource budget before evaluation;
- minimizes the selected scientific marginal variance;
- uses deterministic cost/lexical tie-breaking.

The bounded matrix pivot floor is \(10^{-12}\); this is a numerical guard, not a universal metrology tolerance. Inputs with a different physical scale may need a justified rescaling or revised numerical policy. No jitter is silently added to make an invalid covariance pass.

This reference assumes:
- a fixed, correctly specified linear measurement model;
- Gaussian latent prior and Gaussian residual error;
- residual covariance independent of the latent state;
- complete representation of any shared causes that should be physically interpretable.

A declared residual covariance can account for remaining correlation, but it does not identify its physical mechanism. If the same physical common factor is also included as a latent variable, its contribution must not be double-counted inside \(R\).

## 4. Frozen analytical fixture

The example has:
- two independent prior latent coordinates, scientific \(\theta\) and shared bias \(g\), each with variance 1;
- science devices A/B with loading \([1,1]\);
- a reference action with loading \([0,1]\);
- candidate residual variances 0.01;
- residual covariance 0.003 between science A and the reference, with the remaining off-diagonal terms zero;
- unit action costs and a total budget of two.

The exact subset enumeration selects science A plus the reference. The expected scientific posterior variance is:

\[
0.0137595194177080.
\]

The joint information gain is approximately 4.66422822065 nats. This differs from the diagonal-residual reference because the selected science and reference measurements have positively correlated residuals.

Separate negative control: two scalar duplicate science measurements with residual covariance
\[
R=\begin{bmatrix}0.01&0.0099\\0.0099&0.01\end{bmatrix}
\]
have target posterior variance:
\[
\left(1+\frac{2}{0.0199}\right)^{-1}
\approx0.00985221675.
\]
The independent-residual value would be \(1/201\approx0.00497512438\). The correlated duplicate is less informative, as it should be.

These are constructed Gaussian known-answer calculations, not physical sensor outcomes.

## 5. Integration with the SDCB stack

- #13804 remains the exact finite binary common-cause sequential reference.
- #13831 tests whether two supplied causal/measurement worlds can be distinguished by candidate observations or interventions.
- #13848 supplies a diagonal-prior/residual Gaussian subset reference.
- This reference adds full covariance for bounded Gaussian subset selection.
- The new two-stage categorical oracle separately handles outcome-adaptive selection when the next action can depend on the first result.
- #13805 remains the evidence-transition reference; an optimized design does not authenticate or validate a real receipt.

## 6. References and claim boundary

The 2026 paper by Qi and Baker demonstrates that correlated observation noise can change optimal observation schedules, comparing OU-correlated and IID noise in parameter-estimation design: https://doi.org/10.1016/j.mbs.2025.109571

Kok and Tabandeh's 2026 work addresses short mobile-sensor collocation with a slower reference instrument. It derives covariance expressions for Gaussian process averages and shows that process variability and unequal averaging periods affect calibration/testing uncertainty; C-GDR can outperform OLS when uncertainties are more comparable and the measured process varies more: https://doi.org/10.1002/qre.70319

NIST's current calibration guidance does not mandate one universal recalibration interval. It points to application accuracy requirements, instrument stability, environmental effects, cross-comparisons, and control-chart evidence: https://www.nist.gov/calibrations/recommended-calibration-interval

**Evidence status:** the twelve Rust tests are authored but not executed in this environment. The example, expected-result, and schema will be fetched and parsed from the committed branch; precision-matrix calculations are independently reproduced in a separate numerical implementation. No raw data, physical measurement, or external tournament result is claimed.
