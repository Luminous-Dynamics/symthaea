# AIRROAD-MIMO-HOLDOUT-FALSIFICATION-001 — exact independent model invalidation

**Status:** V0.1 exact-rational MIMO holdout falsification reference  
**Tracking issue:** [#11997](https://github.com/Luminous-Dynamics/symthaea/issues/11997)  
**Composes with:** #11865/#11867 hidden cross-port model pair, #11868/#11887 experiment selection, #11891 pointwise data consistency, #11882/#11885/#11893 continuous envelopes

> This reference falsifies one synthetic candidate under a declared hard uncertainty contract. A model that survives one holdout is not proven true or continuously adequate.

## 1. SOTA boundary

Model validation, set-membership identification and model invalidation are established system-identification methods. MATLAB's System Identification Toolbox explicitly recommends using separate estimation and validation data and inspecting residuals; error-domain model falsification also has a substantial control-oriented literature.

AIRROAD's target is not to invent falsification. It is to bind a MIMO falsification result to:
- a preregistered holdout;
- immutable train/holdout observation IDs;
- one campaign-shared calibration coordinate;
- a separate deterministic measurement-error set;
- exact candidate-specific feasible-set intersection;
- an explicit claim ceiling.

Relevant references:
- García-Violini et al. (2024), *Model invalidation for control-oriented modeling*, Control Engineering Practice 143, 105778: https://doi.org/10.1016/j.conengprac.2023.105778
- Goulet and Smith (2013), Error-Domain Model Falsification: https://doi.org/10.1016/j.compstruc.2013.07.009
- MATLAB frequency-domain identification/validation guidance: https://www.mathworks.com/help/ident/gs/estimate-models-using-frequency-domain-data.html

## 2. Adaptive-design and falsification-budget benchmark

A new preprint directly relevant to AIRROAD's adaptive excitation work is Choudhary (2026-11-05), [Adaptive Experimentation Can Hide Model Error: Detection Limits and Falsification Budget](https://arxiv.org/abs/2611.07529). It formalizes an important distinction: an adaptive experiment can estimate parameters efficiently inside a working model while repeatedly selecting settings where a wrong model still fits. Its own-design power analysis shows that a valid adequacy detector cannot recover information that was never sampled. The paper's empirical results are mixed and include a frozen-policy failure, a fresh-panel failure, and a revised rule whose advantage was not resolved at a longer horizon. We use it as a falsification-budget comparator and warning, not as evidence that any fixed reserve is universally optimal.

A second recent comparator is Niu, Shen and Yong (2026), [Active model discrimination: A multi-stage approach with adaptive partitions](https://doi.org/10.1016/j.automatica.2026.113067), Automatica 190, article 113067. It optimizes separating inputs for affine model families under bounded uncertainties and run-time-revealed information. AIRROAD's next benchmark should compare multi-stage set-based model discrimination against one-step EIG, D-optimal parameter design, fixed sweeps, random exploration and a preregistered falsification reserve.

The required outcome is not simply "which frequency has the largest disagreement?" It is whether the complete sequence of observations makes the intersection of every candidate model with the same shared calibration and measurement-error variables empty, and whether a statistical detector has calibrated power under its actual adaptive sampling law. Parameter learning, acquisition-side detection power, detector validity and final engineering decision value remain separate metrics. See [#12008](https://github.com/Luminous-Dynamics/symthaea/issues/12008).

## 3. Exact model pair

Use the AIRROAD hidden cross-port model pair:
\[
Z_A(s)=\begin{bmatrix}2&0\\0&3\end{bmatrix},\qquad
Z_B(s)=\begin{bmatrix}2&\Delta(s)\\0&3\end{bmatrix},
\]
\[
\Delta(s)=1000\frac{(s^2+1)(s^2+4)(s^2+16)}{(s+5)^7}.
\]

At the three training frequencies \(\omega=1,2,4\) rad/s, the numerator has a zero factor. Both candidate models therefore predict exactly the same 2×2 matrix:
\[
Z_A(j\omega_k)=Z_B(j\omega_k)=\begin{bmatrix}2&0\\0&3\end{bmatrix}.
\]

Training data cannot distinguish them. The holdout frequency is preregistered at \(\omega_h=3\) rad/s, measuring only cross-port element (1,2). Exact complex rational evaluation gives:
\[
\Delta(3j)=
-\frac{401712500}{410338673}
+i\frac{299932500}{410338673}.
\]

## 4. Hard uncertainty semantics

The cross-port observation is affected by one **campaign-shared** real calibration offset:
\[
\xi_{\mathrm{cal}}\in[-1/20,1/20],
\]
which is the same variable at every frequency. It is not reset at the holdout.

A separate holdout sensor-error component has:
\[
|\operatorname{Re}\eta|\le1/20,\qquad
|\operatorname{Im}\eta|\le1/20.
\]

The observation-feasible set for a candidate prediction \(m\) is:
\[
\mathcal Y(m)=m+\Xi_{\mathrm{cal}}+\mathcal E_{\mathrm{meas}}.
\]
This V0.1 control applies to the real component, where the combined admissible half-width is exactly \(1/10\). The same calibration variable remains separately identified in the receipt; the sum of intervals is used only for the scalar intersection calculation.

For candidate A, \(m=0\) and the observed real value has magnitude:
\[
\left|\operatorname{Re}\Delta(3j)\right|=
\frac{401712500}{410338673}>\frac1{10}.
\]
Therefore the holdout is outside candidate A's admitted set, and A is FALSIFIED.

For candidate B, \(m=\Delta(3j)\), so the observation is exactly explained with \(\xi_{\mathrm{cal}}=0\) and \(\eta=0\). B is NOT_FALSIFIED_BY_THIS_HOLDOUT.

The minimum sensor-error half-width needed to rescue candidate A, after using the maximum opposing calibration offset, is:
\[
\frac{401712500}{410338673}-\frac1{20}
=\frac{7623911327}{8206773460}>0.92.
\]
This is a diagnostic only. It must not silently replace the declared 1/20 error bound.

## 5. Exact verifier

The Fraction/Gaussian-rational oracle:
- evaluates \(\Delta(j\omega)\) exactly from the rational transfer expression;
- confirms A and B are identical at training frequencies;
- confirms the preregistered holdout value exactly;
- computes the candidate A and B holdout feasible intervals;
- returns FALSIFIED for A and NOT_FALSIFIED_BY_THIS_HOLDOUT for B;
- computes minimum error inflation without changing the admitted bound;
- validates distinct, immutable training/holdout IDs and campaign/model/topology/clock/port/calibration identities;
- actively rejects train/holdout leakage, post-observation holdout selection, split calibration coordinates, stale identities, unknown error bounds, covariance-to-hard-set promotion and continuous-band claim promotion.

The falsifier's terminal state is deliberately asymmetric:
- FALSIFIED means the declared model class is inconsistent with this evidence and these admitted sets.
- NOT_FALSIFIED_BY_THIS_HOLDOUT means only that this holdout did not reject the candidate.
- NOT_EVALUABLE means the evidence contract is incomplete or inconsistent.

There is no VERIFIED_TRUE state.

## 6. Benchmark extensions

Compare under identical energy, duration and bounded measurement error:
1. fixed log-spaced holdout;
2. maximum model-disagreement holdout;
3. model-disagreement weighted by robust-decision sensitivity;
4. the active one-step/multi-step policies from #11868/#11887.

Report model rejection rate, false-rejection rate under known truth, missed-falsification rate, measurement budget, calibration-dependency preservation and minimum error inflation. Do not optimize fit RMS alone.

## 7. Claim ceiling

V0.1 demonstrates exact falsification of one synthetic candidate under a stipulated hard-set experiment. It does not prove the surviving model true, establish continuous-frequency adequacy, estimate a physical FRF, solve an IQC, or establish robust stability/airworthiness.
