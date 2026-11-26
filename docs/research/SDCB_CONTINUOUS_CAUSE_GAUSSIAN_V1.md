# SDCB continuous common-cause Gaussian design oracle v1

**Status:** bounded exact linear-Gaussian design reference; twelve Rust tests authored, not yet compiled or executed here.  
**Implementation:** \`scripts/research/sdcb_continuous_cause_gaussian.rs\`  
**Fixture:** \`SDCB_CONTINUOUS_CAUSE_GAUSSIAN_V1.example.json\`  
**Expected result:** \`SDCB_CONTINUOUS_CAUSE_GAUSSIAN_V1.expected.json\`

## 1. Motivation

The binary shared-cause reference (#13804) is valuable for exact finite reasoning, but real measurement systems have continuous bias, multiple overlapping common factors, and device-local effects. The paired-world identifiability reference (#13831) captures a different limitation: observational data may not distinguish a scientific effect from a measurement-system explanation.

This increment adds a continuous, small-instance reference that answers a practical experiment-design question:

> Under a fixed reference/measurement budget, which subset of measurements most reduces uncertainty in the scientific quantity after accounting for shared continuous nuisance factors?

This is not a claim that Gaussian sensor calibration or optimal design is new. The contribution is a bounded exact evaluator that makes common-factor-aware reference allocation reproducible and supplies known answers for approximate SDCB policies.

## 2. Model

Let the latent state be:

\[
x=(\theta,z_1,\ldots,z_m)^T
\]

where \(\theta\) is the selected scientific parameter and the \(z_j\) are continuous nuisance factors such as environmental bias, reference-chain bias, batch effect, or device-local state.

Version 1 uses independent Gaussian priors:

\[
x\sim\mathcal N(0,D),\qquad
D=\operatorname{diag}(\tau_1^2,\ldots,\tau_n^2).
\]

A candidate scalar measurement \(a\) has a declared loading vector \(h_a\), cost \(c_a\), and independent residual variance \(r_a\):

\[
y_a=h_a^Tx+\epsilon_a,\qquad
\epsilon_a\sim\mathcal N(0,r_a).
\]

Residuals are conditionally independent across selected actions given the explicit latent vector. Shared errors must therefore be represented as shared latent factors, not hidden again in independent residual noise.

For a selected subset \(S\), the posterior precision is:

\[
\Lambda_S=D^{-1}+\sum_{a\in S}\frac{h_ah_a^T}{r_a},
\qquad
\Sigma_S=\Lambda_S^{-1}.
\]

The exact design objective is target-specific:

\[
S^*=\arg\min_{S:\sum_{a\in S}c_a\le B}
(\Sigma_S)_{jj},
\]

where \(j\) is the declared scientific coordinate and \(B\) is the hard budget.

The oracle separately reports joint Gaussian information gain:

\[
I_S=\frac12\log\frac{\det\Lambda_S}{\det D^{-1}}.
\]

**Important:** joint information gain is a reported diagnostic, not the selection objective. Learning a nuisance factor is useful only insofar as it reduces uncertainty in the selected scientific parameter under this objective. A future EVSI layer can replace the variance objective when a concrete downstream decision and utility are specified.

## 3. Exact bounded optimizer

The Rust reference:
- validates latent/action identities, dimensions, variances, and finite loadings;
- builds information matrices by rank-one precision updates;
- uses Cholesky factorization and triangular solves for covariance marginals;
- enumerates every candidate subset up to 16 actions;
- enforces the hard integer budget before scoring;
- minimizes the target posterior variance;
- breaks equal-value ties deterministically by cost and sorted action IDs;
- reports all marginal posterior variances and joint information gain.

The action cap is intentional. This is an exact small-instance oracle, not the scalable production planner. Large action sets should use a separately evaluated approximation (for example branch-and-bound, greedy c-optimal design, or a learned policy) against this reference on overlapping small cases.

## 4. Frozen analytical fixture

The fixture has one scientific parameter \(\theta\) and one shared bias \(g\), each with prior variance 1:

\[
\theta,g\overset{\mathrm{ind}}{\sim}\mathcal N(0,1).
\]

Two science devices observe the same latent combination:

\[
y_A=\theta+g+\epsilon_A,\qquad
y_B=\theta+g+\epsilon_B,
\]

with residual variance 0.01. A shared reference measures the common factor:

\[
y_R=g+\epsilon_R,\qquad \operatorname{Var}(\epsilon_R)=0.01.
\]

Each action costs one unit; the total budget is two.

**Two science channels only** produce:

\[
\Lambda=
\begin{bmatrix}201&200\\200&201\end{bmatrix},
\qquad
\operatorname{Var}(\theta\mid y_A,y_B)=\frac{201}{401}
\approx0.501246883.
\]

The common bias remains confounded with the scientific parameter. Repeating science channels does not remove that floor.

**One science channel plus the shared reference** produce:

\[
\Lambda=
\begin{bmatrix}101&100\\100&201\end{bmatrix},
\qquad
\operatorname{Var}(\theta\mid y_A,y_R)=\frac{201}{10301}
\approx0.019512669.
\]

The exact optimizer selects \`science-device-a\` and \`shared-reference\` under the frozen budget. The scientific marginal variance falls by approximately 98.05% from its prior value of 1. These are analytic calculations on a declared linear-Gaussian model, not physical sensor results.

The reference check is not automatically an SI-traceable standard. Its provenance, calibration uncertainty, authority, physical suitability, and receipt still require separate qualification.

## 5. What is deliberately not assumed

Version 1 does **not** silently assume that every sensor error is independent. Its conditional residual independence is only valid when all relevant shared causes have been represented in \(z\).

It does not yet include:
- a full non-diagonal prior covariance;
- a general correlated residual covariance matrix \(R\);
- nonlinear or non-Gaussian measurement likelihoods;
- temporal Markov/OU drift or calibration resets;
- unknown loading vectors or inferred factor identity;
- posterior means or observation-value-dependent adaptive branches;
- physical traceability, signatures, execution, or measurement validity.

A hidden residual correlation will make the model overconfident. The benchmark therefore includes deliberate misspecification controls and requires correlated residuals to be either represented explicitly or exposed as model inadequacy.

## 6. Integration path

1. Construct a measurement graph with explicit scientific, shared, reference-chain, batch, and local latent coordinates.
2. Generate candidate science, reference, check-standard, and intervention actions with documented loading and uncertainty assumptions.
3. Run this exact subset oracle for bounded instances.
4. Train or execute a scalable selector; compare its target variance, utility, cost, and regret against the exact result.
5. Bind each real measurement action to the correct reference/device ancestry and calibration context.
6. Execute only authorized actions and accept only independently verified receipts.
7. Update the dynamic evidence ledger only after measurement-assurance checks pass.

The exact design oracle and the EGBT evidence ledger have different responsibilities: one evaluates a declared likelihood model; the other qualifies a specific evidence transition. Neither can certify the other's assumptions.

## 7. References

- NIST, [Recommended Calibration Interval](https://www.nist.gov/calibrations/recommended-calibration-interval): NIST states that it does not prescribe a universal interval and describes measurement-assurance programmes and control-chart evidence.
- NIST Technical Note 2156, [Metrological Traceability: Frequently Asked Questions and NIST Policy](https://www.nist.gov/metrology/metrological-traceability): traceability requires an unbroken calibration chain, with uncertainty contributions and measurement-assurance evidence.
- Kok et al. (2025), [Modelling and determining correlations in sensor networks](https://doi.org/10.1016/j.measen.2024.101793): space-time correlation modelling can change uncertainty calculations relative to i.i.d. sensor-error assumptions.
- Qi & Baker (2026), [Optimal experimental design for parameter estimation in the presence of observation noise](https://doi.org/10.1016/j.mbs.2025.109571): correlated versus uncorrelated observation noise can change optimal observation schedules.
- Yun et al. (2026), [Low-Cost Sensor Calibration for Indoor Air Quality Monitoring](https://arxiv.org/abs/2610.11236): real-data comparison across reference-efficient, location-transfer, long-term drift, and event-conditioned calibration.

## 8. Evidence boundary

The twelve unit tests are authored but have not been compiled or executed in this environment because \`rustc\` and \`cargo\` are unavailable. The expected fixture values are independently checked from the two-by-two precision matrices. No real dataset has been downloaded or benchmarked, and no physical reference/intervention has been run.
