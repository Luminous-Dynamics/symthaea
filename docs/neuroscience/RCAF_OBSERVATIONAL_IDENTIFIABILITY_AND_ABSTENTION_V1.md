# RCAF Executable Observational Identifiability and Abstention v1

**Status:** executable synthetic reference control; no empirical identifiability or biological result is claimed.  
**Related specification:** [RCAF Spatial Nuisance and Identifiability v1](RCAF_SPATIAL_NUISANCE_AND_IDENTIFIABILITY_V1.md).

## 1. Constructive non-identifiability

For clean observations (X_{t,r}), spatial loading (u_r), temporal process (z_t), and amplitude (a), define

[
Y^{target}_{t,r}=X_{t,r}+a z_t u_r,
qquad
Y^{artifact}_{t,r}=X_{t,r}+a z_t u_r.
]

The first world labels the added term a latent target change; the second labels the same term an observation artifact. Since the observed matrices are identical, an observer with access only to those matrices cannot distinguish the causes.

This is a constructive non-identifiability control. It is not a model of a particular scanner, motion process, physiological artifact, or biological mechanism.

## 2. Executable disposition

The standalone Rust reference in `research/rcaf/src/observational_identifiability.rs`:

- generates both counterfactual observation matrices from the same clean input and separable spatial × temporal contribution;
- compares two time-by-ROI matrices at a declared maximum-absolute-difference tolerance;
- reports maximum absolute and root-mean-square differences, comparison size, tolerance and disposition;
- rejects mismatched shapes, non-finite inputs, non-finite generated observations and invalid tolerances;
- never upgrades an observation difference into a biological or causal label.

The disposition is:

| Condition | Disposition |
|---|---|
| Maximum absolute difference ≤ predeclared resolution tolerance | `not_identifiable_abstain` |
| Maximum absolute difference > predeclared resolution tolerance | `observationally_distinct_at_declared_resolution` |

The second disposition only states that the matrices differ at the declared numerical resolution. It does not assert that the difference is biological, causal, meaningful or free of artifact.

## 3. Resolution tolerance is part of the measurement contract

The tolerance must be derived from instrument precision, quantization, known numerical precision and the declared observation process, then frozen before the target contrast is inspected. It must not be tuned after observing the desired outcome.

For real data, a scalar maximum-absolute tolerance may be insufficient when precision varies by ROI, time, acquisition or preprocessing. A future extension should support a predeclared heteroscedastic observation-resolution model and uncertainty-aware equivalence regions. The current scalar tolerance is a transparent synthetic reference, not a universal noise model.

## 4. Validation

The focused workflow runs `rcaf_observational_identifiability`, validates its output against `observational-identifiability.schema.json`, asserts that the exact-equivalence pair requires abstention, asserts that a deliberately perturbed control is only called observationally distinct, and includes the report in the content-addressed execution receipt and archive.

Unit tests cover exact equality, below-resolution and above-resolution differences, invalid tolerance, shape mismatch, non-finite inputs, amplitude overflow and the absence of causal/biological language in the distinct disposition.

## Claim limits

This tests one constructive synthetic equivalence boundary. It does not estimate empirical instrument noise, solve latent causal attribution, prove that two real-world explanations are equivalent, or validate biological/clinical claims. Abstention is the correct outcome only for the declared equivalence/resolution conditions; false abstention on identifiable controls must be measured separately in a broader qualification campaign.
