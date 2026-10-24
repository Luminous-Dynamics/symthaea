# RH-006 Null-Surface Locality and Branch Conditioning — v1

## Status

Research-only design/derivation artifact.

- applicability = not-approved-for-execution
- selection = stop-assumption-failure
- formal p-value = disabled
- formal confidence interval = disabled
- uniform size control = not established

This note derives exact geometry from the existing RH-006 quadratic null and the already recorded synthetic DGP. It does not qualify the Rust implementation and does not constitute empirical validation.

## 1. Exact radial parameterization

The finite-sample equal-risk null is

`Delta(gamma) = gamma'Q gamma + b'gamma + c = 0`.

For a unit direction `u`, write

`gamma = t u`, `t > 0`.

Then

`Delta(tu) = A t^2 + B t + C`

where

`A = u'Qu`

`B = b'u`

`C = c`.

The ray discriminant is

`D(u) = B^2 - 4AC`.

A positive root is admissible only when the quadratic/linear root conditions are satisfied. No root should be imputed when the ray does not intersect the declared null surface.

## 2. New exact branch-conditioning identity

Differentiate the ray equation:

`partial Delta(tu) / partial t = 2At + B`.

At any quadratic root `t`,

`(2At+B)^2 = D(u)`.

Therefore:

`|partial Delta / partial t| = sqrt(D(u))`.

This is an exact identity, not a Monte Carlo approximation.

Consequences:

1. `D(u) < 0`: the ray has no real intersection.
2. `D(u) = 0`: the two branches coalesce into a repeated root and the radial surface derivative vanishes.
3. Small positive `D(u)`: the root exists but is intrinsically sensitive to perturbations in `A,B,C`.
4. First-order root perturbation obeys approximately
   `delta t ~= - delta Delta(tu) / sqrt(D(u))`
   up to the perturbation's projection onto the polynomial coefficients.

Thus the same quantity that controls root existence also controls radial branch conditioning.

## 3. Why this matters for estimated nuisance

The prior RH-006 work observed that estimated nuisance usually retains the oracle positive root conditional on oracle support, even in severe weak-direction cells.

The new identity explains why two apparently different observations can coexist:

- a path may retain a root with a large movement in coefficient space;
- the root can still be mechanically sensitive when `D(u)` is small.

Therefore root-retention rate alone is not a sufficient nuisance-calibration diagnostic.

A stronger report should record:

- root support;
- discriminant magnitude;
- normalized radial derivative;
- root magnitude;
- estimated/oracle root ratio;
- root movement in outcome/effect coordinates.

## 4. Outcome-space locality is more meaningful than coefficient norm

RH-006's weak-direction synthetic construction has, as epsilon approaches zero:

`a2b -> al`

and

`b2a -> 0.25 + 0.5 al`.

For the predeclared weak direction

`u = [0,1,-2] / sqrt(5)`,

the directional feature projection therefore approaches:

`u'Z = (a2b - 2 b2a) / sqrt(5) = -0.5 / sqrt(5) ~= -0.2236068`.

So the outcome-space mean perturbation induced by a weak-direction coefficient radius `t` is approximately

`|u'Z| t ~= 0.2236068 t`.

This converts the previously observed coefficient-space root magnitudes into an interpretable effect scale:

| epsilon | median root | implied RMS/mean-shift scale along weak ray |
|---:|---:|---:|
| `1e-2` | 14.2394 | ~3.18 outcome units |
| `1e-3` | 110.6787 | ~24.75 outcome units |
| `1e-4` | 8895.1428 | ~1,989 outcome units |

These are deterministic transformations of the already recorded root medians and the declared feature construction; they are not new empirical observations.

The crucial result is:

> The severe weak-direction null is not merely far away in coefficient coordinates. It is far away in outcome-space as well.

That makes a local null radius scientifically meaningful: it can be stated in outcome/effect units instead of being an arbitrary coefficient-norm cutoff.

## 5. Recommended local-null coordinate

Define a positive semidefinite effect metric from the held-out feature perturbation:

`G = average_over_target_windows(Z_t' Z_t / test)`.

For a candidate null coefficient vector `gamma`, define

`rho_effect(gamma) = sqrt(gamma' G gamma) / sigma_ref`.

Here `sigma_ref` is a frozen, predeclared outcome-noise or residual scale.

Interpretation:

- `rho_effect = 0.25`: RMS mean perturbation is one quarter of the reference noise scale;
- `rho_effect = 1`: RMS perturbation is comparable to reference noise;
- larger values describe progressively more remote null configurations.

This coordinate is preferable to `||gamma||` because it is invariant to reparameterizations of coefficient coordinates that leave the induced feature-to-outcome perturbation unchanged.

The reference scale must be frozen by the analysis plan. It cannot be selected after examining observed outcomes or bootstrap rejection rates.

## 6. Candidate local admissible set

A local scientific null could be declared as:

`S(r) = { gamma : Delta(gamma)=0, gamma lies in the declared coefficient sign/branch domain, rho_effect(gamma) <= r }`.

A regularity-restricted diagnostic set can additionally track:

`S(r,kappa) = { gamma in S(r) : normalized radial derivative >= kappa }`.

Important distinction:

- `S(r)` is a scientific estimand decision;
- adding `kappa` is a regularity restriction and therefore changes the hypothesis unless the intended estimand explicitly excludes critical points.

The global surface must not be silently replaced by `S(r,kappa)`.

## 7. Continuous set inversion is the preferred geometry

The next inference architecture should treat the null as a set and invert a valid test over that set rather than:

- select one direction;
- choose one positive root;
- take the largest observed grid rejection.

The 2026 Schlemper–Moreira work is directly relevant as a computational precedent: in weak-identification settings, grid inversion can miss disconnected or unbounded confidence-set components, while exploiting polynomial structure permits more reliable inversion. Their setting is IV rather than forecast comparison, so this is a methodological analogy, not a ready-made RH-006 procedure. citeturn937642view1

For RH-006, the natural candidate is:

1. parameterize the admissible quadratic surface;
2. impose the predeclared effect-radius restriction when the estimand is local;
3. retain all surface components and branches;
4. evaluate the candidate test statistic over the entire admissible set;
5. calibrate the supremum only after establishing a valid weak-identification-robust procedure.

A finite spherical grid can remain a convergence diagnostic, but it should not define the inferential object.

## 8. Candidate robust-calibration branches

Three research branches are now worth comparing:

### A. Restricted bootstrap

Retain the existing estimated-nuisance restricted bootstrap.

Purpose: baseline mechanical comparator.

Expectation: likely adequate only in regular strata; no weak-identification guarantee.

### B. Identification-state targeted bootstrap

Construct separate bootstrap transformations for strong, near-weak, and critical identification states and combine them according to a prespecified rule.

Hill's weak-identification-robust bootstrap work is relevant here because it explicitly targets different identification cases rather than relying on a supremum or average transformation, and reports uniform size control in its model-specification setting. The mapping to RH-006 is not automatic and would require a fresh derivation. citeturn332595search0

### C. Set-inversion / test-inversion route

Define a statistic with a valid weak-identification-robust null distribution, then invert over the admissible `gamma)-surface.

This is conceptually the cleanest architecture if a valid statistic can be derived.

The forecast-comparison literature still has to supply the estimator/procedure bridge. Giacomini–White explicitly accommodates estimation uncertainty and general estimation methods for forecast evaluation, while Clark–McCracken's nested forecast work directly studies finite-sample equal-accuracy nulls with nonzero but weak coefficients and allows recursive or rolling estimation. Neither result by itself validates RH-006's exact disjoint-target, fixed-ridge, training-standardized construction. citeturn363719search0turn332595search2

## 9. New falsification tests

Before attempting a formal procedure, the research harness should attempt to falsify the local-surface concept itself.

Required tests:

1. **Radius collapse test**
   Determine the smallest predeclared effect radius for which each weak cell has nonempty null support.

2. **Branch-conditioning test**
   Stratify supported roots by discriminant and normalized radial derivative.

3. **Outcome-coordinate invariance**
   Recompute locality under equivalent feature parameterizations and verify that the effect metric, rather than raw coefficient norm, is stable.

4. **Surface-component test**
   Search for disconnected surface components and verify that numerical inversion does not lose them.

5. **Radius sensitivity**
   Report the full rejection/support path over a frozen radius ladder rather than selecting a favorable radius.

6. **Identification robustness**
   Compare ordinary restricted bootstrap against at least one identification-state targeted candidate.

## 10. Scientific decision boundary

The strongest current interpretation is no longer simply:

> "the weak direction has poor root support."

It is:

> "the finite-sample equal-risk surface develops a weak-curvature, near-critical region in which admissible roots become remote in outcome space and therefore highly sensitive to the null parameterization."

That is a materially stronger and more falsifiable statement.

The correct next gate is consequently:

`surface geometry`
→ `effect-radius support`
→ `branch conditioning`
→ `nuisance calibration by geometry`
→ `continuous/set-valued inversion`
→ `uniform weak-identification calibration`.

No formal p-value or confidence interval should be enabled before all of those gates are satisfied.
