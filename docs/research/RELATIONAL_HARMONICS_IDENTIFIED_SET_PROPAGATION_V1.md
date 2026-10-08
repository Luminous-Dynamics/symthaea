# RH-006 Identified-Set Propagation — v1

## Status

Research diagnostic only.

- formal inference: disabled
- formal p-value: disabled
- formal confidence interval: disabled
- production/selection authorization: disabled

This artifact strengthens the RH-006 identification seam by making identified-set propagation explicit and executable, and by separating identification from downstream decision stability and numerical regularity.

## 1. Semantic correction

The earlier RH-006 witness harness used PI and NI in an overlapping way:

- PI was described as a non-singleton identified set.
- NI was also described as disagreement among observationally equivalent witnesses.

Those statements are both useful, but they are not mutually exclusive as written.

This v1 calculus makes the operational distinction explicit:

- O: directly observable target functional.
- MI: point identified after an explicit, qualified structural/measurement model.
- PI: a non-singleton identified set has actually been characterized under the declared model/assumption class.
- NI: the current analysis has not produced a qualified identified-set characterization.

Thus the existing witness result for C in the canonical adversarial construction should be interpreted as:

C is not point identified; its currently characterized identified set is [-1,1]; therefore the typed level for the set artifact is PI.

The analogous K result with a claimed set of R remains NI unless the repository has a qualified argument that this is the complete identified set.

A witness family is therefore evidence against point identification, but it should not be promoted automatically to a sharp-set characterization.

## 2. Decision is a different gate

A decision can be identified even when the nuisance parameter is not point identified.

The typed state is now the product:

identification level × decision stability × regularity.

In particular:

PI + decision-stable + regular

may authorize a downstream decision claim without authorizing point estimation of the nuisance.

By contrast:

PI + decision-unstable

must fail closed for that downstream decision.

A nonregular boundary is tracked separately rather than being silently collapsed into either stable or unstable.

## 3. Actual RH-006 ray surface

The existing RH-006 finite-sample equal-risk geometry for a fixed ray gamma = t u is:

Delta(tu) = A t^2 + B(s) t + C(s)

with

B(s) = B0 + B1 s

and

C(s) = C0 + C1 s + C2 s^2.

This is the scalar-strength slice of the documented affine joint-DGP construction.

The discriminant is therefore exactly:

D(s) = B(s)^2 - 4 A C(s)

= d0 + d1 s + d2 s^2

where

d0 = B0^2 - 4 A C0

d1 = 2 B0 B1 - 4 A C1

d2 = B1^2 - 4 A C2.

This is the key propagation seam: once the identified set for s is an interval, the discriminant image can be obtained analytically from a quadratic rather than from a dense grid.

## 4. Exact scalar set propagation

For any scalar quadratic p(s), its continuous image over a closed interval [l,u] is obtained from:

- p(l)
- p(u)
- the vertex, when the vertex lies in [l,u].

The script and Rust module use this exact candidate set.

No sampling grid defines the resulting image.

Sampling may still be useful as an independent convergence diagnostic, but it is deliberately outside the definition of the gate.

## 5. Branch-support propagation

For A > 0, the RH-006 ray roots are governed by D, B, and C.

The propagation partition therefore includes roots of:

- D(s), where real branches coalesce;
- C(s), where a root can cross t = 0;
- B(s), a conservative algebraic partition boundary;
- the identified-set endpoints.

Open cells between these critical points have invariant algebraic signs. Evaluating one interior point is sufficient to classify the cell; boundary points are evaluated separately.

The gate records both:

- real-root regime: no-real-branch, double-root, two-real-branches;
- positive-root count: 0, 1, or 2.

This matters because the RH-006 downstream selection is based on an admissible positive branch, not merely on the existence of any real root.

## 6. Regularity

A discriminant-zero point is a nonregular boundary even when the positive-root count is otherwise stable.

The documented identity is:

for Delta(tu) = 0,

|partial Delta / partial t| = sqrt(D).

Hence D = 0 is simultaneously:

- a branch-coalescence boundary;
- a vanishing radial derivative;
- a local conditioning boundary.

This is why decision stability and regularity are deliberately reported as separate dimensions.

## 7. Canonical adversarial propagation

For the existing observational-equivalence construction:

C identified set = [-1,1].

For the downstream discriminant:

D(C) = 0.25 - C^2.

The exact image is:

[-0.75, 0.25].

The identified set reaches:

- positive D: two real branches;
- D = 0: double root;
- negative D: no real branch.

Therefore the downstream branch decision is not invariant over the current C set.

For the contraction surface:

r(C) = sqrt(1 - C^2).

The exact image over [-1,1] is [0,1], while the endpoints are nonregular for the derivative because the slope magnitude |C| / sqrt(1-C^2) diverges there.

This confirms that the identification boundary, regime boundary, and numerical nonregularity remain linked in the canonical witness.

## 8. PI-stable fixture

The executable harness also includes a deliberately different case in which:

- the nuisance is PI over an interval;
- the discriminant stays positive;
- exactly one positive root exists for every admissible nuisance value;
- no discriminant boundary is reached.

That fixture is classified:

PI + decision-stable + regular.

It demonstrates the intended rule: a stable downstream decision can survive partial identification without creating a point estimate.

## 9. Why this is stronger than the previous surface gate

The previous gate used dense sampling to approximate a surface image.

This v1 layer changes the epistemic role of that computation:

previous:
identified set -> numerical grid -> diagnostic classification

now:
identified set -> analytic polynomial image / critical partition -> typed decision + regularity state

A numerical grid can still be added later as an independent cross-check, but a missed narrow boundary or disconnected cell is no longer allowed to define the scientific result.

This is directly aligned with current weak-identification methodology that warns grid inversion can miss pieces of confidence regions and uses polynomial structure and root finding to obtain more reliable set inversion. The analogy is methodological: the RH-006 problem is a finite-sample forecast-risk surface, not an IV confidence set.

## 10. Next extension: effect-radius propagation

The existing RH-006 locality work defines an effect metric:

rho_effect(gamma) = sqrt(gamma' G gamma) / sigma_ref.

On a ray gamma = t u this becomes:

rho_effect(tu) = t sqrt(u' G u) / sigma_ref.

Thus effect-radius support is itself exact along a fixed direction once G and sigma_ref are frozen.

The next extension should propagate the identified nuisance set jointly through:

identified nuisance set
-> discriminant
-> positive-root support
-> effect radius
-> branch conditioning
-> decision.

This will allow the research lane to say not merely whether a branch exists, but whether all admissible branches stay inside a scientifically predeclared local-effect region.

## 11. Post-execution numerical hardening

The implementation was hardened after the v1 execution receipt with two changes:

- quadratic roots now use a cancellation-resistant formulation rather than the naive quadratic formula;
- fixed effect-radius thresholds are converted to exact nuisance-boundary equations. For
  \[
  \rho_{effect}(tu)=t\,\frac{directional\_effect\_scale}{\sigma_{ref}},
  \]
  a frozen radius determines a fixed \(t\), and substituting that value into the RH-006 ray surface leaves a quadratic in the nuisance coordinate.

The repository therefore has an analytic boundary locator for effect-radius crossings without turning a numerical grid into the scientific definition.

The committed v1 execution receipt predates this hardening pass and remains intentionally frozen to its original source hash. The new root-solver/effect-boundary code has only been independently sanity-checked locally; it has not been granted a new hosted compile or execution PASS.

## 11. Required future hardening

Before formal inference is even considered, the next gates should include:

1. multi-parameter nuisance sets rather than only a scalar strength coordinate;
2. disconnected and unbounded set components;
3. interval/box versus sharp-set distinctions;
4. explicit outer- versus inner-approximation labels;
5. dependence-preserving resampling over the identified nuisance class;
6. weak/near-weak/critical identification strata;
7. effect-radius sensitivity over a frozen radius ladder;
8. a full continuous/set-valued null inversion rather than direction-grid selection.

A finite grid should be retained only as a falsification/convergence instrument.

## 12. Set-representation polarity

An identified-set boundary is not complete until its approximation status is typed.

The calculus now distinguishes:

- **witness-only**: admissible examples, with no claim that they exhaust the set;
- **inner approximation**: guaranteed subset of the true set;
- **outer approximation**: guaranteed superset of the true set;
- **sharp**: the exact identified set under the declared assumptions.

This polarity changes which downstream claims are logically licensed:

| set representation | stable decision | unstable decision |
|---|---|---|
| witness-only | not certified | not certified |
| inner approximation | not certified | certified when conflicting decisions occur inside |
| outer approximation | certified when every outer point is stable | not certified |
| sharp | certified | certified |

The asymmetry is deliberate. An outer set can safely certify a universal stable decision because it contains the true set. An inner set can safely certify instability because conflicting decisions inside the subset must also exist in the true set.

This is a set-theoretic claim, not a statistical coverage claim. The approximation guarantee itself must be proved or bound by a separate gate.

## 12. Evidence boundary

This layer does not establish:

- point identification of any RH-006 latent nuisance from real data;
- a valid instrument;
- bootstrap validity;
- uniform size control;
- a formal p-value;
- a confidence interval;
- empirical predictive validity;
- qualification of the existing Rust estimator.

It establishes only that the repository can represent and propagate a characterized scalar identified interval through the documented RH-006 polynomial ray geometry without making the grid itself the scientific definition.

## 13. References

Manski, C. F. (2003), Partial Identification of Probability Distributions.

Tamer, E. (2010), Partial Identification in Econometrics, Annual Review of Economics 2:167–195.

Bontemps, C. and Magnac, T. (2017), Set Identification, Moment Restrictions, and Inference, Annual Review of Economics 9:103–129.

Hill, J. B. (2021), Weak-Identification Robust Wild Bootstrap Applied to a Consistent Model Specification Test, Econometric Theory 37(3):409–463.

Schlemper, G. and Moreira, M. J. (2026), Confidence Sets under Weak Identification: Theory and Practice, arXiv:2604.04279.

The cited work provides methodological precedent for partial identification, weak-identification-aware resampling, and avoiding grid-defined set inversion; none is a plug-in proof for RH-006.

## Boundary

Research-only.

No inference selector is changed.

No p-value or confidence interval is enabled.
