# RH-006 Identification Receipt + Surface Propagation Gate — v1

## Status

Research diagnostic only. The gate cannot authorize RH-006 formal inference.

The purpose is to prevent a scalar latent nuisance from becoming a downstream branch decision merely because an estimator can return a number.

## Receipt model

Every proposed nuisance target should produce a receipt with:

```text
observed_schema_id
target_id
classification
identification_model_id
identification_assumption_digest
observational_equivalence_suite_digest
identified_set_digest
downstream_surface_digest
```

The classification is:

```text
O   = directly observable
MI  = model-identified after explicit qualified assumptions
PI  = partially identified
NI  = not identified
```

The distinction is operational:

```text
O   -> point estimation may proceed
MI  -> point estimation may proceed only under the qualified model
PI  -> retain the set; do not silently collapse it to a point
NI  -> fail closed
```

## Surface propagation

Identification is not finished when a set is produced.

For a downstream function g, compute:

```text
I_g(P) = { g(c) : c in I_C(P) }
```

For a downstream qualitative decision delta, evaluate:

```text
Delta(I_C) = { delta(g(c)) : c in I_C }
```

Then:

- decision-stable means every admissible target value maps to one qualitative regime;
- decision-unstable means the identified set maps to multiple qualitative regimes;
- boundary means the identified set reaches a singular or nonregular surface.

This gives a useful distinction:

```text
point-identification failure != decision-identification failure
```

A target can be PI while the eventual decision is robust if every admissible value maps to the same decision. Conversely, a narrow set can still be unsafe if it straddles a hard branch boundary.

## Executed surface attack

The executable is `scripts/rh006_identification_surface_gate.py`.

It evaluates the adversarial set:

```text
C in [-1, 1]
```

against two surfaces.

### Quadratic branch surface

```text
D(C) = 0.25 - C^2
```

The image is:

```text
D([-1,1]) = [-0.75, 0.25]
```

All three regimes are reachable:

```text
two-real-branches
double-root
no-real-branch
```

Therefore the downstream status is:

```text
decision-unstable-under-identification-set
```

### Contraction-radius surface

```text
r(C) = sqrt(1 - C^2)
```

The image reaches zero at the PSD boundary.

Its derivative is:

```text
|dr/dC| = |C| / sqrt(1-C^2)
```

which diverges as |C| approaches 1.

Therefore the same identified set is also a nonregular conditioning boundary.

## Why this matters

The current RH-006 work already contains cryptographic binding of qualification identity, loss-differential identity, dependence profile, origin schedule, inference plan, and method-selection rule.

Those controls establish evidence provenance and procedural integrity. They do not by themselves establish scientific identification of a latent nuisance.

The new receipt belongs between the observable/equivalence layer and the estimator/inference layer:

```text
observable schema
-> observational equivalence
-> identification receipt
-> identified set
-> surface propagation
-> estimator qualification
-> finite-sample calibration
-> dependence characterization
-> inferential procedure
-> bootstrap
```

## Fail-closed rule

A future RH-006 inference receipt should be unable to authorize execution when:

```text
classification = NI
classification = PI and the downstream decision is not invariant
identified-set surface reaches a declared singular boundary
identification assumptions are absent or unqualified
observational-equivalence suite finds target variation
receipt digest does not bind the exact components
```

This prevents a procedurally perfect but scientifically unidentified nuisance from passing.

## Partial identification is a valid result

Identified sets are first-class inferential objects. The econometric literature explicitly studies inference and decision power under partial identification, including interval-identified parameters. [Canay and Shaikh, 2017; Bugni et al., 2026].

That supports this RH-006 policy:

```text
PI target
-> exact identified set
-> exact downstream set image
-> robust decision only if regime-invariant
```

rather than:

```text
PI target
-> arbitrary representative value
-> ordinary point inference
```

## Identification-rescue boundary

Multiple indicators can aid latent-variable identification, but rank, scaling, and measurement assumptions are part of the identification argument. Current SEM treatments continue to characterize local identification using the rank of the mapping from free parameters to model-implied moments.

Proxy-variable identification likewise depends on explicit relevance, separation, and rank/completeness conditions; prediction alone is insufficient.

Accordingly:

```text
predictive != relevant instrument != valid instrument
correlated proxy != identifying proxy
```

## Next executable target

The next stronger implementation step is a typed Rust `IdentificationReceipt` in `src/partnership/relational_prediction.rs`, consumed by the future inferential binding.

It should carry:

```text
target_id
classification
identification_model_id
assumption_digest
observational_equivalence_digest
identified_set_digest
surface_digest
decision_status
```

Its constructor should require the observational-equivalence suite and refuse to mint a point-identification receipt when admissible witness variation remains.

Only then should a latent nuisance be consumable by the estimator/inference bridge.
