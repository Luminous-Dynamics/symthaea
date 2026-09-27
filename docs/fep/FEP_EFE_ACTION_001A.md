# FEP-EFE-ACTION-001A — action predictive-uncertainty / novelty reference semantics

Issue: #6210  
Parent: #6209

## Status and claim ceiling

This is a docs/data-only synthetic software reference profile. It does not validate the Free
Energy Principle as a scientific theory, establish expected information gain or mutual information,
prove optimal experimental design, establish real-world information gain, or grant
engineering/evidence/execution authority.

The reference corpus is schema `fep-efe-action-001a-reference-v1`.

Canonical SHA-256 of the exact compact UTF-8 JSON bytes:

`3eeb5fe88a4e6893414fcfac9139912ba1312d219a656f99f417e2aeb6a53858`

## Why this source exists

Current `symthaea-fep` has explicit characterization tests demonstrating that, under its current
default EFE path, the software field called the epistemic term is action-invariant and candidate
enumeration contaminates novelty history. A repair needs a constructive target rather than merely
removal of those tests.

This source freezes one bounded known-answer profile. It is not the only legitimate formulation of
predictive uncertainty, epistemic value, or novelty.

## Reference predictive-uncertainty profile

`DiagonalGaussianPredictedEntropyDeltaV1` uses diagonal Gaussian belief precision.

For precision vector `pi` in dimension `d`:

```text
H(pi) = 0.5 * (d + d*ln(2*pi_constant) - sum_i ln(pi_i))
```

For candidate action `a`:

```text
E_proxy(a) = H(pi_pred[a]) - H(pi_current)
```

The current production API historically names this software component `epistemic`, so corpus
expectation fields retain names such as `epistemic_relation` for compatibility. The qualified
semantic claim is narrower: they refer to this exact predicted-entropy-delta proxy.

The scorer is minimized. Therefore, all else equal, an action whose declared predicted precision
implies lower hidden-state entropy has a more negative `E_proxy(a)`.

The corpus favors relational expected answers (`A<B`, `A=B`, reject, identity changed) over
last-bit floating-point values. Binary relations are always written in candidate-list orientation
(first candidate compared with second candidate), so `A>B` and `B<A` are not treated as different
semantic facts but only the former is canonical when the list order is `[A,B]`.

### Information-gain boundary

Standard active-inference expected free energy is commonly decomposed into pragmatic/extrinsic
value plus an epistemic term expressed as expected information gain / conditional mutual
information over future states and observations; equivalently it may be written in risk/ambiguity
form.

This V1 reference does **not** integrate over possible future observations or hypothetical posterior
belief updates. Therefore:

```text
predicted hidden-state entropy change
!= expected information gain
!= conditional mutual information
!= mutual information
!= realized information gain
```

A future expected-posterior-information-gain or mutual-information profile requires a distinct
profile identity, corpus, implementation, and qualification.

## Novelty profile and event ownership

`ActionEventCountNoveltyV1` defines:

```text
N(a) = 1 / (1 + event_count[a])
```

The event class is part of the profile identity. This frozen generic-FEP reference uses:

```text
novelty_event_class = CommittedAction
```

so the corpus specializes the formula to:

```text
N(a) = 1 / (1 + committed_count[a])
```

Candidate scoring is pure. Merely evaluating or selecting a candidate does not change committed
action history.

```text
candidate score != selection != commitment != external execution != observed outcome
```

An explicit generic-FEP commitment event changes `committed_count` under this reference profile.
This follows the live caller audit and ownership already established by #602 / PR #2148 and #606:
`act()` is currently a compatibility commitment/prediction boundary and is not proof of physical
execution.

A domain may later define an `ExecutedAction` or `ObservedOutcome` novelty profile, but that is a
separate profile identity and requires an actual external event/witness from the owning domain.
Generic FEP must not manufacture physical execution evidence merely to update novelty.

## Total reference score

For frozen weights:

```text
G(a) = w_p * P(a) + w_e * E_proxy(a) - w_n * N(a)
```

Raw pragmatic, predicted-entropy-delta, and novelty components remain separately inspectable. A
zero weight removes one component from the total only; it does not erase or mutate the raw
component.

No aggregate score may bypass an external hard feasibility, safety, evidence, or authority
constraint.

## Numeric policy

The source stores decimal inputs as strings. The independent qualifier should use deterministic
high-precision reference arithmetic and the frozen numeric profile:

- decimal precision: 50 digits;
- rounding: `ROUND_HALF_EVEN`;
- relation comparison tolerance: `1e-12`;
- relational expectations preferred over serialized transcendental outputs.

The qualifier may calculate logarithms at the frozen precision. Production Rust may use `f64`, but
production acceptance requires a separately justified numerical refinement/error contract rather
than treating the reference tolerance as an IEEE-754 proof. #5725 owns that reusable boundary.

## Identity

A reference result binds:

- schema/profile version;
- exact predictive-uncertainty profile identity;
- current precision state;
- order-independent candidate-set identity;
- action-conditioned predicted precision inputs;
- pragmatic inputs;
- novelty profile and exact event class;
- committed-action history snapshot for this frozen profile;
- weight profile;
- numeric/tolerance profile.

Changing uncertainty, weights, predictive-uncertainty profile, or novelty event class creates a new
result/profile identity. Enumeration order does not change result identity in the default
set-scoring profile.

## Frozen cases

The ordered corpus is `F01` through `F25`.

It covers:

- positive predicted-entropy-delta discrimination;
- exact proxy ties;
- multi-dimension predictive-uncertainty reduction;
- explicit pragmatic/proxy tradeoffs;
- zero-weight non-erasure;
- committed-action novelty;
- repeated rejected scoring purity;
- explicit commitment updates;
- selection/commitment/execution event-class separation;
- enumeration and set-order invariance;
- duplicate/missing/non-positive/dimension-invalid inputs;
- generation currentness;
- uncertainty/weight identity changes;
- exact score ties;
- hard external constraint precedence;
- software-only claim ceiling;
- explicit representation of the current action-invariant legacy profile.

## Independent qualification

A separate `001A1` qualifier should:

1. bind the exact source head, parent, blobs, digest, schema, profile identity, and `F01…F25` order;
2. reject unknown contract-critical fields/vocabulary;
3. derive Gaussian predicted-entropy differences, committed-event novelty, totals, relations,
   input validity, and identity/currentness outputs from raw inputs before comparing `expected`;
4. verify scoring purity by applying candidate-score calls without commitment events;
5. verify only the declared event class mutates novelty state;
6. verify enumeration invariance;
7. run adversarial mutations for flattening precision, considered/selected/committed/executed event
   confusion, order dependence, invalid precision, duplicate IDs, tie breaking, currentness,
   profile-label promotion, and authority promotion;
8. report only software-reference qualification.

## Production ownership and gate

Do not create a second purity/commitment implementation beside the existing FEP owners.

- #602 / PR #2148 owns pure candidate scoring plus the current committed-action novelty repair;
- #606 owns the additive split of selection, commitment, pure prediction, external execution, and
  actual outcome semantics;
- #604 separately owns TD-vs-direct-learning exclusivity;
- #6222 owns checked hidden-state shape and exact action identity;
- #6216 owns action-conditioned predictive uncertainty for this entropy-delta proxy;
- #6217 owns qualified selection/tie/policy-influence semantics;
- #6221 owns held-out usefulness/calibration.

The corrected reference semantics should be independently qualified before downstream production
work claims conformance to this generation.

```text
001A corrected frozen semantics/reference corpus
-> 001A1 independent oracle
-> consume/qualify #602/#2148 + #606 commitment semantics
-> #6222 checked state/action identity
-> 001C action-conditioned predictive-uncertainty proxy
-> 001D action-selection integration/invariance
-> 001E held-out calibration/ablation
```

Downstream engineering #6203 must not label the current core scorer an expected-information-gain
engine, mutual-information engine, physical epistemic-information-action selector, or execution
authority until a separately qualified profile actually establishes those stronger semantics.
