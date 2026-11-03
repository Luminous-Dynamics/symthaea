# Hybrid PhysicsIR v0.2 — model semantics and claim boundaries

This note documents the additive hybrid-dynamics extension to the existing
solver-independent `ScientificModel`. It does not define a second physics
ontology, a solver interface, an event scheduler, a posterior representation,
or a verification receipt format.

## Compatibility and identity

The existing constructor remains on `SCIENTIFIC_MODEL.v0.1`. A v0.1 model has
no hybrid extension and omits the optional `hybrid_dynamics` field when
serialized. Deserialization defaults a missing field to `None`.

Calling `ScientificModel::with_hybrid_dynamics` promotes the model to
`SCIENTIFIC_MODEL.v0.2`, adds the versioned `HYBRID_DYNAMICS.v0.1` payload,
and validates the complete model. The hybrid payload is included in canonical
serialization and therefore in the parent model digest. A semantic change to
a mode, guard, reset, transition, parameter condition or simultaneous-event
policy changes that identity.

The hybrid payload has one declared physical time coordinate. All mode
derivatives and any time-dependent guard/reset expressions must use that
coordinate. Modes, guards, resets and transitions have stable identifiers and
are sorted during construction for deterministic serialization.

## Core objects

- **Mode:** a discrete physical regime with its vector/algebraic equations,
  explicit parameter roster, assumptions, evidence requirements and
  non-claims.
- **Guard:** a scalar typed physical expression, source mode, crossing
  direction, optional time symbol, parameter roster and applicability/evidence
  references.
- **Reset:** an explicit assignment for every state coordinate, source and
  target modes, named impact/contact law, parameter/time dependence and
  validity/evidence references.
- **Transition:** a source mode, guard, reset, target mode and exact rational
  parameter predicates.
- **Simultaneous-event policy:** either require an independently verified joint
  event/contact certificate or preserve unresolved event-order alternatives.

## The point-impact golden model

The initial golden model declares:

```text
State:      y [length], v [length / time]
Parameters: g [length / time^2], m [mass], e [dimensionless]
Coordinate: t [time]
Auxiliary:  N [force]

Mode free-flight:
    dy/dt = v
    dv/dt = -g

Guard:
    y = 0, crossing positive -> negative

Rebound transition:
    0 < e <= 1  (strict lower and inclusive upper transition predicates are required)
    y_plus = y_minus
    v_plus = -e * v_minus
    target mode = free-flight
    law = Newton normal restitution

Settling transition:
    e = 0
    y_plus = y_minus
    v_plus = 0
    target mode = persistent-contact
    law = declared unilateral normal support

Mode persistent-contact:
    y = 0
    v = 0
    dy/dt = 0
    dv/dt = 0
    N = m * g
```

The e=0 transition is not represented as a zero-speed free-flight rebound:
it enters a distinct persistent-contact mode. This matters for the post-event
vector field and the saltation map.

## Dimensional obligations

The extension reuses the existing `PhysicalType`, `ScientificExpression`,
`ScientificQuantity`, equation, assumption and evidence contracts.

- Each additive guard term must be dimensionally compatible.
- Reset outputs must match the target state-coordinate type.
- A reset must explicitly assign every state coordinate; there is no implicit
  identity carry-through.
- Every mode/guard/reset expression parameter must appear in the corresponding
  explicit parameter roster.
- Mode derivatives must use the model's declared hybrid time coordinate.
- A time-dependent guard/reset must explicitly declare that same time
  coordinate.
- Newton restitution must be a known real dimensionless parameter.
- Rational transition predicates are exact, positive-denominator, reduced
  rationals; no floating threshold is smuggled into the model. The validator
  intersects bounds/equalities/exclusions and rejects provably empty domains.
  Integer-typed parameters are checked against the integer lattice, rather than
  assuming every nonempty real interval contains an admissible integer.

The extension does not yet create a full symbolic inequality/proof engine for
all parameter-domain coverage. Declared predicates describe transition
conditions; they do not prove that runtime parameter values satisfy them or
that the roster of transitions is globally exhaustive.

## Evidence boundaries

Canonical model semantics are separate from:
- branch and branch-completeness certificates;
- event-time interval enclosures and event-order graphs;
- saltation/reset sensitivity receipts;
- solver numerical run receipts;
- posterior probabilities or conditional branch likelihoods;
- empirical measurement and physical validation.

The simultaneous-event policy does not prove that two events are simultaneous.
Equal timestamps or overlapping event-time enclosures remain unresolved unless
a joint event/contact law and an independently verified joint-event certificate
establish the semantics.

The v0.2 schema is a model representation, not an integrator or a claim of
physical validity. The 1D fixture is analytical and synthetic. It does not
qualify friction, simultaneous impacts, multiple contact branches, compliant
contact, parameter-family saltation, or a real engineered embodiment.

## Integration path

The intended division is:

1. Symthaea's `ScientificModel` binds canonical physical meaning and
   dimensional assumptions.
2. Symtropy executes a declared model projection and emits independent
   numerical / branch / event evidence.
3. Symthaea's inference layer consumes those evidence artifacts to estimate
   branch-conditioned uncertainty, without inferring branch probabilities from
   solver convergence frequency.
4. Independent verifiers assess the relevant receipts and claim ceilings.

Backend projection must report omitted or transformed semantics. A successful
solver run is not proof that the model meaning was preserved, and a formally
well-typed model is not empirical confirmation.
