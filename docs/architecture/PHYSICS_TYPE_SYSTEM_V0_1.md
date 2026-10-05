# Physics Type System v0.1

Status: Architecture RFC
Related: #6862, #6865, #6866, #6867, #6868

## Thesis

Symthaea should have a physics type system, but it should not be merely a units library and it should not replace Rust's ordinary type system.

The repository already contains several partially overlapping forms of physical typing:
- DimensionalSignature and dimensional inference
- constructive dimension-aware GP candidate generation
- TensorDescriptor and MetricSignature
- EquationNode
- Lie/discrete symmetry descriptors
- physics-domain classification
- numerical validation
- symbolic analysis
- formal verification backends

The missing layer is a single compositional physical judgment that can travel with a candidate from discovery through validation, simulation, and proof.

## Design principle

Use a gradual, refinement-oriented physical type system.

Unknown information stays unknown. Proven incompatibility is rejected. Stronger claims require stronger evidence.

Desired progression:

```text
Unknown
  -> partially inferred
  -> dimensionally typed
  -> geometrically typed
  -> physically refined
  -> numerically validated
  -> symbolically established
  -> formally discharged
```

The mathematical/physical type and epistemic status remain separate concepts, but an evidence receipt can bind them together.

## Physical type axes

### Quantity kind

Examples: Length, Mass, Time, Velocity, Acceleration, Momentum, Force, Energy, Power, Charge, Temperature, Angle.

This is distinct from dimensions. Energy and torque have identical SI dimensions but are not automatically interchangeable physical quantities.

### Dimension

Reuse the existing seven-base-dimension exponent representation. Multiplication adds exponents, division subtracts them, powers scale them, and addition/subtraction require compatible dimensions.

### Unit and scale

Separate coherent SI representation from display/conversion units. Unit conversion must not silently change physical meaning.

### Scalar domain

Track integer, rational, real, complex, interval/bounded numeric, and approximate floating-point domains where they affect admissible operations or proof strategy.

### Geometry and tensor shape

Reuse tensor rank, covariant/contravariant index positions, tensor symmetries, ambient metric signature, and manifold/coordinate context when known.

### Domain refinements

Track positive, nonnegative, nonzero, bounded, normalized, periodic, continuous, and discrete properties. These are required for safe use of log, inverse powers, square roots, denominators, and similar operators.

### Dynamics semantics

Track independent variable, derivative order, continuous-vs-discrete time, ODE-vs-PDE role, and spatial differential order where relevant.

### Symmetry metadata

Track Lie-group representation, discrete symmetries, gauge/local-vs-global status, and parity/time-reversal behavior as metadata until a backend can justify stronger proof-relevant claims.

## Core typing rules

- Energy + Energy -> Energy
- Energy + Length -> TypeError
- Mass * Acceleration -> Force
- Force * Length -> Energy
- Velocity * Time -> Length
- Energy / Time -> Power
- sin(Angle) -> Dimensionless
- sin(Length) -> TypeError
- log(Energy) -> TypeError
- log(Energy / EnergyReference) -> Dimensionless
- d(Length)/dt -> Velocity
- gradient(ScalarField) -> CovectorField
- tensor contraction requires compatible index positions
- equality requires compatible physical type, not merely equal raw dimensions

Dimensional consistency is necessary evidence, never a proof that an equation models the real world correctly.

## Unknowns and fail-closed behavior

Unknown is not Dimensionless.

If a variable has no physical annotation, the checker should return Unknown rather than silently treating it as dimensionless.

An operation should be Valid when compatibility is established, Invalid when incompatibility is established, and Unknown when the available context is insufficient.

## Integration architecture

### Phase 1 — PhysicalType kernel

Add a small serializable pure-data type that composes existing descriptors rather than duplicating them.

Conceptually:

```text
PhysicalType = { kind, dimension, scalar_domain, geometry, refinements, dynamics }
```

Do not begin with a giant generic Rust type.

### Phase 2 — Typed EquationNode

Give every equation AST node an inferred physical type and structured diagnostics with source paths.

### Phase 3 — Typed discovery

Extend typed_generation.rs from constructive dimensional generation toward typed evolutionary search. Physical type should become a generative constraint, not merely a post-hoc rejection filter.

### Phase 4 — Typed differentiation

Derivative operators should transform physical types automatically, creating a direct semantic bridge between calculus and physical meaning.

### Phase 5 — Typed numerical methods

Annotate integrators and discretization schemes with input/output types, continuous-vs-discrete semantics, approximation assumptions, and known invariant obligations.

A numerical method must not silently inherit the truth of a continuous equation.

### Phase 6 — Proof bridge

Generate formal obligations from the same canonical typed representation used for executable evaluation.

### Phase 7 — Discovery receipts

Publication-grade results should bind candidate digest, physical type judgment, grammar/configuration digest, search seed, discovery IC manifest, holdout seed, holdout IC manifest, symbolic status, formal status, recognition result, and negative-control result.

## What this unlocks

Typed discovery reduces wasted search, rejects nonsensical equations before expensive simulation, distinguishes equal-dimension-but-different quantities, strengthens tensor physics, gives proof backends canonical semantics, and creates an interoperability boundary for external scientific tools.

## Relationship to Lanyon

Lanyon explicitly describes a long-term goal of building a type system for the physical universe and a pipeline from formal specification to implementation and proof.

Symthaea should pursue a complementary role: a discovery-aware physical type system that remains useful when hypotheses are incomplete or unknown.

No partnership or endorsement is implied by this RFC.

## Non-goals

- Do not replace Rust's type system.
- Do not encode every physical theorem into the type checker.
- Do not require complete metadata before exploration can begin.
- Do not equate dimensional consistency with physical correctness.
- Do not collapse epistemic status into mathematical type.
- Do not duplicate existing unit and tensor abstractions.

## First vertical slice

1. PhysicalType data structure.
2. Quantity-kind enum.
3. Reuse DimensionalSignature.
4. TypeJudgement::{Valid, Invalid, Unknown}.
5. Arithmetic rules for Add/Sub/Mul/Div/Pow.
6. Basic function/domain checks.
7. Time-derivative dimension inference.
8. Integration with random_expr_with_dimension.
9. Structured diagnostics.
10. Deterministic serialization/digest for evidence receipts.

The success criterion is not a larger unit catalog. It is that the same candidate receives the same physical judgment everywhere it appears.

## References

- Lanyon AI, Our Vision (2026): https://www.lanyon.ai/blog/vision/
- Lanyon AI, Formulary (2026): https://www.lanyon.ai/research/formulary/
- Modelica Language Specification, Unit Expressions: https://specification.modelica.org/master/unit-expressions.html
- uom Rust crate: https://docs.rs/uom/
- dimensioned Rust crate: https://docs.rs/dimensioned/
- Bobbin et al., Formalizing dimensional analysis using the Lean theorem prover (2025): https://arxiv.org/abs/2509.13142
- SAIUnit, Nature Communications (2025): https://www.nature.com/articles/s41467-025-58626-4
