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

Separate coherent SI representation from display/conversion units. Unit conversion must not silently change physical meaning. A unit descriptor carries an explicit affine transform to the chosen canonical representation. Offset-bearing units must not be modeled as pure multiplicative scales.

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

Implemented on the RFC branch through exact head `4d05209b226ea6a7738d1ad6587b88bad8302cbd`:

1. Shared `symthaea-types::physical_type` kernel with `PhysicalType`.
2. Quantity-kind enum separated from dimensional signature.
3. Shared seven-base-dimension `PhysicalDimension`.
4. `TypeJudgement::{Valid, Invalid, Unknown}`.
5. Arithmetic rules for Add/Sub/Mul/Div/Pow in the physics bridge expression layer.
6. Function/domain checks for sin/cos/exp/log/sqrt.
7. Explicit unknown dimensions; unknown is never encoded as dimensionless.
8. Explicit quantity annotations so same-dimension quantities such as Energy and Torque remain semantically distinct.
9. Deterministic canonical JSON bytes plus BLAKE3 digest for physical-type receipts.
10. Typed simulation quantities and typed multi-physics ports, while preserving legacy string APIs.
11. Explicit stage-to-stage typed connections so topology is not inferred from a globally unique signal name.
12. Ontology-neutral semantic identifiers for future QUDT/SysML/domain-catalog mappings.
13. Rational affine unit transforms, including scale and offset.

Still intentionally outstanding from this list: direct typed differentiation, typed EquationNode, typed-generation integration, and formal/executable equivalence binding.

The shared kernel is deliberately free of solver and AST dependencies. Expression inference remains in `symthaea-physics-bridge`, which consumes the common semantic types.

The success criterion is not a larger unit catalog. It is that the same candidate receives the same physical judgment everywhere it appears.

## References

- QUDT Catalog, latest published release (3.5.2, September 2026): https://www.qudt.org/catalog/qudt-catalog.html
- QUDT Quantity Kinds: https://www.qudt.org/doc/2026/09/DOC_VOCAB-QUANTITY-KINDS.html
- QUDT Dimension Vectors: https://www.qudt.org/doc/2026/09/DOC_VOCAB-DIMENSION-VECTORS.html
- FMI 3.0.2 Specification, UnitDefinitions: https://fmi-standard.org/docs/3.0.2/
- Lanyon AI, Our Vision (2026): https://www.lanyon.ai/blog/vision/
## Engineering integration

Engineering is a primary consumer of the physical type system, not a downstream application.

The existing engineering faculty already contains independent physical models for:

- structural mechanics and factor-of-safety evaluation;
- radial distribution power flow;
- thermofluid flow and head loss;
- control stability and transient metrics;
- circuit power checks;
- acoustics and optics;
- signal processing;
- materials and aging;
- robotics and multibody dynamics;
- CAD/fabrication geometry;
- digital-twin telemetry;
- solver-agnostic multi-physics orchestration.

The current interface is partly stringly typed. For example, simulation parameters and metrics use unit strings, and coupled solver stages exchange `consumes`/`produces` names as free-form strings. The transition path is additive: typed fields coexist with legacy fields, so adapters can migrate without a flag-day API break.

The type-system integration should therefore make the physical quantity itself first-class:

```text
TypedQuantity {
    id
    physical_type
    value_or_domain
    uncertainty
    provenance
}
```

A multi-physics edge should become an explicit typed topology:

```text
producer stage + output port
      -> physical compatibility judgment
      -> optional explicit transform
      -> consumer stage + input port
```

The v0.1 simulation bridge implements this boundary with `TypedPhysicalPort` and `TypedPhysicalConnection`. Unknown semantics fail closed during typed-topology validation.

This prevents physically incompatible coupling from surviving until an external solver is invoked.

### Requirements become measurable contracts

An engineering requirement should be able to express a typed quantity and admissible bound, while keeping the natural-language statement for human review.

Example:

```text
REQ-STRUCT-001
quantity: MaximumStress
type: Stress
constraint: <= allowable_stress
criticality: Blocking
evidence: FEA + material data
```

The type system does not decide whether the allowable stress itself is justified. It ensures the comparison is physically meaningful.

### Engineering discovery

Engineering models should become discoverable dynamical systems.

Candidate discovery can operate over:

- structural oscillators and load-response models;
- RLC and power-system dynamics;
- control plants;
- thermal/flow networks;
- material degradation trajectories;
- vehicle and manipulator dynamics.

The same discovery protocol applies:

```text
engineering model
  -> cold candidate discovery
  -> independent holdout
  -> physical type check
  -> symbolic establishment
  -> formal proof where supported
  -> solver simulation
  -> telemetry comparison
  -> safety-case evidence
```

### Multi-physics interoperability

Keep the internal model solver-neutral.

FMI is a useful adapter boundary because FMI 3.0 defines Model Exchange, Co-Simulation, and Scheduled Execution and includes unit/type metadata in model descriptions. It should remain an optional interoperability layer, not a dependency of the Symthaea core. citeturn972582search0turn972582search2

SysML v2 is similarly relevant at the systems-engineering boundary. The OMG specification provides formal semantics for requirements and other system aspects, plus machine-readable quantities-and-units and requirement-derivation libraries. Symthaea should map to that ecosystem at import/export boundaries rather than reproduce SysML internally. citeturn909167search0turn909167search4

### Important distinction

Engineering evidence should preserve the same epistemic ladder as scientific discovery:

```text
typed
  != validated
  != simulated
  != verified
  != certified
  != physically true
```

A solver result can discharge a narrowly defined engineering obligation inside its model envelope, but it must not silently elevate the claim beyond that envelope.

## Model maturity is a separate axis

Physical semantics answer what a model means. They do not answer how much trust the model has earned.

A model descriptor should therefore carry an independent maturity/evidence envelope. A useful initial vocabulary is:

- TextbookAnalytic
- ValidatedNumerical
- CalibratedEmpirical
- ResearchPrototype
- FrontierHypothesis
- SyntheticInstrumental

This matters because Symthaea already spans very different regimes: continuum solvers, calibrated nuclear models, particle-physics calculations, quantum chemistry, engineering solvers, and frontier modules such as holography, stochastic QED, tensor networks, and quantum Darwinism.

The critical invariant is:

dimensionally valid != physically validated != engineering qualified

A numerical pass must not upgrade a model's maturity. Catalog recognition must not upgrade it. A formal theorem about the mathematics must not automatically certify the empirical model. A frontier hypothesis may still be mathematically interesting and discoverable; it simply carries a different evidence envelope.

Conceptually:

```text
ScientificModel {
    physical_type
    model_maturity
    epistemic_status
    provenance
    executable_semantics
}
```

Engineering safety gates can require a minimum maturity/evidence class without preventing exploratory research from using lower-maturity models.

Related: #6871.
## Interoperability design note

QUDT 3.5.2 publishes distinct vocabularies for Quantity Kinds, Units, and Dimension Vectors. FMI 3.0.2 likewise carries unit exponents plus scale factors and offsets in model descriptions.

Symthaea therefore keeps the core representation compact and solver-neutral:

```text
PhysicalType
  = quantity kind
  + optional dimension
  + optional unit transform
  + scalar domain
  + refinements
  + optional external semantic identifier
```

A future QUDT adapter can map `SemanticIdentifier { namespace, identifier }` to a QUDT QuantityKind/Unit URI without making QUDT a runtime dependency of the shared type crate.

The current v0.1 dimension representation remains seven-base-SI for compatibility with the existing physics bridge. Angle/radian semantics are represented through `QuantityKind::Angle`; extending the dimension vector with an explicit angle component should be deliberate schema evolution rather than an incidental field addition.