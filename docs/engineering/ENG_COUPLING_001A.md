# ENG-COUPLING-001A — Canonical Cross-Domain Transfer Contract

Issue: #6180  
Parent: #6173  
Reference corpus: `docs/engineering/data/eng_coupling_001a_reference_v1.json`  
Canonical corpus SHA-256 (UTF-8 bytes as committed): `f74b17ec813c3e8a43389489f22abb0b6fffe7af78086ea1262206153f586753`

## Status

This is a source/data contract only. It defines deterministic synthetic semantics for cross-domain engineering transfers. It does **not** execute solvers, validate physical models, create FIELD observations, satisfy requirements, commission hardware, certify products, authorize procurement/build, or create actuation authority.

## Governing theorem

```text
source-domain result
+ exact semantic transfer definition
+ target-domain expectation
-> one bounded cross-domain proposition
```

The following shortcuts are forbidden:

```text
field names match != same physical quantity
units compatible != physically valid mapping
stage A converged + stage B converged != coupled convergence
coupled numerical convergence != model applicability
mapped field != conserved transfer
synthetic result != physical observation
```

## Contract

A claim-bearing `EngineeringCouplingProfileV1` binds, directly or by canonical reference:

- schema and profile identity;
- source and target model subjects and generations;
- exact configuration / geometry generations where material;
- intended use and claim ceiling;
- execution direction (`OneWay`, `BidirectionalIterative`, `CoSimulation`, or separately qualified extension);
- ordered transfer definitions;
- source/target quantity identities;
- units and exact conversion semantics, including affine transforms;
- source/target frames and sign/orientation conventions;
- spatial support and exact interface/region identity;
- intensive/extensive semantics where relevant;
- instantaneous, averaged, integrated, or cumulative semantics;
- time grid/window/clock semantics;
- mapping operator identity and parameters;
- conservation profile and explicit source/sink/loss terms;
- uncertainty transformation and dependence/common-mode references;
- applicability and currentness dependencies;
- coupled residual/convergence policy where iterative;
- exact authority ceiling.

Friendly strings may be retained for display/compatibility but cannot carry the claim.

## Independent disposition axes

The reference corpus deliberately avoids a universal `pass` flag.

### Semantic compatibility

`Compatible`, `QuantityMismatch`, `UnitTransformMissing`, `FrameMismatch`, `SupportMismatch`, `TimeBasisMismatch`, `MappingUnsupported`, `IncompleteSemanticBinding`.

### Conservation

`NotApplicable`, `ExactUnderDeclaredModel`, `ConservativeWithinTolerance`, `NonConservativeByDeclaredSourceOrSink`, `ApproximateWithResidual`, `Unknown`.

### Uncertainty and dependence

`CompleteTransfer`, `IncompleteUncertaintyTransfer`, `DependenceViolation`, `UnknownUncertainty`.

### Currentness and applicability

`Current`, `ReviewRequired`, `Stale`, `ApplicabilityUnsupported`.

### Coupled numerical state

Source-stage, target-stage, coupling-residual, conservation-residual and warning state remain separate. The corpus uses bounded summary dispositions only for the synthetic cases: `NumericallyConverged`, `NotConverged`, `WarningIneligible`.

### Authority

`RejectPromotion` and `ComparisonOnly` protect the boundary between synthetic/model evidence and physical/authority-bearing claims.

## Reference corpus

The frozen corpus contains **42 ordered cases** covering:

- exact and missing unit conversions;
- affine temperature conversion;
- pressure/force and flux/rate support semantics;
- frame/sign transforms;
- distributed-to-lumped mappings;
- mapping and geometry generation currentness;
- time integration, interpolation, lag, and extrapolation;
- exact conservation, declared source/sink behavior, hidden loss, and double counting;
- stage vs coupling convergence;
- solver warning and applicability separation;
- transformed uncertainty, missing uncertainty, and common-mode dependence;
- source/target/mapping generation drift;
- stale-result injection;
- synthetic-to-FIELD and synthetic-to-physical-authority promotion attacks.

Positive mathematical controls include:

```text
10 mm = 0.01 m
25 degC = 298.15 K
1000 Pa * 0.2 m^2 = 200 N
500 W/m^2 * 0.4 m^2 = 200 W
50 W * 20 s = 1000 J
linear uncertainty scale: 10 ±2, scale 3 -> 30 ±6
```

These are software/reference facts only.

## Anti-duplication

ENG-COUPLING does not own another quantity, frame, uncertainty, solver, observation, ETK, or domain-physics ontology. Production integration should reference the canonical owners.

## Qualification sequence

```text
#6180 source/data freeze
-> #6181 independent stdlib-only oracle
-> hosted exact-head qualification
-> typed production integration
-> legacy MultiPhysicsRequest adapter with explicit semantic-loss report
-> benign integrated pilot #6182
```

The independent qualifier must derive outcomes from raw case inputs, not trust stored `expected` fields.

## Claim ceiling

A future exact-head PASS may establish only faithful deterministic software semantics for the frozen synthetic transfer corpus: quantity/unit/frame/support/time mapping, conservation accounting, uncertainty/dependence handling, currentness/applicability, and bounded coupled numerical dispositions.

It establishes no physical correctness of any coupled model, external-solver validity, engineering safety, requirement satisfaction, commissioning, certification, procurement/build readiness, or physical actuation authority.