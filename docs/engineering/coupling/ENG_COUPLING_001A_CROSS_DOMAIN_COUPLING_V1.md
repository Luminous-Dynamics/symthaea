# ENG-COUPLING-001A — Cross-Domain Coupling Reference v1

## Status

Reference/source subject for ENG-COUPLING-001 / issue #6173 and source child #6176.

This document freezes a synthetic semantic contract for engineering model coupling. It is not a solver validation, physical conservation proof, model-credibility receipt, safety case, commissioning record, requirement-satisfaction receipt, or actuation authority.

Canonical JSON SHA-256:

`36a79c8cabab9a99f4c5ad4d67330c107a4417c99a1790c019b0a384b02c38ba`

## Ownership boundary

Domain owners define the physics/models on each side of an interface. SE-SEM owns canonical quantities, units, frames and interface primitives. ENG-COUPLING owns only the exact transfer relationship. `symthaea-sim-bridge` owns orchestration/execution. SE-VV/SE-MODEL own applicability, credibility, UQ and discrepancy consequences. FIELD/SE-OBS own physical observations.

Therefore:

```text
field names match
!= same physical quantity

units compatible
!= coupling physically valid

stage convergence
!= coupling convergence
!= model applicability
!= model credibility
!= physical validation
```

## Required coupling identity

A claim-bearing coupling profile binds:
- exact source and target model subjects/generations;
- coupling directionality;
- canonical exchanged quantity;
- source/target units and any affine transform;
- source/target frames, sign and orientation;
- spatial support and exact interface region;
- mapping/interpolation/reduction operator identity;
- conservation/source/sink/residual profile;
- temporal synchronization and interpolation/extrapolation policy;
- stage convergence and coupling-residual convergence separately;
- uncertainty transform and dependence/common-mode refs;
- applicability/currentness dependencies;
- intended-use and claim ceiling.

Changing any claim-relevant item creates a new coupling generation or requires explicit applicability transfer.

## Directionality

```text
OneWay
BidirectionalIterative
CoSimulation
OtherQualifiedProfile
```

Execution order alone does not establish directionality semantics.

## Mapping and support

A point value, line load, surface flux, volume source and lumped port are not interchangeable. Mapping operators are exact semantic subjects. Examples include identity, affine unit conversion, frame transform, interpolation, conservative projection, surface/volume integration, distributed-to-lumped reduction and externally qualified adapters.

```text
mapped value
!= conserved transfer
```

## Conservation

Where applicable preserve mass, energy, charge, momentum, angular momentum, species mass or another exact conserved quantity independently.

Possible states:

```text
NotApplicable
ExactUnderDeclaredModel
ConservativeWithinTolerance
NonConservativeByDeclaredSourceOrSink
ApproximateWithResidual
Unknown
```

Unknown never silently becomes conserved. Declared sources, sinks and residuals remain visible.

## Temporal semantics

Bind source/target time grids, synchronization, interpolation/extrapolation, lag/delay, iteration order and convergence/stopping policy.

```text
stage A stable + stage B stable
!= coupled loop stable
```

## Uncertainty and dependence

A transformed value without a declared uncertainty transformation is not a complete uncertainty transfer. Common upstream uncertainty and derived-field dependence remain explicit. Multiple outputs derived from one upstream result are not independent evidence.

## Currentness

A source/target model, geometry, material state, mapping operator, calibration or boundary-condition change can stale the coupling. Friendly labels do not preserve semantic identity across such changes.

## Evidence boundary

```text
SyntheticCoupledField
!= FIELD PhysicalObservation

numerical agreement with one observation
!= physical validation
```

Model/source/target/interface residuals remain separately reportable.

## Positive reference profiles

The canonical corpus contains three synthetic positive/reference profiles:
1. Celsius-to-Kelvin affine temperature transfer;
2. conservative thermal-power surface transfer under an iterative profile;
3. framed structural displacement transfer to a sensor/control frame.

They establish only known-answer semantic structure.

## Hostile corpus

The canonical JSON contains 24 cases covering unit errors, affine-unit errors, frame/sign mismatch, spatial-support mismatch, quantity mapping, temporal integration, conservation loss, declared sources/sinks, directionality, coupling convergence, applicability, stale generations, geometry drift, time-grid mismatch, extrapolation, mapping identity, dependence, uncertainty loss, common modes, source double counting, stale-stage reuse, synthetic-observation laundering, prediction/observation agreement and no-authority synthetic closure.

## Relationship to current multiphysics runtime

The desired migration is:

```text
EngineeringCouplingProfileV1
        ↓ qualified semantic reference
MultiPhysicsRequest / CoupledSimulationStage
        ↓ execution
SimulationResult + coupling diagnostics
```

Current friendly `consumes` / `produces` strings may remain compatibility/display data initially. They are not sufficient claim-bearing coupling identity.

## Claim ceiling

A future qualification PASS may establish only deterministic source-contract conformance for the frozen synthetic coupling semantics. It does not establish physical model validity, physical conservation, solver correctness beyond separately qualified evidence, safety, requirement satisfaction, commissioning, build readiness, or physical actuation authority.
