# Symthaea Physical Agency RFC v0.1

**Status:** Draft architecture RFC  
**Base reviewed:** `2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`  
**Execution posture:** simulation-first; no new hardware authority

## Purpose

Symthaea already has specialist physics, causal-reasoning, simulation, digital-twin,
formal-safety, control, perception, and HAL layers. The missing abstraction is not
another laser/acoustics/plasma controller. It is a common language for describing a
*desired physical state transition*, comparing physically different mechanisms that
could realize it, and proving that a proposed intervention is sufficiently understood
and authorized before it can approach an actuator boundary.

The architectural invariant is:

```text
CanModel(effect) != CanPropose(effect) != CanExecute(effect)
```

Understanding a harmful or hazardous physical phenomenon must never imply execution
authority. Threat models are therefore represented separately from executable
interventions.

## Existing anchors to reuse

Physical Agency is deliberately additive. It should compose existing capabilities
rather than duplicate them:

- `symthaea-causal-reasoning`: structural causal models, interventions, graph surgery,
  and counterfactual reasoning.
- `symthaea-sim-bridge`: normalized solver requests/results, multi-physics coupling,
  uncertainty, and execution provenance.
- `symthaea-digital-twin`: telemetry-backed asset state and prediction-error/free-energy
  tracking.
- `symthaea-formal-safety`: proof obligations and evidence references independent of a
  specific proof assistant.
- `symthaea-control`: estimation, LQR/LQG/MPC, stability primitives, and future runtime
  safety-filter integration.
- `symthaea-hal`: the existing below-cognition watchdog/e-stop/bounds interlock. This
  remains authoritative for its supported hardware.
- Specialist physical domains such as `symthaea-acoustics`, `symthaea-optics`,
  `symthaea-thermofluids`, `symthaea-circuits`, and `symthaea-physics`.

`Physical Agency` must not rename or repurpose `symthaea-field-dynamics`, whose current
ontology is consciousness-field dynamics rather than physical actuator fields.

## Three-plane separation

Physical agency keeps three kinds of state separate even when they refer to the same
phenomenon.

### Epistemic plane

What Symthaea currently believes about reality:

```text
observation + provenance + uncertainty + validity region + timestamp
```

A belief is not an authorization.

### Causal plane

What Symthaea predicts would happen under an intervention:

```text
Belief(S_t) x do(intervention) -> Distribution(S_t+dt)
```

A prediction is not an authorization.

### Authority plane

What the system is currently permitted to execute, under which evidence, envelope,
and expiry.

```text
proposal + evidence + discharged obligations + authority + freshness -> permit
```

No permit means no executable transition.

## Core vocabulary

The first implementation should define a small, stable physical-effect ontology rather
than device commands.

`DesiredTransition` describes the requested state change. It carries an objective,
target-region reference, effect class, permitted modalities, required authority,
reversibility expectation, uncertainty budget, energy budget, and time budget.

`PhysicalModality` names broad mechanism families such as mechanical, acoustic,
photonic, thermal, electric, magnetic, fluid, plasma, chemical, and coupled/multiphysics.
It does not encode a specific device.

`EffectKind` names state transformations such as observe, characterize, communicate,
translate, rotate, heat, cool, excite, illuminate, constrain, separate, join, deposit
material, remove material, alter flow, and coupled/custom transformations.

`ProposedIntervention` is a mechanism-specific candidate produced by a planner or
adapter. It is always unqualified.

`ThreatScenario` describes an effect that Symthaea must predict or defend against. It
has no conversion path to `ProposedIntervention`.

`ExecutionPermit` is a later runtime capability. Its constructor must remain private to
the qualification path and its initial representation must not implement
`Deserialize`; serialized bytes must never mint actuator authority.

## Authority classes

Authority is monotonic and fail-closed:

```text
SimulationOnly
    < PassiveObservation
    < DiagnosticExcitation
    < ReversibleActuation
    < ControlledEnergyTransfer
    < IrreversibleMaterialChange
```

The default is `SimulationOnly`. Unknown authority is never promoted implicitly.
A permit at one class cannot authorize an intervention requiring a higher class.

## Physical-effect compilation

The long-term compiler pipeline is:

```text
Goal
  -> DesiredTransition
  -> mechanism expansion
  -> counterfactual portfolio
  -> solver / model ensemble
  -> uncertainty + validity checks
  -> Pareto filtering
  -> safety obligations
  -> authority qualification
  -> runtime safety filter
  -> actuator-specific adapter
```

The first tranche ends before the final two stages.

## Model ensemble and epistemic parallax

No single model should become an oracle. Candidate consequences may be evaluated by
several independent model families:

```text
analytical | numerical | learned | empirical | HDC-analogical
```

Agreement is evidence. Disagreement is a first-class signal that may trigger more
sensing, another solver, human review, or abstention.

Every model contribution should carry a validity region. Leaving that region raises
uncertainty and may invalidate execution qualification.

## Simulation evidence boundary

The existing `symthaea-sim-bridge` distinction between `DryRun`, `Unknown`, and
`ExternalSolver` provenance should be preserved and strengthened.

For physical-agency qualification:

- `DryRun` may test orchestration but must not discharge an execution-safety obligation.
- `Unknown` execution provenance is non-evidence for execution qualification.
- Solver-backed safety evidence must come from `ExternalSolver` results with complete
  backend/version/input/output/parser provenance.
- Backend capabilities must default to unsupported/false when not declared.

## Abstention and information-seeking actions

Abstention is a normal planner result, not an error. At minimum the planner must be able
to return:

```text
Observe | Simulate | AskForReview | Wait | Retreat | SafeShutdown | NoQualifiedAction
```

Where additional information could materially alter the decision, active perception may
be selected before manipulation. The system should prefer changing what it knows before
changing the world when decision-relevant uncertainty is too high.

## Runtime safety architecture

Physical Agency does not bypass platform safety.

The intended composition is:

```text
cognition
  -> unqualified physical proposal
  -> qualification / proof obligations
  -> execution permit
  -> runtime safety filter
  -> existing platform interlock / HAL
  -> hardware
```

Independent hardware interlocks remain authoritative even when every upstream layer is
green.

## Initial crate boundaries

### `symthaea-physical-effects`

A small dependency-light domain crate containing canonical effect vocabulary, authority
classes, budgets, validation, and proposal/threat-model types. It contains no planner,
solver, HAL, or device code.

### `symthaea-physical-agency`

A later composition crate for mechanism discovery, counterfactual portfolios, model
comparison, Pareto selection, abstention, and qualification inputs. It consumes existing
causal, simulation, digital-twin, and formal-safety crates.

No `symthaea-physical-runtime` crate should be added until simulation-only qualification
has demonstrated that the abstractions are stable.

## Phase-A invariants

1. No dependency from the new physical-agency crates to `symthaea-hal`.
2. Default authority is `SimulationOnly`.
3. Threat models have no executable conversion API.
4. Qualified runtime capabilities are not deserializable from untrusted data.
5. `DryRun` and unknown-provenance simulation results cannot discharge execution gates.
6. Unknown backend capabilities are treated as unsupported.
7. Invalid/non-finite budgets and probabilities fail closed.
8. A model outside its validity region cannot silently retain prior confidence.
9. Abstention is always a legal planner outcome.
10. Existing platform interlocks remain below and independent of cognition.

## Proposed stacked PR sequence

```text
PA-00  RFC and invariants (this document)
PA-01  symthaea-physical-effects: canonical types + validation
PA-02  sim-bridge backend capability negotiation (conservative defaults)
PA-03  physical-agency: candidate portfolio + Pareto + abstention
PA-04  formal-safety qualification adapter + non-serializable execution permit
PA-05  PHYSIS v0 simulation-only benchmark
PA-06  acoustic reference adapter (simulation / low-energy research only)
PA-07  optics reference adapter (sensing / simulation first)
PA-08  existing plasma-domain integration as simulation/threat-model input
PA-09  digital-twin system identification and model-validity tracking
PA-10  active perception / value-of-information planner
```

Each PR should remain independently reviewable, preserve the workspace DAG, add tests for
new safety-critical types, and avoid expanding actuator authority as a side effect of
adding modeling capability.

## Phase-A exit gate

Phase A is complete only when Symthaea can take a modality-neutral `DesiredTransition`,
generate multiple simulated mechanism candidates, preserve model disagreement and
uncertainty, abstain when evidence is insufficient, compile safety obligations, and prove
through tests that no execution-capable value can be obtained without the qualification
boundary.

That establishes physical intelligence as a causal/evidence architecture before any
new hazardous-energy hardware integration is considered.
