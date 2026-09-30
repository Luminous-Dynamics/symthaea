# SWA-001 Spatial Intelligence & Adaptive Workspace Architecture

Status: proposed integration architecture

This document records where workspace/environment intelligence belongs in the existing Symthaea stack. The goal is composition, not another parallel domain engine.

## Existing substrate

The current repository already contains:

- symthaea-fabrication-kernel::building for thermal load, structural stress, occupancy, comfort, and energy consumption across multi-scale horizons.
- symthaea-digital-twin for auditable engineered-system twin state and telemetry primitives.
- engineering simulation bridges and safety/evidence abstractions.
- spatial cognition benchmarks for spatial relations, landmarks, perspective taking, and path updating.
- resource-allocation and active-inference machinery.
- institutional failure learning lineage for prediction -> outcome -> error -> revised constraint.

## Composition

A workspace is modeled as a coupled socio-technical environment:

PhysicalState
-> WorkspaceRepresentation
-> Predictions
-> CandidateInterventions
-> CounterfactualEvaluation
-> Human/authorized Decision
-> Intervention
-> Outcome
-> PredictionError
-> IFL ConstraintCandidate

### PhysicalState

Geometry and zones, thermal conditions, acoustic conditions, lighting, electrical load/power quality, structural condition, water/resource state, network dependencies, and equipment health.

### UseState

Aggregate occupancy, zone utilization, room/resource availability, demand patterns, and scheduling constraints. Individual performance inference is explicitly out of scope.

### HumanAgencyState

Explicitly supplied preferences, accessibility requirements, privacy choices, collaboration/deep-work preferences, and sensing consent. These are agency inputs, not latent personality labels.

### EconomicState

Lifecycle cost, utilization, maintenance burden, fixed commitments, stranded-cost exposure, reconfiguration cost, exit cost, and alternative-use value. Long-lived commitments must expose uncertainty rather than assuming demand.

### ResilienceState

Dependencies, spare capacity, degraded modes, islanding/fallback options, recovery time, and concentration risk.

### LearningState

Prediction, actual outcome, prediction error, mechanism hypotheses, counterevidence, counterfactuals, and learned constraints with complete lineage.

## Candidate types

The first adapter layer can introduce:

- WorkspaceAsset
- WorkspaceZone
- WorkspaceObservation
- WorkspacePreference
- WorkspaceIntervention
- WorkspacePrediction
- WorkspaceOutcome
- WorkspacePredictionError
- WorkspaceConstraintCandidate

These should reference existing engineering, digital-twin, temporal-prediction, and IFL types where practical.

## Multi-objective discipline

Do not reduce the workspace to a single opaque optimization score.

Keep separate dimensions such as:

- human agency
- privacy
- accessibility
- comfort
- deep-work capacity
- collaboration
- adaptability
- resilience
- lifecycle cost
- resource efficiency
- environmental impact
- maintenance burden

Candidate interventions should expose tradeoffs and uncertainty so humans/authorized governance can choose among them.

## WeWork-derived reversibility invariant

For any proposed long-lived workspace commitment, preserve:

1. expected demand range
2. commitment duration
3. fixed vs variable cost
4. utilization assumptions
5. maintenance burden
6. reconfiguration cost
7. exit/termination cost
8. alternative-use value
9. downside/liquidity scenario
10. evidence supporting the assumptions

The system should be able to ask whether a lower-commitment or more modular design achieves similar outcomes before recommending an irreversible commitment.

## Privacy and agency invariants

- No covert sensing.
- No employee productivity scoring.
- No inferred sensitive traits from ambient environmental data.
- Preferences are not blanket consent.
- Aggregate environmental telemetry must remain distinct from individual records.
- Recommendations never become authorization automatically.
- Physical actuation requires an explicit authorized control boundary.

## Qualification path

G0: deterministic schemas and serialization
G1: replayable predictions
G2: exact prediction/outcome pairing
G3: uncertainty and contradiction preservation
G4: deterministic counterfactual fixtures
G5: privacy/agency invariant tests
G6: longitudinal recurrence and reconfiguration-cost fixtures
G7: Mycelix receipt projection preserving complete lineage

## Implementation rule

Prefer adapters around existing crates over a new physics or simulation engine. The workspace layer should become a composition boundary connecting physical models, cognitive inference, simulation, economic/resilience reasoning, and Mycelix durable coordination.

Related: Symthaea issue #6624; Mycelix issue #3634.
