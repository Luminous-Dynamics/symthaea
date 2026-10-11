# SWA-001 Spatial Intelligence & Adaptive Workspace Architecture

Status: proposed integration architecture

This document records where workspace/environment intelligence belongs in the existing Symthaea/Mycelix stack. The goal is composition, not another parallel domain engine.

## Existing related work — important discovery

This capability is more developed than the initial proposal suggested.

### Symthaea substrate

The repository already contains:

- `symthaea-fabrication-kernel::building` for thermal load, structural stress, occupancy, comfort, and energy consumption across multi-scale horizons.
- `symthaea-digital-twin` for auditable engineered-system twin state and telemetry primitives.
- `symthaea-engineering` as a composition facade combining digital twins, simulation, causal reasoning, formal safety, materials aging, memory, and the Global Workspace.
- `symthaea-sim-bridge` for solver-agnostic FEA/CFD/multibody/circuit/process simulation.
- spatial cognition and perception infrastructure.
- resource-allocation and active-inference machinery.
- institutional failure learning lineage for prediction -> outcome -> error -> revised constraint.

The existing `TwinState` is already especially close to the required primitive: it stores identity, asset class, telemetry, prediction residuals, epistemic/aleatoric uncertainty, free-energy breakdown, health, and intervention candidates.

### Mycelix substrate

Mycelix already has an unusually direct environment/commons substrate:

- `mycelix-workspace`: ecosystem orchestration, shared bridge infrastructure, SDKs, routing, migration, and hApp composition.
- `mycelix-commons`: property, housing, maintenance, cooperative membership, community land trusts, governance, water, food, transport, mutual aid.
- `mycelix-energy`: energy projects, investment/ownership transitions, grid production/consumption and peer-to-peer energy coordination.
- `mycelix-hearth`: household/kinship coordination, shared resources, care, autonomy, rhythms and presence.
- `mycelix-identity`: identity, recovery and assurance infrastructure.
- `mycelix-attribution`: privacy-preserving usage receipts, attestations and stewardship lineage.
- `mycelix-governance` and shared bridge infrastructure for authorized decisions and cross-cluster routing.

Therefore SWA should **not** create new housing, energy, identity, commons, or governance stacks. It should connect these existing systems through a narrow evidence/decision boundary.

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

## Boundary architecture

**Symthaea owns inference and simulation:**

observe -> represent -> predict -> simulate -> compare -> propose -> learn

**Mycelix owns durable coordination context:**

identity -> consent -> authorization context -> decision -> receipt -> contestability -> durable provenance

A Mycelix receipt must never turn a Symthaea prediction into an observation or grant physical authority.

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

## Immediate next implementation slice

Build one deterministic fixture spanning a small building:

1. a `BuildingTwin` supplies physical predictions and residuals;
2. `symthaea-digital-twin::TwinState` supplies uncertainty/health/intervention context;
3. an intervention candidate changes a zone-level configuration;
4. a counterfactual evaluator compares at least comfort, energy, resilience, cost and reversibility independently;
5. IFL records prediction -> outcome -> error -> constraint lineage;
6. Mycelix records consent/authorization context and a durable decision receipt;
7. replay proves that the same fixture produces the same decision evidence without requiring the same authority holder.

This fixture should be the reference integration test before any real-world actuation work.

Related: Symthaea issue #6624; Mycelix issue #3634.


## 8. Boundary calibration against existing Mycelix primitives

A deeper repository audit confirms that the workspace architecture should compose several existing Mycelix primitives rather than introduce a new generic consent or attribution stack:

- mycelix-commons/zomes/boundary-contracts already models explicit, revocable boundaries with private terms and public status summaries. This is the natural consent/scope primitive for workspace interventions where a human or steward explicitly permits a class of action.
- crates/mycelix-bridge-entry-types already provides schema-versioned BridgeEventEntry and BridgeQueryEntry for cross-cluster evidence/routing. A workspace decision can use the bridge event mechanism for durable lineage without turning an inference into policy.
- mycelix-attribution UsageReceipt is specifically a dependency-usage attribution primitive. It must not be repurposed as a physical-workspace authorization or decision receipt merely because it is called a receipt.
- Mycelix identity remains the actor-binding layer; governance remains the authorization/policy layer; Symthaea remains the inference/simulation layer.

Therefore the intended boundary is:

Symthaea evidence -> bridge/event projection -> identity-bound decision context -> explicit boundary/authorization -> physical actuation -> observed outcome

A projection must preserve whether each field is a model-derived prediction, scenario assumption, observation, residual, or unresolved contradiction. Unsupported dimensions remain unknown rather than being synthesized into a composite score.

### External digital-twin calibration

The broader digital-twin ecosystem already demonstrates mature patterns for telemetry, thermal/energy simulation, spatial models, time-series replay, and 3D operational views. The differentiating research question here is not whether a building can be visualized or simulated; it is whether physical simulation can remain explicitly separated from consent, authorization, provenance, contestability, and institutional learning. The architecture therefore treats those as separate trust domains rather than adding another dashboard layer.
