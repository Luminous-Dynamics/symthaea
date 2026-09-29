# Strategic IR Consolidation

**Status:** Architecture proposal / migration contract  
**Scope:** Symthaea strategic reasoning, game-theory, UCL semantic frames  
**Audited:** 2026-09-29

## Purpose

Symthaea currently contains multiple useful but overlapping strategic implementations:

- `crates/domains/symthaea-game-theory`: focused two-player normal-form analysis.
- `crates/core/symthaea-core/src/hdc/game_theory.rs`: broader normal-form/game-theory algorithms embedded in the HDC core.
- `crates/domains/symthaea-economics/src/game.rs`: auditable 2×2 economic/game primitives.
- `crates/core/symthaea-core/src/hdc/ucl_cross_domain_frames.rs`: semantic frames for TRADE, CONFLICT, FEEDBACK_LOOP, NORM_ENFORCEMENT, COOPERATION, and ADAPTATION.

The repository should not add another independent game-theory implementation. Instead, these capabilities should converge behind a canonical Strategic Intermediate Representation (Strategic IR), with compatibility adapters preserving existing public APIs during migration.

## Architectural invariant

> Semantic interpretation determines the model; strategic solvers analyze the model; neither semantic confidence nor solver confidence grants authority to act.

The dependency direction is:

```
HDC / UCL
   |
   v
Semantic Interpretation
   |
   v
Strategic IR
   |
   +--> Normal-form / extensive-form / repeated / stochastic models
   |
   +--> temporal strategic properties
   |
   +--> causal / counterfactual analysis
   |
   v
Solver adapters
   |
   v
StrategicResult + evidence + provenance
   |
   v
Integral evaluation
   |
   v
Authority / execution boundary
```

## Canonical vocabulary

The first Strategic IR should define stable semantic contracts rather than solver-specific data structures:

```text
StrategicProblem
ValidatedGame
StrategicState
Player / Agent
Action
ActionAvailability
JointAction
Policy
Strategy
StrategyProfile
PreferenceModel
InformationModel
Observation
BeliefState
TransitionModel
Horizon
StrategicResult
AnalysisTask
AnalysisMethod
Completeness
UncertaintyBudget
VerificationEvidence
```

The distinction between Action, Policy, Strategy, and StrategyProfile is intentional:

- Action: one move at one decision point.
- Policy: mapping from information/context to actions.
- Strategy: an agent's complete contingent behavior under a game model.
- StrategyProfile: one strategy for every relevant agent.

## Solver boundary

Solvers implement a common capability contract:

```rust
pub trait StrategicSolver {
    fn capabilities(&self) -> SolverCapabilities;
    fn method(&self) -> AnalysisMethod;
    fn supported_tasks(&self) -> &'static [AnalysisTask];

    fn solve(
        &self,
        game: &ValidatedGame,
        task: AnalysisTask,
    ) -> Result<StrategicResult, AnalysisError>;
}
```

The canonical layer must not encode assumptions of a particular solver.

Initial adapters should cover:

- pure-strategy Nash;
- best-response analysis;
- dominance;
- 2×2 mixed Nash;
- zero-sum/minimax;
- existing fictitious-play implementation;
- later CFR/MCCFR and temporal model checking.

An algorithm must be named according to what it actually computes. In particular, an implementation using fictitious play must not expose an API named as an LP solver.

## Result evidence

A strategic result is not just an equilibrium:

```rust
pub struct StrategicResult {
    pub result: ResultValue,
    pub assumptions: Vec<Assumption>,
    pub observations: Vec<ObservationRef>,
    pub beliefs: Vec<BeliefRef>,
    pub method: AnalysisMethod,
    pub solver_version: String,
    pub completeness: Completeness,
    pub uncertainty: UncertaintyBudget,
    pub verification: Vec<VerificationEvidence>,
    pub provenance: ProvenanceGraph,
}
```

Where useful, solver quality should preserve separate quantities:

- equilibrium residual;
- regret;
- exploitability;
- convergence status;
- sampling error;
- approximation bounds;
- explored/search coverage.

These must not be collapsed into a single confidence number.

## Exact versus approximate computation

Canonical finite-game fixtures should prefer exact representations where practical (integer/rational payoffs). Floating-point and stochastic algorithms must carry explicit approximation/reproducibility metadata.

At minimum:

```text
Reproducibility
  Deterministic
  SeededDeterministic
  Stochastic
  ExternalDependency
  NonReproducible
```

and:

```text
Completeness
  Exhaustive
  ProvenComplete
  BoundedSearch
  Sampled
  Heuristic
  Unknown
```

## UCL compilation boundary

UCL frames are semantic representations, not strategic solvers.

The first semantic-to-strategic compilers should be:

```text
TRADE        -> exchange / bargaining problem
COOPERATION  -> coordination / public-good problem
CONFLICT     -> general-sum strategic interaction
```

The compiler must be allowed to return an under-specified model. Missing preferences, information, legal constraints, or action availability must become explicit model uncertainty rather than fabricated defaults.

This gives the pipeline:

```
UCL frame
  -> semantic bindings
  -> model hypothesis
  -> StrategicProblem
  -> validation
  -> solver
```

HDC similarity must never be treated as an equilibrium computation.

## Temporal strategic reasoning

Equilibrium is only one analysis task. The IR should reserve a first-class property layer for:

```text
Equilibrium
Reachability
Safety
Liveness
Avoidance
CoalitionAbility
```

And a task layer:

```text
EvaluateStrategy
SynthesizeStrategy
FindEquilibrium
PredictTrajectory
VerifyProperty
FindCounterexample
```

This makes strategic ability distinct from equilibrium and from prediction.

## Verification evidence

Results should be able to carry machine-checkable evidence:

```text
EquilibriumCertificate
ReachabilityWitness
SafetyCertificate
CounterexampleTrace
InvariantCertificate
ApproximationBound
```

A counterexample trace is especially important for model criticism because it converts a failed property into an inspectable trajectory rather than a scalar failure.

## Uncertainty separation

The IR should distinguish:

```text
observation uncertainty
semantic uncertainty
preference uncertainty
belief uncertainty
parameter uncertainty
model uncertainty
computational uncertainty
```

This is intentionally compositional. A solver can be mathematically exact for a model while the model itself remains uncertain.

## Migration plan

### Phase A — Freeze semantics

1. Define Strategic IR types and invariants.
2. Add validation for finite normal-form games.
3. Add canonical equilibrium certificates.
4. Add property/metamorphic tests.
5. Do not remove existing APIs.

### Phase B — Adapters

1. Adapt `symthaea-game-theory` to Strategic IR.
2. Adapt `symthaea-economics::game`.
3. Adapt core normal-form implementation.
4. Mark duplicated algorithms as compatibility paths.
5. Correct misleading algorithm names/documentation.

### Phase C — Extraction

Move mathematical reasoning out of HDC-specific modules while keeping HDC/UCL as semantic producers.

Target dependency:

```
HDC/UCL -> Strategic IR -> Strategic mathematics
```

not:

```
Strategic mathematics -> HDC implementation details
```

### Phase D — Advanced solvers

Add solver plugins only after the IR is stable:

- CFR/MCCFR;
- regret minimization;
- replicator dynamics;
- stochastic games;
- Bayesian games;
- coalition reasoning;
- temporal strategic model checking.

External Rust solvers may be wrapped through adapters, but their representations must not become Symthaea's canonical ontology.

### Phase E — Experiment and DKG integration

Every nontrivial analysis should be representable as an auditable experiment:

```text
GameFingerprint
ParameterSet
SolverMetadata
Seed
Hypothesis
Runs
StrategicResult
AnalysisReceipt
TrajectoryFingerprint
```

The DKG should store reproducible experiments and evidence, not private runtime beliefs.

## Validation status correction

The six UCL cross-domain frames are implemented in the current tree and have an integration test file. However, that integration test is currently listed in `scripts/orphan-tests-quarantine.txt`.

Therefore the accurate capability state is:

```text
specification      present
implementation     present
unit validation    present
integration tests  present but quarantined
property testing   not yet established
production use     not established by this audit
```

Roadmap documents should use this evidence-oriented status instead of describing the frames as simply missing.

## Acceptance criteria for the consolidation

The consolidation is complete only when:

1. One canonical Strategic IR exists.
2. Existing public game-theory APIs can be represented without semantic loss.
3. All solver outputs identify their method and completeness.
4. Approximate methods expose approximation/reproducibility metadata.
5. Equilibrium and strategic ability are distinct analysis tasks.
6. UCL-to-Strategic translation is explicit and provenance-bearing.
7. HDC similarity is never used as mathematical equilibrium evidence.
8. Strategic analysis cannot cross the authority boundary directly.
9. At least one property-based/metamorphic validation suite protects solver invariants.
10. Stale roadmap claims are replaced by capability evidence.

## Non-goals

This pass does not:

- choose a single equilibrium concept for all domains;
- make strategic analysis authoritative;
- equate simulation with empirical evidence;
- introduce a single scalar strategic confidence;
- force all games into normal form;
- immediately depend on an external CFR library;
- remove existing APIs before adapter coverage exists.

The purpose is to establish the reasoning boundary first, then increase mathematical and computational sophistication without creating another parallel architecture.


## 2026-09-29 ecosystem review: information sets and solver evidence

A current Rust ecosystem review reinforces two additional design constraints.

First, modern CFR implementations expose the game as a state machine with explicit player/chance/terminal nodes and information sets, while keeping the solving algorithm behind that interface. The current `cfr` crate reports regret information and regret bounds, and validates structural properties such as consistent information-set action sets and perfect recall before solving. citeturn0search0turn0search8

Second, the current `mccfr` ecosystem explicitly separates public information, private information, turns, sampling, regret updates, and averaged strategy queries. This is a strong signal that Symthaea should not model imperfect-information strategy as a function of omniscient world state. citeturn0search1turn0search4

Accordingly, the next IR hardening tranche should make these invariants explicit:

- an agent receives an `AgentContext`, not arbitrary `WorldState`;
- every information set has a well-defined legal-action set;
- strategies are complete contingent plans over information sets, while policies are decision procedures;
- solver validation rejects malformed information structures before mathematical analysis;
- perfect-recall requirements belong to the capability contract of solvers that require them, rather than becoming a universal assumption of the IR;
- regret, exploitability, convergence, and approximation bounds remain separate evidence fields.

A further ecosystem pattern is worth preserving: safe/depth-limited subgame solving composes belief/world partitioning and local re-solving as orthogonal layers rather than conflating them. That supports treating belief updates, model restriction, and solver choice as composable stages in Strategic IR rather than embedding them into one solver-specific type. citeturn0search3turn0search7

These observations strengthen the existing architectural decision not to make an external CFR crate the canonical ontology. External solvers can be adapters once the IR can faithfully express their required game-state, action, information-set, and evidence contracts.


## 2026-09-29 information-structure hardening

The next implementation tranche now makes the information-set boundary executable rather than documentary.

The canonical domain module adds:

- DecisionState with explicit player, information-set identity, and legal actions;
- InformationSet with explicit member decision states;
- InformationStructure::validate() for structural integrity;
- rejection of duplicate state IDs and information-set IDs;
- rejection of empty or duplicate information-set membership;
- rejection of unknown or multiply assigned decision states;
- rejection when a decision state's declared information set disagrees with its containing set;
- rejection when members of one information set belong to different players;
- rejection when members expose different legal-action vocabularies;
- solver compatibility checks for imperfect-information support;
- explicit PerfectRecallEvidence gating for solvers that require perfect recall.

This mirrors the current CFR ecosystem: game builders validate inconsistent information-set action sets and perfect-recall requirements before solving, while newer MCCFR abstractions treat public/private information and perfect-recall properties as explicit solver-facing contracts. citeturn0search0turn0search3turn0search4

Importantly, the Strategic IR does not declare perfect recall globally. A model may be representable without satisfying a particular solver's recall requirements; the solver adapter must state its own requirements and the model must provide corresponding evidence.

The implementation also avoids assuming that a numeric state ID is a vector index. State identity is validated explicitly, and references are resolved by identifier. This prevents sparse or reordered identifiers from becoming accidental memory-layout contracts.

Current branch implementation status:

```text
agent-local policy contract       present
contingent strategy validation    present
information-set validation        present
solver capability declaration     present
perfect-recall gating             present
CI verification                   not yet observed
```

This is still a migration-stage contract. It does not yet prove perfect recall, construct extensive-form transitions, or provide a CFR solver; those remain separate model/solver responsibilities.

### 2026-09-29 derived perfect-recall evidence

The extensive-form boundary now derives `PerfectRecallEvidence::Verified` from the
reachable state graph rather than accepting a caller-supplied boolean as proof.

Validation first requires:
- every information-structure decision state to have a corresponding decision node;
- every decision node to have a corresponding information-structure entry;
- every information-set member to be reachable and to be a decision node;
- the reachable graph to be acyclic for this migration-stage finite-tree model.

The verifier then replays every reachable history and compares, for each information
set, the acting player's prior `(information_set, action)` sequence across all member
states. Opponent and chance transitions are intentionally omitted from that recalled
sequence. A solver that requires perfect recall can therefore consume evidence derived
from the game model instead of trusting an unattested flag.

This is deliberately stronger than a solver capability declaration but narrower than a
universal Strategic IR invariant: games with repeated states, transpositions, or
history-dependent information encoders need an explicit history model before they can
be admitted without this finite-tree restriction. This keeps the current boundary
sound while leaving room for a later `StateId`/history abstraction.

External CFR work reinforces this direction: current Rust CFR tooling validates malformed
trees and perfect-recall violations during game construction, while MCCFR separates
`EmbeddedHistory` and `PerfectRecall` as explicit attestations rather than treating
recall as an incidental solver detail.

### 2026-09-29 action availability and recall boundary refinement

Action ordering is not semantic: extensive-form validation compares decision-node
transitions and information-set legal actions as sets, while still rejecting
duplicates. This prevents serialization order from changing the strategic model.

The current finite-tree recall verifier proves consistency of a player's prior
information-set/action sequence. It intentionally does not yet claim that every
piece of information previously observed is represented in the information-set
identity. That stronger property requires an explicit observation/history encoder.

This matches current MCCFR practice: modern implementations separate the game state,
public/private information, and an encoder that maps states into information-set
identifiers. Perfect recall is then an attestation about replay consistency rather
than merely a property inferred from an integer information-set ID.

Accordingly, the next IR boundary should introduce a typed history/observation
encoder before adding CFR/MCCFR adapters. Until then, `verify_perfect_recall`
should be understood as a strong finite-tree structural check, not a complete
semantic observation proof.


### 2026-09-29 semantic information encoder boundary

The extensive-form layer now treats the declared information-set partition and the semantic mechanism that produces that partition as distinct contracts. A typed `HistoryEvent` stream and `InformationEncoder` boundary are the intended seam for public/private observation and recalled-history semantics. Structural `PerfectRecallEvidence` remains valid as finite-tree action/infoset evidence, while semantic observation consistency is verified separately against an encoder. The migration-stage state graph remains a true tree; shared children/transpositions are not admitted until history is first-class.
