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

    fn solve(
        &self,
        game: &ValidatedGame,
        task: &AnalysisTask,
    ) -> StrategicResult;
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
