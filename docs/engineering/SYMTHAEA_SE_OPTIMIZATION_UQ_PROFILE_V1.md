# Symthaea Systems Engineering Optimization & UQ Profile v1

Status: architecture profile only. No optimizer result, calibrated model, sensitivity ranking, reliability estimate, engineering evidence, qualification, certification, or physical authority is established by this document.

Tracking: #3677, #3681, #3686

## 1. Purpose

Define a reproducible, evidence-bounded optimization and uncertainty layer for Symthaea systems engineering.

The goal is to let Symthaea explore multidisciplinary design spaces, uncertainty, sensitivity, calibration, trajectories, and discrete planning while preserving exact model/configuration lineage and never confusing numerical optimization with engineering qualification.

Canonical laws:

```text
optimum
!= validated design

Pareto-efficient
!= safe

numerically feasible
!= requirement satisfied

posterior/calibrated model
!= physical truth

sensitivity ranking
!= causal fact
```

## 2. Separation of concerns

Optimization/UQ should not become a monolithic solver layer.

Use specialized engines through one common engineering contract:

- OpenMDAO for multidisciplinary coupling, design analysis, derivatives, and optimization;
- Dakota for uncertainty quantification, sampling, sensitivity, reliability, calibration, and selected optimization;
- CasADi for nonlinear dynamic/control/trajectory optimization;
- Pyomo later for algebraic, mixed-integer, scheduling, allocation, and operations-research problems;
- Dymos later for OpenMDAO-aligned trajectory optimization.

Symthaea remains responsible for engineering semantics, provenance, reasoning, proposal generation, and interpretation — not for replacing mature optimization engines.

## 3. Core semantic types

Future implementation should distinguish at least:

```text
DesignVariable
DesignSpace
Objective
ConstraintIntent
DesignPointRef
EvaluationRef
TradeStudy
OptimizationRun
ParetoRelation
UncertaintyCampaign
DistributionSpec
CorrelationSpec
SensitivityResult
CalibrationResult
ReliabilityResult
TrajectoryStudy
```

No optimizer-generated type should contain authority states such as:

```text
Approved
Safe
Qualified
Verified
Certified
```

## 4. Design-variable semantics

Every design variable should bind:

- stable semantic ID;
- exact SE node/configuration target;
- units;
- domain/type;
- lower/upper bounds or discrete alternatives;
- baseline value;
- transform/scaling where applicable;
- source/rationale;
- validity conditions;
- whether it is controllable, uncertain, observed, or derived.

Changing a design variable creates or references a new exact engineering configuration. It must not mutate the identity of an already-evaluated configuration.

## 5. Objectives and constraints

An optimizer's objective and numerical constraints are not automatically identical to stakeholder requirements.

Freeze:

```text
Requirement
!= optimizer ConstraintIntent
```

A `ConstraintIntent` should retain an explicit link to the source requirement/assumption where one exists, together with the projection method and any approximation/lossiness.

Similarly:

```text
objective function
!= value judgment
!= engineering approval policy
```

Multi-objective studies should preserve trade-offs rather than collapsing them into one hidden scalar unless an explicit, versioned scalarization policy is part of the study.

## 6. Common evaluation artifact

Every evaluated design point should bind:

- exact SE graph/model/configuration snapshot;
- exact design-variable vector + units;
- objective/constraint schema revision;
- exact analysis tool/adapter versions;
- environment/execution identity;
- raw inputs and outputs;
- normalized responses;
- convergence/failure status;
- uncertainty metadata;
- derivative provenance where applicable;
- wall-clock/resource policy;
- parent trade-study/run identity.

Failed, infeasible, nonconverged, or interrupted evaluations are first-class artifacts and must not silently disappear from the study history.

## 7. OpenMDAO role

OpenMDAO is the preferred first multidisciplinary analysis/optimization orchestrator.

Use it for:

- coupled multidisciplinary systems;
- design-variable/objective/constraint declaration;
- derivative-aware analysis;
- gradient/non-gradient optimization;
- DOE/trade-space studies;
- case recording;
- large engineering coupling graphs.

Every OpenMDAO case must map back to an exact engineering configuration.

Canonical laws:

```text
OpenMDAO driver success
!= engineering success

optimizer termination success
!= global optimum
!= requirement satisfaction
```

## 8. Dakota role

Dakota is the preferred first UQ/calibration/sensitivity/reliability campaign engine.

Use it for:

- epistemic/aleatoric propagation;
- sampling;
- global/local sensitivity;
- calibration/parameter estimation;
- reliability/margins-and-uncertainty studies;
- multifidelity sampling;
- selected optimization workflows.

Dakota 6.24 provides a structured JSON input path, but that input route is currently experimental and must be version-gated and qualified rather than treated as a stable interchange standard.

Every Dakota campaign should preserve:

- exact distributions;
- correlations/dependencies;
- sampling method;
- seeds;
- sample counts/stopping policy;
- model and response definitions;
- restart/checkpoint artifacts when used;
- parser/input mode;
- raw results and intermediate representations where useful.

## 9. Uncertainty semantics

Do not reduce engineering uncertainty to one confidence score.

Support at least:

```text
Interval
Distribution
EpistemicBound
AleatoricVariability
Correlation/Dependency
SensitivityIndex
ConfidenceInterval
CredibleInterval
UnknownOrUnmodeled
```

Every uncertainty representation needs provenance and method identity.

Where epistemic and aleatoric uncertainty cannot be cleanly separated, preserve the ambiguity rather than inventing precision.

## 10. Calibration vs validation

Freeze:

```text
calibrated against observations
!= validated for intended use
```

A calibration result should identify:

- observations/data used;
- data configuration/source identity;
- parameters estimated;
- priors/bounds;
- likelihood/error model where applicable;
- calibration method;
- posterior/estimate;
- residuals/fit diagnostics;
- intended validity envelope.

Using the same data for calibration and validation must be machine-visible.

## 11. Sensitivity vs causality

Freeze:

```text
sensitive parameter
!= causal factor
```

Sensitivity results describe dependence of model outputs on model inputs under a study definition. They do not by themselves establish real-world causality.

Where causal reasoning is desired, hand the sensitivity result to the causal-analysis layer as candidate information rather than promoting it to a causal edge.

## 12. CasADi role

Use CasADi where symbolic/algorithmic differentiation and nonlinear dynamic optimization are valuable, particularly:

- control-system optimization;
- model predictive control studies;
- robotics;
- dynamic-system parameterization;
- trajectory optimization;
- constrained nonlinear programs.

Preserve discretization/transcription, NLP solver identity, tolerances, KKT/residual information, initial guesses, and local-solution status.

Freeze:

```text
local optimum
!= globally optimal design
!= robust controller
```

## 13. Pyomo role

Add later for problems dominated by discrete/algebraic operations-research semantics:

- scheduling;
- resource allocation;
- network design;
- unit commitment/planning;
- mixed-integer decisions;
- logistics;
- portfolio/configuration selection.

Preserve incumbent, objective bound, optimality gap, timeout, solver identity, and termination reason.

Freeze:

```text
timeout incumbent
!= proven optimum
```

## 14. Dymos role

Dymos may be added after the generic OpenMDAO contract qualifies for trajectory-specific work.

It should inherit the same exact model/configuration, derivative, solver, and execution provenance contracts rather than establishing a parallel trajectory authority model.

## 15. SE graph integration

The optimization/UQ layer consumes exact engineering semantics.

Possible mappings:

```text
Component parameter -> DesignVariable
Requirement-derived numerical intent -> ConstraintIntent
ResourceBudget -> constraint/response
Configuration -> DesignPointRef
AnalysisArtifact -> response/evaluation input
OperationalObservation -> calibration input
Change -> stale-study applicability review
```

Do not let an optimizer directly mutate accepted SE graph semantics.

A candidate point may generate a proposed configuration change, which follows the normal proposal/review path.

## 16. HDC/LTC and cognitive coupling

Only after deterministic optimization/UQ contracts qualify, cognition may assist with search.

HDC/LTC may:

- suggest promising design-space neighborhoods;
- cluster failure/infeasible regions;
- propose variable couplings;
- recall structurally similar studies;
- propose surrogate/model structure;
- identify temporally evolving trade spaces.

They may not:

- delete constraints silently;
- redefine accepted objectives silently;
- hide failed/infeasible points;
- claim global optimality;
- promote a candidate design to qualified status.

## 17. Candidate evidence boundary

Optimization/UQ outputs are computation artifacts.

Some may later become candidate evidence when bound to an accepted verification/assurance plan and exact model/configuration.

Canonical path:

```text
trade study / UQ / calibration result
-> bounded computation artifact
-> possible CandidateEvidence
-> ETK admission/currentness
-> possible claim support
```

Never:

```text
optimizer selected design
-> requirement satisfied
```

## 18. Proposed PR sequence

```text
SE-OPT-000  optimization/UQ profile (this document)
SE-OPT-001  common design-space/evaluation artifact protocol
SE-OPT-002  OpenMDAO read/execute adapter
SE-OPT-003  Dakota 6.24 campaign adapter
SE-OPT-004  uncertainty representation/propagation contract
SE-OPT-005  CasADi control/trajectory adapter
SE-OPT-006  Pyomo operations-research adapter
SE-OPT-007  HDC/LTC proposal coupling
SE-OPT-008  SE Gym optimization/UQ corpus
```

Implementation depending on SE graph/model identities remains blocked until required lower SE contracts qualify.

## 19. Qualification requirements

Every adapter/campaign path should demonstrate:

```text
exact tool identity
+ exact model/configuration identity
+ exact variable/objective/constraint identity
+ exact options/seeds
+ bounded execution
+ preservation of failures/infeasible points
+ raw artifact identity
+ normalized result reproducibility
+ uncertainty-method identity
+ immutable postflight
```

For derivative-based methods, include independent derivative/finite-difference checks where practical.

For stochastic methods, qualify statistical/calibration behavior rather than demanding byte-identical result sequences when nondeterminism is inherent.

## 20. SE Gym benchmark families

Initial frozen benchmark families should include:

1. thermal-structural sizing;
2. battery mass/endurance/thermal trade;
3. controller tuning with robustness constraints;
4. circuit power/noise/cost trade;
5. building energy/cost/resilience trade;
6. water-network energy/reliability trade;
7. deliberately multimodal/nonconvex problem;
8. deliberately infeasible problem;
9. uncertain-model calibration problem;
10. stale optimum after a model/configuration change;
11. mixed discrete/continuous planning problem;
12. failed solver/nonconvergence preservation test.

Measure separately:

- objective reproduction;
- constraint reproduction;
- feasibility correctness;
- derivative agreement;
- design-point lineage accuracy;
- uncertainty calibration;
- sensitivity reproducibility;
- failure preservation;
- stale-result detection;
- false-authority rate.

## 21. Non-goals

- no universal optimizer;
- no hidden scalar engineering-quality score;
- no optimizer-output-to-requirement shortcut;
- no posterior-to-truth shortcut;
- no sensitivity-to-causality shortcut;
- no HDC-similarity-to-constraint deletion;
- no optimization-result-to-fabrication/deployment authority;
- no second ETK.

## 22. Closure criterion

The optimization/UQ program succeeds when Symthaea can define and execute a reproducible multidisciplinary trade study over exact engineering model/configuration snapshots, preserve every evaluation and uncertainty assumption, explain why candidate points dominate or violate constraints, detect when a model change makes old results stale, and hand bounded computation/candidate-evidence artifacts into SE/ETK while keeping distinct:

```text
what was optimized
what was assumed
what was evaluated
what failed
what uncertainty remains
what configuration a result belongs to
what evidence it may support
what authority exists
```
