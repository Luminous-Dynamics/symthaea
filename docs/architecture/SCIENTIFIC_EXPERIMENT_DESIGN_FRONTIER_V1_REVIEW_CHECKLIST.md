# SCI-009 — Experiment Design Frontier v1 Review Checklist

Use this checklist to review the SCI-009 architecture contract only. It does not qualify any existing experiment planner.

## Scope and ownership

- [ ] SCI-009 owns generic experiment-comparison/planning mechanics, not universal scientific objectives.
- [ ] Domain science continues to own experiment semantics, measurement models, safety policy, ethics/consent, and action authorization.
- [ ] SCI-007 remains the falsifier-contract owner.
- [ ] SCI-008 remains the observation/uncertainty owner.
- [ ] SCI-006 remains the dependency-topology owner.
- [ ] SCI-004 remains the prospective/adaptive experiment-contract owner.

## Baseline preservation

- [ ] Query-by-committee disagreement remains a named baseline rather than being relabeled EIG.
- [ ] Coverage-aware disagreement remains distinct from EIG.
- [ ] Spark Engine mutual-information EIG remains scoped to its exact signature-level outcome model.
- [ ] Falsification-first design is supported without requiring Bayesian probabilities.
- [ ] Random/cost-only/simple baselines remain usable for planner comparison.

## Candidate identity

- [ ] Experiment candidate identity binds scientifically material target/intervention/setup/measurement/control semantics.
- [ ] Human-readable names are not identity roots.
- [ ] Material candidate revision creates new identity or explicit revision relation.
- [ ] Candidate identity remains distinct from execution identity and experiment-contract identity.

## Planning snapshot

- [ ] Planning decisions bind the exact information state available when the decision was made.
- [ ] Hypothesis/proposition/falsifier set is exact.
- [ ] Belief/prior/model state identity is exact when used.
- [ ] Evidence/dependency generation is retained where material.
- [ ] Resource/availability state is retained where material.
- [ ] Planner implementation/profile identity is retained.
- [ ] New evidence creates a new planning snapshot rather than mutating the old one.

## Prediction coverage

- [ ] Total hypothesis count is distinguishable from finite/evaluable prediction count.
- [ ] `None`, invalid, and non-finite predictions cannot silently enter numerical scoring.
- [ ] Missing predictions do not default to zero-valued predictions.
- [ ] Coverage is diagnostic/admissibility information, not itself scientific value.
- [ ] High disagreement under sparse coverage cannot hide that sparsity.

## Expected information gain naming

- [ ] `ExpectedInformationGain` requires an explicit probabilistic model.
- [ ] Prior/current hypothesis distribution is bound.
- [ ] Outcome/likelihood model is bound.
- [ ] Observation space/discretization/density representation is bound.
- [ ] Noise/uncertainty model is bound where material.
- [ ] Numerical approximation method is bound where material.
- [ ] Planner implementation and execution lineage are bound.
- [ ] Variance, entropy proxy, ensemble spread, uncertainty reduction heuristic, and disagreement retain honest names.

## Model misspecification

- [ ] Planner-model uncertainty can remain explicit.
- [ ] Prior sensitivity can be represented.
- [ ] Likelihood-model sensitivity can be represented.
- [ ] Discretization/model-family sensitivity can be represented.
- [ ] OOD applicability is not hidden.
- [ ] Hypothesis-set incompleteness can remain unresolved.
- [ ] Planner calibration is not assumed from mathematical form alone.

## Scientific-value coordinates

- [ ] Expected information gain can be one coordinate.
- [ ] Disagreement can be one coordinate.
- [ ] Falsification power can be one coordinate.
- [ ] Causal-model discrimination can be one coordinate.
- [ ] Defeater-resolution value can be one coordinate.
- [ ] Replication/triangulation value can be one coordinate.
- [ ] Measurement-quality improvement can be one coordinate.
- [ ] OOD-boundary information can be one coordinate.
- [ ] No coordinate is automatically a universal truth/confidence score.

## Cost / time / feasibility

- [ ] Scientific value remains distinct from cost.
- [ ] Scientific value remains distinct from duration.
- [ ] Feasibility remains distinct from desirability.
- [ ] Unknown cost/resource values do not become zero.
- [ ] Equipment/sample/personnel constraints can remain explicit.
- [ ] Destructive sample use can remain explicit.
- [ ] Scheduling/setup/changeover dependencies can remain explicit.

## Safety / rights / ethics

- [ ] Safety is not automatically modeled as a soft negative utility.
- [ ] Hard safety constraints can exclude candidates before ranking.
- [ ] Rights/consent/ethics/governance constraints can be hard admissibility gates.
- [ ] Caller-provided `safe=true` cannot grant execution eligibility.
- [ ] Planner recommendation cannot itself execute the experiment.
- [ ] Scientific desirability cannot override independent action authority.

## Pareto semantics

- [ ] Coordinate direction (maximize/minimize) is explicit.
- [ ] Unknown/unavailable values remain explicit.
- [ ] Dominance requires no-worse on all declared comparable coordinates and better on at least one.
- [ ] Frontier identity binds the exact comparison profile.
- [ ] Dominated candidates remain auditable.
- [ ] Inadmissible candidates remain separate from dominated candidates.
- [ ] Frontier size/cardinality is descriptive, not evidence strength.

## Scalar preference policies

- [ ] Scalarization is optional, not the generic default.
- [ ] Weight/threshold/normalization semantics are versioned and exact.
- [ ] Lexicographic policies are allowed and explicit.
- [ ] Budget-constrained policies are allowed and explicit.
- [ ] Minimax/regret policies are allowed and explicit.
- [ ] Changing preference semantics creates a new policy identity.
- [ ] Policy-specific utility is not stored as universal `experiment_quality`.

## Adaptive / sequential design

- [ ] Adaptive science remains compatible with SCI-004 preregistration.
- [ ] Candidate generation/action space is frozen where confirmatory authority requires it.
- [ ] Planner/update rule is frozen where material.
- [ ] Stopping rule is frozen prospectively where material.
- [ ] Actual observations enter through SCI-008/relevant qualification layers.
- [ ] Every replan creates a new immutable planning event.
- [ ] Simulated rollout observations are explicitly labeled simulated.
- [ ] Simulated outcomes cannot be laundered into observed evidence.

## Expected versus realized planning value

- [ ] Historical expected value remains immutable after execution.
- [ ] Realized information/entropy change can be compared later.
- [ ] Predicted vs realized cost/duration can be compared later.
- [ ] Expected vs realized applicability/measurement quality can be compared later.
- [ ] Bad realized outcome does not automatically prove ex-ante planning error.
- [ ] Good realized outcome does not automatically qualify planner calibration.

## Dependency-aware planning

- [ ] Shared instrument/calibration/model/data dependencies remain visible.
- [ ] Superficially different experiments do not automatically count as independent evidence opportunities.
- [ ] Replication/triangulation objectives can consume SCI-006 topology.
- [ ] Learned grammar/model dependencies can influence prospective cleanliness and independence assessment.

## Novelty

- [ ] Novelty is not a universal `novel: bool` or scientific-value shortcut.
- [ ] Novelty/search value retains exact corpus/cutoff/query/model/tool scope.
- [ ] Absence of prior match within a search scope is not universal novelty proof.

## First implementation slice

- [ ] First shared Rust slice is non-executing.
- [ ] First slice does not compute EIG.
- [ ] First slice does not compute safety.
- [ ] First slice does not authorize action.
- [ ] First slice focuses on candidate refs, planning snapshots, coordinate availability/direction, feasibility state, Pareto profile/frontier.
- [ ] Existing domain planners enter later through explicit adapters.

## Anti-authority checks

Reject the architecture if it permits any of these without a separately qualified domain policy:

- [ ] universal `best_experiment: bool`;
- [ ] universal `experiment_quality: f64`;
- [ ] planner output -> execution capability;
- [ ] EIG -> scientific truth;
- [ ] low cost -> safe;
- [ ] high disagreement -> calibrated uncertainty;
- [ ] no prediction -> zero prediction;
- [ ] unknown risk -> zero risk;
- [ ] simulated outcome -> observation;
- [ ] post-hoc utility weight change retaining preregistered authority;
- [ ] dropped failed/rejected/dominated candidates from audit history.

## Review exit question

Approve SCI-009 architecture only if the answer is yes:

Does the contract support intelligent, adaptive experiment selection while preserving enough model, uncertainty, coverage, falsification, dependency, feasibility, resource, safety, preference-policy, and planning-history structure that no single heuristic score can silently become universal scientific or action authority?