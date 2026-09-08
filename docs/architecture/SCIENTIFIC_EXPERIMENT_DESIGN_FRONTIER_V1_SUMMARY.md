# SCI-009 — Scientific Experiment Design Frontier v1 Summary

SCI-009 defines the common experiment-selection boundary for the Scientific Method Kernel.

It does not implement an optimizer and does not authorize experiments.

## Core distinction

The contract preserves:

- candidate experiment != executable experiment;
- feasibility != desirability;
- prediction coverage != information gain;
- disagreement != calibrated uncertainty;
- expected information gain != realized information gain;
- falsification power != posterior entropy reduction;
- cost != scientific value;
- risk constraint != ordinary utility penalty;
- planner recommendation != execution authority.

## Existing baselines remain valid and distinct

SCI-009 keeps three important planning families separate:

1. Conjecture Engine query-by-committee prediction variance;
2. #662 coverage-aware disagreement, if later qualified;
3. Spark Engine mutual-information EIG under its documented signature-level probabilistic model.

A falsification-first planner is also first-class and does not require Bayesian probabilities.

The shared kernel does not rename one mechanism into another.

## EIG naming discipline

A calculation may be called expected information gain only when it binds the relevant probabilistic objects, including the hypothesis/belief distribution, outcome/likelihood model, observation representation, uncertainty/noise model where material, approximation method, implementation identity, and execution lineage.

Variance, ensemble spread, entropy proxies, and heuristic uncertainty reduction retain their own names.

## Default multidimensional comparison

SCI-009 prefers a Pareto surface when multiple non-commensurate objectives exist and no qualified scalar preference policy has been declared.

Candidate coordinates may include:

- information gain;
- disagreement;
- falsification power;
- causal-model discrimination;
- defeater resolution;
- replication/triangulation value;
- measurement quality;
- cost;
- duration;
- resource burden;
- feasibility;
- risk/admissibility.

Unknown coordinates remain unknown.

Inadmissible candidates remain distinct from merely dominated candidates.

## Scalar policies are allowed, but scoped

Domains may explicitly use policies such as EIG-per-dollar, budget constraints, lexicographic priorities, weighted utility, or minimax/regret.

Those policies must be exact/versioned and their output is policy-specific—not a universal experiment-quality score.

## Safety boundary

Some safety, consent, rights, ethical, environmental, or governance constraints are hard admissibility boundaries rather than costs that information gain can buy its way through.

The intended chain is:

candidate -> admissibility/feasibility -> scientific comparison/frontier -> planner proposal -> separate authorization -> execution.

## Adaptive science

SCI-009 composes with SCI-004 adaptive preregistration.

After each real observation, the system creates a new planning snapshot and replans under the prospectively frozen update policy.

Simulated MAP-world rollouts remain explicitly simulated and never become observed evidence.

## Expected versus realized value

A later meta-scientific layer can compare predicted planning value with realized outcomes, including information gain, cost, duration, measurement quality, and applicability.

Historical planning decisions are not rewritten after outcomes are known.

## First implementation slice

The first shared Rust tranche should be neutral and non-executing:

- `ExperimentCandidateRefV1`;
- `ExperimentPlanningSnapshotRefV1`;
- `PlanningCoordinateV1`;
- `ExperimentFeasibilityStateV1`;
- `ParetoComparisonProfileV1`;
- `ExperimentDesignFrontierV1`.

It should consume externally produced coordinates rather than computing Bayesian EIG, safety, scientific validity, or action authority.

## Integration path

SCI-009 follows:

SCI-007 falsifier semantics -> SCI-008 uncertainty-bearing observations -> SCI-009 experiment design.

Later it can feed:

- SCI-010 symbolic-discovery tournament;
- SCI-012 Genesis causal interventions;
- SCI-013 falsification campaigns;
- SCI-014 Theory Atlas discriminating-experiment queues.

## Non-claims

SCI-009 does not establish that any current planner is calibrated, optimal, safe, ethically admissible, cost accurate, causally identified, or scientifically superior.

It does not choose universal utility weights or grant execution authority.