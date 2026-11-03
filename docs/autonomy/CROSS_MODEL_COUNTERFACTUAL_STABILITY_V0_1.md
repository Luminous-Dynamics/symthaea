# Cross-model counterfactual stability v0.1

**Status:** finite action-set model-sensitivity evaluator implemented; no model comparison results reported  
**Code:** `symthaea_vehicle::model_stability`  
**Research motivation:** 2026 world-model and prediction-to-planning surveys emphasize rollout error, model exploitation, horizon drift and action-dependent other-agent responses.

## Question

Does a selected action remain the best feasible action when the *same* finite action set is evaluated under multiple declared dynamics/model profiles? Or does the apparent best action reverse when friction, actuator, actor, sensor or other model assumptions change?

A planner can exploit a simulator/model error: predicted reward improves while actual outcomes deteriorate. A single-model counterfactual audit cannot reveal ranking instability across plausible model variants. This module supplies a narrow diagnostic for that failure mode.

## Input contract

A decision record contains:
- exact decision and scenario IDs;
- scenario, initial-state, environment and forecast-horizon digests;
- one metric ID, one metric unit ID and direction;
- selected action ID and tie tolerance;
- caller assertions that the action set and model set are complete;
- multiple declared model profiles, each identified by model artifact and profile digests;
- for every model/action pair: exact command payload and digest, feasibility, objective value, evaluator identity/artifact, execution context digest, run receipt ID and output digest.

Every model profile must evaluate exactly the same action IDs with exactly the same command payloads. The per-action context fingerprint binds decision/scenario, initial state, environment, horizon, metric definition, model artifact/profile and evaluator identity. Execution receipt IDs must be unique.

The structural identities do not prove that models are independent or physically valid. The caller must ensure all model profiles use compatible metric semantics, units, horizon, initial state, exogenous conditions and observation policies.

## Outcomes

- `SelectedActionBestInEveryModel`: selected action is feasible and best under every declared profile.
- `SameAlternativeBestInEveryModel`: the same different action is best under every profile.
- `BestActionChangesAcrossModels`: the best feasible action changes across model profiles.
- `SelectedActionInfeasibleInAtLeastOneModel`: the selected action violates declared feasibility under at least one profile.
- `NoFeasibleActionInAtLeastOneModel`: at least one model profile has no feasible candidate.
- `Indeterminate`: action/model set is declared incomplete or any feasibility remains unknown.

Stable action-ID tie-breaking makes each model's best-action selection independent of candidate insertion order. The report exposes per-model best action, per-model selected-action regret, best-action agreement rate, selected-action feasibility counts and worst-case selected-action regret (only where defined).

## Interpretation

The key diagnostic is rank reversal. If a candidate wins only in one model profile but loses in another, the decision is model-sensitive. The correct response is not automatically to average the outputs: preserve each profile, identify the physical assumption causing the reversal and investigate whether the assumption is supported by measurements.

The report does not:
- prove the model profiles span reality or are independent;
- establish a true global optimum;
- turn simulator-relative feasibility into physical safety;
- estimate event prevalence;
- replace model validation, uncertainty quantification or robust control.

The module requires a common objective ID and unit to compare actions across model profiles. Even with matching identifiers, the caller must check that objective implementations have consistent semantics.

## Benchmark protocol

Construct frozen model-profile sets with named, bounded discrepancies, such as:
- dry / nominal / low friction and asymmetric axle grip;
- nominal / delayed brake actuation and steering-rate perturbations;
- nominal / biased localization and stale timestamps;
- actor-script timing and speed variations;
- alternative but independently reviewed vehicle dynamics parameter sets.

Use the same scenario, initial state, road, actor script and candidate action payloads under all model profiles. Run each profile in its own declared execution context and obtain exact receipts through Symtropy.

Report:
- best-action agreement rate;
- fraction of models where selected action is infeasible;
- fraction of models with a better feasible alternative;
- worst-case selected-action regret;
- action ranking reversals;
- per-profile result and discrepancy assumption.

Do not average outcomes from incompatible model/metric scales. If profiles are not comparable, report them separately and classify the cross-profile result as unsupported.

## Controls

Ten unit tests are authored for:
- best-action reversal across model profiles;
- selected action remaining best across all profiles;
- incomplete action/model set indeterminacy;
- selected-action infeasibility in one model;
- unknown feasibility;
- mismatched action sets and command payloads;
- input-order-invariant canonical report;
- report tamper detection;
- duplicate model-profile rejection.

Tests have not been compiled or executed in this session.

## Integration boundary

Symtropy #3387 remains the execution/replay owner. Symthaea owns model-family design, action-set semantics, ranking interpretation and family-level statistics. Do not create a parallel vehicle physics engine. The first implementation can consume outputs from named versions of the existing Symthaea `BicycleModelSimulator` and later add independent/higher-fidelity model profiles where qualified.

References:
- https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0
- https://link.springer.com/article/10.1007/s44163-026-02122-1
- https://link.springer.com/article/10.1007/s10462-026-11604-8
