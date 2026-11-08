# Bounded counterfactual decision-quality evaluation v0.1

**Status:** finite action-set evaluator implemented; no vehicle decision-quality results reported  
**Code:** `symthaea_vehicle::counterfactual_decision`  
**Measurement reference:** NIST Measurement Science for Automated Vehicles (2026)

## Question

A collision outcome from one chosen trajectory does not tell us whether the controller could have selected a better feasible action. Conversely, a collision-free run does not show that the selected action was optimal or even robust.

This module answers one deliberately bounded question:

> Within this declared finite set of action candidates, under this named evaluator and one declared scalar metric, was a better feasible action available than the selected action?

This is a local/model-relative counterfactual comparison, not a proof of global optimality or physical safety.

## Input contract

A `CounterfactualDecisionInput` includes:
- decision, scenario and failure-family IDs;
- scenario artifact digest, initial-state digest, environment profile digest, dynamics profile digest and execution-protocol digest;
- frozen controller artifact digest;
- separate evaluator artifact digest and evaluator ID;
- non-empty external independence-evidence reference;
- one named metric ID, direction and tie tolerance;
- selected action ID;
- caller assertion that the candidate set is complete;
- candidate actions with exact steering/throttle/brake command, action digest, model-relative feasibility, objective value, evaluator ID and execution receipt/digest.

All candidate receipts must use the same evaluator ID. Controller and evaluator artifact digests must be distinct. Every action digest is recomputed from the exact command payload. Every action also carries a recomputable execution-context digest over the shared scenario, initial state, environment, dynamics, protocol, evaluator artifact/build and metric definition. The validator rejects any action whose context digest differs. Execution receipt IDs must be unique and execution digests must be 64-hex strings.

These are structural checks, not cryptographic authentication. The independence evidence reference is not dereferenced or verified. Distinct artifact digests do not prove architectural or common-cause independence.

## Outcomes

- `NoBetterFeasibleActionFound`: selected action is feasible and no better feasible candidate was found within the declared set and tie tolerance.
- `BetterFeasibleActionFound`: a different feasible action has a better value under the named objective; positive regret is reported.
- `SelectedActionInfeasible`: the selected command is marked infeasible under the declared evaluator/model.
- `NoFeasibleAction`: no candidate in the declared set was feasible.
- `Indeterminate`: the set is declared incomplete or one or more candidate feasibility results are unknown.

The module does not turn incomplete search into an optimum claim. It also refuses to rank alternatives if the evaluator identity changes mid-set.

## Objective semantics

Only one scalar objective is evaluated per report. This avoids hidden weighted composites. Examples of separate objective runs include:
- minimum peak lateral acceleration;
- maximum predicted clearance margin;
- minimum predicted stopping distance beyond a stop boundary;
- minimum route-progress delay;
- minimum command effort proxy.

Different objectives may disagree. Retain separate reports and Pareto comparisons; do not collapse safety, legality, progress, comfort and energy into one opaque reward.

## Integrity / reproducibility

The report retains the entire input, all candidate command payloads and action/evaluation receipts. Its validator recomputes the finding and checks the BLAKE3 output digest. The digest is an integrity fingerprint, not a signature or a verified simulator attestation.

The record binds each action to the same declared decision-time state, scenario, environment, dynamics, execution protocol and evaluator build via the shared context digest. The caller must still ensure that the execution backend actually used those values, that sensor/map information policies matched, and that horizon/termination criteria are comparable. Structural digests do not authenticate the backend or prove these experimental conditions.

## Relation to NIST's four-tier measurement programme

NIST's 2026 Measurement Science for Automated Vehicles programme describes:
1. real-time surrogate safety metrics (e.g. TTC, PET, DRAC);
2. forward simulation to predict likely outcomes;
3. comparison of the selected action with counterfactual alternatives;
4. diagnosis of repeated decision weaknesses across scenarios.

This module is a narrow data/evaluation contract for the Tier-3 comparison. It does not implement Tier 1 metrics, forward simulation, exhaustive action search or Tier-4 diagnosis. Symtropy remains the execution/replay owner; Symthaea owns the action-set design, criterion, statistics and interpretation.

Reference: https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0

## Controls

Thirteen unit tests are authored for:
- better feasible action and positive regret;
- no better action / zero regret;
- correct direction for higher-is-better metrics;
- incomplete action-set indeterminacy;
- unknown candidate-feasibility indeterminacy;
- infeasible selected action;
- no feasible action;
- duplicate IDs, mixed evaluators and controller/evaluator digest aliasing;
- malformed shared context digests and mismatched per-action context receipts;
- deterministic tie-breaking independent of candidate insertion order;
- action payload mutation rejection;
- report tamper rejection even after digest recomputation.

Tests have not been compiled or executed in this session.

## Claim ceiling

A result means only that a better feasible action was or was not found inside the declared candidate set, under the named objective and evaluator model. It does not prove that the set was exhaustive, the model was physically correct, the chosen action was safe in the real world or that the controller is globally optimal.
