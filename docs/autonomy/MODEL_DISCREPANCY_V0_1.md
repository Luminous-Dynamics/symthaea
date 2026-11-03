# SAM Horizon-Conditioned Model Discrepancy Protocol v0.1

**Protocol ID:** `SAM-MODEL-RESIDUAL-001`  
**Status:** research evaluator contract; source implementation not yet compiled or run  
**Owner:** Symthaea  
**Execution owner:** Symtropy #3387 and interactive actor work #3471  
**Related:** Symthaea #10252, #10256, #10266, #10083, #10043 and #6283; Symtropy #1173 and #3385

## Purpose

Measure whether world-model error changes with rollout horizon and decision context, and whether the planner selects actions whose predicted utility is systematically more attractive than their realized utility under a declared reference execution.

A model with low one-step mean error can diverge at longer horizons, fail within a particular actor/failure family, or contain an optimistic bias that an optimizer exploits. Point-forecast error and planner decision quality are different measurands and must be reported separately.

The first implementation is an offline Rust evaluator in `symthaea_vehicle::model_discrepancy`. It consumes caller-supplied prediction/reference rows and decision/action outcomes. It does not implement a predictor, simulator, reference solver, geometric uncertainty set, or physical safety monitor.

## Input contract

### Model and evaluator identity

Each report binds:
- predictor model ID and artifact digest;
- reference execution model ID and artifact digest;
- outcome evaluator ID and artifact digest;
- planner ID and artifact digest;
- sampling-profile digest;
- protocol ID and digest.

A distinct model ID or artifact digest does not prove model independence. Reference suitability and common-mode dependencies are tracked separately by #10266 and Symtropy #1173.

### Residual row

Each residual is identified by scenario, failure family, actor policy, split, shift regime, initial state, environment, dynamics, action history, observation-availability history, forecast issuance time, horizon, metric and unit.

The evaluator uses:
` signed_residual = predicted_value - realized_value `

Positive residual means the predictor's numeric value exceeds the matched reference value; whether that is optimistic or pessimistic depends on the metric's semantics. Always retain the metric ID, unit and declared favorable direction in the experiment registry.

A row key is scenario × actor-policy × action-history × metric × horizon. Duplicate keys are rejected. Rows for a paired horizon-growth claim must share the same scenario, actor policy, action history, forecast issuance and full context signature. Context mismatch is rejected. No absent horizon is interpolated or imputed.

The caller also supplies a frozen `ExpectedResidualSlot` schedule independently of reference outcomes. Every expected slot must have exactly one typed outcome row. The evaluator rejects duplicate expected slots, outcome rows not in the schedule, and expected slots with no outcome row. The schedule binds scenario, family, actor policy, split/shift, initial state, environment, dynamics, action history, observation availability, forecast issuance, horizon, metric and unit. Its source and digest must be frozen before outcomes are revealed; the evaluator checks exact slot equality but cannot authenticate the caller's sampling design or prove schedule completeness.

### Decision/action row

Each decision binds the same initial state, environment, dynamics, actor policy, action history, observation availability, forecast time, utility, selection receipt, candidate-set digest and selection/reveal sequence.

Each candidate binds its action ID, command digest, feasibility, predicted utility, typed reference-outcome status, optional realized utility, and optional execution/evaluator receipts.

Reference-outcome status is one of Observed, Missing, Censored, TimedOut or Unsupported. An Observed feasible action requires a finite realized utility and execution/evaluator receipts. An Observed infeasible or indeterminate action may retain its feasibility evidence without inventing a utility. A non-observed outcome must have no realized utility or fabricated receipt and must remain feasibility-indeterminate. If any candidate reference outcome is non-observed, the decision becomes IncompleteReference and no ranking reversal or regret is emitted.

For residual rows, Observed requires finite predicted and realized values; Missing, Censored, TimedOut and Unsupported retain a typed expected slot without a realized value or prediction-set membership. These rows count in horizon/family denominators but do not enter numeric residual metrics. If an expected slot is omitted entirely, the evaluator rejects the report.

The evaluator:
- recomputes the decision-context digest and exposes public digest-building helpers for callers;
- recomputes the action-set digest from sorted action ID / command digest pairs;
- rejects duplicate action and receipt identities;
- requires the selection commit sequence to precede the reference-outcome reveal sequence;
- treats missing/censored/timed-out/unsupported reference outcomes as IncompleteReference;
- treats unknown candidate feasibility as indeterminate;
- compares only the supplied finite candidate set;
- reports selected-action regret, predicted-vs-realized ranking reversal and optimism gap when comparison is determinate.

Selection/reveal sequence numbers and receipts are caller-supplied assertions. This evaluator cannot cryptographically prove that the planner was blinded or that the reference outcome was independently generated. The separate protocol #10266 addresses reference suitability and common-mode disclosure.

## Outputs

### Horizon residual summary

For each expected horizon:
- total expected outcome rows and observed/missing/censored/timed-out/unsupported counts;
- count and failure-family count;
- mean/median signed residual and median/p90/p95 absolute residual over Observed rows only;
- worst observed failure-family p95 absolute residual;
- prediction-set membership/coverage counts over Observed rows only.

When no Observed outcomes exist for a horizon or family, numeric metrics are `None`, never zero. Family/horizon summaries retain the same outcome counts and observed-only metrics. These are descriptive statistics; no confidence interval or real-world population inference is generated by this module.

Paired horizon growth is calculated only between adjacent scheduled horizon slots for the same scenario × actor policy × action history × metric, and only when both slots are Observed. A Missing/Censored/TimedOut/Unsupported intermediate slot breaks adjacency; the evaluator never bridges it. Positive absolute-residual growth indicates worsening error at the later observed horizon. The exact pair is retained.

### Planner discrepancy audit

For each decision, the evaluator reports:
- predicted best action within the supplied feasible candidate set;
- realized best action within that same set;
- ranking reversal;
- selected-action regret under realized utility;
- optimism gap: predicted selected-action advantage minus its realized advantage;
- status: comparable, incomplete reference, indeterminate feasibility, selected action infeasible, or no feasible action;
- caller assertion that the finite candidate set is complete.

The finite-set result is not a global optimum. If any action's feasibility is indeterminate, no ranking or regret claim is emitted for that decision. If the selected action is infeasible, the best feasible alternatives may be shown but selected-action regret is not computed as if it were an admissible choice.

### Report integrity

The report retains canonical raw input rows, recomputes all aggregates, and validates a BLAKE3 digest. This provides deterministic integrity/recomputation, not authentication, source provenance, reference independence or attestation.

## Analytic and adversarial fixtures

Required first fixtures:

1. Perfect model: predicted and realized values equal at all horizons; every residual is zero.
2. Known constant bias: signed residual has a known sign and magnitude at every horizon.
3. Known horizon growth: absolute residual grows by a fixed amount; paired growth matches the analytic delta.
4. Sign control: positive and negative signed bias remain distinct.
5. Missing horizon: no interpolation or synthetic pair is produced.
6. Context mismatch: changed environment, action history, actor policy, state, forecast time, split or shift rejects a paired claim.
7. Duplicate scenario/action-history/metric/horizon key is rejected.
8. Missing, censored, timed-out and unsupported candidate reference outcomes produce IncompleteReference; no ranking/regret is emitted. The residual expected-schedule denominator retains each incomplete slot.
9. Ranking reversal: planner-predicted best action loses under realized utility; the reversal and regret are exposed.
10. Optimism attack: inject an optimistic selected-action utility and verify optimism gap rises.
11. Outcome leakage: reveal sequence at or before selection commit is rejected.
12. Unknown feasibility: ranking/regret remains indeterminate.
13. Action-set digest mutation and report mutation are detected.
14. Input-order invariance: canonical reordering leaves the report digest unchanged.
15. Extreme finite numeric values cannot silently overflow summaries into non-finite outputs.

## Benchmark sequence

1. Analytic one-dimensional motion and stopping fixtures.
2. Synthetic crossing actor policies under Symtropy #3471, executed/replayed via #3387.
3. Held-out action/horizon experiments with deliberate model optimism and context perturbations.
4. Waymo Open Motion and Argoverse 2 Motion under official splits for forecasting residuals; point-prediction accuracy stays separate from set coverage (#10256).
5. WOD-E2E and Fail2Drive for their distinct closed-loop long-tail decision/generalization tasks.
6. Cross-model profile interventions from #10252 (friction, actuator delay, localization and actor-policy variants).
7. Context-qualified empirical or physical comparison only after model-suitability, rights, instrumentation and approval gates.

## Prospective targets (not results)

- At matched compute, at least 20% lower p95 OOD residual on a frozen family-disjoint set versus the strongest faithfully reproduced baseline, with no more than 1% relative in-distribution p95 regression.
- At matched empirical coverage, at least 20% lower median/p95 prediction-set measure than a frozen heuristic tube with no worse adequately sampled worst-family coverage (#10256).
- On a paper-faithful Fail2Drive replication, at least 25% relative reduction in paired OOD success-rate degradation, with ID success no more than 1 percentage point worse.
- At matched compute and fixed finite action sets, at least 10% lower median and p95 counterfactual regret without increased infeasibility or deadline misses.
- Zero accepted context-mismatched pairs, future-outcome leakage, duplicate row keys or missing reference outcomes recoded as success.

These targets require preregistration, exact baseline reproduction, denominators, uncertainty and named hardware. They are not existing results, safety thresholds or certification criteria.

## Claim ceiling

The evaluator measures caller-supplied prediction/reference discrepancies and finite-set decision comparisons. It does not establish that the reference model represents physical reality, that the admitted model set spans reality, that the planner is globally optimal, that prediction sets have conditional/OOD validity, or that a vehicle is safe or qualified. No ISO 26262 / ISO 21448, roadworthiness, public-road, collision-risk reduction or SAE Level 4/5 claim is made.
