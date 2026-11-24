# SEMI-ROBUST-EVSI-001 — finite-set worst-case EVSI v0.1

**Issue:** #13541  
**Implementation:** symthaea-semiconductor::benchmark::robust_decision_loss_evsi  
**Base:** point-EVSI scorer #13540  
**Status:** implementation and independent-oracle tests authored; exact-head validation pending.

## Research thesis

A point-scenario EVSI score can be overconfident when prior prevalence, assay sensitivity/specificity, appraiser effects, common-reference faults, measurement cost or turnaround are estimated from sparse/selected-only data.

This module extends point-EVSI with **finite-set worst-case scenario optimization**. It does not claim a general distributionally robust optimization guarantee.

For candidate \(i\) and declared scenario \(\theta\):

\[
V_i(\theta)=R_{0,i}(\theta)-R_{1,i}(\theta)-C_i(\theta)-D_i(\theta).
\]

The robust candidate score is:

\[
V_i^{robust}=\min_{\theta\in\Theta_i} V_i(\theta).
\]

The portfolio maximizes the sum of candidate robust values under the same qualified-measurement budget and group-coverage constraints.

Scenario weights are optional descriptive metadata; they are not used in this minimax objective. A weighted expected-scenario objective would be a different, separately named policy.

## Typed evidence

Each candidate contains:
- stable candidate/event identity, group, instrument, measurand and unit;
- decision deadline and measurement turnaround;
- the same prospective calibration/MSA readiness record as the point-EVSI scorer;
- a declared central scenario ID;
- a finite scenario set;
- a completeness flag and canonical scenario-set digest.

Each scenario binds:
- unique scenario ID;
- optional descriptive weight;
- scenario-specific measurement cost and delay cost;
- a full decision-loss model (prior state probabilities, action-loss matrix, assay likelihood matrix);
- source-provenance SHA-256;
- scenario SHA-256.

Scenario identity validation requires consistent state/action/assay-outcome IDs across the uncertainty set. The numeric priors, losses, likelihoods and costs may differ.

## Status and fail-closed semantics

- An incomplete global or per-candidate scenario set returns UNKNOWN, not a score based on the scenarios that happened to arrive.
- Invalid scenario provenance/digest, duplicate IDs or inconsistent state/action/outcome identities are rejected.
- Unknown/rejected readiness and deadline misses remain ineligible in every scenario.
- A candidate with a positive central value but non-positive worst-case value is marked NON_POSITIVE_WORST_CASE; it is not selected unless a predeclared group-coverage quota forces it.
- The portfolio report records the binding worst-case scenario ID, central scenario, best-case scenario and every scenario's full point-EVSI decomposition.
- The exact portfolio optimizer retains the point-EVSI candidate cap (at most 20) and deterministic tie-breaking; oversized portfolios are rejected, not silently approximated.

## Tests

Ten unit tests cover:
1. worst-case score equals the minimum per-scenario net value;
2. central-positive / worst-case-negative candidate is not selected;
3. per-candidate incomplete uncertainty set yields UNKNOWN;
4. globally incomplete uncertainty set yields UNKNOWN;
5. unknown calibration readiness excludes a candidate across scenarios;
6. state/action/outcome identity changes are rejected;
7. descriptive scenario weights do not alter minimax value;
8. scenario ordering does not change the robust report digest;
9. exact robust portfolio respects group quotas and budget;
10. scenario provenance tampering is rejected.

Two independent integration tests:
- recompute Bayes risk and net EVSI for each scenario independently, then verify the worst-case binding scenario/value;
- independently enumerate every feasible portfolio subset under capacity/group quotas and compare candidate IDs plus worst-case objective.

Exact-head rustfmt, strict Clippy, unit/integration tests and SEMI-BENCH-001 reference workflow are still required.

## Benchmark plan

Use a preregistered scenario matrix:
- prevalence lower/central/upper;
- assay sensitivity and specificity lower/central/upper;
- appraiser and instrument variation;
- shared-reference/common-cause faults;
- measurement and turnaround cost ranges;
- a subset of scenarios with delayed result beyond the decision deadline;
- one subgroup with weaker assay performance.

Compare:
- point-EVSI;
- finite-set robust-EVSI;
- maximum predictive variance / interval width;
- defect-risk sampling;
- random/stratified audit;
- fixed-frequency sampling.

Report per-scenario and worst-case decision loss, robust regret, selected portfolio, escaped-defect upper bound, false holds, inspection capacity, deadline misses, group coverage and the frequency with which the binding worst-case scenario differs from the central scenario.

## Claim ceiling

This is finite-set worst-case optimization over caller-declared scenarios. It is not robust to unknown scenarios outside that set and does not estimate real fab economics or assay uncertainty by itself. No equipment-control, wafer-disposition or process-change authority is added.
