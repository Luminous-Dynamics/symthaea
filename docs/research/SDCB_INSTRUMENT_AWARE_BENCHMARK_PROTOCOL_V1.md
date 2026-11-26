# SDCB instrument-aware benchmark protocol v1

**Status:** preregistration-style protocol; no tournament has been run.  
**Implementation under test:** `scripts/research/sdcb_instrument_aware.rs`  
**Reference fixture:** `SDCB_INSTRUMENT_AWARE_DESIGN_V1.example.json`

## 1. Evaluation claim

Test whether a scientific decision agent can distinguish uncertainty about the scientific system from uncertainty about the measurement channel, then choose calibration, science measurement, replication, or abstention based on downstream decision value.

The target claim is deliberately narrower than “better scientific intelligence”:

> Under matched evidence and resource budgets, an instrument-aware policy reduces held-out decision regret under persistent measurement-channel uncertainty without wasting resources on decision-irrelevant calibration, accepting hypothesis-leaking calibration data, or violating evidence and safety constraints.

No single composite score is permitted.

## 2. Controlled task families

Generate paired worlds with the same scientific hypothesis/action problem but different observation channels:

| Family | Measurement process | Required behaviour |
|---|---|---|
| F0: known channel | Instrument state known or calibrated exactly | Do not spend budget on redundant calibration |
| F1: persistent bias | Repeated readings share one latent offset/gain state | Avoid treating same-channel repeats as independent evidence |
| F2: calibration helps | Reference standard identifies channel state; science alone is confounded | Select calibration when its downstream decision value justifies cost |
| F3: no decision value | Calibration reveals channel state but every plausible state implies the same best action | Reject decision-irrelevant calibration |
| F4: shifted standard | Calibration channel quality differs under a held-out scenario | Preserve robust regret and uncertainty; do not trust nominal accuracy alone |
| F5: independent replication | A second instrument has a distinct channel state | Distinguish same-device repeatability from cross-device reproducibility |
| F6: stale calibration | Channel drifts between calibration and science measurement | Defer this to a transition-state implementation; v1 must abstain from claiming support |
| F7: contaminated reference | Calibration output is spuriously predictive of the scientific hypothesis | Reject the candidate under the conditional-independence contract |

F0–F4 and F7 can be expressed in the current finite static-state model. F5 requires a joint state with two instrument channels. F6 requires explicit instrument-state transitions and is out of scope for the current Rust reference. Keep these capability boundaries visible in results.

## 3. Matched methods

Use the same prior, candidate catalogue, scenario set, action utility, information access, and branch-wise resource cap for every method. Report each method's assumptions.

Minimum arms:

1. No experiment / current best action.
2. Fixed science-only sequence.
3. Repeated same-instrument measurement.
4. Calibration-first heuristic.
5. Greedy EIG over all candidate measurements.
6. EIG with oracle-known calibration state (diagnostic upper bound, not a deployable baseline).
7. Decision-focused one-step EVSI and exact two-stage finite oracle on tractable tasks.
8. Finite-scenario robust maximin EVSI.
9. R-I and R-IDeA where representativeness/proxy assumptions are valid.
10. Multi-fidelity BO comparators where fidelity costs and likelihoods are matched.
11. Instrument-aware two-stage policy.

Keep the exact finite oracle labelled as an upper-bound/reference implementation. Do not compare internal acquisition values across EIG, R-IDeA, EVSI, and worst-case utility.

## 4. Metrics

Report separate metric families:

**Decision quality:** expected terminal utility, Bayes decision regret, worst-scenario regret, robust value of information, calibration-attributable decision gain, and false confidence in a wrong action.

**Measurement-channel inference:** posterior log loss/Brier score for instrument state, calibration error, interval coverage, drift detection delay (only for transition-state variants), and performance by calibration regime.

**Scientific inference:** held-out prediction loss, identifiability-conditioned mechanism recovery, false mechanism-claim rate, and calibrated abstention. Do not award mechanism-recovery credit when the task is observationally non-identifiable.

**Efficiency:** science measurements, calibration measurements, total cost, wall-clock, energy where measurable, cost to first correct decision, and the componentwise Pareto frontier. Report branch-wise maximum cost as well as expected cost.

**Integrity and reliability:** calibration leakage accepted (must be zero), hard budget violations (must be zero), stale/misbound/duplicate observation acceptance, unauthorized mutations, independent replay success, and provenance completeness.

The agent should not be rewarded merely for learning the instrument state. The metric that matters is whether calibration changes a justified scientific decision or improves a declared scientific inference target.

## 5. Negative controls and ablations

Freeze these controls before parameter tuning:

- Repeated same-channel measurements under a shared, unobserved offset must not create false confidence about the scientific hypothesis.
- A calibration likelihood that depends on the scientific hypothesis after conditioning on instrument state must be rejected.
- Calibration must not win when channel state is already known and it cannot improve any terminal decision.
- A high-EIG science probe with zero decision value must not beat a lower-EIG calibration-plus-science policy when the latter improves terminal utility.
- A nominally accurate calibration standard that reverses under a plausible held-out scenario must not be treated as certain.
- Every possible first-outcome branch must obey the hard cumulative resource cap.
- A malformed, stale, duplicate, or unauthorized measurement receipt must cause zero scientific-state and instrument-state mutation once EGBT integration exists.

Ablate shared-state modelling, calibration-channel constraints, scenario robustness, branch-wise budgets, and the second-stage adaptive choice independently. Never remove a hard safety check merely to improve average utility.

## 6. Statistical protocol and SOTA gates

- Use paired task seeds and identical candidate sets; begin with at least 20 seeds per synthetic family.
- Hold out complete channel models, bias families, and scientific mechanisms rather than randomly splitting episodes from the same generator.
- Freeze all kernel widths, acquisition weights, calibration thresholds, proxy training, and stop rules on a validation split.
- Use task-family clustered bootstrap intervals and paired comparisons; preregister multiplicity correction for multiple arms.
- Publish all attempts, failures, costs, model assumptions, and source revisions.
- Require independent reproduction on held-out task families before any SOTA claim.

Proposed gates (targets, not achievements):

- 100% agreement with the exact finite oracle on the frozen small known-answer suite.
- Zero accepted calibration-leakage controls and zero branch-wise budget violations.
- At least 10% lower held-out decision regret than the strongest matched EIG-only and repeated-measurement baselines on F1/F2, with a paired 95% interval excluding zero.
- No material regression against EIG-only in F0, where calibration is unnecessary.
- At least 10% lower total resource cost at matched decision quality on one calibration-sensitive multi-fidelity family.
- Non-inferior predictive calibration and no increase in false scientific-mechanism claims.
- External reproduction using a second implementation or independent team.

The 10% thresholds are project targets, not accepted industry standards. A result on synthetic SDCB worlds is not an external benchmark win.

## 7. External benchmark use

Use Petri-bench for method integrity and experiment quality, ActiveSciBench for mechanism-discovery transfer, and Autonomous Science Bench for complete campaign workflows. Use ASI-Bench/AstaBench for broader project-level autonomy only as a separate track. Do not retrofit calibration scores into upstream leaderboard metrics or compare raw scores across different benchmark protocols.

First run the controlled instrument-aware synthetic suite. Then select externally benchmarked tasks only where the observation process, calibration evidence, and decision utility can be represented without altering the upstream task's scoring or hiding information from the agent.

## 8. Implementation boundary

Current Rust v1 supports static persistent instrument state, binary outcomes, two experiment stages, finite ambiguity scenarios, and hard branch-wise budgets. It does not support drift transitions, independent multi-device states, continuous measurements, posterior model training, EGBT receipts, or HDC/CfC integration. Those are successive research increments, not assumed capabilities.

**Qualification:** the protocol is authored; no benchmark tournament or SOTA comparison has been executed.
