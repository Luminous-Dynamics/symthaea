# VIS-DVPC-025: equal-budget closed-loop activation and acquisition frontier

**Status:** synthetic mechanism benchmark; implementation validation pending exact-head Rust CI.  
**Tracking:** [issue #8686](https://github.com/Luminous-Dynamics/symthaea/issues/8686).  
**Claim ceiling:** no real-world safety, empirical superiority, or state-of-the-art claim.

## Research question

Does a decision-value acquisition policy create positive downstream net value under the same all-in budget while refusing activation when evidence is shifted, incomplete, dependent, low-value, or not provenance-valid?

This deliberately joins a selection loop and an activation gate in one scorer. Every candidate is selected from the visible action table. Only after selection does the simulator look up that action's outcome, charge actual costs, update the evidence state, recompute the gate, and decide which actions may be selected next.

## Frozen fixture

- 13 hand-authored campaigns cover positive opportunity, high uncertainty with low value, ranking reversal, cost inversion, source-to-target shift, row dependence, missing/censored evidence, safe-but-useless evidence, actual cost overrun, negative realized value, independent cluster units, no information, and repeated fixed-sample peeking.
- Each campaign has three measurement candidates and one activation candidate; the public action table and evaluator-only outcome table are separate hashed files. Oracle rows freeze per-observation risk bits and bounded-value samples, validate them against the aggregate summaries, and never place those samples in the visible table.
- The public table contains estimates, uncertainty, estimated cost, budget, target and the declared type of evidence unit. Realized value, actual cost, sample counts, failures, value summaries, target calibration, provenance and sealed-access controls are evaluator-only.
- Thirteen campaigns are a known-answer adversarial fixture, not thirteen independent empirical replications. The seeds identify frozen cases; they do not support confidence intervals or population claims.

## Gate in this version

Two gates are compared. The fixed gate uses a one-sided Clopper–Pearson upper bound for failure risk with alpha 0.025 plus a bounded-value Hoeffding lower bound with alpha 0.025. The anytime gate consumes the frozen individual observations in action-return batches and uses the same preregistered risk-mixture likelihood-ratio e-process and bounded-value betting e-process as VIS-DVPC-024; each process must cross 1/alpha = 40. The anytime gate checks at actual batch-arrival boundaries, not fictional unobserved stopping times. Both modes use risk cap 5%, value floor 500 milli-units, target calibration, valid provenance, and no sealed access. The two alpha allocations of 0.025 use a union bound for the combined error budget.

Only declared IID trials or independent-cluster units may contribute. Row-dependent samples and evidence labelled as a repeated fixed-sample peek are excluded rather than counted as independent. Source-to-target mismatch and provenance/sealed-access failures deny activation. Missing results do not become zero-cost successes.

**Composition boundary:** the campaign scorer now contains a standalone implementation of the same preregistered e-process formulas used in VIS-DVPC-024, applied to per-observation raw streams and selected action batch boundaries. This avoids counting a fixed-sample interval as valid under repeated peeking. The code is not yet a shared production library with VIS-DVPC-024, and it does not cryptographically verify the synthetic provenance booleans. Independent-cluster labels are explicit test inputs, not proof of independence; no deployment guarantee follows.

## Baselines

- **fixed_fallback:** take no measurement and retain the safe fallback.
- **forced_adapt:** intentionally bypass the certificate; its false authorizations are reported and it is not a compliant policy.
- **highest_uncertainty:** choose the highest-uncertainty affordable measurement.
- **cost_first:** choose the lowest-cost affordable measurement.
- **decision_value_fixed:** choose by declared marginal test value minus estimated cost, then apply the fixed-sample gate.
- **risk_aware_value_fixed:** decision value with a simple predicted-risk penalty/risk-cap filter, then fixed gate.
- **decision_value_anytime:** same acquisition score but activation requires both anytime-valid e-processes to cross.
- **risk_aware_value_anytime:** risk-penalized acquisition with the anytime-valid gate.

All baselines get the same visible candidate set, budgets and evaluator outcomes. The scorer does not expose oracle outcomes to the policy before a selected action. The hindsight oracle is deliberately omitted from deployable baseline ranks.

## Accounting and output

For each campaign-policy pair, output includes selected actions, gate status/reason, activation, false authorization, budget overrun, actual cost, gross realized value, net value, risk upper bound, value lower bound, and the counts used by the gate.

Net value is realized activation value minus (all actual measurement and activation costs multiplied by 100 milli-value units). An abstaining policy that spends money but never activates can therefore have negative net value; a policy that never spends gets zero rather than being credited as a discovery success. Actual costs, not only planning estimates, control budget-overrun reporting.

The output is a metric vector, not a single weighted leaderboard score. Keep risk, realized value, cost, activation coverage, harmful activations, budget overruns and false authorizations separately visible.

## Known-answer acceptance gates

1. Frozen visible-action and evaluator-only oracle files pass SHA-256 verification.
2. The scorer emits exactly 104 campaign-policy rows and eight policy summaries.
3. The intentionally unsafe forced_adapt baseline is recorded with 13 false authorizations in the no-evidence fixture.
4. The fixed decision-value policy authorizes a safe, positive-value campaign and records positive net value. The anytime decision-value policy consumes a second batch, crosses the time-uniform risk/value evidence thresholds, then authorizes without false authorization.
5. The row-dependent and repeated-fixed-peek controls do not authorize the high-uncertainty baseline.
6. Actual cost overruns cannot disappear from the ledger.
7. Tests and scoring run against the exact commit in the dedicated workflow; the workflow publishes raw rows and a source/fixture/output receipt.

A green mechanism test establishes only that the implementation matches these frozen examples. It does not prove the gate's assumptions on real inputs, guarantee target transportability, validate cryptographic receipts, or show that a method outperforms industry systems.

## Research positioning and next benchmarks

The structure follows the direction of current end-to-end scientific-agent evaluations: budgeted experiment campaigns, updated decisions after observations, and separate discovery/learning outcomes. Autonomous Science Bench 0.1 evaluates closed-loop campaigns from published lab experiments and explicitly compares outcomes and costs under shared budgets; its six-task pilot also warns that three trials per task make small leaderboard differences inconclusive. [Benchmark](https://www.autonomous-science-bench.ai/) · [contribution protocol](https://www.autonomous-science-bench.ai/contribute).

AutoSciBench motivates continuously generating and verifying new benchmark cases rather than treating a finite fixture as permanently sufficient. Its 2026 preprint reports that generated tasks exposed shortcuts and lowered solver accuracy relative to human-curated tasks on some domains. [Paper](https://arxiv.org/abs/2610.05140).

For risk control, the same-distribution assumption cannot be hand-waved: high-probability risk control under covariate shift requires explicitly justified weighting/transport assumptions. [Almeida et al., PMLR 2025](https://proceedings.mlr.press/v266/almeida25a.html). For broad scientific-agent comparisons, AstaBench reports end-to-end outcomes alongside cost per problem; its scores are not numerically comparable to this small synthetic mechanism fixture. [AI Index 2026 summary](https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf).

### Advancement criteria beyond this fixture

- Factor the shared e-process logic from VIS-DVPC-024 into a common Rust module and make this scorer call that shared code; then add an independently validated cluster-level e-process adapter instead of relying on the fixture unit label.
- Replace synthetic provenance booleans with verified, source-bound evidence receipts and negative controls for byte mutation, swapped source/target and stale receipts.
- Move from summary-only outcomes to immutable raw per-trial outcomes and an independent aggregate verifier; prove the action log contains no unselected oracle rows.
- Run paired campaigns on the same seeds at multiple predeclared budgets and held-out shift families; resample at the independent task/source/cluster level.
- Compare against random, uncertainty, cost-normalized uncertainty, diversity, plug-in EVSI and robust EVSI on identical information, tools, compute and all-in costs.
- Proposed internal targets (not industry standards): zero false authorizations on all frozen negative controls; at least 10% greater net decision value at equal cost on two held-out task families with the paired task/cluster interval excluding zero and no risk-cap violation, or at least 20% lower cost at non-inferior decision quality and risk.
- Publish failure cases and IID robustness tax. Do not claim SOTA until the benchmark is externally reproducible, the baselines are faithful, and results generalize beyond synthetic fixtures.
