# VIS-DVPC-025: equal-budget closed-loop activation and acquisition frontier

**Status:** synthetic mechanism benchmark; implementation validation pending exact-head Rust CI.  
**Tracking:** [issue #8686](https://github.com/Luminous-Dynamics/symthaea/issues/8686).  
**Claim ceiling:** no real-world safety, empirical superiority, or state-of-the-art claim.

## Research question

Does a decision-value acquisition policy create positive downstream net value under the same all-in budget while refusing activation when evidence is shifted, incomplete, dependent, low-value, or not provenance-valid?

This deliberately joins a selection loop and an activation gate in one scorer. Every candidate is selected from the visible action table. Only after selection does the simulator look up that action's outcome, charge actual costs, update the evidence state, recompute the gate, and decide which actions may be selected next.

## Frozen fixture

- 13 hand-authored campaigns cover positive opportunity, high uncertainty with low value, ranking reversal, cost inversion, source-to-target shift, row dependence, missing/censored evidence, safe-but-useless evidence, actual cost overrun, negative realized value, independent cluster units, no information, and repeated fixed-sample peeking.
- Each campaign has three measurement candidates and one activation candidate; the public action table and evaluator-only outcome table are separate hashed files.
- The public table contains estimates, uncertainty, estimated cost, budget, target and the declared type of evidence unit. Realized value, actual cost, sample counts, failures, value summaries, target calibration, provenance and sealed-access controls are evaluator-only.
- Thirteen campaigns are a known-answer adversarial fixture, not thirteen independent empirical replications. The seeds identify frozen cases; they do not support confidence intervals or population claims.

## Gate in this version

The mechanism gate uses a one-sided Clopper–Pearson upper bound for failure risk with alpha 0.025, plus a bounded-value Hoeffding lower bound with alpha 0.025. It requires the risk upper bound to be at most 5%, a net-value lower bound of at least 0.5 normalized units (500 milli-units), target calibration, valid provenance, and no sealed access.

Only declared IID trials or independent-cluster units may contribute. Row-dependent samples and evidence labelled as a repeated fixed-sample peek are excluded rather than counted as independent. Source-to-target mismatch and provenance/sealed-access failures deny activation. Missing results do not become zero-cost successes.

**Composition boundary:** this v1 computes the fixed-sample gate inside the integration scorer. It does not yet invoke the VIS-DVPC-024 e-process implementation as a shared library, nor does it cryptographically verify the synthetic provenance booleans. The peeking control is rejected; it is not evidence that the sequential gate has been integrated. Wiring the anytime-valid and cluster-aware gate implementations into the same campaign adapter remains an explicit next gate.

## Baselines

- **fixed_fallback:** take no measurement and retain the safe fallback.
- **forced_adapt:** intentionally bypass the certificate; its false authorizations are reported and it is not a compliant policy.
- **highest_uncertainty:** choose the highest-uncertainty affordable measurement.
- **cost_first:** choose the lowest-cost affordable measurement.
- **decision_value:** choose by declared marginal test value minus estimated cost.
- **risk_aware_value:** decision value with a simple predicted-risk penalty and risk-cap filter.

All baselines get the same visible candidate set, budgets and evaluator outcomes. The scorer does not expose oracle outcomes to the policy before a selected action. The hindsight oracle is deliberately omitted from deployable baseline ranks.

## Accounting and output

For each campaign-policy pair, output includes selected actions, gate status/reason, activation, false authorization, budget overrun, actual cost, gross realized value, net value, risk upper bound, value lower bound, and the counts used by the gate.

Net value is realized activation value minus (all actual measurement and activation costs multiplied by 100 milli-value units). An abstaining policy that spends money but never activates can therefore have negative net value; a policy that never spends gets zero rather than being credited as a discovery success. Actual costs, not only planning estimates, control budget-overrun reporting.

The output is a metric vector, not a single weighted leaderboard score. Keep risk, realized value, cost, activation coverage, harmful activations, budget overruns and false authorizations separately visible.

## Known-answer acceptance gates

1. Frozen visible-action and evaluator-only oracle files pass SHA-256 verification.
2. The scorer emits exactly 78 campaign-policy rows and six policy summaries.
3. The intentionally unsafe forced_adapt baseline is recorded with 13 false authorizations in the no-evidence fixture.
4. The decision_value policy authorizes a safe, positive-value campaign and records positive net value.
5. The row-dependent and repeated-fixed-peek controls do not authorize the high-uncertainty baseline.
6. Actual cost overruns cannot disappear from the ledger.
7. Tests and scoring run against the exact commit in the dedicated workflow; the workflow publishes raw rows and a source/fixture/output receipt.

A green mechanism test establishes only that the implementation matches these frozen examples. It does not prove the gate's assumptions on real inputs, guarantee target transportability, validate cryptographic receipts, or show that a method outperforms industry systems.

## Research positioning and next benchmarks

The structure follows the direction of current end-to-end scientific-agent evaluations: budgeted experiment campaigns, updated decisions after observations, and separate discovery/learning outcomes. Autonomous Science Bench 0.1 evaluates closed-loop campaigns from published lab experiments and explicitly compares outcomes and costs under shared budgets; its six-task pilot also warns that three trials per task make small leaderboard differences inconclusive. [Benchmark](https://www.autonomous-science-bench.ai/) · [contribution protocol](https://www.autonomous-science-bench.ai/contribute).

AutoSciBench motivates continuously generating and verifying new benchmark cases rather than treating a finite fixture as permanently sufficient. Its 2026 preprint reports that generated tasks exposed shortcuts and lowered solver accuracy relative to human-curated tasks on some domains. [Paper](https://arxiv.org/abs/2610.05140).

For risk control, the same-distribution assumption cannot be hand-waved: high-probability risk control under covariate shift requires explicitly justified weighting/transport assumptions. [Almeida et al., PMLR 2025](https://proceedings.mlr.press/v266/almeida25a.html). For broad scientific-agent comparisons, AstaBench reports end-to-end outcomes alongside cost per problem; its scores are not numerically comparable to this small synthetic mechanism fixture. [AI Index 2026 summary](https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf).

### Advancement criteria beyond this fixture

- Integrate the real VIS-DVPC-024 anytime-valid gate and an independent-cluster gate into the shared campaign loop, not just their synthetic verdicts.
- Replace synthetic provenance booleans with verified, source-bound evidence receipts and negative controls for byte mutation, swapped source/target and stale receipts.
- Move from summary-only outcomes to immutable raw per-trial outcomes and an independent aggregate verifier; prove the action log contains no unselected oracle rows.
- Run paired campaigns on the same seeds at multiple predeclared budgets and held-out shift families; resample at the independent task/source/cluster level.
- Compare against random, uncertainty, cost-normalized uncertainty, diversity, plug-in EVSI and robust EVSI on identical information, tools, compute and all-in costs.
- Proposed internal targets (not industry standards): zero false authorizations on all frozen negative controls; at least 10% greater net decision value at equal cost on two held-out task families with the paired task/cluster interval excluding zero and no risk-cap violation, or at least 20% lower cost at non-inferior decision quality and risk.
- Publish failure cases and IID robustness tax. Do not claim SOTA until the benchmark is externally reproducible, the baselines are faithful, and results generalize beyond synthetic fixtures.
