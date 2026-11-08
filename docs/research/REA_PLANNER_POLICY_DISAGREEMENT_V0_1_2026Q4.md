# REA planner policy disagreement benchmark v0.1 — 8 November 2026

**Status:** original synthetic math-only planner input plus separate grader-only obligation effects and exact arithmetic oracle. No human adjudication, public planner execution, real observation, likelihood calibration or policy superiority is claimed.

**Owner:** [#11162 — REA-RUST-052](https://github.com/Luminous-Dynamics/symthaea/issues/11162), following [#11125](https://github.com/Luminous-Dynamics/symthaea/issues/11125) and [#11127](https://github.com/Luminous-Dynamics/symthaea/pull/11127).

## Why a policy-disagreement case is necessary

A one-experiment arithmetic example can validate EIG and EVSI formulas, but it cannot test whether different objectives choose different actions. This v0.1 case forces different policies to disagree for explainable reasons.

The agent-visible file is a full planner input conforming to #11127. The separate grader file contains outcome-to-obligation effects and expected policy rankings. It must never be included in the planner prompt or tool context.

## Three candidate observations

1. **static_function_probe** — strongest nominal information about vulnerability, but its outcomes do not close either mandatory proof obligation.
2. **verify_deployment_manifest** — no information about the vulnerability hypothesis, but it can establish or contradict exact deployment binding.
3. **controlled_reproduction** — less nominal EIG than the static probe, but its discrimination remains stronger under the conservative scenario and it can establish or contradict behavior.

All three are synthetic, authorized, reversible, read-only proposals. Predicted outcomes never update readiness.

## Mathematical oracle

With prior P(vulnerable)=0.5, utility matrix mitigate=(3,-6), no_action=(-20,0), and equally weighted nominal/conservative likelihood scenarios:

| Metric | Static probe | Manifest | Controlled reproduction |
|---|---:|---:|---:|
| Nominal EIG (bits) | 0.5310044064 | 0 | 0.2780719051 |
| Conservative EIG (bits) | 0.0002885582 | 0 | 0.1887218755 |
| Expected scenario EIG (bits) | 0.2656464823 | 0 | 0.2333968903 |
| Lower-tail CVaR-EIG, alpha=0.25 | 0.0002885582 | 0 | 0.1887218755 |
| Normalized planner cost | 0.3833333333 | 0.1033333333 | 0.7333333333 |
| Point planner utility (EIG / cost) | 1.3852288863 | 0 | 0.3791889615 |
| Robust lower-tail utility (CVaR-EIG / cost) | 0.0007527606 | 0 | 0.2573480121 |
| Expected EVSI (synthetic utility units) | 0.775 | 0 | 0.05 |
| Mandatory obligation updates | 0 | 1 | 1 |

Point and robust planner utilities divide the selected information-gain statistic by normalized cost: cost = w_time × (time / time_budget) + w_energy × (energy / energy_budget) + w_risk × (risk / risk_budget). The table uses the #10074/#10105 weighting and budgets from the agent-visible planner config. These values are independently computed mathematical expectations, not empirical probabilities or planner API results. EVSI is shown as expected value before any additional action-cost penalty; no public EVSI planner result is claimed.

## Expected first-action disagreement

- Point EIG: static function probe.
- Expected-scenario EIG: static function probe.
- Lower-tail robust EIG: controlled reproduction.
- Expected EVSI: static function probe.
- Obligation-closure per second: deployment manifest.
- Combined gated sequence: deployment manifest, then controlled reproduction.

The static probe is not “wrong”; it optimizes a different objective. The key result is that information-oriented policies can rationally rank it first while the claim remains unready. The proof-obligation policy selects the manifest first, then reproduction. Readiness becomes true only after both observed outcomes are independently admitted and satisfy their respective obligations.

## Qualification split

The exact arithmetic is tested independently of the public production planners. Synthetic Evidence must not be relabelled as independent observation to invoke the public planner. Actual observed-Evidence integration remains a later separate qualification lane.

The grader view is development-only and public in this PR; it is not a hidden test set. A future hidden split must be generated independently and withheld from evaluated systems.

## Claim ceiling

This PR qualifies only schema/fixture consistency and deterministic mathematical golden values. It does not execute the existing point, robust or decision-aware planner APIs, validate probability calibration, establish source authority, execute experiments, or demonstrate a real-world policy winner.
