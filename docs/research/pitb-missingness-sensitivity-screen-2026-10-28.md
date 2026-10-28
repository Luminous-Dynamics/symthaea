# Independent PITB missingness and estimand screening (28 October 2026)

**Status:** exploratory synthetic screen only. Not a production Rust result, not confirmatory statistical qualification, and not evidence of PITB transfer capability.

## Reproduction

From the repository root:

    python3 scripts/research/pitb_missingness_sensitivity_v0.py --studies 1000 --bootstrap-reps 299 --seed 20261028 --out /tmp/pitb-missingness.json

This screening run used Python 3.13.5 and NumPy 2.3.5, 1,000 studies per scenario, 299 component-bootstrap replicates per study, baseline mean 0.35, family ICC 0.10, paired-outcome association fraction 0.50, and requested family-effect SD 0.08. The profiles are synthetic (40 components, balanced sizes of 4, or a heavy-tailed profile with sizes 1, 4 and 15); they are not the unknown audited PITB component sizes.

## Findings

- **MCAR target-row omission:** 15% random omissions produced 95.5% conditional interval coverage in this run and a mean estimated effect of -0.034 percentage points. This does not authorize complete-case deletion from the real confirmatory ledger.
- **Outcome-dependent omission:** the synthetic generator omitted a target with probability 60% when the baseline succeeded and treatment failed, and 5% otherwise. It retained about 88.5% of rows on average. Although the true target-weighted effect was zero, complete-case analysis had +7.27 percentage points mean bias; interval coverage was 57.2%; the lower bound rejected zero in 40.5% of studies; and the full +10pp utility gate passed in 23.1%.
- **Informative component size and effect:** with family effects centered to a zero target-weighted mean, the equal-component effect averaged +8.47 percentage points. This is an estimand difference, not a discrepancy that should be resolved by selecting the more favorable metric.
- **Size-dependent runtime outage:** every assignment remained in the data, while unavailable treatment attempts were scored as non-solves. The estimand was the operational system's success probability, not latent solver capability. The mean operational target effect was -12.31 percentage points in this scenario.

The MNAR case is deliberately adversarial. It demonstrates the danger of a complete-case analysis under one specified selection process; it does not establish that the real benchmark has MNAR data or that missing outcomes can be inferred from observed data. The outage scenario is an observed system failure, not missingness, and must not be analyzed as if failed arms vanished from the denominator.

## Decision

Keep the confirmatory data contract fail-closed: every frozen eligible target and arm needs an explicit outcome row; not-attempted, timeout and tool-unavailable statuses remain non-solves; absent required proof receipts invalidate a claimed success; post-outcome exclusions are forbidden. Use the target-weighted finite frozen test set as the primary estimand, with equal-component weighting diagnostic only.

A material unresolved implementation gate remains: the scorer must consume the audited cross-family connected-component manifest as its bootstrap/resampling key before component-count qualification. Twenty named families are not twenty independent components if dependency links collapse them into fewer groups. The new machine-readable calibration plan and CI negative controls codify this requirement, but the plan is not a simulation result.

The findings align with established cluster-trial guidance that the estimand must match the weighting and that informative cluster size can change the question being estimated: [estimands in cluster-randomized trials](https://pmc.ncbi.nlm.nih.gov/articles/PMC9908044/) and [cluster trial design guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10555937/). Missing-outcome methods must also be matched to explicit assumptions, with clustered structure preserved where relevant: [BMJ 2025 practical guide](https://www.bmj.com/content/391/bmj-2025-084194).
