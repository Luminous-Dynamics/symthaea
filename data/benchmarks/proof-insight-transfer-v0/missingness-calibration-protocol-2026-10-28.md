# PITB selective-missingness and informative-component-size calibration

**Status:** preregistered design only. This is not simulated evidence, not a Rust test result, and not a proof-transfer measurement.

## Why this is the next qualification gate

PITB's current weak-null grid handles positive and negative family effects whose target-weighted average is zero, but its source assumptions still omit selective attrition and alternative component-size mechanisms. Those omissions matter because the primary endpoint weights eligible targets, while uncertainty is estimated by resampling clusters. If size, effect, missingness, or dependency structure is informative, a plausible-looking confidence interval can answer a different question from the one intended.

This is not a niche edge case. Published guidance on cluster-randomized trials emphasizes specifying whether inference concerns the average participant/target or the average cluster, because these estimands can differ when cluster size is informative. It also warns that common estimators can be biased under informative cluster sizes. See [Estimands in cluster-randomized trials](https://pmc.ncbi.nlm.nih.gov/articles/PMC9908044/) and the [2023 cluster-trial design guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10555937/). The BMJ's 2025 practical guide also highlights prespecified covariates linked to outcome missingness and the need to consider clustered missing-data methods ([BMJ 2025](https://www.bmj.com/content/391/bmj-2025-084194)).

## 1. Freeze the estimand before choosing the estimator

The primary estimand is the finite, target-weighted mean paired difference over the entire frozen set of eligible test-target assignments:

`delta_target = sum_t (success_treatment - success_baseline) / N_frozen_eligible_targets`

Every frozen eligible assignment stays in the denominator. The secondary equal-component-weighted estimand is diagnostic only and cannot replace the primary result after observing outcomes. This makes the target population explicit; it does not claim that the frozen benchmark is a random sample from every future theorem family.

The primary treatment effect is exact-target semantic-gate success, not compilation, subgoal count, or model self-report. A timeout, not-attempted result, or tool-unavailable result with a valid recorded run envelope is a non-solve for that arm, not an absent observation. The two arms must receive the same frozen resource ceilings; actual usage and failure reasons remain recorded.

## 2. Fail-closed rules for genuinely missing evidence

- A frozen target row or required arm-outcome row omitted from the ledger invalidates the confirmatory run. Do not silently delete it, impute success, or report a complete-case rate as the primary endpoint.
- A required checker or semantic-gate receipt missing for a claimed success blocks that success. An absent receipt is evidence incompleteness, not a zero-cost solve.
- Explicit recorded runtime failures (timeout, not attempted, or tool unavailable) stay in the assignment denominator as non-solves.
- Eligibility, exclusions, family membership, and test-denominator membership must be frozen before any arm outcomes are seen. Post-outcome exclusion is forbidden.
- If several source families are joined by an audited cross-family relation into one dependency-connected component, inferential resampling must use the connected component. The confirmatory minimum of 20 applies to independent connected components, not the raw number of family IDs.

A primary ledger that fails one of these controls is invalid or no-claim; an analyst must not repair it by weakening a gate.

## 3. Registered stress scenarios

The machine-readable plan defines twelve required adversarial scenarios. They cover random target-row omission, whole-component omission, difficulty-related non-attempts, arm-differential outages, size-informative component outages, outcome-dependent omission (MNAR), missing proof receipts, post-outcome exclusions, effect heterogeneity correlated with component size, and cross-family dependency collapse. The nominal rates 0%, 5%, 15%, and 30% form the sensitivity grid for executable follow-on work; the scenario list is the minimum contract, not a claim that all factorial cells have already been run.

Two classes must not be conflated:

1. **Assigned task fails:** the ledger has the target and arm row with a valid run envelope and a recorded failure status. This remains a non-solve in the primary performance endpoint.
2. **Outcome/evidence is missing:** the required assignment or proof of success is absent. This is a ledger/evidence failure and blocks confirmatory qualification.

## 4. What can and cannot be identified

MCAR and MAR stress simulations may be informative under their explicit generators, but a missing-at-random label is not established by the observed ledger. In the MNAR scenario, missingness depends on latent potential outcomes. The complete-case estimator is not permitted as confirmatory evidence, and no single inverse-probability adjustment can recover the missing outcomes without additional assumptions and measured covariates. Treat that arm as an adversarial sensitivity case: report bias/coverage only under a fully declared data-generating model, and never interpret synthetic identification as empirical identification.

## 5. Calibration matrix and go/no-go outcomes

Run the following dimensions as a prespecified grid once the cross-family relationship audit has supplied real independent component sizes:

- independent component counts: 20, 30, 40, and 60;
- component-size profiles: balanced, the validation-derived empirical distribution, and a prespecified heavy-tailed profile;
- nominal missingness rates: 0%, 5%, 15%, and 30%;
- the existing ICC and paired-potential-outcome association grid;
- weak-null family-effect heterogeneity, including effects correlated with component size;
- the twelve mandatory scenario classes in the JSON plan.

The run schedule, seeds, repetition count, bootstrap count, and output hashes must be frozen before inspecting calibration outcomes. Begin with a resource-only profiling pilot and freeze a feasible compute budget before opening metric summaries; do not tune the generator or decision thresholds to make the gate pass. The qualification run should report Monte Carlo uncertainty, interval availability, unconditional false-claim rates, conditional interval diagnostics, bias, coverage, interval width, and gate power.

**No-go conditions:** any missing frozen target/arm row; missing required receipt supporting a claimed success; post-outcome eligibility changes; unresolved component leakage; fewer than 20 independent connected components; or failure of the existing simultaneous strict-null / weak-null gates. Do not lower thresholds to overcome a no-go.

The current primary scorer resamples its declared family aggregates; the new plan deliberately makes connected-component IDs a required prerequisite. Confirmatory qualification must remain blocked until the audited component manifest and the scorer's resampling key agree. Counting 20 family labels where the audited graph contains only 14 independent components is not a valid way to satisfy the floor.

## 6. What this work does not claim

This is the calibration protocol and machine-readable scenario contract only. It contains no empirical missingness rates, no simulated calibration result, no power result, no theorem-transfer outcome, and no SOTA win. The next implementation step is an independently testable simulator that applies these frozen scenarios to the primary decision function and verifies that every missing row, missing receipt, and component collapse causes its required no-go or fail-closed behavior.
