# NANO-013 design-weighted scoring

This scorer turns the independent, evaluator-sealed probability sample into explicit estimates. It complements, but does not replace, the symthaea-nanotech-audit structural integrity audit.

## Run

From the repository root:

~~~sh
cargo run -p symthaea-nanotech-audit --bin nano-rescue-score -- <run-directory>
~~~

For a full MADE export, also run the general artifact auditor:

~~~sh
cargo run -p symthaea-nanotech-audit -- <run-directory>
~~~

The scoring CLI independently reruns the NANO-013 contract audit and refuses to score if the bundle is absent or has findings. For a full MADE export, run the general auditor as well; its **artifact_integrity_pass** concerns artifact consistency only and is not a scientific-performance pass.

The scorer reads **rescue-controller/audit-sampling-manifest.json**, **rescue-controller/decision-ledger.jsonl**, and **rescue-controller/blinded-audit-results.jsonl**. It independently checks raw manifest digest, sample membership, sample/result cardinality, parent/trajectory/stratum identities, inclusion probabilities, reference status/label consistency, action-to-outcome reconciliation, policy revision, contiguous per-trajectory decision sequences, leakage flags, and policy-label blinding. It fails closed for malformed records, duplicate/missing sampled windows, mixed cost units, and digest mismatches.

## Estimators

For sampled window \(i\), let \(\pi_i\) be its preregistered inclusion probability and \(w_i=1/\pi_i\). The scorer reports Horvitz–Thompson weighted totals and ratio estimates:

- **Policy reference-query fraction:** \(\sum_i w_i q_i / \sum_i w_i\), including windows whose reference calculation subsequently failed. This measures expensive reference-query behavior, not whether a reference answer was obtained.
- **Policy intervention fraction:** the weighted fraction with any action other than plain \(continue\). It includes conservative continuation, stop/abstain and reference requests, and therefore measures a different trade-off from query fraction.
- **Reference-outcome coverage:** \(\sum_i w_i 1[\text{reference evaluated}] / \sum_i w_i\).
- **Severe-event intervention recall:** weighted severe outcomes on which the policy took any non-continue action, divided by weighted severe outcomes with successful reference labels.
- **Severe-event reference-query recall:** weighted severe outcomes on which the policy explicitly requested a reference, divided by weighted severe outcomes with successful reference labels. It is narrower than intervention recall.
- **False-intervention rate and false-query rate:** separately measure interventions or reference requests among known non-severe outcomes.
- **Known-label false-alarm burdens:** report both known non-severe interventions and known non-severe reference queries divided by the weighted eligible-window population. If any sampled labels are missing, these are lower bounds on the corresponding total burdens.
- **Physical-integrity failure rate:** weighted known integrity failures divided by the weighted number of outcomes with a known integrity label.
- **Cost accounting:** the report separates recorded operational policy cost, observed probability-sample reference cost, and an inclusion-weighted hypothetical full-frame reference-cost equivalent. That last extrapolation estimates what it would cost to reference-evaluate every eligible window at the sampled per-window costs; it is not the actual audit expenditure, a bill, or a measured compute budget. Policy and reference costs retain separate units and should not be added unless reconciled to a common basis.

A failed or unavailable reference is never recoded as a negative event or a safe physical outcome. If any labels are missing, recall and integrity rates are labelled as conditional on successful references and the report emits a warning. Missingness may be informative, so the conditional rates cannot be promoted to population claims without an explicit missing-data strategy or recovered reference outcomes.

## Uncertainty

The scorer forms an approximate percentile interval by resampling complete **parent_group_id** clusters with replacement, preserving all sampled windows within each selected parent. It uses a deterministic seed per metric for reproducibility. It withholds an interval below 10 represented parent groups, and also withholds it when fewer than 90% of bootstrap replicates have a defined estimate. Fewer than 20 independent parent groups triggers an additional warning.

This is an approximate cluster bootstrap—not an exact variance estimator for every stratified finite-population design, not a guarantee under arbitrary dependence, and not a substitute for a prospectively justified sample-size/power plan. The independent unit must actually be the declared parent-system cluster; assigning many correlated windows separate IDs does not create independence.

## Synthetic and promotion gates

The current fixture is deliberately tiny and is a contract example, not a performance dataset. It should demonstrate weighted arithmetic and visible label-missingness, not pass the benchmark.

Proposed scientific promotion targets remain separate:
- at least 90% severe-event recall at no more than 20% of eligible windows queried, with false alarms, lead time and missed events reported;
- nominal 90% intervals showing a prespecified 85–95% grouped empirical coverage range when sample size supports it, with interval width and worst-group coverage;
- no violation of preregistered physical-integrity hard gates in any material/system group;
- paired, parent-system-grouped comparisons against fixed, random-query, uncertainty-only, novelty-only and cost-aware baselines;
- at least two independent system/material families, including a meaningful held-out shift family;
- at least 20% more reference-validated useful simulation time per all-in budget than the strongest eligible baseline, supported by uncertainty at the group level.

These are proposed thresholds, not measured Symthaea results. A scorer exit code of zero means only that it produced a report from internally consistent scoring inputs; it is not a pass/fail promotion decision.

## Research comparators

The benchmark should compare global uncertainty and distance/novelty baselines against per-atom or environment-conditional calibration where the model stack supports it. Ho, Ortner and Wang's 2026 work introduces learned environment-dependent quantile calibration for machine-learned interatomic potentials: https://www.nature.com/articles/s41524-026-02080-3. A 2026 Au(100)/water/NaOH interface study reports localized errors despite improvements in bulk-like regions after fine-tuning: https://www.nature.com/articles/s41524-026-02051-8. These results motivate comparator arms; they do not demonstrate Symthaea calibration, transfer, or physical reliability.
