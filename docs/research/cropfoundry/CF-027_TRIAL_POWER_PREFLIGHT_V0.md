# CF-027: pre-registered field-trial power and replication adequacy

**Status:** screening calculator and benchmark contract. It does not evaluate crop efficacy. The checked-in example deliberately uses synthetic variance assumptions and cannot be used to decide the size of an actual farm trial.

## Why it matters

Crop improvement is not demonstrated by a promising prediction or a few better-looking plots. A confirmatory trial must have an endpoint, estimand, smallest practically important effect, justified variance estimate, allocation unit, randomization/blocking plan, replication, site-season scope, attrition plan and analysis that matches the design.

Three relevant sources motivate this work:

- Lark et al. (2025), [*Designing on-farm trials: an example with interventions to improve micronutrient status of grain crops*](https://www.nature.com/articles/s44264-025-00101-0), found that farm-level complete-block replication was critical to detect plausible effects with power at least 0.8, while regional precision required spatial replication across farm clusters. Its Ethiopia design is a methods example, not a sample-size prescription for Gauteng.
- Sichinga-Ligowe et al. (2024), [*An introduction to conducting responsible and reproducible agricultural research*](https://doi.org/10.1079/cabireviews.2024.0058), highlights replication, random allocation, blocking, power and matching analysis to design.
- McKenzie (2025), [*Designing and analysing powerful experiments*](https://doi.org/10.1111/1475-5890.70003), discusses how estimands, residual variance, intracluster correlation, compliance and attrition influence power in experiments.

## Paradigm shift

Make *evidence acquisition itself* an optimized, constrained engineering problem. Before a nutrient, breeding, irrigation or post-harvest intervention uses scarce farm resources, determine whether the planned experiment could answer its primary question. A trial that cannot distinguish a meaningful improvement from noise is not an inexpensive success; it is expensive ambiguity.

CF-027 ties CropFoundry's other gates together:

- CF-018 establishes whether treatment assignment and causal interpretation are defensible.
- CF-022 tests irrigation policy mechanics on synthetic traces, but cannot show field efficacy.
- CF-023/024/026 define the relevant post-harvest and nutritional outcomes and their measurement boundaries.
- CF-027 asks whether a real confirmatory design can estimate those outcomes with adequate power and coverage.

## Runnable preflight

From the Symthaea workspace root:

```bash
cargo run -q -p xtask -- cropfoundry-trial-power-plan \
  --input docs/research/cropfoundry/trial-power-input-example-v0.json \
  --out reports/cropfoundry/trial-power-example-v0.json

cargo test -p xtask cropfoundry_trial_power
```

The receipt pins the exact input bytes with SHA-256 and preserves the protocol and variance-evidence identifiers. These identifiers are not independently resolved by the tool. It also records the randomization method and declared blocking unit, while explicitly not modelling efficiency gains from blocking. The receipt includes side-by-side sample-size sensitivity for 80% and 90% target power, in addition to the requested target.

### Method supported in v0

The implementation supports a balanced, two-arm parallel design and uses the usual normal-approximation sample size for a difference between independent means:

`n0 = ceil(2 * (z_(1 - alpha_adj/2) + z_power)^2 * residual_sd^2 / minimum_effect^2)`

where `alpha_adj = familywise_alpha / co_primary_endpoint_count` uses a conservative Bonferroni adjustment. It then applies the simple equal-cluster design effect `DE = 1 + (plots_per_cluster - 1) * ICC`, inflates for expected attrition, and rounds the per-arm count up to whole clusters.

This is a screening approximation. It does **not** model blocked-trial variance reduction, spatial autocorrelation, repeated measures, unequal cluster sizes, nonlinear/binary outcomes, treatment interactions, non-inferiority margins, finite-cluster degrees of freedom, adaptive stopping, or a network of heterogeneous sites. The tool must emit those limitations, and the analysis must be redone with a design-specific method or simulation when they matter.

### Synthetic example

The checked-in example assumes (illustratively) a 0.5 kg-per-plot minimum effect, residual SD 1.0 kg, two-sided family-wise alpha 0.05, target power 0.80, five plots per farm cluster, ICC 0.10 and 10% attrition. The calculation gives:

- 63 independent-plot equivalents required per arm before clustering/attrition;
- design effect 1.4;
- 98 recruited plots per arm before whole-cluster rounding;
- 20 farm clusters × 5 plots = 100 plots per arm; **200 plots across both arms**.

Those numbers are arithmetic outputs under invented assumptions, not a proposed local sample size. If the residual SD, cluster size, ICC, attrition or minimum effect changes, the result changes. A statistician must confirm the final design with the intended analysis and the actual variance evidence.

## Benchmark gates and targets

1. **Protocol completeness:** all required fields present, exact input digest emitted, alpha/power and units validated, cluster/attrition arithmetic reproducible.
2. **Numerical validation:** standard-normal quantile test vectors, formula test vectors, multiple-endpoint adjustment test, cluster/attrition edge cases, fail-closed invalid input cases.
3. **Adequacy review:** provisional target power at least 0.80; also report 0.90 sensitivity. Use family-wise alpha no greater than 0.05 across declared co-primary endpoints. These are study-planning targets, not universal claims.
4. **External scope:** do not claim cross-site or cross-season transfer from a single site-season. At least two site-seasons is a provisional minimum signal for such a claim, but precision across a target region may require much wider and more representative spatial sampling.
5. **Field confirmation:** preregister estimand, allocation, blocking, exclusions, treatment compliance, measurement protocol and analysis before outcomes. Report estimates and confidence intervals regardless of significance; do not substitute post-hoc observed power for the effect estimate.

The correct benchmark is not “maximize sample size.” It is **maximize decision-relevant information subject to budget, access, farmer burden, site coverage and safety**, while preserving a design that can credibly answer the question.

