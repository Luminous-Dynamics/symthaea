# Aviation Public-Data Evaluation Protocol (APDEP-001)

**Status:** proposed evaluation contract; not yet implemented or evaluated  
**Companion:** [AECB-001](./AVIATION_ENTERPRISE_BENCHMARK.md), [forecast-to-optimization track #8626](https://github.com/Luminous-Dynamics/symthaea/issues/8626)  
**Purpose:** compare aviation forecasts and decision policies without confusing predictive accuracy, simulator outcomes, causal effects, and operational savings.

## 1. Falsifiable claims

This protocol separates three claims that must never be merged into a single score.

1. **Forecast claim:** with only information available at a declared decision time, a model predicts future flight outcomes better calibrated baselines on a locked temporal test set.
2. **Policy-replay claim:** given the same forecasts, objective, resources, and declared simulated action-effect model, a policy obtains better replay value than competing policies. This is conditional on the model, not evidence that the intervention works in an airline.
3. **Operational-effect claim:** a real intervention reduces harm in practice. Public retrospective flight records alone do not identify this causal effect; it requires credible quasi-experimental identification or a prospective controlled rollout with operator authorization.

**Primary null:** the proposed method adds no decision value over the strongest feasible baseline under identical information, budgets, hard constraints, and evaluation cases. The benchmark must be able to retain that null.

## 2. Data and permissible scope

### Public outcomes

Use the U.S. DOT Bureau of Transportation Statistics (BTS) *Reporting Carrier On-Time Performance (1987-present)* data as the principal public source: https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ. The BTS table describes monthly reporting-carrier data and includes scheduled/actual times, delays, cancellations, diversions, taxi times and cause fields; the public page reports over 235 million records and 109 fields, and its latest-month availability changes over time. Record the source access date, selected data month, exact field list, download URL, checksum, and any provider corrections for each run. Reference download page: https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr+cr4s14zn0pr&gnoyr_VQ=FGJ.

BTS offers outcome data, not a complete operational-control record. It generally does **not** expose the full decision-time state of crew legality, passenger itineraries, qualified part inventory, maintenance work packages, gate-resource assignments, actions considered, rejected alternatives, or the interventions an operator actually chose. Never infer these latent fields from the final outcome.

NASA's Turbofan Engine Degradation Simulation-2 is a separate run-to-failure prognostics dataset generated using C-MAPSS and described as data under realistic flight conditions: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/. It can support a clearly separate engine-health sub-benchmark. It is not airline-wide maintenance work-order data, and it does not supply an airline's dispatch/maintenance authority or lifecycle cost ledger. Keep engine-health scores out of the flight-operations score unless a predeclared, validated mapping exists.

### Dataset tiers

- **Tier P (public observed outcomes):** BTS/DOT observations for forecast quality and retrospective outcome stratification only.
- **Tier S (simulated decisions):** explicit scenarios and action-effect assumptions for counterfactual policy comparison; report sensitivities to those assumptions.
- **Tier O (authorized operator evidence):** timestamped operational decision logs, resource constraints, interventions, costs, and outcomes under data-use approval. Required to make credible operational-effect claims.

Agreement between BTS and DOT reporting layers is a consistency diagnostic, not independent replication of the same operating system.

## 3. Decision-time contract and leakage prevention

Every run declares a decision timestamp and a feature-availability rule before training. Every field must be classified as: known at schedule publication, known at decision time from historical aggregates, known only during/after operation, or provenance/identifier only.

Prohibit as predictive features any field that is produced by or revised after the decision time unless the intended task explicitly forecasts a later decision with a well-defined update time. In particular, actual arrival/departure times, realized delay minutes, actual wheels-off/on, realized cancellation/diversion status, reported delay/cancellation causes, and post-event summaries are target/evaluation data—not predictors for a pre-departure policy. Do not quietly use them through rolling aggregates.

Mandatory safeguards:

1. **Time-respecting splits:** no random flight-level split as the headline result. Train on past periods, tune on a later validation interval, and evaluate on a locked future interval. Report exact month boundaries.
2. **Grouped generalization:** in addition to the temporal holdout, report held-out airport, carrier, and airport-week/carrier-month strata where sample size supports them. Keep related flights and derived aggregates on the correct side of the split.
3. **As-of aggregates:** compute carrier, route, airport, and tail-history features using only prior observations. Unit-test the cutoff at the exact timestamp, including timezone and overnight-flight cases.
4. **No target leakage in preprocessing:** fit imputation, encoders, scaling, calibration, feature selection, and threshold selection on training/validation data only.
5. **Negative controls:** add a deliberately future-derived canary feature. The pipeline must flag it or the test must fail; it must not silently report an improved model.
6. **Auditability:** save feature schema, availability class, dataset checksums, split manifest, training configuration, random seeds, dependency lock, model digest, and metric implementation version in a run manifest.

If the 2026 public forecast-to-optimization paper is used as a reproduction target, preserve its published objective, carrier-month decision unit, panel construction, budget semantics, and evaluation formula before comparison. Do not tune a different objective until it reproduces the reported 97.95%–99.60% oracle-value range. The range is specific to that paper's setting; it is not a generic aviation SOTA threshold.

## 4. Forecast tracks and baselines

Do not collapse all outcomes into a single “readiness” number. Define distinct tasks with separate targets and metrics:

| Task | Required baseline ladder | Primary measures |
|---|---|---|
| Arrival/departure delay | seasonal historical median/quantiles; route/airport/carrier historical baseline; calibrated regression model; strong boosted-tree baseline | MAE, RMSE, quantile loss, interval coverage, tail error |
| Cancellation/diversion risk | historical base-rate; regularized logistic model; strong calibrated tree model | Brier score, log loss, PR-AUC, calibration by regime |
| Carrier-month service-harm prediction (paper reproduction) | exact published baseline; simple equal allocation; historical-rate allocation | objective fidelity, calibration, ranking fidelity, regret vs hindsight oracle |
| Engine-health/prognostics (separate NASA track) | published dataset-specific baselines and prior-art protocols | dataset-specific prognostic metrics; no conversion to airline savings without a validated bridge |

Use comparable feature availability, tuning budgets, training windows, and compute reporting for all models. Report calibration by carrier, airport, season, and disruption regime—not only global accuracy. A ranking improvement is not a decision improvement unless the declared allocation objective also improves.

## 5. Decision-policy ladder

For each frozen decision instance and objective, compare:

1. do nothing / no discretionary intervention;
2. equal or proportional allocation under the same budget;
3. historical rule-based allocation;
4. simple score/rank policy using the forecast from the best calibrated baseline;
5. forecast-then-optimize with a transparent exact or standard solver where tractable;
6. integrated resource-constrained model (when its data and assumptions are genuinely available);
7. proposed Symthaea-guided policy, using the same input information and consuming no additional action budget;
8. hindsight oracle, explicitly labeled **non-deployable upper bound**, never a production competitor.

The optimizer, solver, or Symthaea layer must not see test outcomes during selection. A model-generated recommendation is advisory; only the independent deterministic checker decides whether modeled hard constraints are satisfied. Safety- or legality-critical constraints are never converted into soft penalties to improve the score.

## 6. Objective and harm accounting

Publish the complete objective equation and units before scoring. Keep mutually exclusive service-harm categories mutually exclusive: for example, a cancelled flight must not also be counted as a non-cancelled late arrival unless the published reference objective explicitly defines that overlap. Preserve component ledgers so weights cannot hide a regression.

At minimum report:

- total objective and each cost/harm component;
- cancellations, diversions, late arrivals, and severe-delay tail rates;
- resource budget consumed and number/fraction of decisions changed;
- performance by carrier, airport, season, geography, disruption type, and missing-data regime;
- mean, median, 90th/95th/99th percentile loss and conditional value-at-risk (CVaR) at declared tail levels;
- infeasible-plan rate, hard-constraint violations (target: zero), abstention/indeterminate rate, and stale/missing-evidence rejection count;
- wall-clock, peak memory, input size, and deterministic replay status.

Do not call a weighted score “cost savings” unless the monetary cost mapping is sourced and validated. If the map is hypothetical, label the objective in normalized or synthetic units.

## 7. Policy evaluation and uncertainty

Run budget fractions 5%, 10%, 20%, and 30% when reproducing the carrier-month paper, while preserving that paper's exact denominator and allocation rules. For all new objectives, state the resource unit (e.g., actions available per carrier-month), eligible population, and capacity constraints before running.

Report at least three comparisons separately:

- **Prediction value:** forecast metrics vs the strongest forecast baseline.
- **Decision value:** realized *replay* objective under a fixed declared simulator/action-effect model vs baselines.
- **Causal value:** measured intervention effect in Tier O data. Never infer this from forecast improvement or synthetic policy replay.

Use paired comparisons on identical cases and seeds. Bootstrap at the correct cluster level (such as carrier-month or airport-week) rather than treating correlated flights as independent. Show confidence intervals and effect sizes, not just point estimates. Predeclare primary outcomes, stress regimes, exclusion rules, and stopping criteria before inspecting final test results. If several policies or metrics are searched, disclose the search and correct for selection where appropriate.

The hindsight oracle is only meaningful when it uses the same objective and constraints with realized information; it is not an achievable baseline. Report regret to the oracle plus absolute harm components, because a strong oracle-relative ratio can coexist with a poor absolute result.

## 8. Scenario and robustness protocol

Public outcome data cannot estimate the effect of an action that was never observed or identified. In Tier S, declare the action-effect model and explore a sensitivity grid rather than choose a single convenient effect size. Vary disruption severity, forecast error, resource scarcity, action effectiveness, lead time, cost ratio, and correlated multi-flight shocks.

Mandatory metamorphic/fault-injection cases include:

- remove a qualified part, legal crew, compatible aircraft, or maintenance window: no policy may return an infeasible plan;
- make an evidence record stale, unsigned, mismatched, or unavailable: the verifier must reject or return indeterminate, never silently accept;
- increase a declared action cost or resource requirement while holding other inputs fixed: the optimal objective must not improve because of the worsening prerequisite (where the model semantics guarantee this);
- shift the forecast distribution without changing the truth: report calibration deterioration and decision regret;
- introduce correlated disruptions and train/test regime shift: measure tail loss and failure modes;
- compare perfect forecasts, noisy forecasts, and miscalibrated forecasts to separate optimization quality from forecast quality;
- test the policy when every available plan is worse than abstaining: the system must be able to recommend no action.

A simulator result is valid only for its declared equations and assumptions. No synthetic result is evidence of airworthiness, dispatch authorization, maintenance release, regulatory certification, causal savings, or performance on an airline's actual operation.

## 9. Targets: what “better than SOTA” means here

Targets are gates to attempt, not results already achieved:

- **Leakage:** zero known look-ahead features; all temporal-split and future-canary tests pass.
- **Constraint validity:** 100% valid or explicitly rejected/indeterminate outputs in the frozen benchmark; zero silent hard-constraint violations.
- **Baseline competitiveness:** beat the strongest feasible forecast and policy baseline on the predeclared primary outcome, with a paired 95% interval excluding no improvement, and no predeclared severe-tail or subgroup regression.
- **Practical significance:** establish a minimum worthwhile effect with domain experts before testing; do not retrofit a universal 5% threshold to objectives with different scales.
- **Reproduction:** match the published result only to the extent the paper's protocol can be faithfully reproduced; publish a deviation ledger when it cannot.
- **Robustness:** publish regret and CVaR under predefined stress/shift grids; uncertainty-aware policies must prove their net value after any conservatism cost.
- **Runtime:** report latency and memory as functions of instance size, including an explicit deadline; optimality claims are restricted to instances where exactness is established.
- **Independent audit:** a separate implementation (preferably outside the producer's repository and maintained by a different party) checks plans, costs, manifests, and negative controls before promotion from research.

A 5% improvement may be a useful *synthetic screening target* for AECB, but is not a general public-data SOTA threshold. Absolute outcome improvements, forecast metrics, and oracle-value ratios are different quantities and must not be merged.

## 10. Promotion gates and claim language

A run can be promoted only after the input/data manifest, feature availability ledger, objective, baseline set, locked split, executable command, independent verification results, and artifact digests are published.

Suggested claim vocabulary:

- **“Implemented”** — code exists at a specified revision.
- **“Executed”** — exact command, environment, input digest, exit status, and output artifact are recorded.
- **“Reproduced”** — a published protocol was reconstructed and deviations quantified.
- **“Improved in retrospective evaluation”** — a locked observed-outcome evaluation beat specified baselines under stated uncertainty.
- **“Reduced operational harm”** — authorized Tier O evidence and credible causal design support the claim.
- **“Certified/approved”** — only when the relevant competent authority and scope support that exact claim.

Do not describe a skipped CI run as a pass, a queued run as evidence, an optimizer objective as monetary savings, or retrospective correlations as causal intervention effects.

## 11. External references

- BTS, *Reporting Carrier On-Time Performance (1987-present)*: https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ
- BTS, field-selected download: https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr+cr4s14zn0pr&gnoyr_VQ=FGJ
- BTS 2026 reporting directive: https://www.bts.gov/explore-topics-and-geography/modes/aviation/number-40-technical-directive-reporting-time
- Cha and Kwon (30 September 2026), *Data-driven optimization for airline service quality improvement under delay and cancellation risk*: https://doi.org/10.3934/jimo.2026188
- Purohit, Feuerle and Hecker (19 October 2026), *Pre-arrival OCC decision support for aircraft turnaround coordination*: https://doi.org/10.3389/frai.2026.1977034
- NASA PCoE, *Turbofan Engine Degradation Simulation-2*: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
