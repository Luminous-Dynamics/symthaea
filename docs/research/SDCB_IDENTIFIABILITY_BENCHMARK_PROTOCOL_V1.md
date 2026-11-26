# SDCB identifiability, intervention, and common-cause tournament protocol v1

**Status:** preregistration proposal; no tournament or physical experiment has been run.  
**Exact reference:** `scripts/research/sdcb_identifiability_oracle.rs`  
**Related work:** SDCB common-cause oracle #13804; EGBT joint common-cause ledger #13805; real-sensor plan #13802.

## 1. Primary question

Can an agent distinguish a scientific effect from a shared measurement-system cause, recognize when the available observations cannot do so, and choose a valid, affordable intervention that makes the explanations distinguishable?

Primary claim to test:

> Under matched candidate actions, evidence, budgets, and stopping rules, an identifiability-aware SDCB policy reduces false scientific/mechanistic conclusions and decision regret versus agreement, i.i.d. fusion, and information-gain-only policies, while preserving performance when sensors truly are independent.

The oracle is an evaluator/control, not a production scientific agent.

## 2. Task families

| Family | Ground-truth construction | Required behavior |
|---|---|---|
| EQ0 exact observational equivalence | Two different science/common-cause worlds produce identical distributions for every allowed observation | Report non-identifiability; do not infer a mechanism from repeated agreement |
| EQ1 intervention-separable | EQ0 plus one valid intervention with different world-conditioned distributions | Select the intervention when decision value and budget justify it |
| EQ2 near-equivalence | Observation distributions differ by very small TV distance | Quantify weak separation; do not equate nonzero TV with practical identifiability |
| EQ3 non-separating intervention | Candidate intervention leaves the paired-world distributions unchanged | Reject the claim that the intervention identifies the cause |
| EQ4 intervention contamination | The manipulation also changes the measurand or another causal parent | Detect/use an expanded world model; do not credit a violated exclusion assumption |
| EQ5 sham intervention | Sham and active intervention are randomized and labelled | Separate manipulation effect from handling/time/operator effects |
| EQ6 genuinely independent devices | Errors are conditionally independent, with no shared cause | Avoid unnecessary common-cause penalties and calibration expenditure |
| EQ7 correlated measurand | Physical field is correlated while measurement errors are independent | Do not misclassify signal correlation as shared sensor error |
| EQ8 multiple common causes | Shared environment, reference chain, batch, firmware/power, and local device effects coexist | Attribute only what is identifiable; preserve residual uncertainty |
| EQ9 reference-chain transfer | Devices share some, but not all, reference ancestry | Track chain identity and do not treat common ancestry as independent replication |

EQ0–EQ3 should first be generated as small exact categorical worlds. EQ4–EQ9 require richer structural models and/or measured metadata. Never label injected perturbations as observed physical failure mechanisms.

## 3. Matched comparator arms

All policy arms receive the same hypothesis set, observations, device registry, candidate experiments, cost vector, priors, and hard budget.

1. Majority vote and cross-device agreement.
2. I.i.d. Bayesian sensor fusion (deliberate negative control).
3. Hierarchical/random-effects shared-plus-local calibration.
4. Joint-state EIG over scientific, common-cause, and device states.
5. Hypothesis-only EIG and greedy EIG-per-cost.
6. EVSI / decision-focused acquisition with abstention.
7. Fixed reference schedule and check-standard/control-chart triggers.
8. Random feasible experiment/intervention.
9. Robust finite-scenario maximin.
10. Common-cause-aware SDCB.
11. Exact one-step and, later, exact two-stage finite oracles as evaluator references.

Do not compare EIG, EVSI, and maximin scores as if they shared a common unit. Compare realized decision utility, regret, cost, false-claim rate, and resource-quality frontiers.

## 4. Metrics

**Identifiability**
- paired-world equivalence detection accuracy;
- TV separation and, where suitable, Jensen–Shannon divergence;
- intervention discrimination power;
- fraction of declared causal alternatives remaining observationally indistinguishable;
- false identification rate on exact-equivalence controls.

**Scientific decision quality**
- expected utility and regret against the exact finite oracle;
- wrong-decision rate and selective risk;
- abstention rate, coverage, and risk-versus-coverage curve;
- false scientific-effect and false mechanism-claim rates.

**Measurement assurance**
- common and device-local posterior Brier/log scores where ground truth exists;
- interval coverage and calibration of predictive uncertainty;
- drift-detection delay and false alarms;
- reference-query count, reference-chain diversity, and calibration age/validity.

**Execution/evidence integrity**
- hard-budget violations;
- invalid, stale, wrong-device, wrong-chain, duplicate, or revoked receipt acceptance;
- zero-mutation rejection rate;
- intervention execution/receipt mismatch;
- replay agreement and provenance completeness.

Do not collapse the dimensions to a single weighted score. Report a Pareto frontier.

## 5. Statistical protocol and gates

- Freeze generators, world pairs, action sets, priors, costs, abstention utility, and stopping rules before policy tuning.
- Use complete mechanism families and device/reference-chain cohorts as held-out units; do not split adjacent time windows randomly.
- Start with at least 20 paired seeds per synthetic family; power the confirmatory run for a preregistered minimum effect and confidence-interval width.
- Use paired task-level comparisons, cluster bootstrap by world/task family, and multiplicity correction across confirmatory endpoints.
- Publish all seeds, rejected receipts, abstentions, resource use, and negative controls.
- Re-run a subset with an independent evaluator implementation.

Proposed targets, not achievements or industry standards:
1. 100% agreement with exact finite known-answer and paired-world equivalence controls.
2. Zero false identification claims on EQ0/EQ3/EQ4 negative controls.
3. Zero budget, data-leakage, or intervention-label violations.
4. At least 10% lower held-out decision regret than matched hypothesis-only EIG on preregistered EQ1/EQ2/EQ8 tasks, with paired 95% interval excluding zero.
5. Non-inferior decision value on EQ6/EQ7 independent/physical-correlation controls.
6. At least 10% lower reference/intervention cost at matched decision utility on one preregistered real-data track.
7. 100% zero-mutation behavior for invalid EGBT receipts.

A target is not a result. The 100% gates apply to finite exact controls; they must not be advertised as universal empirical guarantees.

## 6. Real-data integration without overclaiming

### Inha 2026 indoor sensor calibration dataset

Use the authors' S1 reference-efficiency, S2 location-transfer, S3 chronological-drift, and S4 event-conditioned tracks. Their paper compares iTransformer, TimeXer, DLinear, XLinear, TimeMixer, SenDaL, TESLA, and the authors' lightweight temporal model. The paper reports best average RMSE for its model on S1 across five variables, and strong transfer/drift/event results; the model's RMSE still degrades over time. Reproduce the authors' protocol before adding SDCB acquisition policies.

The dataset is suitable for reference-budget, location-transfer, drift, and event-conditioned selection. It does **not** provide direct ground truth for whether correlated residuals arose from a common reference chain, shared power/firmware, or a genuinely correlated physical measurand. Do not count its five locations as five independent laboratories or its low-cost/reference sensor pairs as a fully instrumented causal intervention study.

Dataset release: https://github.com/inhaKDD/Indoor-Sensor-Calibration-Dataset  
Paper: https://arxiv.org/abs/2610.11236

### UCI Gas Sensor Array Drift

Use the chronological batches for drift/channel robustness. Its 16 channels are a single sensor array, not 16 independent deployed instruments. The dataset terms are research-only and exclude commercial use; do not mirror or use commercially without permission.

### Additional field-study comparator

A 2026 multi-modal calibration/drift paper evaluates five custom devices at two Brussels sites, using source-device, intra-location, and inter-location transfer. It reports that transfer rankings depend on pollutant and that some transferred calibrations do not improve on the raw sensor. Its dataset is not openly archived in the paper's data-availability statement, and its injected drift is not observed physical ageing. Use it as a literature comparator, not as a downloadable benchmark unless access is granted.

## 7. How to compare with current practice

Separate three competitions:

1. **Prediction/calibration:** compare RMSE/MAE/bias, uncertainty coverage, drift degradation, location transfer, latency, memory, and energy with published models. Reproduce the upstream split and native metrics.
2. **Experiment selection:** compare fixed/random, representative, EIG, EVSI, hierarchical calibration, robust, and SDCB policies under identical evidence and reference budgets.
3. **Evidence qualification:** compare invalid-evidence rejection, provenance, replay, stale-context rejection, and zero-mutation behavior. Most predictor benchmarks do not test these properties.

The competitive objective is not to claim a universal win on a single RMSE column. It is to improve the decision-quality/resource Pareto frontier while lowering false scientific claims and maintaining calibration quality on independent-sensor and correlated-measurand controls.

## 8. Reproducibility record

Every run should pin:
- dataset release and asset digest;
- source commit and policy/model digest;
- raw data licence and redistribution permission;
- split manifest and reference-query transcript;
- calibration/model hyperparameters and training budget;
- candidate action registry and costs;
- task/world seeds and exact oracle outputs;
- hardware, software, latency, memory, and energy telemetry;
- all exclusions, failures, abstentions, and rejected receipts.
