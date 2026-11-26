# SDCB real-sensor calibration and drift benchmark plan v1

**Status:** executable evaluation plan; no raw dataset has been downloaded or benchmarked in this change.  
**Parent matrix:** SDCB scientific-agent benchmark matrix.  
**Related references:** instrument-aware design #13762; drift-aware multi-instrument design #13793; common-cause design #13804; common-cause evidence transition #13805.

## 1. Purpose and claim boundary

Synthetic finite oracles are necessary for exact correctness, identifiability, calibration-leakage, common-cause, and atomicity controls. They are not sufficient to show that an experimental-design agent improves calibration on real measurements.

This plan adds two real-data evaluation anchors with complementary properties:

1. **Inha multi-location indoor air-quality calibration dataset (2026):** paired low-cost/reference measurements, five locations, six months, context metadata, and four author-defined evaluation scenarios.
2. **UCI Gas Sensor Array Drift datasets:** long-term chemical-sensor-array drift, concentration labels in the extended variant, and ten chronological batches.

The intended claim is narrow:

> Under matched reference-measurement budgets, SDCB's decision-aware policies reduce calibration/decision loss or reference consumption while maintaining or improving held-out calibration quality under location, time, and event shift.

This plan does not claim that either dataset establishes a real common-cause ground truth, nor that a finite binary latent-state reference is a complete physical sensor model. The SDCB synthetic common-cause challenge remains a separate, labelled counterfactual subtrack.

## 2. Dataset inventory and intended roles

### 2.1 Inha multi-location indoor air-quality calibration dataset

Official repository: https://github.com/inhaKDD/Indoor-Sensor-Calibration-Dataset  
Paper: https://arxiv.org/abs/2610.11236  
Release: https://github.com/inhaKDD/Indoor-Sensor-Calibration-Dataset/releases/tag/Dataset_v1

The repository describes a six-month collection from 6 January to 6 July 2026 in one university laboratory, with five locations (East, West, South, North, Center), CO2, CO, PM1, temperature, and relative humidity, 1-second acquisition resampled to 1-minute intervals, paired low-cost and reference measurements, and door/window/occupancy/business-hours context.

The authors define four relevant scenarios:
- S1: reference-efficient calibration across observed locations;
- S2: transfer to an unseen location;
- S3: long-term drift;
- S4: event-conditioned shifts, including door/window changes.

The public release metadata identifies asset AAAI27_sensors_calibration_datasets.zip, 717,543,457 bytes, with SHA-256 2327515344c9abf4cb8373ffbdc1e7fda6b11774d4bf25252eced334948c581a. Pin this exact release asset and verify the digest after download. Do not silently switch to a later release or a paper-derived reconstructed copy.

**License caution:** the GitHub repository exposes an MIT LICENSE, but the dataset ZIP is a separate release asset and the repository listing does not clearly state a distinct data licence/redistribution grant. Before redistribution or use in a commercial product, inspect the release/package terms and seek clarification from the dataset authors if necessary. Until then, download from the official release for permitted research use and do not mirror the raw archive.

**What it can test:** reference-budget allocation, temporal calibration, cross-location transfer, event-conditioned calibration, uncertainty under observed shifts, and how policy choice changes calibration quality per reference query.

**What it cannot establish by itself:** that a residual correlation is caused by a shared calibration chain, power supply, firmware, or environmental sensor error. The locations share one laboratory and real indoor air quality can itself be spatially/temporally correlated. Distinguish correlation in the measurand from correlation in sensor error.

### 2.2 UCI Gas Sensor Array Drift datasets

Primary dataset: https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset  
Concentration-labelled extension: https://archive.ics.uci.edu/dataset/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations

The archive contains 13,910 measurements from a 16-channel metal-oxide gas-sensor array, six analytes, and a 36-month collection period. Data are organized into ten batches corresponding to chronological periods. The extended dataset includes gas concentration labels and a 128-feature representation (eight features per channel).

**Critical interpretation:** this is a 16-channel array in a controlled gas-delivery platform, not 16 independent deployed instruments. Use it for sensor-channel drift, temporal batch transfer, concentration prediction/classification, and channel-corruption robustness. Do not use it to claim independent-device replication or a physical network common-cause benchmark.

**License caution:** the UCI dataset description explicitly limits use to research purposes and excludes commercial purposes. Cite the required original calibration/drift papers. Obtain permission before any commercial use.

### 2.3 Optional follow-on data

The 2026 review of metal-oxide sensor drift identifies additional long-duration and gas-mixture datasets, including a 62-sensor long-term e-nose series and dynamic/turbulent gas-mixture data. Treat these as follow-on candidates only after checking their source, access conditions, labels, and licence. They are not prerequisites for v1.

## 3. Evaluation tracks

Keep the dataset authors' native tasks and metrics intact. Add SDCB policy evaluation as an additional track; do not relabel a calibration RMSE score as an experimental-design win.

### Track A — reference-efficient calibration (Inha S1)

**Question:** how much held-out calibration quality can be obtained from a constrained number of reference measurements?

- Reveal low-cost sensor readings and authorized context to all policies.
- Hide reference measurements from the agent except when selected as a calibration/query action.
- Reserve a fixed, inaccessible reference test set for evaluation.
- Evaluate reference-query budgets at preregistered fractions (for example 1%, 5%, 10%, and 20% of eligible training reference records); also report absolute query counts.
- Compare random, fixed-cadence, representative stratified, uncertainty-only, EIG-per-cost, one-step EVSI, R-I/R-IDeA where its assumptions fit, robust decision-aware SDCB, and the exact small-instance oracle on a downsampled subset.
- Score prediction/calibration on the same hidden test records at each budget.

### Track B — location transfer (Inha S2)

**Question:** can a policy decide where a limited number of reference observations are most valuable when moving to an unseen location?

- Use leave-one-location-out evaluation over all five locations.
- The held-out location's low-cost stream/context is visible according to the upstream protocol; its reference values remain hidden except for explicitly budgeted adaptation queries.
- Train/validate only on the other four locations.
- Report zero-shot transfer and budgeted adaptation separately.
- Do not randomly split minute-level rows across train and test; that leaks near-duplicate temporal context.

Five rotations are the primary outer test units. Report each location, not only the mean.

### Track C — temporal drift and calibration freshness (Inha S3; UCI batches)

**Question:** when is a reference observation stale, and when is re-calibration worth its cost?

- Use forward-only chronological splits. Earlier periods train; later periods are evaluation. No future record can influence preprocessing, normalization, hyperparameters, model selection, or calibration thresholds.
- For Inha, use the authors' S3 protocol where reproducible; otherwise preregister chronological cutoffs and disclose deviations.
- For UCI, preserve the ten batch order; use expanding-window forward chaining. A final batch may be held out once as a confirmatory test after all choices are frozen.
- Compare fixed recalibration schedules, check-standard/control-chart triggers, predictive-uncertainty triggers, drift-aware instrument-state policies, and no-recalibration controls.
- Evaluate drift detection delay, false alarms, calibration degradation, reference queries consumed, and time/cost to recover a preregistered error tolerance.

### Track D — event-conditioned calibration (Inha S4)

**Question:** do abrupt environmental changes invalidate a previously adequate calibration or change which reference measurement is most useful?

- Build event windows from door/window context and other metadata that is actually provided.
- Keep event labels and time windows disjoint across training, validation, and test episodes.
- Compare event-blind calibration with event-aware calibration and active reference selection.
- Do not infer a causal effect of door/window changes from observational metadata alone. Report event-conditioned associations unless an intervention or defensible causal design supports a causal claim.

### Track E — common-cause challenge (controlled counterfactual, separate from real-data scores)

Neither real dataset provides a complete labelled ground truth for all shared calibration-chain, power, firmware, batch, or environmental causes. Therefore:
- start from real records, preserve the unmodified real-data score as the primary result;
- create a separate synthetic challenge with known shared perturbation g_t and device/channel-specific perturbations c_d,t;
- vary common-mode bias, common-mode drift, local drift, independent noise, measurand correlation, and missing/corrupted channels independently;
- use paired random seeds and publish the perturbation generator and parameter distribution for the public development suite;
- reserve hidden generator seeds and mechanism combinations for confirmation;
- never combine the counterfactual results with the unmodified real-data calibration RMSE.

This tests whether the agent detects and handles a known shared-cause intervention. It does not turn synthetic perturbations into observations of real physical common causes.

## 4. Matched baselines

### Calibration / prediction models

For the Inha dataset, include the authors' published lightweight temporal model and the named comparators available in the paper/release, including iTransformer, TimeXer, DLinear, XLinear, TimeMixer, SenDaL, and TESLA where reproducible. Confirm code, hyperparameters, preprocessing, and split implementation before marking a baseline reproduced.

For UCI, include the published drift-compensation baseline families and the 2026 LiteSensor-Net/KD-DM reference if its implementation and chronological protocol can be independently reproduced. Its reported metrics are classification/edge-deployment results, not experimental-design results; compare them only on their declared task.

Add simple and interpretable controls:
- identity/no-calibration and static affine calibration;
- per-location/per-device calibration;
- pooled calibration;
- hierarchical/random-effects calibration;
- persistence and fixed-cadence recalibration;
- check-standard/control-chart recalibration.

### Experiment-selection policies

Under matched candidate pools, reference access, resource budgets, and stop rules:
1. fixed reference schedule;
2. random reference selection;
3. stratified/representative reference selection;
4. prediction-disagreement/query-by-committee;
5. EIG and EIG-per-cost;
6. R-I/R-IDeA where target-representativeness/proxy assumptions are met;
7. one-step EVSI;
8. robust finite-scenario/maximin EVSI;
9. multi-fidelity/reference-cost-aware policy;
10. common-cause-aware instrument policy;
11. exact finite oracle on downsampled episodes.

The calibration model and the experiment-selection policy are separate factors. Run a factorial comparison where feasible: each policy should be paired with the same downstream calibration model, and each calibration model should be paired with the same query policy. This isolates whether gains come from better calibration modelling or better experiment selection.

## 5. Metrics

Report separate score families:

**Calibration and prediction**
- RMSE and MAE in the original physical units, per variable and location;
- normalized RMSE only as a secondary aggregation (state the normalization);
- predictive log score/Brier where the output is probabilistic;
- interval coverage and width, calibration error, and tail-error metrics;
- calibration performance before and after shift.

**Decision quality**
- declared task utility and Bayes decision regret;
- EVSI / robust value of information;
- reference-query cost to a preregistered calibration tolerance;
- regret relative to exact finite oracle on downsampled episodes;
- abstention and escalation quality.

**Measurement assurance**
- calibration freshness and drift detection delay;
- false recalibration alarms and missed drift;
- common/local state recovery only on tasks where the corresponding truth is known;
- false independent-confirmation rate on the injected shared-cause track;
- uncertainty coverage with and without common-mode perturbations.

**System and resource efficiency**
- reference queries, number of calibration actions, compute, wall time, peak memory;
- energy only with measured hardware telemetry;
- branch-wise maximum resource consumption as well as expected cost;
- replay success and evidence integrity once EGBT is connected.

Do not collapse these into one score. A model may win calibration RMSE while losing query efficiency, or win accuracy while making unjustified confidence claims.

## 6. Leakage, uncertainty, and statistical protocol

- Pin release URL/tag, raw archive digest, preprocessing revision, split manifest, and generator seed commitments.
- Treat the time series as grouped observations. Keep contiguous temporal blocks, event episodes, and location groups intact across splits.
- Fit preprocessing, feature scaling, imputation, and calibration transformations only on training data.
- Keep reference test values inaccessible to the policy, calibration model, hyperparameter tuning, and stopping rule.
- Use paired query-policy seeds on the same eligible reference candidates.
- Bootstrap by location, event episode, or temporal block—not by individual minute rows.
- Report all five location rotations and all ten UCI batches where applicable.
- Freeze baseline implementations and tuning budgets before confirmatory runs.
- Publish failed, aborted, and high-cost runs; do not report only the best seed.
- Use confidence intervals and preregistered multiplicity correction across policy/model comparisons.

For a single five-location dataset, leave-one-location-out is informative but is not broad cross-building generalization. Require at least one independently collected site/dataset before claiming transfer to unseen buildings or regions.

## 7. Proposed acceptance gates

These are SDCB project targets, not established industry standards or achieved results:

1. Exact agreement with the finite oracle on all small known-answer episodes.
2. Reproduction of the dataset's published split and baseline scores within a preregistered tolerance before adding a SDCB claim.
3. Non-inferior held-out RMSE/MAE versus the strongest reproduced calibration model at the same reference budget.
4. At least 10% lower reference-query cost to achieve the same preregistered error tolerance than the strongest matched fixed/EIG-only policy on at least one shift-sensitive track, with paired confidence intervals reported.
5. At least 10% lower decision regret than the strongest matched EIG-only policy on a preregistered task-utility evaluation, without worsening calibration coverage.
6. Zero false independent-confirmation claims on the hidden common-cause challenge.
7. Zero invalid/stale/misbound evidence mutations in the EGBT negative-control suite and at least 99% acceptance on valid receipt controls.
8. Independent reproduction on a second dataset or by a second implementation before any SOTA claim.

A synthetic common-cause win is not a real sensor-network result. A calibration-model win is not automatically an experiment-selection win. An upstream benchmark score must retain its native definition.

## 8. Data access and licensing status

- **Inha dataset:** public GitHub repository and release exist; the repository displays an MIT LICENSE. The release ZIP does not have a clearly separate data-licence statement in the repository listing. Check the dataset-specific terms before redistribution or commercial use; do not infer that all data inherits the code licence.
- **UCI Gas Sensor Array Drift:** UCI's dataset description says research use only and excludes commercial purposes. Cite the required papers; obtain permission before commercial deployment.
- No raw dataset was downloaded or processed when this plan was authored. Data hashes, row counts, missingness, exact split files, and reproduction of the published baselines remain execution gates.

## 9. Sources

- Inha dataset repository and scenario descriptions: https://github.com/inhaKDD/Indoor-Sensor-Calibration-Dataset
- Inha dataset release: https://github.com/inhaKDD/Indoor-Sensor-Calibration-Dataset/releases/tag/Dataset_v1
- Yun et al., *Low-Cost Sensor Calibration for Indoor Air Quality Monitoring: A Dataset, Evaluation Scenarios, and a Lightweight Model* (8 Oct 2026): https://arxiv.org/abs/2610.11236
- UCI Gas Sensor Array Drift Dataset: https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset
- UCI Gas Sensor Array Drift at Different Concentrations: https://archive.ics.uci.edu/dataset/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations
- Kim et al., *Drift-Robust Lightweight Deep Learning on Open Gas Sensor Benchmarks: A Reproducible Architecture Study with CBRN Applicability Mapping*, *Molecules* 31(11), 1884 (1 June 2026): https://doi.org/10.3390/molecules31111884
