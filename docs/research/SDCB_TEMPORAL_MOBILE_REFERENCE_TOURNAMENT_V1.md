# SDCB temporal covariance and mobile-reference tournament v1

**Status:** preregistration proposal. No raw dataset was downloaded, no external benchmark was run, and no physical sensor was tested in this change.

## 1. Primary hypothesis

At matched measurement budgets, a temporal, full-covariance, decision-aware planner can reduce target scientific uncertainty or decision loss relative to static covariance and IID baselines, while preserving nominal uncertainty coverage, calibration-chain validity and hard resource limits.

Test three distinct sources of gain:

1. temporal observation model (OU / state-space / unequal averaging windows);
2. covariance-model uncertainty (mixture and robust support-set handling);
3. adaptive selection (next measurement depends on the actual observation history).

Use a factorial design so an improvement is not incorrectly attributed to the planner when it came from a better calibration model.

## 2. Task families

| ID | Family | Required control |
|---|---|---|
| TC0 | Static IID and diagonal-covariance controls | Reproduce the earlier Gaussian references |
| TC1 | Scalar OU repeated readings | Repetitions inside one correlation time must not be counted as IID |
| TC2 | Overlapping and duplicate time windows | Covariance matrix must preserve overlap; duplicate windows must not create fictitious independent evidence |
| TC3 | Disjoint windows with variable gaps | Cross-covariance must decay with elapsed time and fitted tau |
| TC4 | Unequal averaging times | Fast/mobile and slow/reference observations must refer to their actual integration kernels |
| TC5 | Multiple shared OU factors | Separate environment/reference/power/clock effects with cross-factor covariance |
| TC6 | Device-local drift plus shared drift | Distinguish independent local errors from common-cause state |
| TC7 | Covariance hyperparameter uncertainty | Include within-model and between-model variance; test uncertain tau, variance and residual covariance |
| TC8 | Abrupt regime change / changepoint | Detect when a previously fitted covariance model no longer predicts |
| TC9 | Calibration/reset intervention | Model state changes and exclusion restrictions; do not assume perfect reset |
| TC10 | Partial reference-chain ancestry | Preserve which calibration links are shared and which are independent |
| TC11 | Process-vs-instrument confounding | Do not attribute real measurand autocorrelation to sensor bias |
| TC12 | Model misspecification | Heavy tails, nonstationary drift, nonlinear response and hidden common factors |
| TC13 | Adaptive two-stage reference choice | Compare contingent policy with best fixed second action under identical budgets |
| TC14 | Evidence and replay attacks | Wrong chain, stale state version, duplicate receipt, revoked verifier and expired context must reject atomically |

TC0–TC4 have analytic or direct finite controls. TC5–TC14 require their declared generator, state-space or independent reference evaluator. Publish each family separately.

## 3. Matched baselines

All methods receive identical training data, candidate actions, reference budget, timestamps, device identities, averaging windows, calibration ancestry, outcome resolution and terminal utility.

### Temporal and calibration baselines
- IID Gaussian fusion (deliberate negative control);
- static full-covariance Gaussian conditioning;
- scalar/multivariate OU Kalman filter;
- continuous-time Kalman filter / state-space model;
- Gaussian-process regression with matched kernel family;
- hierarchical random-effects calibration;
- fixed-interval check-standard/control-chart policy;
- OLS, GDR, X-GDR, Y-GDR, YB-GDR and C-GDR for the short-collocation regression task, under the paper's assumptions.

### Experiment-design baselines
- random feasible and fixed-cadence reference allocation;
- A-, D-, c- and E-optimal design where objectives apply;
- Fisher-information / profile-likelihood design for nonlinear models;
- expected information gain;
- one-step and finite-horizon EVSI;
- R-IDeA / robust decision-aware design where the assumptions match;
- plug-in covariance maximin and Bayesian model averaging;
- exact full-covariance and exact finite adaptive oracles on small instances;
- HDC/CfC policy evaluated only against the exact references.

### Edge uncertainty comparator
Kaparounakis and Stanley-Marbell's 2026 processor-native uncertainty propagation is a comparator for uncertainty-propagation latency, memory and energy at equal output quality. Their reported 42.9x and 94.4x speedups are on their FPGA platforms and their specified thermal-sensor workload. They are not a measurement-selection result and must not be compared with SDCB wall-clock results without a matched hardware, accuracy and workload protocol.

Digital metrological twins and self-calibrating hydraulic sensor networks (dos Reis et al., 2026) are architectural comparators. That paper reports no research dataset, so it is not a numeric benchmark baseline.

## 4. Real-data anchors and strict limits

### 4.1 Kok & Tabandeh short-collocation study

Reproduce the paper's model and simulation cases before proposing an improvement. Its central challenge is a fast sensor briefly collocated with a slower reference while the physical process changes during unequal averaging periods. Compare OLS and the paper's GDR variants, including C-GDR, only on matched assumptions and parameter regimes.

Preserve the paper's conditional finding: OLS can work when reference uncertainty is much smaller; C-GDR can improve results when uncertainty is more comparable and process variability is larger. Do not claim C-GDR is universally superior.

### 4.2 Inha multi-location indoor air-quality dataset

Use the S1–S4 scenarios in the existing SDCB_REAL_SENSOR_BENCHMARK_PLAN_V1.md:

- reference-efficient calibration;
- unseen-location transfer;
- chronological drift;
- event-conditioned shifts.

The current release describes five locations within one laboratory, not five independent laboratories. Real indoor measurand covariance and sensor-error covariance must remain separate. Use forward-only chronological splits and leave-one-location-out rotations. Confirm release-asset data terms before redistribution or commercial use.

### 4.3 UCI Gas Sensor Array Drift

Use the ten chronological batches for channel-drift and concentration-transfer tasks. The 16 channels are in one array; they are not 16 independent deployed devices. The current official UCI page is internally inconsistent: its narrative says the dataset is for research purposes and excludes commercial use, while its structured licence field states CC BY 4.0 and permits reuse for any purpose. Treat the licence as unresolved; quarantine this dataset from redistribution and commercial use until UCI and/or the dataset authors clarify which terms govern.

### 4.4 Controlled synthetic counterfactual track

Create known common-cause mechanisms with explicit ground truth: shared reference drift, common power/clock, common environment, device-local ageing, and changes in the actual measurand. Hold out mechanisms and combinations, not merely random rows. This is the only track where the generator can authoritatively label the causal source.

## 5. Metrics — report separately

**Scientific and calibration quality**
- target bias and RMSE;
- nominal 90% and 95% interval coverage by device, location, time block, regime and task family;
- interval width / sharpness;
- predictive log score and CRPS where distributions are available;
- parameter- and decision-specific uncertainty, not only joint entropy.

**Measurement assurance**
- false independent-confirmation rate;
- stale calibration / wrong ancestry acceptance rate;
- invalid intervention acceptance rate;
- evidence rejection atomicity and exact replay;
- reference-chain uncertainty contribution retained in the final result.

**Temporal performance**
- drift detection delay and false-alarm rate;
- calibration degradation after change point;
- time to recover a target error/coverage tolerance;
- sensitivity to integration-window mismatch and overlap.

**Decision and resources**
- expected terminal utility and regret against exact oracle;
- target variance or decision loss per unit reference cost;
- total and maximum path cost;
- measurement count, elapsed time, memory, latency and energy;
- number of physical reference collocations and travel/setup cost.

No weighted scalar may compensate for a false traceability claim, invalid evidence acceptance, safety/authority violation or hard-budget breach.

## 6. Proposed gates (targets, not achievements)

- 100% agreement with frozen exact analytic covariance and finite-policy references within declared numerical tolerances.
- Zero invalid windows, covariance matrices, stale receipts or out-of-budget paths accepted in negative controls.
- Zero false independent-confirmation claims in shared-cause and duplicate-window controls.
- Nominal 90% and 95% coverage assessed by task family and chronological holdout, with confidence intervals published; pooled coverage alone is insufficient.
- Approximate policy regret no greater than 2% on tractable held-out exact-oracle tasks.
- At least 10% lower target decision loss per equal reference cost than the strongest matched static/IID baseline on preregistered held-out common-cause and short-collocation families; paired confidence intervals must exclude zero before making the claim.
- Non-inferiority to the strongest matched temporal calibration baseline on native real-data tasks.
- No empirical SOTA statement until paper baselines, dataset protocols and data licences are independently reproduced.

These are internal research gates, not standards or demonstrated results.

## 7. Statistics and leakage controls

- Use chronological train/validation/test splits; never randomly mix adjacent minute-level records across splits.
- Use block bootstrap or episode-level resampling to preserve autocorrelation.
- Hold out entire devices, reference chains, locations, time periods and common-cause mechanisms where the dataset permits.
- Report each task-family result and each location/device; do not publish only a pooled mean.
- Preregister task generators, seeds/commitments, outcome bins, model families, hyperparameter search budget, costs, utilities and statistical tests.
- Freeze the calibration model when comparing selection policies, and freeze the selection policy when comparing calibration models.
- Treat scenarios generated from observed residuals as estimated scenarios, not physical causal labels.
- Keep synthetic, paper-reported, externally reproduced, locally executed, hosted-CI and physically measured evidence classes distinct.

## 8. Competitive differentiation

The contribution to test is not a new claim to OU models, Kalman filters, optimal experimental design, C-GDR, self-calibration, digital metrological twins or processor-native uncertainty arithmetic. Those are existing research directions.

The potential SDCB differentiation is the jointly evaluated chain:

**time-windowed process model → shared/local dynamic state inference → covariance-model uncertainty → target-decision-aware adaptive measurement choice → hard pathwise budget → reference-chain evidence qualification → independently replayed belief transition.**

An improvement claim must survive the factorial controls and the strongest matched baseline, not merely an IID baseline.
