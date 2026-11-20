# AIRROAD battery prognostics benchmark

**Programme:** AIRROAD-BATTERY-PROGNOSTICS-001  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11584  
**Status:** benchmark contract; no reproduction or model improvement is claimed

## Parent and composition

- Mission-conditioned battery capability and mass-coupled sizing: [#7805](https://github.com/Luminous-Dynamics/symthaea/issues/7805).
- Cross-fidelity electric-aircraft validation: [#11445](https://github.com/Luminous-Dynamics/symthaea/issues/11445).
- Coupled mission electro-thermal solver: [#11411](https://github.com/Luminous-Dynamics/symthaea/issues/11411).
- Mission operating policy and repeated sorties: [#11467](https://github.com/Luminous-Dynamics/symthaea/issues/11467).

## Purpose

Create a reproducible, leakage-safe battery-prognostics benchmark using public eVTOL cell telemetry, evaluating state of charge (SOC), state of health (SOH), and threshold-based remaining useful life (RUL) under phase-aware mission duty histories.

This programme owns dataset reconstruction, prediction-time feature rules, target construction, leakage-safe cell-level splits, baseline comparisons, uncertainty and decision relevance. It does not create a battery electrochemistry model, pack thermal solver, BMS, aircraft controller, maintenance authority or battery safety case.

## Why this benchmark matters now

Yan and Wiston (Batteries, 24 August 2026) published a leakage-safe cell-held-out benchmark reconstructed from the public CMU eVTOL battery dataset:

- 22 Sony-Murata VTC6 cells and 441 protocol-consistent C/5 reference-performance-test anchors.
- Three cells with ambiguous/non-monotone diagnostic-capacity trajectories excluded from the primary health analysis, leaving 19 cells.
- Outer leave-one-cell-out validation, matched 20-mission histories, prediction-time feature controls, and right-censored RUL handling.
- Random Forest achieved 0.620 percentage-point SOH MAE, compared with 1.252 for the mission-phase Transformer and 1.471 for Attention-LSTM-MoE.
- Best RUL MAE reported was 24.067, 46.527 and 122.198 missions at the 90%, 85% and 80% SOH thresholds respectively; error and model differences were threshold-dependent.
- Row-random splitting was substantially more optimistic than cell-held-out validation.

The study's own limitations matter: it is a laboratory cell-level dataset, not a fielded aircraft pack; the primary health benchmark uses only 19 cells; the three excluded cells narrow the inferential population; and model classes were not tuned under a fully equal computational budget. The 90/85/80% thresholds are study scenarios, not aircraft retirement requirements.

References:
- Yan & Wiston (2026), article: https://doi.org/10.3390/batteries12090322
- Public source dataset: Bills et al. (2023), *A battery dataset for electric vertical takeoff and landing aircraft*, Scientific Data 10, 344.
- Study repository: https://github.com/MusaWiston/Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML

## Paradigm shift

Move from **aggregate cell accuracy on random rows** to **operationally honest, unseen-cell prognostics**:

- Mission-phase features preserve takeoff, cruise, landing and charging stress rather than erasing sequence structure.
- Features must be available at the prediction timestamp; no future mission, target-derived label, cell identity or post-event information may leak into predictors.
- Split by physical cell, not adjacent samples; keep all pre-processing, model selection and calibration within each training fold.
- Treat RUL as a censored time-to-event quantity. A cell that has not reached the selected threshold by the end of observation is right-censored, not labelled as failed at the dataset boundary.
- Compare model complexity against strong tree ensembles under matched information and, separately, matched compute budgets.
- Evaluate prediction calibration and decision value, not MAE alone.

## Benchmark protocol

### Dataset reconstruction

1. Pin the original public CMU dataset revision and download/source URI, license/usage terms, raw file hashes and dataset documentation.
2. Preserve raw telemetry and source fields. Record cell ID as grouping metadata only, not as a predictor.
3. Reconstruct C/5 RPT anchors from the declared published protocol; do not choose a convenient capacity point or use a score-based fallback.
4. Reproduce the study's 22-cell source roster, 441 anchors and 19-cell primary health roster; preserve the three excluded cells and run sensitivity analysis rather than deleting them from all reporting.
5. Document voltage/current sign conventions, clock/mission indexing, missingness, phase labels, quality flags and any data corrections.
6. Rebuild SOC, SOH and 90/85/80% RUL targets chronologically. Do not extrapolate SOH beyond bracketing RPT anchors. Retain right-censoring and event support explicitly.

### Evaluation splits and leakage controls

- Primary: 19-fold leave-one-cell-out on the published primary health roster.
- Secondary: all 22 cells, with the three ambiguous trajectories handled through explicit alternate target definitions / censored or excluded sensitivity cases; never silently treat ambiguous labels as verified events.
- No random row split as a headline metric. Retain it only as an anti-baseline showing the degree of leakage optimism.
- Inner validation, feature selection, preprocessing, hyperparameter search, early stopping and probability/interval calibration must exclude the outer held-out cell.
- Matched-history primary comparison: same 20-mission causal history for Random Forest, gradient-boosted trees, temporal Transformer and LSTM/MoE.
- Report current-mission-only and whole-mission aggregate controls separately; do not mix their scores with matched-history comparisons.
- Freeze model configurations and evaluation protocol before observing outer-fold results.

### Models and baselines

At minimum:
1. Persistence / last-observation baseline for SOC and SOH.
2. Coulomb-counting or physics-informed SOC reference where input telemetry supports it.
3. Linear / ridge regression and simple state-space reference.
4. Random Forest (required strong baseline).
5. Gradient-boosted tree baseline (XGBoost / LightGBM only in an offline reference adapter if license and environment allow).
6. Mission-phase Transformer.
7. LSTM / GRU and LSTM-MoE.
8. Optional HDC/CfC representation model only after the baseline corpus and data-access contract are stable; it must receive the same prediction-time information and holdout cells.

Research/reference tooling may use a pinned Python implementation outside production Rust. Production battery/mission capability types remain Rust-first. Do not make Python a runtime or deployment dependency.

## Metrics

### SOC
- MAE/RMSE in percentage points, phase-end and mission-end.
- Signed bias, maximum absolute error and error by phase / current regime.
- Recursive drift for increment-prediction approaches.
- Physical bound violation rate and coulomb-counting reconciliation error.

### SOH
- MAE/RMSE in percentage points, signed bias, per-cell distribution and cell-bootstrap confidence intervals.
- Error by SOH regime, history length, ambient/temperature and charging/discharge profile.
- Prediction interval coverage/width, calibration error and between-cell transfer uncertainty.
- Compare against 0.620 percentage-point RF MAE as a published reference, not an automatic target or directly matched guarantee.

### RUL
- Threshold-specific MAE/RMSE and signed bias at 90%, 85% and 80% SOH scenario thresholds.
- Number of event-observed cells and samples per threshold; right-censored population reported separately.
- Time-dependent discrimination/calibration or survival scoring where appropriate.
- Do not score unobserved threshold crossings as exact failure events.
- Treat thresholds as benchmark scenarios, not regulatory or maintenance limits.

### Operational decision quality
- Early-warning lead time and false-alarm rate.
- False-safe / missed-threshold frequency with exact sample counts and confidence intervals.
- False-replacement burden and unnecessary maintenance exposure.
- Whether uncertainty can change reserve/capability decisions or should trigger abstention.
- Model runtime and memory are secondary; they cannot compensate for leakage or weak unseen-cell generalization.

## Proposed targets and claim boundaries

These are internal research targets, not certification or maintenance limits.

- Reproduce the public paper's dataset roster, 441 valid RPT anchors and 19-cell primary split, or publish every discrepancy.
- Reproduce the reported baseline order and per-cell-held-out results within a declared tolerance before claiming a methodological advance.
- Zero target/future/cell-identity leakage in a frozen negative-control corpus.
- Zero artificial RUL failures for right-censored cells.
- All folds must be cell-disjoint and deterministic.
- Publish per-cell errors, bootstrap uncertainty, threshold event support and all excluded/ambiguous cells.
- To claim improvement over the published baseline, beat the 0.620 percentage-point RF SOH MAE on the same dataset, target construction, 19-cell roster, outer LOCO folds and prediction-time features, with uncertainty intervals and a preregistered comparison. Also report performance on the 22-cell sensitivity roster.
- Improved average error must not increase false-safe frequency or destroy interval calibration; report the Pareto trade-off.

A 19-cell held-out-cell experiment is still a small laboratory dataset. It cannot establish unseen chemistry, pack topology, thermal management, aviation duty-cycle generalization or service-life validity.

## Integration with aircraft engineering

Qualified prognostics outputs may inform the mission-conditioned battery-capability envelope under #7805 and uncertainty scenarios under #11445. Keep cell-level labels distinct from pack-level SOH, pack power capability and aircraft reserve.

No prognostic output may directly authorize flight, extend a component's life, set maintenance intervals, relax BMS protections or claim battery safety. A pack/aircraft deployment needs separate pack-level characterization and independent approval.

## Claim ceiling

Reproducing a published benchmark is not a new SOTA result. Beating a published score is meaningful only with identical data, targets, splits, predictors and statistical treatment. Cell-level laboratory performance is not aircraft-pack validation or an approved maintenance/airworthiness determination.

## Initial execution sequence

1. Pin the public CMU dataset source, license/usage terms, raw-data revision and SHA-256 manifest.
2. Retrieve the 2026 study's code and supplementary material; record exact revisions and unavailable dependencies.
3. Reconstruct the 441 protocol-consistent RPT anchors independently and publish discrepancies against the published protocol.
4. Freeze the 19-cell primary LOCO split, 22-cell sensitivity cohort, targets, censoring rules and 20-mission predictor histories before fitting.
5. Reproduce persistence, Random Forest, Transformer and LSTM/MoE results with per-cell outputs.
6. Only after faithful reproduction, introduce Symthaea/HDC/CfC as a matched-information additional model.
7. Publish a manifest for dataset, code, model, split, preprocessing, calibration, target and output digests.

## Evidence boundary

The CMU source is cell-level laboratory evidence, not a complete aircraft pack, aircraft installation, in-service fleet or certification dataset. It must not be promoted into pack-level power capability or aircraft reserve without separate pack characterization and validation.


## Raw-pipeline reconstruction audit (required before model reproduction)

The authors' public preprocessing specification adds important implementation details that must be preserved:

- Required raw telemetry fields include `time_s`, `Ecell_V`, `I_mA`, `QDischarge_mA_h`, `QCharge_mA_h`, `Temperature__C`, `cycleNumber` and `Ns`.
- The source convention uses positive current for charge and negative current for discharge. Do not infer a different sign convention from the feature tables.
- `cycleNumber` is not globally unique after concatenation. Reconstruct cycle instances when the counter changes or the recording clock moves backwards; do not group all equal raw counters together.
- The checked-in phase map assigns `Ns=4,5,6` to take-off, cruise and landing. A valid mission must contain that ordered subsequence.
- Gaps greater than 60 seconds are flagged and excluded from duration, energy and coulomb integration; missing required/non-finite raw fields fail the cell rather than being silently imputed.
- The capacity-test detector is protocol-based, not score-based: at least 70% of discharge time within 0.39–0.81 A; starting voltage at least 4.10 V; final/minimum voltage at most 2.55 V; voltage span at least 1.40 V; discharge capacity between 1.50 and 3.75 Ah; discharge duration at least 7,200 seconds. RPT cadence is audited against 50 ± 5 missions but must not be used to force an incomplete cycle into an RPT.
- SOH is interpolated only between bracketing valid RPT anchors. It is not forward-filled, backward-filled or extrapolated beyond the final valid RPT.
- RUL threshold crossings that are not observed remain right-censored; the final recorded mission is not an artificial failure.
- SOC/health target-construction fields must remain outside the predictor allow-list.

**Critical source-repository warning:** its README states that previously archived feature tables and metrics came from a different preprocessing path and are not evidence for the corrected raw pipeline. Therefore, do not copy checked-in legacy feature tables or archived score CSVs into the reproduction and call the result a reproduction. Rebuild from the public raw CMU archive using the pinned corrected pipeline, capture a new manifest, rerun every model, and compare results to the 2026 article. Record any discrepancy. This distinction is a required evidence gate, not a documentation nicety.

Sources:
- Study's preprocessing/target specification: https://github.com/MusaWiston/Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML/blob/main/preprocessing_and_targets.md
- Study repository and explicit legacy-artifact warning: https://github.com/MusaWiston/Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML
- Public raw dataset landing page (access may require following CMU KiltHub's download workflow): https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830
