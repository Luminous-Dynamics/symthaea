# SDCB QUANT real-data metrology tournament v1

**Status:** dataset and protocol identified; data files have not yet been downloaded, parsed or analyzed in this execution. This is a preregistration, not a benchmark result or physical validation.

## 1. Why QUANT is the first real-data target

The CEDA archive lists the QUANT project as a public dataset of 52 commercial low-cost air-quality devices deployed at three UK sites:
- Manchester Air Quality Supersite (MAQS), urban background;
- London Air Quality Supersite (LAQS), urban background;
- Fishergate AURN, York, roadside.

The catalogue reports 270 files / 7 GB, with measurements spanning 10 December 2019 to 31 October 2022. The archive includes NetCDF and CSV files and lists pollutant and meteorological channels, including NO2 and PM2.5. CEDA states that the data are publicly accessible under the Open Government Licence v3 and that the archive was restored on 1 April 2026 after the November 2025 archive incident. The catalogue explicitly says CEDA has not performed quality control on the provider-supplied data. Cite the archive as Lacy, Diez and Edwards (2023), using the catalogue's citation instructions.

This is unusually relevant because the data offer multiple devices, device types, urban/roadside contexts, and a long chronological range. It can evaluate cross-device and cross-site generalization without pretending that 52 channels are 52 independent laboratories.

## 2. Critical distinction: public QUANT data versus the 2026 short-collocation reference series

Kok and Tabandeh (2026), *Uncertainty Evaluation for Testing and Calibrating Mobile Sensors Only Using Short Collocation Periods*, Quality and Reliability Engineering International 42(7), 4298–4309, DOI 10.1002/qre.70319, is the closest recent scientific baseline.

The article uses a QUANT-based NO2 case study and describes:
- one selected sensor and an accurate reference analyzer from July 2021 to October 2022;
- a reference time resolution of 1 minute;
- a sensor time resolution of 5 minutes;
- comparison of unequal reporting/averaging windows under a time-varying measurand;
- OLS, GDR-family and corrected generalized distance regression (C-GDR) comparisons.

However, the article's Data Availability Statement says the data supporting its findings are available from the corresponding author on reasonable request. Do not assume the article's paper-specific one-minute reference series is present in the public CEDA files merely because the public low-cost QUANT archive is present. The exact reference series, instrument uncertainty, collocation metadata and averaging semantics must be checked or requested from the authors before claiming a reproduction of their one-minute/five-minute case.

The article also assumes independent identically distributed measurement errors per instrument in its base model. SDCB's common-cause/reference-chain/device-drift extension is a deliberately broader model and must be compared first on the article's stated assumptions, then on separate dependence/misspecification tracks.

### Published known-answer results to reproduce before claiming improvement

The paper's six-case simulation offers frozen regression/calibration controls:
- The nominal expected sensor-performance pass percentage is 95%.
- When process variability is ignored, reported percentages drop to 72.9% in the DU-ML-YH, DU-MH-YH, SU-ML-YH and SU-MH-YH scenarios; incorporating the process variability restores these to 95%.
- In the DU-MH-YL scenario, the reported parameter-interval coverage CP(a)/CP(b) is 94.7%/94.8% for OLS and 93.8%/93.7% for C-GDR.
- In the SU-ML-YH scenario, OLS reports 38.8%/37.4% parameter coverage, while C-GDR reports 91.7%/91.4%.
- In the SU-MH-YH scenario, OLS reports 76.2%/74.4%, YB-GDR 95.7%/95.7%, and C-GDR 93.7%/93.5%.

These published values establish an important comparison rule: **do not claim OLS, GDR, C-GDR or SDCB is universally best.** Reproduce the instrument-uncertainty/process-variability regimes individually. Then test whether SDCB improves calibration coverage, decision loss or collocation cost under matched assumptions and held-out data. The table values are targets for replication, not results from our code.


## 3. Acquisition and custody manifest

Before downloading or processing, freeze:
- CEDA catalogue UUID and retrieval date;
- archive directory, exact file URL and file name;
- catalogue-provided file size and checksum where available;
- licence/citation text and archive restoration notice;
- instrument-list and metadata file versions;
- selected site, device, pollutant and date range;
- NetCDF dimensions, variables, units, time standard/calendar, fill values and QC-flag semantics;
- a file SHA-256 generated after retrieval;
- software/toolchain and parser version.

Do not download the full 7 GB archive merely to start. Use the archive directory's JSON listing / DAP capabilities to identify a single modest device/channel file and its matching instrument metadata first. Then acquire only the preregistered site/device/channel subset. Retain raw files immutably; derived and filtered tables get separate content digests.

**Access status at protocol authoring:** the CEDA catalogue and directory listings were browsed and inspected. Attempts to retrieve a small instrument CSV through the CEDA file endpoint returned an unsupported binary content-type error to the browser integration and failed in the file-download environment. No NetCDF/CSV bytes were acquired, no dataset checksum was computed, and no data parsing or numerical analysis has been performed. This is an access blocker, not a dataset-quality finding. Retry using CEDA's documented archive access / DAP workflow when a supported authenticated or direct-file path is available; do not substitute guessed file URLs or claim a download.

## 4. Data preparation contract

The raw data are not IID rows. Each record must retain:
- site and deployment/platform identity;
- manufacturer/model/device serial or stable archive device ID;
- pollutant channel and units;
- original time coordinate, time zone/calendar and conversion;
- acquisition/reporting interval and integration semantics where documented;
- provider QC flag and QC reason, with raw flag preserved;
- calibration algorithm/version and instrument metadata;
- environmental covariates and missingness indicators;
- collocation/reference identity and reference ancestry where available.

Do not assume a timestamp labels an instantaneous point sample. If the record represents an interval average, preserve its start/end and averaging kernel. Where the native data provide only hourly or longer averages, do not reconstruct minute-scale truth by interpolation and then report that interpolation as a reference measurement.

Do not silently remove bad or missing readings. Publish a primary QC-qualified cohort and a sensitivity cohort that includes flagged/uncertain observations with explicit uncertainty treatment. CEDA performs no archive-level QC, so provider flags and protocol-level exclusions must be separately interpreted and versioned.

## 5. Tournament splits

Freeze splits before model tuning.

### Split Q1 — chronological within-device

Train on early time, validate on a subsequent disjoint interval, test on the final chronological interval. Do not randomly shuffle timestamps. Prevent windows crossing split boundaries and leave a declared embargo/gap where integration windows or temporal correlation could leak across partitions.

### Split Q2 — leave-one-device-out

Hold out complete devices, including their complete time histories. Tune on the remaining devices. This tests transport to unseen units rather than interpolation of the same sensor's future.

### Split Q3 — leave-one-platform/manufacturer-family-out

Hold out device families where the inventory supports it. Report the actual number of held-out families; do not merge unlike product families to manufacture sample size.

### Split Q4 — leave-one-site-out

Train on two sites and evaluate on the third. Report urban-background and roadside performance separately. The three sites are three locations, not three independent laboratories.

### Split Q5 — compound regime shift

Use evaluator-controlled holdouts for season/weather/maintenance/deployment transitions, with all thresholds and shift labels frozen before evaluation. The agent may see only the permitted public context, not the held-out regime labels.

All split assignments, device IDs, site IDs, event-time ranges and cohort inclusion rules must be committed in a machine-readable manifest.

## 6. Baselines and ablations

Compare only on matched channels, units, timestamps/windows, training support and reference assumptions.

**Standards and measurement conformance**
- CEN/TS 17660-1:2021 for gaseous pollutants (including NO2) and CEN/TS 17660-2:2024 for PM10/PM2.5, using their published editions and licensed test procedures.
- ISO/IEC Guide 98-3:2008 / JCGM 100 (GUM) uncertainty evaluation and reporting; JCGM 101 Monte Carlo propagation where needed.
- Keep device-level standard classification separate from SDCB's network-level inference. The published CEN/TS scopes focus on individual sensor systems at fixed sites; they do not by themselves validate common-cause dependence, cross-device reference ancestry, or network-wide decision integrity.
- Do not claim standards compliance from the QUANT archive alone. Freeze the standard edition, obtain the complete normative test requirements, instrument conditions and acceptance criteria, and execute the required laboratory/field procedures.

**Measurement and regression baselines**
- naive timestamp join / IID residual baseline (diagnostic negative control only);
- ordinary least squares (OLS);
- generalized distance regression (GDR);
- C-GDR and the other method variants reported by Kok and Tabandeh where inputs and assumptions match;
- hierarchical/random-effects calibration;
- control-standard / control-chart recalibration where the data support it.

**Temporal covariance baselines**
- stationary scalar OU;
- vector continuous-time Kalman/state-space models;
- Gaussian-process temporal covariance using the kernel families and fitting assumptions declared in the protocol;
- static full-covariance conditioning;
- IID and diagonal covariance as negative controls, never as the assumed truth.

**SDCB ablations**
1. scalar OU window covariance only;
2. vector state dynamics with full process cross-covariance;
3. unequal/overlapping interval observations;
4. covariance-model mixture with within- and between-model variance;
5. open-world model adequacy and abstention;
6. shared reference-chain + device-local factor conditioning;
7. full conditional joint Gaussian likelihoods and correlated outcome bins;
8. adaptive experiment selection using the existing policy oracle.

Keep physical pollutant-process covariance separate from shared measurement-system causes, reference-chain uncertainty, device-local drift and residual acquisition noise. Residual correlation alone does not identify a physical cause.

## 7. Standards-conformance and traceability lane

Before asserting that SDCB improves deployment quality, qualify the individual sensor systems against the published, applicable standard:
- CEN/TS 17660-1:2021 covers performance evaluation/classification of low-cost sensor systems for gaseous pollutants in ambient air, including NO2.
- CEN/TS 17660-2:2024 covers particulate matter, including PM10 and PM2.5.
- ISO/IEC Guide 98-3:2008 / JCGM 100 provides the general framework for evaluating and expressing measurement uncertainty; JCGM 101 covers propagation of distributions using Monte Carlo methods where appropriate.

The published CEN/TS specifications focus on individual sensor systems under prescribed laboratory/field testing at fixed sites. They are not a substitute for a network-level benchmark. Keep two claim lanes:
1. **Device conformance:** reproduce the applicable standard's individual-device tests with the exact licensed edition, calibration/field protocol and acceptance criteria.
2. **Network decision assurance:** quantify shared reference, environmental, power, firmware, device-local and time-window dependence; test network predictive uncertainty, common-cause false confirmation, spatial/site transport and decision value.

Metrological traceability is a separate property: a measurement result must be related to a reference through a documented, unbroken calibration chain, with each calibration contributing to uncertainty. A digital ID/version envelope or verified software receipt is not itself a calibration certificate or proof of traceability. Report instrument identity, reference-standard certificates, calibration intervals, uncertainty contributions and ancestry separately from the statistical predictive model.

The current CEN committee is reviewing successor prEN versions; use the published CEN/TS editions as the declared baseline until a successor is formally published/adopted. Do not silently mix draft criteria into a claimed standards-conformance result.

## 8. Metrics and decision outcomes

Report by pollutant, site, device family, time regime and held-out split:
- bias, MAE and RMSE;
- calibration slope/intercept and residual distribution diagnostics;
- log predictive density and CRPS where a predictive distribution is available;
- nominal 90% and 95% interval coverage, interval width/sharpness and coverage-risk curves;
- uncertainty of calibration slope/intercept and its empirical coverage;
- false rejection rate for correctly performing sensors and missed fault rate for injected/known bad states where truth labels exist;
- detection delay and abstention rate under drift;
- reference measurements / collocation time consumed per qualified decision;
- terminal decision loss and experiment-selection regret against the exact small-world policy oracle;
- unsupported model/covariance cases, missing data and all failed/invalid episodes.

A high correlation or low RMSE is not a metrological traceability certificate. State separately whether the result establishes predictive performance, calibration-parameter coverage, sensor conformance, traceability, causal diagnosis or physical validation. The public archive alone does not automatically supply ground truth for all of these claims.

## 9. SDCB-specific research questions

1. Does interval-integrated OU/vector state covariance reduce bias and over-rejection relative to point-sample or IID models when acquisition periods differ?
2. Does full cross-factor covariance prevent false independent confirmation among devices sharing a site, power/environment, reference or calibration ancestry?
3. Does the model-mixture total variance improve coverage under uncertainty about process time constants and residual structures?
4. Does open-world adequacy detect a new drift/common-cause regime rather than force the closest registered covariance model?
5. Does a state-conditional joint-bin likelihood preserve the dependence structure and alter experiment selection correctly?
6. Does the adaptive policy reduce terminal decision loss per equal reference/collocation cost relative to OLS/GDR/C-GDR and fixed schedules?
7. Do the gains survive leave-one-device-out, leave-one-family-out and leave-one-site-out tests?

The critical outcome is not whether SDCB wins one RMSE table. It is whether the system makes better-calibrated, lower-cost scientific decisions without claiming independence, adequacy or traceability it has not established.

## 10. Acceptance gates (targets, not results)

- 100% agreement with frozen analytical and synthetic numerical fixtures within tolerance.
- Zero silent time-window substitutions or undocumented resampling.
- Zero acceptance of invalid covariance or stale model/calibration/device context.
- Zero false-independence claims on duplicate/common-cause controls.
- Nominal 90% and 95% predictive coverage evaluated by device/site/regime, with intervals and sharpness also reported.
- At most 2% approximate-policy regret on exact small-instance cases.
- At least 10% reduction in terminal decision loss per equal reference cost against the strongest matched baseline, with paired confidence intervals excluding zero, before making a superiority claim.
- All acceptance thresholds must be frozen before outer test evaluation.

These are research gates. No QUANT benchmark result, dataset acquisition, physical test or SOTA comparison is claimed in this document.

## 11. Sources

- CEDA, QUANT dataset catalogue, UUID ae1df3ef736f4248927984b7aa079d2e: https://catalogue.ceda.ac.uk/uuid/ae1df3ef736f4248927984b7aa079d2e
- CEDA archive directory: https://data.ceda.ac.uk/badc/deposited2023/QUANT/
- CEN/TS 17660-1:2021, Air quality — Performance evaluation of air quality sensor systems — Part 1: Gaseous pollutants in ambient air.
- CEN/TS 17660-2:2024, Air quality — Performance evaluation of air quality sensor systems — Part 2: Particulate matter in ambient air.
- ISO/IEC Guide 98-3:2008 / JCGM 100, Guide to the Expression of Uncertainty in Measurement (GUM).
- ISO/IEC Guide 98-3 Supplement 1 / JCGM 101, Propagation of distributions using a Monte Carlo method.
- Kok, G. and Tabandeh, S. (2026), *Uncertainty Evaluation for Testing and Calibrating Mobile Sensors Only Using Short Collocation Periods*, Quality and Reliability Engineering International 42(7), 4298–4309. DOI: https://doi.org/10.1002/qre.70319
- Terminal-Bench-Science live task set and contribution process: https://github.com/harbor-framework/terminal-bench-science/ and https://www.terminal-bench-science.ai/contribute
- ASI-Bench task catalog: https://asibench.apexin.ai/tasks/catalog/
