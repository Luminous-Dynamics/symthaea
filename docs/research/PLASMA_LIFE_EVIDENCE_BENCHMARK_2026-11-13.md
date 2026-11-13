# PLASMA-LIFE-001 — Bond-life evidence, censoring and accelerated-aging transfer

**Date:** 2026-11-13  
**Issue:** [#11971](https://github.com/Luminous-Dynamics/symthaea/issues/11971)  
**Status:** S0 synthetic evidence contract; no physical durability result

## 1. Research thesis

A plasma-treated joint can be evaluated at different points in its manufacturing/service lifecycle. The measurements collected at those points do not necessarily estimate the same physical quantity.

This benchmark makes three clocks explicit:

1. **Treatment-to-bond delay** — elapsed time after plasma activation and before adhesive/coating application. This estimates a bondability window under a declared storage environment.
2. **Post-bond environmental age** — elapsed time after assembly/cure while stored or exposed to declared conditions. Cross-sectional destructive measurements at these ages estimate strength retention or acceptance fraction.
3. **Under-load service time** — elapsed time while a bonded joint is under a declared mechanical/environmental stress until failure, censoring or a competing terminal event. This supports a survival / reliability estimand if the test design justifies it.

A 30-day treatment-to-bond storage result is not 30-day bonded-joint service life. A destructive lap-shear result measured at an age is not automatically a time-to-failure event. This distinction is the central S0 contribution.

The issue is an application-specific statistical evidence contract. It does not implement a general survival-analysis library or a plasma degradation model.

## 2. Ownership and reuse

- [PLASMA-SURFACE-001 #11795](https://github.com/Luminous-Dynamics/symthaea/issues/11795) owns the exact material/adhesive application, bondability window, acceptance floor and physical product outcomes.
- [PLASMA-OED-001 #11907](https://github.com/Luminous-Dynamics/symthaea/issues/11907) owns which experiment to run next and decision-value selection.
- [PLASMA-VM-001 #11847](https://github.com/Luminous-Dynamics/symthaea/issues/11847) owns process telemetry to downstream outcome prediction.
- [PLASMA-REGION-TRUTH-001 #11952](https://github.com/Luminous-Dynamics/symthaea/issues/11952) owns measurement support, spatial join and registration semantics.
- MAT-013 #4344 and MAT-INTERFACE-001 #6227 own canonical specimen and interface-state identity.
- Shared statistics owners retain general estimators. The current default statistics crate is version 0.1.0 and does not expose the later Kaplan–Meier/Cox APIs described in pending-release evidence. S0 therefore validates observation semantics only; it does not assume those APIs have been merged or qualified.
- The standalone proof-time survival baseline in POD-Meta #9652 / draft PR #9658 is a domain-specific research artifact and is not treated as a released generic plasma dependency.

No second generic survival engine is introduced.

## 3. Research and standards basis

### Material and adhesive dependence

Péczi-Kovács and Kókai (20 October 2026) compare argon and nitrogen atmospheric-plasma treatment on PP, HIPS, PVC, PC and PET. Both treatments improve wettability and bond performance relative to untreated specimens, but the lowest water contact angle does not consistently produce the strongest bond. Argon gives a wider damage-free processing window in the tested conditions, and the ranking depends on polymer/adhesive combination.

Morgese et al. (2024) report that storage performance depends on polymer/additive profile. In their studied systems, plasma-treated additive-bearing PE partially lost adhesion after 30 days, while specific primer-coated specimens retained performance through two months. These are material-specific observations, not universal life laws.

### Accelerated-aging limitations

ASTM D1183-03(2019) states that no accelerated procedure perfectly correlates with service and that no single or small set of laboratory conditions simulates all service conditions. ASTM D2919-01(2023) describes stressed adhesive-joint durability under environmental and temperature conditions and notes the relevance of stress, temperature and relative humidity. ASTM D1151-00(2022) addresses moisture and temperature exposure and expresses retained strength relative to original strength.

References:
- Péczi-Kovács & Kókai (2026), DOI [10.1007/s44245-026-00396-4](https://doi.org/10.1007/s44245-026-00396-4).
- Morgese et al. (2024), DOI [10.1007/s11998-024-00938-z](https://doi.org/10.1007/s11998-024-00938-z).
- [ASTM D1183-03(2019)](https://store.astm.org/d1183-03r19.html).
- [ASTM D2919-01(2023)](https://store.astm.org/d2919-01r23.html).
- [ASTM D1151-00(2022)](https://store.astm.org/d1151-00r22.html).

These are method-family anchors. A qualified laboratory must select the licensed current edition and exact procedure suitable for the joint. This benchmark claims no compliance.

## 4. The critical measurement distinction

### A. Cross-sectional destructive strength

Different specimens are assigned to distinct pre-bond delay or post-bond aging times and then destructively tested. The outcome is a strength measurement, failure mode and acceptance status at that sampled age.

Permitted estimands include:
- strength distribution at the declared age;
- quantiles and uncertainty;
- fraction meeting the application acceptance floor;
- failure-mode distribution;
- retention relative to a matched reference population.

These are not survival curves. A specimen tested destructively at day 30 was not necessarily observed to fail at day 30. It was measured at day 30.

### B. Sustained-load time-to-failure

A joint is exposed to a declared load and environment. The observed event is the first failure according to a predeclared failure criterion. Exact failure time can support an event-time observation.

### C. Periodic inspection

If a joint is known-good at time t0 and first observed failed at t1, the event is interval-censored in (t0, t1]. Do not assign the first failed inspection time as an exact failure time unless the physical observation supports that precision.

### D. Right-censoring and unknown outcomes

A specimen still intact at the planned test end is right-censored at its last valid at-risk time. A specimen lost, instrument-interrupted, or with unknown status is not automatically a valid non-informative censor. The censoring reason and mechanism must remain explicit.

### E. Competing failure modes

At an actual failure event, preserve the adjudicated failure cause, such as treated-interface adhesive failure, cohesive adhesive failure, cohesive substrate failure, coating/substrate interface failure, environmental/corrosion-assisted failure, or mixed/unclassified failure. The taxonomy must be selected for the actual joint. Do not infer cause from the treatment recipe.

## 5. Accelerated-aging transfer contract

An accelerated-life claim must bind:
- exact substrate and adhesive profile, material lot, cure and conditioning;
- physical stress variables and calibrated exposure history (temperature, humidity, load, chemical exposure, cycle profile and duration);
- reference condition and units;
- acceleration model family, version, fitted range and assumptions;
- the observed stress/time envelope and the requested prediction horizon;
- failure-mode / fracture-surface evidence across stress levels;
- the uncertainty and model-discrepancy assessment;
- held-out material lot / day / device identities.

A temperature-acceleration formula is not sufficient evidence by itself. A change in dominant failure mode, adhesive transition, substrate softening, chemical pathway or other physical regime invalidates a single-mechanism scale factor until a qualified model accounts for the change.

S0 does not admit extrapolation beyond the measured stress envelope. A future release may permit extrapolation only through a separately reviewed, evidence-bearing mechanism-transfer qualification.

## 6. S0 observation contract

Each record binds:
- specimen IDs and declared independent physical count;
- canonical material, adhesive and process profile;
- one explicit clock kind and time origin;
- one measurement modality and claim kind;
- terminal state and event/censor interval;
- last known-good / first known-failed times where interval-censored;
- censor reason and design receipt where right-censored;
- failure cause and adjudication evidence where an event occurred;
- stress/environment history and calibration identity;
- raw observation and protocol digests;
- lot/day/device/operator/preparation independence groups;
- accelerated model, reference condition, stress support, prediction horizon and mechanism evidence where transfer is claimed.

### Clock / modality compatibility

| Claim | Permitted clock | Permitted modality | Terminal state |
|---|---|---|---|
| Strength at age | PRE_BOND_DELAY or POST_BOND_AGE | DESTRUCTIVE_CROSS_SECTION | STRENGTH_AT_AGE |
| Time to failure | UNDER_LOAD_SERVICE | SUSTAINED_LOAD | EVENT, RIGHT_CENSORED, COMPETING_EVENT |
| Interval-observed failure | UNDER_LOAD_SERVICE | PERIODIC_INSPECTION | INTERVAL_CENSORED |

Other combinations are blocked or rejected. This table is deliberately restrictive; a future physical protocol can add modalities only with an explicit measurement model and review.

## 7. Frozen known-answer benchmark

The S0 fixture contains 18 synthetic records:

| ID | Expected | Purpose |
|---|---|---|
| DESTRUCTIVE_STRENGTH_AS_SURVIVAL | REJECTED | Destructive strength falsely labelled as time-to-failure |
| TREATMENT_DELAY_AS_SERVICE_AGE | REJECTED | Pre-bond storage mixed with service time |
| POST_BOND_AGE_WITHOUT_LOAD | REJECTED | Cross-sectional aging promoted to under-load life |
| EXACT_EVENT | PASS | Exact sustained-load failure event |
| RIGHT_CENSORED_INTACT | PASS | Valid intact-at-end right-censor |
| INTERVAL_CENSORED_INSPECTION | PASS | Last-good / first-failed interval preserved |
| COMPETING_FAILURE_CAUSE | PASS | Adjudicated competing failure cause retained |
| UNKNOWN_OUTCOME_AS_CENSOR | BLOCKED | Unknown outcome is not valid censoring |
| INFORMATIVE_POLICY_CENSOR | BLOCKED | Policy/resource stopping cannot be assumed non-informative |
| ZERO_EVENT_TAIL | REJECTED | Unsupported median/q90 life assertion |
| DUPLICATE_SPECIMEN | REJECTED | Repeated specimen identity inflates independent count |
| MIXED_CLOCKS | REJECTED | Clock type and analysis claim disagree |
| ACCELERATED_OUTSIDE_RANGE | BLOCKED | Prediction exceeds observed stress/time support |
| MECHANISM_SHIFT | REJECTED | A changed mechanism invalidates one-factor transfer |
| NO_MECHANISM_EVIDENCE | BLOCKED | No failure-mode evidence for accelerated transfer |
| STRESS_HISTORY_GAP | BLOCKED | Required calibrated stress history is missing |
| MATERIAL_LOT_POOLING | REJECTED | Incompatible material/adhesive cohorts pooled without transfer |
| CROSS_SECTIONAL_ACCEPTANCE | PASS | Correctly retained destructive strength-at-age data |

PASS means only that the synthetic evidence record is structurally admissible. It does not mean product acceptance, measured bond durability or service-life qualification.

## 8. Statistical baselines and comparison protocol

Once an actual dataset exists and the estimand is frozen, compare:
1. Cross-sectional strength distributions by treatment delay and post-bond age.
2. Complete-case summaries as a deliberately limited diagnostic baseline.
3. Product-limit survival for valid right-censored life cohorts under stated censoring assumptions.
4. Aalen–Johansen cumulative incidence when competing failure causes exist.
5. Weibull and lognormal parametric lifetime models with tail diagnostics.
6. Cox proportional hazards with proportionality and calibration diagnostics.
7. Accelerated failure-time / stress-covariate models only within supported stress regimes.
8. Mechanism-stratified models where failure mechanisms differ.
9. The incumbent primer / conventional treatment and the best commercial plasma reference at the same service horizon.

No estimator is declared the winner in advance. Do not report a life number without its risk set, censoring assumptions, cause structure, supported horizon and uncertainty.

## 9. Metrics and prospective targets

Report independently:
- survival or cumulative incidence at predeclared service horizons;
- cause-specific cumulative incidence;
- restricted mean survival time over the observed horizon;
- median/q90 only if the observed curve identifies the quantile;
- time-dependent Brier score / integrated Brier score where valid;
- cross-sectional strength quantile error and acceptance fraction;
- prediction interval and lower-bound coverage;
- calibration by material lot, adhesive system, failure mode and stress regime;
- mechanism-shift detection and false acceleration-transfer rate;
- false-qualified / abstention rate;
- specimen use, duration and destructive-test cost.

Prospective targets (not standards or achieved results):
- 100% of records declare clock, modality and terminal status;
- zero destructive strength-at-age records promoted to time-to-failure;
- zero unknown outcomes treated as valid non-informative censoring;
- zero unsupported stress/time extrapolations;
- zero single-factor acceleration claims after a demonstrated failure-mechanism shift;
- zero unsupported tail quantiles;
- ≥90% nominal interval coverage on adequately powered independent holdouts, reported per stratum;
- ≥95% detection of planted mechanism shifts at published false-abstention and coverage levels.

No single composite score may hide false qualification, tail non-identifiability or mechanism changes.

## 10. Physical promotion protocol

Before a laboratory study:
- freeze one substrate/adhesive system and intended service environment;
- decide whether the trial is cross-sectional strength retention, sustained-load life or periodic inspection;
- predeclare event, censor and competing-cause rules;
- select licensed methods appropriate to the actual joint and stress;
- size the experiment from pilot event rates/variance, target precision and failure-mode stratification;
- preserve invalids, interruptions, lost specimens and excursions by identity;
- include non-accelerated references and overlapping stress levels;
- adjudicate fracture surfaces/modes at each stress;
- reserve independent lots/days for blinded validation.

## 11. Claim ceiling

S0 validates only clock/event/censor/provenance and accelerated-transfer refusal semantics on synthetic records. It does not estimate real bond life, establish a degradation law, qualify accelerated aging, prove mechanism invariance, predict service life, improve adhesion, save inspection cost, certify a product or authorize equipment operation.


## 12. External durability SOTA comparators (reviewed 13 November 2026)

The durability benchmark must compare against actual adhesive-life and fatigue-lifetime methods, not only surface-wettability regression.

### A. Plasma-treated TPO / silicone BIPV service-life model (2026)

Zhu et al., *Surface and Coatings Technology* 522 (15 February 2026), DOI [10.1016/j.surfcoat.2026.133164](https://doi.org/10.1016/j.surfcoat.2026.133164), study a TPO-to-silicone adhesive interface for building-integrated photovoltaic structures.

The article reports:
- interface bond strength increasing from 0.23 MPa untreated to 1.25 MPa after plasma modification;
- lap-shear strength increasing from 0.42 MPa to 1.84 MPa;
- plasma parameter exploration across air, O₂, Ar, time and power conditions;
- a generalized Eyring model combining numerical simulation and accelerated stress/humidity/thermal aging;
- a reported model accuracy of 98% and a predicted 18.17-year service life at 45 °C and 30% RH.

These values are the authors' reported results for their specific TPO/glass/silicone BIPV structure. They are not directly comparable to PP/PE lap-shear results, a different adhesive, or an unvalidated Luminous process. The 98% headline must not be treated as an external predictive-coverage result until its exact metric, training/test split, stress-condition holdout and uncertainty are audited.

**Benchmark consequence:** generalized Eyring multi-stress coupling is a direct competitor baseline, not a novelty claim for Luminous. Our challenge should test whether such a model transfers across unseen temperature/RH/load combinations, substrate/adhesive lots and independently adjudicated failure mechanisms. Compare in-range prediction, held-out stress-condition prediction and lower-bound calibration separately.

Source: [ScienceDirect article record and abstract](https://www.sciencedirect.com/science/article/abs/pii/S0257897226000162).

### B. Hygrothermal CFRP adhesive fracture and Weibull fatigue (2026)

Vigón et al., *Applied Sciences* 16(21), 10810 (2026), DOI [10.3390/app162110810](https://doi.org/10.3390/app162110810), investigate Mode I fracture and fatigue behaviour of carbon/epoxy adhesive joints after 60 °C / 70% RH exposure for 1, 2, 4 and 12 weeks.

The study is valuable because it distinguishes:
- cross-sectional static DCB fracture toughness by aging cohort;
- fatigue crack initiation life under cyclic loading;
- subsequent crack propagation rate;
- run-out specimens treated as right-censored observations in a Weibull regression model;
- fracture-surface evidence showing that the relative cohesive/interlaminar failure contributions change with aging.

The reported static G_IC response is non-monotonic: it reaches a minimum after two weeks and increases at longer aging, while fatigue-initiation resistance and crack-growth behaviour follow different trends. The authors discuss competing moisture plasticisation and adhesive post-curing explanations; they do not claim the mechanism is uniquely resolved.

**Benchmark consequence:** compare strength-at-age distributions and fatigue survival as different endpoints. Use run-outs as censored, and use fracture-surface adjudication to test mechanism transfer. A single monotonic degradation curve or one scalar "retention" score would conceal the observed behaviour.

Source: [Open-access article](https://www.mdpi.com/2076-3417/16/21/10810).

### C. Weibull and Cox survival with aging covariates (2026)

*Survival Analysis of Fatigue Lifetime in Aged Elastomers Using Weibull and Cox Regression Models*, first published 2 July 2026, DOI [10.1111/ffe.70344](https://doi.org/10.1111/ffe.70344), treats specimens that fail outside the gauge length as right-censored and compares Weibull and Cox models. It adds aging-related material descriptors (including crosslink density) and uses a Cox frailty term to account for population heterogeneity.

The authors report consistent predictions within the data-supported region and greater covariate flexibility for the Cox model, while noting limited specimen counts and uncertainty. Elastomer fatigue is an adjacent statistical comparator rather than a direct plasma-bonding dataset.

**Benchmark consequence:** include both parametric Weibull and semiparametric Cox/frailty baselines once the actual event-time cohort supports them. Report calibration by aging stratum and specimen/lot cluster; do not confuse model fit with validated extrapolation.

Source: [Open-access article](https://onlinelibrary.wiley.com/doi/10.1111/ffe.70344).

### D. Defect interactions and multiple evidence modalities (2025 review)

Dachev et al., *Materials* 18(12), 2724 (10 June 2025), DOI [10.3390/ma18122724](https://doi.org/10.3390/ma18122724), review defects and reliability in adhesive bonding, including kissing bonds, porosity, voids, poor cure, substrate failures, environmental aging and the difficulty of modelling interactions among defects and stochastic variability.

**Benchmark consequence:** plasma durability must not be reduced to a temperature-time stress model alone. Record adhesive cure, bondline geometry, defect indicators, contamination, substrate variability and mode adjudication. Compare multi-method inspection only at each method's real support; an NDT output is not a destructive strength label. 

Source: [PubMed record](https://pubmed.ncbi.nlm.nih.gov/40572857/).

### SOTA comparison matrix

| Baseline / comparator | Correct target | What it tests | Key limitation |
|---|---|---|---|
| Generalized Eyring multi-stress model (TPO/silicone BIPV, 2026) | Service life under coupled stress/humidity/thermal conditions | Stress interaction and service-life prediction | Material/adhesive-specific; audit exact validation split and metric |
| Weibull regression of fatigue initiation with run-outs (CFRP, 2026) | Cycles to crack initiation | Censoring-aware fatigue and aging effects | Specific Mode I DCB geometry and stress range |
| Weibull + Cox/frailty (aged elastomer, 2026) | Fatigue cycles to failure | Parametric vs covariate-adjusted survival | Adjacent material class; small/censored cohort |
| Cross-sectional strength at age | Strength distribution after predeclared exposure | Retention, quantiles and accepted fraction | Not time-to-failure |
| Non-parametric survival / Aalen–Johansen | True event-time and competing-cause data | Censoring-aware event probability | Censoring assumptions; no extrapolation by default |
| Mechanism-stratified / discrepancy-aware transfer | Held-out stress and material regimes | Whether one acceleration relationship remains valid | Requires sufficient failure-mode and stress coverage |

The same raw cohort must not be used to tune the acceleration model, choose its failure mode and claim independent validation. Preserve training, calibration and blinded confirmation by whole specimens/lots/days/stress regimes.
