# MARINE-MEASUREMENT-VOI-IDENTIFIABILITY-001
## Evidence-first instrumentation, parameter identifiability and value-of-information design

**Status:** research specification v0.1.0  
**Created:** 11 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** #11430 (trial registries), #11518 (clock drift), #11532 (gas-resolved emissions), #11493 (causal identifiability), #11510 (3-DOF coupling)

> This programme chooses measurement portfolios and decides whether data can support a parameter/decision. It does not assume that more sensors make a model valid. It is not a physical qualification or metrology certificate.

## 1. Paradigm shift: measure what the decision needs, not everything that can be measured

A marine digital twin can become more complicated while remaining less identifiable. Correlated channels, inferred values, hidden signal processing, clock drift, unrecorded operating states and insufficient operating excitation can make a large dataset unable to distinguish the physical parameters it claims to estimate.

The correct sequence is:

1. Define the decision and the physical quantities required to make it.
2. Define the observation operators and sensor boundaries for each quantity.
3. Audit independence, algebraic dependence, provenance, calibration, timebase, dynamic range and operating excitation.
4. Determine structural and practical identifiability before calibration.
5. Select the next measurement or experiment by its expected reduction in decision uncertainty/regret.
6. Calibrate only parameters supported by the measurements.
7. Abstain or reduce the model when the required quantity is not identifiable.

The goal is not the most instrumented ship. It is the smallest trustworthy measurement portfolio that changes decisions enough to justify its lifecycle cost.

## 2. Published methodological benchmark: four-tanker measurement adequacy study

Coșofreț et al., *Measurement Adequacy and Parameter Identifiability Before Calibration of a Physics-Informed Ship-Performance Model: A Methodological Case Study on Four Tanker Voyages*, JMSE 14(20), 1940 (17 October 2026), DOI 10.3390/jmse14201940.

The published study audits 207 records (0.5–5 hours each) across four voyages of one tanker before model calibration. The onboard sensor/acquisition architecture is undocumented. The paper reports:
- exact algebraic dependence among speed-over-ground, distance/duration and longitudinal current fields, leaving two independent quantities rather than three;
- a five-parameter observation model with structural sensitivity rank at most two;
- the available records practically support only one scalar correction, κ, while the environmental composite is not practically identifiable on those voyages;
- a one-dimensional scale predicts remaining-route fuel totals within the reported range, but the authors explicitly do not interpret κ as a physical efficiency or independently identified hull/engine state;
- route-level totals can close while segment-level relationships remain unstable;
- the paper does not claim ISO 19030 conformity; its reported speed/power inputs are not shown to meet the standard's independent speed and shaft-torque traceability conditions.

The processed 207-record dataset, traceability workbook, code, checksums and verification scripts are deposited under Zenodo DOI 10.5281/zenodo.23235788 with CC BY 4.0. The original commercial workbook and sailing-leg performance pages remain proprietary.

This is a methodology benchmark, not proof that every ship-performance model has the same rank deficiency. Reproduce the tests and preserve the authors' claim ceiling.

## 3. Sensing and measurement portfolio as a design variable

Evaluate candidate measurement channels against a named output/decision, not by a generic "sensor coverage" score.

Candidate channels include:
- independent speed through water (e.g. Doppler log) and independent GNSS speed over ground;
- shaft torque and RPM with traceable calibration;
- engine fuel supply/return mass flow, temperature, density and lower calorific value;
- engine load, cylinder/load profiles and fuel index, while checking whether fields share upstream computation;
- true/apparent wind vector, measurement height, vertical profile and device-local flow where needed;
- wave radar/buoy observations, heading, current, IMU and vessel motions;
- draft, trim, heel, loading state and cargo mass;
- rudder angle, propeller torque/thrust/RPM and propulsion mode;
- WAPS commanded state and independently observed achieved state;
- battery current, voltage, temperature, state of charge and independently closed energy boundary;
- engine-resolved CH4/N2O test-bed, ELM or CEMS channels with method-specific calibration and uncertainty.

Every channel must identify the quantity, observation operator, physical location, unit, sample rate/basis, uncertainty, calibration, timebase, provenance, rights, limitations and whether it is measured, derived, inferred or manually entered.

## 4. Structural and practical identifiability gates

### 4.1 Structural dependence

- Detect duplicate/algebraically derived fields (e.g. distance / duration = speed over ground).
- Build an explicit observation graph, separating independent sensors from transforms and shared upstream calculations.
- Construct the parameter-to-observation sensitivity matrix/Jacobian over the declared model and state domain.
- Report rank, null directions, singular values and parameter combinations, not only a solver's convergence.
- A parameter that is structurally unidentifiable cannot be recovered by more samples from the same observation operator.

### 4.2 Practical identifiability / excitation

- Assess singular-value spectrum, condition number, profile likelihood or posterior geometry across the observed operating domain.
- Test whether wind angle, sea state, loading, speed, shaft load, device state and engine operating point vary enough to identify the target parameters.
- Separate poor excitation from a bad sensor; do not infer a fault solely from an inconsistent relationship.
- Freeze holdout voyages before calibration and prevent future observations from entering the adaptation window.

### 4.3 Measurement adequacy

- Verify independent speed, shaft torque/RPM or qualified fuel-derived delivered-power path for the intended use.
- Preserve sensor uncertainty, bias, drift, resolution, bandwidth, dynamics and cross-channel covariance.
- Reject a sensor portfolio that only identifies an aggregate scale when the decision requires separate hull, propeller, wind-device or engine effects.
- Aggregated voyage fuel agreement is not proof of segment-level identifiability.

## 5. Value-of-information (VOI) objective

For candidate measurement portfolio M, define its expected value by the expected reduction in a declared decision loss/regret after observing the measurement, less lifecycle cost and operational burden:

NetVOI(M) = E[DecisionLoss before measurement] - E[DecisionLoss after measurement M] - LifecycleCost(M) - OperationalBurden(M)

The decision loss may include lifecycle fuel/energy/cost/GHG, service misses, model uncertainty or false design selection. Choose the portfolio that provides the greatest positive net value subject to installability, safety, calibration access, data rights, crew workload and maintenance constraints.

Do not use posterior entropy reduction as the only objective when different uncertain parameters have very different decision consequences. A sensor that reduces uncertainty in an irrelevant parameter may have little or no decision value.

The calculation must expose:
- prior/model discrepancy and parameter domain;
- candidate sensor specification, placement and sampling plan;
- likelihood/measurement model and cross-channel dependence;
- expected information gain and expected decision-loss reduction;
- installation, calibration, maintenance, communications, power and replacement cost;
- sampling/telemetry/storage/crew burden;
- expected downtime and failure probability;
- uncertainty in the VOI estimate itself.

## 6. Adaptive measurement and experiment design

Use a staged approach:
- **Stage A — passive audit:** inspect current channels, identities, algebraic dependencies and excitation.
- **Stage B — targeted installation:** add the sensor that resolves the highest-impact ambiguity.
- **Stage C — designed excitation:** plan safe, pre-approved operating windows that distinguish remaining parameters (e.g. varied load/speed/wind/device state).
- **Stage D — independent qualification:** validate the identified quantities on held-out voyages or physical tests.
- **Stage E — portfolio retirement:** remove or downgrade sensors whose data have no marginal value, while maintaining statutory/safety instrumentation and required redundancy.

An adaptive design may choose the next experiment to maximize expected reduction in decision regret (Bayesian experimental design), but the experiment must be safe, operationally authorized and recorded before execution.

## 7. Falsifiable benchmark protocol

Build a reference corpus of synthetic, published and permissioned measurement cases:
- independent sensors with known truth and calibrated uncertainty;
- algebraically dependent channels;
- shared upstream computed channels mislabeled as independent;
- missing calibration, stale calibration, clock drift and unknown sample rate;
- weak versus rich operating excitation;
- structural rank deficiency;
- practical ill-conditioning and high parameter covariance;
- measurement noise, bias, drift, lag, saturation and dropout;
- an apparently good route-total fit with wrong segment-level dynamics.

Compare:
- no-screening model calibration;
- feature-correlation screening only;
- physics sensitivity/rank/observability audit;
- uncertainty-aware Bayesian calibration after gating;
- adaptive measurement/experiment design based on decision VOI.

## 8. Proposed acceptance targets

Research targets, not achieved results:
- 100% detection of the declared algebraic-dependence negative controls.
- Zero structurally unidentifiable parameters reported as physically identified.
- 100% reporting of sensitivity-matrix rank, null directions and condition diagnostics for declared parameter sets.
- Zero model calibration promoted to decision-grade when mandatory observation operators are unverified.
- 100% explicit separation of measured, derived, inferred and manual fields.
- Zero future-observation leakage in prospective adaptation tests.
- Known-answer agreement for synthetic VOI scenarios.
- Sensor portfolio selection must outperform the "instrument everything" and "minimum sensor count" baselines on net decision value, not just information volume.
- On independent cases, expected decision regret after measurement selection must fall by more than the combined uncertainty in the VOI estimate.

## 9. Industry and standards comparison

- ISO 19030-2 specifies a default minimum sensor set and signal quality for comparing conventional fixed-pitch-propeller displacement ships to themselves over time; it emphasizes independent speed-through-water and a delivered-power path based on shaft torque/RPM or a qualified fuel-derived alternative. It is not a universal WAPS trial standard.
- CIMAC's *Monitoring Systems for Marine Hybrid Propulsion Systems* guideline already recognizes the importance of data quality, reliability, semantics and system integration. Our opportunity is to make sensor adequacy decision-specific and quantify the marginal value of each added channel.
- The 2026 four-tanker study establishes that calibration should be preceded by an identifiability audit. Luminous should extend this into adaptive instrumentation/experiment design for WAPS and hybrid cargo vessels, while preserving the study's careful distinction between structural, practical and physical validity.

We should not claim to invent sensor selection or Bayesian experimental design. Demonstrate a better decision-value benchmark using marine-specific observation dependencies, energy/propulsion boundaries and service-level outcomes.

## 10. Luminous system integration

- Symthaea: observation dependency graph, identifiability, decision-loss definition, VOI policy and abstention.
- Symtropy: deterministic sensitivity/Jacobian calculations, parameter estimation and physical replay within qualified domains.
- Sol-Atlas: sensor placement, coordinate frames, field-of-view/line-of-sight and installation geometry.
- Mycelix: instrument identity, calibration traceability, supplier certificates, software/firmware revisions, access rights and lifecycle maintenance provenance.

## 11. Sources

- Coșofreț et al., JMSE 14(20), 1940 (17 October 2026): https://doi.org/10.3390/jmse14201940
- Zenodo reproduction package (CC BY 4.0): https://doi.org/10.5281/zenodo.23235788
- ISO 19030-2:2016, https://www.iso.org/standard/63774.html
- CIMAC, Monitoring Systems for Marine Hybrid Propulsion Systems (2021): https://www.cimac.com/cms/upload/Publication_Press/WG_Publications/CIMAC_WG20_Guideline_Monitoring_Systems_for_Marine_Hybrid_Propulsion_Systems_Nov_2021.pdf
- Kroner et al., Metrology for reliable fuel consumption measurements in the maritime sector, Measurement 2024, DOI 10.1016/j.measurement.2024.114161: https://doi.org/10.1016/j.measurement.2024.114161

## 12. Claim ceiling

A sensor portfolio's calculated VOI is conditional on its prior, likelihood, decision loss and cost assumptions. Identifiability is model- and observation-specific. A parameter estimate is not physical truth. A complete sensor list is not proof of calibration, independence or valid acquisition. No study result establishes WAPS fuel savings, vessel safety, class/flag compliance or fleet transfer.
