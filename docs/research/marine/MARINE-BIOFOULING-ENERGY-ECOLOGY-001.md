# MARINE-BIOFOULING-ENERGY-ECOLOGY-001
## Co-design hull performance, in-water cleaning, coating integrity and biosecurity

**Status:** research specification v0.1.0  
**Created:** 11 November 2026  
**Implementation workstream:** #11608  
**Related:** MARINE-ACOUSTIC-ECOLOGY-001 (#11603), MARINE-HYDRODYNAMIC-VALIDATION-001 (#11442), MARINE-SERVICE-ROBUSTNESS-001, ISO 19030 hull/propeller performance trend methods.

> Research and evidence specification only. It is not a cleaning authorization, legal applicability decision, invasive-species clearance, coating certification or ship-performance claim.

## 1. Paradigm shift: manage the hull as a living, route-exposed system

Biofouling management is often treated as a fixed maintenance interval or as an anti-fouling coating procurement decision. The more useful engineering problem is to co-design:
- hull form and niche-area accessibility;
- coating/anti-fouling system, material, age and repairability;
- route, idle time, anchorage, water temperature and salinity exposure;
- fouling growth and inspection uncertainty;
- resistance, shaft power, fuel, cavitation and underwater-noise penalties;
- cleaning method, waste capture, organism/biocide release and local authorization;
- off-hire, drydock, cleaning and service-schedule impact.

The design objective is not simply the lowest roughness or fuel use. It is the best lifecycle cargo service under hydrodynamic, coating, environmental and biosecurity constraints.

## 2. Regulatory and industry snapshot

### 2.1 IMO Biofouling Guidelines and forthcoming work

IMO Resolution MEPC.378(80) adopted the revised 2023 Guidelines for the control and management of ships' biofouling to minimize the transfer of invasive aquatic species. They are guidelines, not a completed mandatory global biofouling convention.

MEPC 83 approved a new output to develop a legally binding standalone framework. PPR 13 (9–13 February 2026) agreed foundational elements and a correspondence group. IMO's biofouling page identifies a target for the PPR Sub-Committee to provide a draft legal framework and recommendations by MEPC 89 in mid-2029. These are development milestones, not current mandatory global requirements.

IMO approved MEPC.1/Circ.918 guidance on in-water cleaning in 2025. Its purpose includes supporting safe and environmentally responsible cleaning. Cleaning may damage anti-fouling coatings and release waste, hazardous substances or invasive organisms.

### 2.2 Energy, noise and maintenance interaction

The IMO recognizes that hull fouling increases resistance, fuel costs, air pollution and GHG emissions, and that hull/propeller biofouling management can contribute to reduced underwater radiated noise.

The GreenVoyage2050 hull-cleaning portal describes cleaning performance as dependent on fouling severity, vessel, operation and trading area; its 1–5% fuel-reduction range is an indicative technology-portal estimate, not a universal result.

### 2.3 Recent research

- Hadžić et al., *Biofouling of ships and offshore structures: research trends and future pathways*, Ocean Engineering 347 (15 February 2026), DOI 10.1016/j.oceaneng.2025.123988. The review covers hydrodynamics, carbon intensity, roughness inspection, mitigation and maintenance; it identifies hull inspection and maintenance as under-explored in its research mapping.
- A 2025 scenario-based analysis covering 52,636 vessels across six ship types estimates that moderate fouling can increase annual fuel consumption/GHG by an average 20–30% under the study's scenario and model assumptions. This is a modeled fleet estimate, not measured performance for every ship or a transferable saving.
- ISO 6319:2026 addresses conducting and documenting in-water cleaning of biofouling on ships. The full licensed standard text and applicability must be reviewed before encoding detailed conformity checks.

## 3. Falsifiable hypotheses

**H1 — Fouling progression is route- and duty-cycle-dependent.** Idle time, anchorage, water temperature/salinity, trading area, coating system, niche geometry and speed profile affect fouling growth. A universal calendar interval will be suboptimal across routes.

**H2 — Condition-triggered inspection can beat fixed-interval cleaning.** Increased shaft power/fuel at matched speed and loading, degraded ISO 19030 trend indicators or acoustic changes may trigger inspection, but only after weather, sea state, trim, hull/propeller state and machinery causes are considered.

**H3 — Cleaning efficacy and environmental containment are coupled.** An in-water cleaning method that restores hydrodynamic performance but releases organisms, biocides or coating particles may be environmentally worse than a slower or more expensive contained method.

**H4 — Niche areas matter.** Sea chests, bow thrusters, rudders, propellers, appendages and recesses can create transfer and performance risks not represented by average hull coverage.

**H5 — Energy and noise can improve together, but must be measured.** Hard/shell fouling can alter turbulence, cavitation and radiated noise. A cleaning intervention may affect both resistance and acoustic signatures, but the causal effect must be separately measured.

## 4. Evidence model

Every hull-condition record should bind:
- vessel identity, configuration and service fingerprint;
- hull-zone identity, geometry and inspected-area fraction;
- coating/anti-fouling-system family, revision, application date, repair and damage history;
- inspection date, water/body location, route exposure and inactivity history;
- imaging/sensor identity, calibration, image/trace digest, coverage and viewing limitations;
- fouling classification (slime, weed, soft/hard animal fouling, unknown), coverage, height/roughness or proxy, uncertainty and assessor confidence;
- invasive-species identity/confidence when confirmed by an appropriate taxonomic method; never infer species from a generic fouling class;
- niche-area inspection disposition and uninspected area;
- resistance/shaft-power/acoustic observations with environmental and operating condition;
- cleaning event, method, contractor, authorization, containment, filtration, waste manifest, discharge route, organisms/biocides/particles, coating damage and post-cleaning inspection.

A hull-wide condition statement must disclose the inspected-area denominator. A sample of accessible flat plate is not proof of clean sea chests or appendages.

## 5. Matched performance protocol

Use the vessel's declared service contract and same-vessel performance history. Compare pre/post-cleaning or coating changes under matched speed-through-water, shaft power/RPM, draft/trim, loading, sea state, wind/current and propeller/rudder state.

ISO 19030-1/-2 provides a useful framework for measuring changes in hull and propeller performance on the same vessel over time within its applicable scope. It is not an absolute comparison score between different hull designs.

For each intervention, report:
- shaft power and fuel change at matched speed/service;
- predicted/observed resistance change and method;
- uncertainty, environmental correction and exclusion reasons;
- acoustic spectrum/uncertainty when a linked ISO 17208 trial exists;
- lost time, cost, coating damage and cleaning waste disposition;
- inspected-area coverage and unresolved niche zones.

Do not infer daily speed-fuel response from annual fleet averages, or call a before/after comparison causal when route, load, sea state or configuration changed.

A structural before/after condition match is not performance qualification. It does not establish calibration, sensor authenticity, causal attribution, or a physical fuel-saving result.

## 6. Benchmark ladder

| ID | Intervention | Isolated contribution |
|---|---|---|
| F0 | Baseline coating/maintenance policy | Reference condition |
| F1 | Route-aware idle-time exposure and condition tracking | Fouling growth prediction |
| F2 | Improved niche-area access and inspection coverage | False-negative reduction |
| F3 | Condition-triggered inspection schedule | Value of inspection |
| F4 | Coating/material variant with matched operating history | Surface-system effect |
| F5 | In-water cleaning with capture/filtration | Energy restoration vs release control |
| F6 | Drydock/contained cleaning alternative | Lifecycle and off-hire comparison |
| F7 | Cleaning plus matched acoustic trial | URN interaction |
| F8 | Route-level biosecurity policy | Invasive-species risk management |

These are ablations and alternatives, not a requirement to install every measure.

## 7. Proposed targets (research goals, not results)

- 100% vessel, coating, inspection and cleaning-event provenance.
- 100% inspected-area coverage denominator and explicit uninspected areas.
- 0 unapproved or undocumented in-water cleaning actions in the test corpus.
- 100% cleaning events linked to method, containment/discharge route and waste disposition.
- 0 unknown species or cleaning-release states silently converted to zero risk.
- Report inspection false-negative rate, coverage, classification uncertainty, coating damage, organism/biocide/particle release, off-hire time and post-cleaning fuel/shaft-power change separately.
- A measured hydrodynamic benefit is claimed only when it exceeds combined measurement/model uncertainty and recurs on independent matched voyages.
- Any biological-release claim requires appropriate sampling/taxonomic evidence and chain of custody, not only a cleaning contractor's visual report.

Do not use the modeled 20–30% fleet penalty or indicative 1–5% portal savings as a design guarantee or target.

## 8. Industry and SOTA comparison

Existing methods include IMO MEPC.378(80) biofouling guidance, MEPC.1/Circ.918 in-water cleaning guidance, ISO 19030 same-ship hull/propeller performance trend methods, and specialist inspection/cleaning equipment. ISO 6319:2026 (published March 2026) gives a new method/documentation reference for conducting and documenting in-water cleaning.

The Luminous opportunity is a lifecycle evidence thread that joins route exposure, surface/coating revision, inspection coverage, species/waste evidence, measured resistance/shaft-power change, acoustic effect, maintenance cost and delivered cargo service. It should interoperate with qualified inspection and cleaning providers rather than invent organism identification or cleaning authorization. A July 2026 physics-guided incremental-learning paper provides a new comparison method for detecting gradual performance degradation and post-cleaning recovery, but its data are not permissioned for sharing; reproduce only the stated method with an independently admitted dataset. ISO 19030-1 is scoped to conventional fixed-pitch propeller ships and must not be presented as directly applicable to PWC waterjets or tug azimuth/VSP systems without a validated extension.

## 9. Standards and sources

- IMO 2023 Biofouling Guidelines, Resolution MEPC.378(80): https://www.imo.org/en/ourwork/environment/pages/biofouling.aspx
- IMO guidance on in-water cleaning, MEPC.1/Circ.918: available from the IMO Biofouling page.
- IMO PPR 13 (9–13 February 2026): https://www.imo.org/en/mediacentre/meetingsummaries/pages/ppr-13th-session.aspx
- ISO 6319:2026, conducting and documenting in-water cleaning of biofouling on ships: https://www.iso.org/obp/ui?_escaped_fragment_=iso%3Astd%3Aiso%3A6319%3Aed-1%3Av1%3Aen
- IMO GreenVoyage2050 hull cleaning: https://greenvoyage2050.imo.org/technology/hull-cleaning/
- Hadžić et al., Ocean Engineering 347 (2026), DOI 10.1016/j.oceaneng.2025.123988: https://doi.org/10.1016/j.oceaneng.2025.123988
- Scenario analysis of hull-fouling GHG penalties (2025), DOI 10.1016/j.apor.2025.104870: https://doi.org/10.1016/j.apor.2025.104870
- ISO 19030-1/-2: https://www.iso.org/standard/63774.html and https://www.iso.org/standard/63775.html

## 10. Claim ceiling

A fouling classifier pass is not species identification. A hull inspection is not full-surface coverage. A cleaning record is not proof of zero biological release. A fuel reduction after cleaning is not causal without matched conditions and uncertainty. ISO 6319 documentation does not replace local statutory authorization, port rules or operator biosecurity obligations.


## 11. SOTA update — physics-guided incremental drift and post-cleaning recovery (14 November 2026)

### 11.1 New comparator

Lang et al., *Physics-guided machine learning for ship biofouling assessment in support of maritime decarbonization*, Transportation Research Part D, Volume 156, July 2026, Article 105364, DOI https://doi.org/10.1016/j.trd.2026.105364.

The study combines a physics-informed warm-up set representing ideal, unfouled calm-water vessel states with incremental model updates over sliding temporal windows. It applies updated models at reference conditions to quantify performance change and reports improved tracking of gradual degradation and rapid post-cleaning recovery compared with conventional retraining and ISO 19030 assessment in its study.

The publisher's record states that the authors do not have permission to share the data. This is a methodological comparator, not an open reproduction dataset. Its ship-performance context does not establish transfer to PWC waterjets or harbour-tug azimuth/VSP configurations.

Source: https://www.sciencedirect.com/science/article/pii/S1361920926001574

### 11.2 Attribution boundary and causal confounders

A rising fuel/shaft-power residual or a speed loss can trigger inspection, but it does not independently diagnose biofouling. Competing causes include:
- wind, waves, current, water density/temperature, depth and route;
- payload, displacement, draft, trim and speed/control changes;
- propeller/impeller/nozzle/azimuth/VSP damage, wear, cavitation or ventilation;
- intake blockage, debris, cooling degradation or machinery efficiency changes;
- battery SOH, generator state, engine/gearbox/inverter/motor changes;
- hull/coating repair, sensor relocation, recalibration or data-pipeline changes.

Separate: observed performance residual; model/domain uncertainty; candidate-cause attribution; independent inspected hull-zone evidence; cleaning authorization/capture/waste evidence; and matched post-intervention response. The estimator must be able to abstain when causes are not identifiable.

Incremental updates must be time-causal: no future telemetry, post-cleaning observations, future inspection labels or revised calibration may leak into prior decisions. Preserve each baseline model and its historical predictions; an update is a versioned successor with explicit training window, input cutoff, configuration and source digests.

### 11.3 Standards and applicability correction

- **ISO 6319:2026** (published March 2026) covers planning, conducting and documenting in-water cleaning of external submerged surfaces, including hull and niche areas and all levels of biofouling, with or without capture. It does not cover internal piping. https://www.iso.org/standard/85997.html
- **ISO 20679:2025** (published January 2025) provides independent performance testing procedures for in-water cleaning systems, including efficacy/safety and capture/processing/disposal of debris. It does not establish universal cleaning-performance criteria; authorities remain responsible for those. https://www.iso.org/standard/86393.html
- **ISO 19030-1:2016** was confirmed current in 2022 and defines same-vessel performance-trend principles for conventional fixed-pitch propeller ships. Do not claim its direct conformance for PWC waterjet or tug azimuth/VSP architectures without a validated extension. https://www.iso.org/standard/63774.html
- IMO **MEPC.378(80)** (2023) and **MEPC.1/Circ.918** (29 April 2025) remain key biofouling-management / in-water-cleaning guidance references. https://www.imo.org/en/ourwork/environment/pages/biofouling.aspx

Keep cleaning execution/documentation, independent cleaning-system efficacy testing and hull/propulsor performance trending as different evidence families.

### 11.4 South African practical engagement context

The IMO's Regional In-Water Cleaning Technology Demonstration and Workshop took place in Cape Town on 5–8 October 2026. It brought agencies from 12 countries, industry and South African technology providers together. The National Department of Transport hosted with SAMSA and TNPA as implementing agencies. IMO reported live demonstrations by licensed South African service providers; SAMSA's 21 October sector report describes inspection, cleaning, capture, filtration, verification and documentation.

- IMO (8 October 2026): https://www.imo.org/en/mediacentre/pages/whatsnew-2511.aspx
- SAMSA sector report (21 October 2026): https://blog.samsa.org.za/2026/10/21/samsa-participates-in-test-biofouling-regional-in-water-cleaning-technology-demonstration-and-workshop/

This is evidence of a local provider/regulatory ecosystem to investigate, not an endorsement, invitation, permit or data-access grant to Luminous. First map provider capabilities and permissions; do not assume all ports, coatings or cleaning systems are authorized.

### 11.5 Future v0.2 implementation requirements

Do not silently add unreviewed fields to the v0.1.0 schema. A new version should add explicit, separately typed references for:
- normalized performance observation, raw trace, calibration and reference-condition fingerprint;
- model identity/version, training-data cutoff/window, source/configuration digests and update parent;
- time-of-decision availability, forecast/observation timing and evidence freshness;
- residual interval, uncertainty method and model-domain disposition;
- candidate causes and confounder coverage (weather, load, propulsion, condition, sensor);
- independent hull-inspection evidence and coverage;
- post-cleaning matched observation, cleaning authorization, capture and waste receipts;
- ATTRIBUTED, NOT_ATTRIBUTED, INDETERMINATE and OUT_OF_DOMAIN outcomes that cannot be conflated with simple evidence-record validity.

Reuse MARINE-SHIFT-001 #8171 for conditional calibration under hydrodynamic regime shift; do not create another conformal-prediction engine. The marine v0.2 work should adapt that qualified owner to distinguish performance residual from cause attribution and support abstention.

## 12. Product-family transfer limits

- **PWC-SIT:** inspect hull and niche areas as well as intake, grate, impeller, stator/nozzle and reverse system. Distinguish fouling from debris blockage, ventilation/cavitation, impeller damage and engine/motor/controller changes. ISO 19030 conventional-propeller indicators are not direct waterjet-PWC conformance.
- **HARBOUR-TUG:** separate hull resistance and propeller/azimuth/VSP fouling from shaft/gearbox/engine/battery/drive efficiency and dispatch mix. Match energy per completed job at comparable draft, load, speed, assist/escort task and environmental state.
- **Both:** a measured cleaning-related energy recovery needs independent pre/post matched-condition evidence with uncertainty. Cleaning legality, species risk, capture, disposal and coating integrity remain non-compensatory gates.

Linked product owners: Symthaea PWC #12095 / installed waterjet #12135; tug #12096 / EMS #12136; contextual evidence join #11612; marine configuration/lifecycle #6154.
