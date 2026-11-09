# Marine structural health, condition-based inspection and value-of-information benchmark — V0.1

**Status:** synthetic observation/decision contract and benchmark. No structural solver, damage-growth prediction, NDT procedure or inspection authority is implemented here.  
**Parent:** [MARINE-STRUCTURAL-HEALTH-001 #11089](https://github.com/Luminous-Dynamics/symthaea/issues/11089)

## 1. The central evidence separation

This benchmark separates four evidence layers:

1. **Environment and excitation:** wave/wind/sea state, speed, heading, draft/loading, temperature, salinity and wet/dry exposure.
2. **Structural response:** strain, acceleration, vibration, deflection and acoustic-emission events.
3. **Direct condition inspection:** qualified UT thickness, visual/NDT indications, coating/EN/EIS tests or composite-specific inspection, with procedure, coverage, calibration and uncertainty.
4. **Engineering decision:** a qualified method and competent authority interpret the evidence against the exact structure, design revision, material and acceptance criteria.

A wave event is not damage. A strain anomaly is not a crack. An AE hit is not crack size. EIS/EN is not remaining wall thickness. A visual ROV survey cannot see through a laminate or coating. Missing sensor data cannot become a “no damage” result.

## 2. Reuse and authority boundary

- HYDRO-CO-DESIGN #8082 / MARINE-CAT #7719 own qualified hull/structural loads and design envelopes; this issue does not solve stress, fatigue crack growth or corrosion kinetics.
- Symtropy COMPONENT-DIGITAL-THREAD #3331 owns exact component identity, material/process provenance, NDT and repair/reinspection records.
- Symtropy AERO-SERVICE-LOOP #4056 owns generic inspection/repair/return-to-service task semantics.
- MARINE-SERVICEABILITY #11049 owns vessel task access and service context.
- MARINE-POWER-TRANSIENT #11028 and MARINE-DESIGN #7703 own operating/mission load context.
- MARINE-LIVEABOARD-EVIDENCE-JOIN #10988 binds same-configuration/missions.
- Sol-Atlas MARITIME-ATLAS #115 may provide read-only spatial coverage/freshness evidence.

The structural-health layer owns marine-specific observation, uncertainty, condition disposition, inspection planning and value-of-information contracts. It does not create a new structural model, NDT interpretation engine, reliability model, class/survey approval system or maintenance-release authority.

## 3. Research and industry context

Seyffert et al., *Structural condition-based maintenance for naval vessels: an end-to-end perspective*, Data-Centric Engineering 7 (2026), e39, published online 16 September 2026, DOI https://doi.org/10.1017/dce.2026.10073, describes an end-to-end chain from environment/load monitoring through response, structural damage assessment and maintenance decisions. It emphasizes uncertainty from fabrication quality, loading and environmental history. AE is promising for fatigue monitoring; AE + electrochemical noise (EN) may support corrosion monitoring; EIS/EN can provide coating/corrosion indicators. The paper also highlights unresolved practical issues in in-service EIS, robust localization/characterization and sensor-distance/geometry effects.

Silionis & Anyfantis, *Value of Information-based assessment of strain-based thickness loss monitoring in ship hull structures*, Ocean Engineering 357 (1 June 2026), 125434, DOI https://doi.org/10.1016/j.oceaneng.2026.125434, evaluates monitoring through decision consequence, exceedance risk, false-alarm costs, inspection timing and monitoring cost. It motivates an explicit VoI test, not a universal sensor or interval prescription.

The NSRP FY26 Technology Investment Plan identifies condition-based monitoring, corrosion detection, enhanced NDT, AUV hull inspection, corrosion/delamination assessment and early structural-health prognostics as industry R&D priorities. It is a priority list, not proof of technology maturity or small-craft economics: https://www.nsrp.org/wp-content/uploads/2026/03/NSRP-2026-Technology-Investment-Plan-FINAL.pdf

A 2026 marine centrifugal-fan condition-assessment study reports weighted F1 0.9965 on six equipment fault modes and visual [email protected] 0.861 for cracks / 0.827 for corrosion. These are fan/compartment results and are not transferable hull-damage targets: https://doi.org/10.1016/j.ijnaoe.2026.100746

## 4. Paradigm shifts to benchmark

- **Calendar-only inspection -> risk- and VoI-informed inspection:** choose where/when/how to inspect based on the consequence of a wrong decision, uncertainty, exposure and acquisition cost. Never exceed governing periodic/survey requirements.
- **Single-sensor alarms -> context-conditioned evidence:** align load/environmental context with structural response and direct inspection.
- **Point RUL -> calibrated uncertainty and abstention:** preserve model uncertainty, discrepancy, applicability and out-of-domain state. No actionable RUL without qualified degradation model and validation.
- **Sensor count -> decision-value coverage:** choose sensors for observable failure modes and decisions, with blind zones, calibration, cable/power, access, corrosion exposure and false alarms included.
- **Digital twin -> inspection-aware evidence thread:** exact region, datum, structure/material revision, sensor/inspection history, as-built lineage, coverage and authority decisions.
- **Monitoring everything -> selective monitoring plus targeted NDT:** evaluate whether a few contextual sensors and planned inspections outperform dense permanent instrumentation after full lifecycle cost.

## 5. Material families and modality scope

Keep metallic, composite and hybrid structures separate:
- Marine aluminium/steel: fatigue hot spots, weld/detail stress, general/localized corrosion, coating loss and thickness reduction.
- Fibre-reinforced composite: delamination, impact damage, moisture ingress, bond/joint degradation and laminate-specific NDT.
- Bonded/bolted/hybrid joints: interface degradation, fastener corrosion and load-transfer changes.

Modality boundaries:
- Wave radar, anemometer, GNSS, speed/heading, draft/load, temperature and wetness contextualize excitation/exposure; they do not directly detect damage.
- Strain/FBG/accelerometers measure response at instrumented locations; interpretation requires qualified structure models, sensor placement, calibration and load context.
- AE can detect active events; source localization depends on geometry, coupling, propagation, noise and spacing. An AE hit is not a crack size.
- EIS/EN are coating/corrosion indicators; site representativeness and in-service interpretation remain maturity concerns. They do not replace UT thickness/NDT.
- UT/PAUT and other NDT are direct inspection evidence only within the qualified procedure, operator, coverage and uncertainty envelope.
- ROV/AUV visual survey documents accessible surfaces; it does not prove hidden/internal condition or complete coverage.
- Composite NDT methods are laminate/geometry-specific; do not transfer metal-plate acceptance rules.

## 6. Condition and decision semantics

The independent evaluator must distinguish:
- `valid_within_declared_scope`: only the exact qualified inspection/criterion has support; this is not global “hull healthy”.
- `inspection_required`: an anomaly, exposure or consequence makes a specified inspection/review prudent under the supplied decision policy.
- `indeterminate`: critical identity, data quality, coverage, calibration, uncertainty or applicability is insufficient.
- `unsupported`: requested interpretation is outside the available validated method/material/geometry/operating envelope.
- `rejected_by_qualified_evidence`: a qualified direct inspection explicitly fails its supplied acceptance criterion or identity check.

Next actions are recommendations only: continue the existing approved plan, target inspection, request competent structural review, escalate to the governing authority, or no recommendation because evidence is indeterminate/unsupported. No system may autonomously extend an approved inspection interval, declare seaworthiness or authorize repair/deferment.

## 7. VoI contract

Compare fixed-calendar inspection, risk-ranked targeted NDT, threshold-triggered inspection, probabilistic update and VoI policy on identical prior, exposure, sensor likelihood, consequence model, inspection cost and downtime boundary. A perfect-information oracle is a theoretical upper bound only.

For a qualified decision model:
- expected loss before information = minimum expected consequence across permitted actions under current information;
- expected loss after information = expected minimum consequence after observing the proposed information;
- expected value of sample information (EVSI) = before-information expected loss minus after-information expected loss;
- net VoI = EVSI minus inspection/sensor acquisition, installation, calibration, maintenance and downtime costs within the declared boundary.

Do not calculate or rank VoI where probabilities, consequence costs, likelihoods, authority constraints or the cost boundary are missing/unqualified. Return `unknown` rather than invent a benefit. Synthetic numerical fixtures use fictional cost points only and are not currency or risk recommendations.

## 8. Metrics and fair comparison

Report separately by material, failure mechanism, severity, geometry and environment:
- detection and false-positive rate only against labelled/qualified reference data;
- localization error and coverage;
- thickness/strain bias, RMSE and uncertainty interval coverage against reference inspection;
- probability calibration, out-of-domain and abstention rates;
- inspection coverage and blind-zone fraction;
- lead time to independent confirmation;
- false alarm / missed-damage consequences;
- EVSI, net VoI, inspection cost, downtime and sensitivity to prior/likelihood/model error;
- time to competent review and closure;
- sensor failure, calibration and requalification burden;
- p50/p95/worst outcomes.

Never pool corrosion, fatigue cracking and composite delamination into one headline F1. Never compare a monitoring policy with different priors, costs, inspection methods or authority limits.

## 9. Acceptance targets — goals, not achieved results

- Every observation binds exact component/region, configuration, datum, method and evidence class.
- Zero “no indication” dispositions when coverage, calibration, data quality or detectability is insufficient.
- Zero false-safe outcomes on wrong-identity, stale calibration, out-of-domain, missed-coverage and qualified-NDT-negative controls.
- 100% of damage/RUL outputs preserve uncertainty, model applicability and validation refs; no unqualified point RUL.
- Exact agreement with independent oracle on all finite synthetic condition/VoI cases.
- No VoI policy recommends exceeding a governing inspection/survey limit.
- A sensor/inspection strategy wins only if net decision benefit exceeds its total acquisition and lifecycle cost on held-out cases without increasing missed-critical-damage risk.
- No safety, hull-life or inspection-interval percentage claim before measured and qualified evidence.

## 10. Validation ladder

1. Synthetic known-answer and negative-control contract tests.
2. Laboratory coupons/reference specimens using qualified artificial defects and calibrated instrumentation.
3. Sensor/NDT trials across declared geometry, material, thickness, coating and environment.
4. Instrumented structural component/mock-up or scale model with controlled loading and independent reference inspection.
5. Full-scale vessel trial comparing continuous observations with independent periodic inspection and measured load/weather histories.
6. Longitudinal fleet exposure before reliability, damage-growth or inspection-interval claims.

## 11. Executable Rust primitives

A dependency-free crate now exists at `crates/domains/symthaea-marine-structural-health`. It deliberately implements only rules that can be evaluated from caller-supplied evidence:
- exact region and structure-revision identity;
- qualified method, data-quality, calibration and coverage gates;
- conservative measurement-interval comparison against authority-supplied bounds;
- no-indication-within-declared-scope semantics;
- VoI arithmetic when the prior/likelihood/consequence/cost model is declared qualified;
- advisory recommendation mapping where mandatory inspection remains dominant.

It does not calculate structural stress, fatigue growth, corrosion kinetics, signal interpretation, NDT results, RUL, inspection intervals or repair approval. It creates no default acceptance limits. A direct no-indication result is limited to the declared region/revision, method, criterion and coverage.

Fourteen unit tests are authored for the narrow primitives, including uncertainty crossing a lower limit, fully outside intervals, stale calibration, incomplete coverage, wrong region, superseded structure revision, missing criterion, research-only method and mandatory-inspection precedence over negative VoI. **These tests are written but have not been compiled or executed in this environment** because Rust/Cargo are unavailable. The branch remains subject to exact-head Cargo test, Clippy and formatting before any software PASS is claimed.

## 12. Claim ceiling

A synthetic PASS validates only contract semantics. Sensor alerts, ML scores, digital twins, AE events, coating proxies, visual surveys and RUL estimates do not establish structural adequacy, safe operating limits, seaworthiness, class/flag approval or permission to defer an inspection. Structural decisions remain with qualified engineers, inspectors, surveyors and competent authorities under the actual vessel's applicable requirements.
