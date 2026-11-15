# PLASMA-MEDICAL — industry and state-of-the-art benchmark
**Date:** 15 November 2026  
**Related work:** [PLASMA-MEDICAL-001](https://github.com/Luminous-Dynamics/symthaea/issues/12236), [PLASMA-MEDICAL-002](https://github.com/Luminous-Dynamics/symthaea/issues/12265)  
**Purpose:** research positioning and prospective benchmark design; not a medical-device specification or clinical protocol

## Executive conclusion

The medical cold-atmospheric-plasma market is no longer an empty field. Several commercially available or clinically used device families exist, including argon plasma jets, microwave-driven plasma torches and ambient-air surface/discharge devices. In parallel, 2026 research is advancing multijet applicators and more rigorous cross-device safety characterization.

Luminous should not compete on the unsupported claim that it can make the first medical plasma jet. The stronger opportunity is an **evidence-first, metrology-native platform** that can compare devices and configurations fairly, show where exposure lands, capture uncertainty and interface effects, and connect those measurements to tissue response and clinical endpoints without pretending those layers are interchangeable.

A critical update: DIN SPEC 91315:2025-07 is already a current technical rule for medical CAP source characterization. We should map to it and extend its structured, traceable implementation—not invent a competing generic standard or claim superiority over it.

## 1. What the industry already offers

The following is a landscape map, not a head-to-head ranking. Manufacturer product claims, independent physical characterization, clinical evidence and regulatory status are separate evidence classes.

| Device / family | Publicly described architecture | Documented differentiators | Evidence / interpretation ceiling |
|---|---|---|---|
| **kINPen MED** | Argon atmospheric-pressure plasma jet (APPJ) | INPTDAT lists <3.5 W, 1 MHz, argon at 4–6 standard L/min and reported gas temperature 35–38 °C; catalogue describes it as a commercially available Class IIa medical device | Strong incumbent clinical/research reference. Specifications are catalogue data; not a universal dose definition or an independent matched efficacy comparison. |
| **Adtec SteriPlas** | Microwave-driven plasma torch / torch-style wound system | Adtec announced EU MDR certification in May 2025; its published clinical material and device instructions describe use in wound and infection management | A real commercial comparator, not a blank-market opportunity. Verify exact model, IFU revision, target-jurisdiction status, accessories and permitted indications before comparison. Do not transfer claims from one SteriPlas model/IFU revision to another. |
| **PlasmaDerm** | Direct ambient-air dielectric-barrier discharge (DBD) | Manufacturer describes geometry-specific spacers and two handpieces for large/small wound surfaces; manufacturer states MDR and ISO 13485 status | Manufacturer-reported product/design and certification claims. Clinical claims must be evaluated against independent, indication-specific study data and the current device documentation. |
| **plasma care** | Ambient-air surface micro-discharge (SMD) | Manufacturer describes a sterile disposable spacer, RFID-based single-use control, portable battery operation and a declared skin treatment area up to 73 cm² per spacer | The 73 cm² is a manufacturer-declared product treatment-area figure; it must not be treated as equivalent to a separately measured active-surface-treatment-area (ASTA) result from a defined test method. |
| **Research multijet (2026)** | Helium-based, multi-outlet plasma jet with interchangeable two- and four-jet applicators | A 7 November 2026 paper reports DIN SPEC 91315:2025-07-based characterization, surface coverage, liquid evaporation, ex-vivo temperature, electrical characterization, cytotoxicity and genotoxicity testing | Important emerging research comparator, not evidence of a clinically approved or clinically effective product. Its measured multi-jet coverage was not linear with outlet count. |

Primary/public sources:
- kINPen MED INPTDAT catalogue: https://www.inptdat.de/kinpen%C2%AE-med
- Adtec MDR certification announcement (May 2025): https://adtechealthcare.com/mdr-certification/
- Adtec product and clinical material: https://adtechealthcare.com/
- PlasmaDerm product technology: https://plasmaderm.de/en/product-technology
- plasma care product: https://www.terraplasma-medical.com/en/product/
- 2026 multijet safety study: https://www.nature.com/articles/s41598-026-78776-9

### Do not compare headline treatment-area numbers naively

Published reviews and manufacturer descriptions use different terms and methods for coverage. Examples include an applicator's physical footprint, a manufacturer's indicated treatment area, the KI-starch effective/active surface area, an agar inhibition zone, and the actual wound area covered during scanning. These are not identical quantities. Some reviews also differ in the number of device families they count, depending on whether they count technologies or individual certified products.

The benchmark must therefore report at least:
- applicator footprint;
- measured active surface treatment area (ASTA) under a named method and threshold;
- spatial map and coverage fraction at a declared target geometry;
- treated clinical area per unit time under the declared protocol;
- gaps, overlaps and hotspot distribution;
- the exact device model, head, spacer, consumable and IFU revision.

A manufacturer's declared area is an important product specification, but it is not an independent ASTA measurement or proof of uniform biological exposure.

## 2. The most important 2026 research signal: standardization is still an open problem

Boeckmann et al., *Scientific Reports* (25 March 2026), directly compared a plasma jet (kINPen MED) and a surface micro-discharge device (plasma care) in vitro. They reported materially different cell responses under their study conditions; treatment-path geometry changed cell response even when the nominal device and exposure time were held constant. The work emphasizes gas-only controls for plasma jets. It also found that plasma-conditioned PBS generated indirectly through a grid did not provide a universal cross-device comparison: biological effects scaled nonlinearly with liquid volume and treatment time, and indirect exposure omitted short-lived reactive species and non-chemical plasma components.

Source: https://www.nature.com/articles/s41598-026-45037-8

This changes our programme in three ways:

1. **There is no simple universal conversion from device settings or treatment time to biological dose.**
2. **A single indirect liquid assay is not an adequate universal proxy for direct tissue exposure.** Direct CAP and PAL/PAW need separate intervention identities and validation.
3. **Geometry belongs in the exposure record.** Scanning path, nozzle motion, standoff, active footprint, liquid layer, target curvature and local environment may change the result.

The paper's own limitations also matter: some conditions were not replicated in three independent repeats. Use it as compelling evidence of a standardization problem, not as the final answer or a new universal comparison protocol.

## 3. A new SOTA reference: the 7 November 2026 multijet paper

Ossowska et al. published a safety assessment of a helium multijet device, explicitly conducted in accordance with DIN SPEC 91315:2025-07.

Important observations from the reported setup:
- interchangeable two- and four-outlet applicators were tested;
- active surface area was measured using a KI-starch assay with separate concentrated and full-area thresholds;
- at 10 mm and four minutes, the paper reports full ASTA of 1,311 mm² for the two-jet applicator versus 787 mm² for the four-jet applicator;
- at 8 mm, the four-jet ASTA was 540 mm² versus 443 mm² for the two-jet applicator (22% greater);
- short-exposure blind spots were observed between individual plume targets;
- the authors report geometry-, distance-, voltage- and time-dependent thermal behaviour;
- ex-vivo porcine-skin temperature, evaporation, electrical characteristics, in-vitro cytotoxicity and a keratinocyte micronucleus/genotoxicity assay were included;
- the paper reports no increase in micronuclei relative to untreated controls in the tested assay and minimal cytotoxicity for the tested exposure envelope.

Source: https://www.nature.com/articles/s41598-026-78776-9

Interpretation:
- A multi-outlet head is not simply a single jet multiplied by the number of outlets.
- More emitters do not guarantee proportional active area, uniformity, lower treatment time or lower biological risk.
- Inter-plume interaction, per-emitter chemistry, spacing, ambient flow, target distance, electrical coupling, scanning path and heat accumulation must be measured.
- The study is a preclinical safety assessment. It does not establish clinical efficacy, a medical indication, regulatory authorization for Luminous, or a transferable treatment dose.

This is a particularly relevant challenge for Luminous's miniature modular plasma-array programme. Array qualification must include emitter-to-emitter distribution and failure detection, not just total generator output.

## 4. DIN SPEC 91315:2025-07 should be our interoperability baseline

DIN Media lists DIN SPEC 91315:2025-07 as the current technical rule, published July 2025, replacing DIN SPEC 91315:2014-06. Its 69-page scope covers general requirements for CAP sources for medical applications. The public contents include:
- DBD (including surface micro-discharge and volume DBD), plasma jets, plasma torches and corona concepts;
- treatment duration, treatment distance and effective treatment area;
- physical assessment including electrical current, UV radiation and gas emissions;
- biological assessment including ex-vivo pig-ear surface temperature, pH and transepidermal water loss, cell-culture cytotoxicity and genotoxicity, and microbial testing;
- chemical testing of liquid species including nitrate, nitrite, hydrogen peroxide, hypochlorous acid and pH;
- test reporting and example protocols.

Source: https://www.dinmedia.de/en/technical-rule/din-spec-91315/392129615

**Strategic decision:** map relevant existing Symthaea fields to DIN SPEC test-report concepts, preserve the exact standard edition and method identifiers, and represent any added measurements as explicitly Luminous extensions. We should not claim “DIN compliant” until a qualified test laboratory has performed the applicable tests and a standards expert has reviewed the mapping. A JSON schema cannot confer conformity.

## 5. Clinical SOTA: promising signals, still not a universal wound-treatment effect

The evidence is moving, but study populations, device architecture, comparators and endpoints remain heterogeneous.

- **2026 chronic-wound meta-analysis:** 12 RCTs, 477 participants. The authors report pooled improvements in wound-area and healing-rate outcomes and a reduction in bacterial load, while identifying heterogeneity by plasma technology and limitations in study quality. DOI: 10.1016/j.jtv.2026.101017. https://pubmed.ncbi.nlm.nih.gov/42150381/
- **2026 DFU-focused meta-analysis:** three RCTs, 107 participants and 126 ulcers. It reports an early wound-area signal, but no sustained between-group bacterial-load advantage; the authors call for larger, longer RCTs. https://pubmed.ncbi.nlm.nih.gov/41264397/
- **New 2026 DFU RCT:** an open-label randomized trial assigned 50 people with Wagner grade 2 DFUs to standard care or standard care plus helium-based CAP. Treatments were twice weekly for three weeks; wound size and targeted bacterial cultures were assessed through day 21. The authors reported greater wound contraction and bacterial-load reduction in the CAP group. It is a short, open-label study, so it does not establish long-term complete closure, recurrence prevention, limb preservation or survival benefit. https://pubmed.ncbi.nlm.nih.gov/42463739/
- **Venous-ulcer randomized trial:** an air-CAP jet study with 60 adults found numerically greater wound-area reduction but did not demonstrate statistically significant superiority at week 17 (P=0.30); complete healing was also not significantly different (P=0.361). Immediate microbial burden reductions do not establish clinical superiority. https://pubmed.ncbi.nlm.nih.gov/41859502/
- **NIPP-RD III:** a Phase III protocol for prevention of radiation dermatitis during breast radiotherapy, not tumour treatment. The registry update on 8 October 2026 reports recruitment complete, study continuing, n=141 and no Phase III results posted. https://drks.de/search/en/trial/DRKS00032560/details

These data justify research, not an across-the-board treatment claim. Clinical effects must remain tied to wound etiology, device, modality, protocol, standard care, comparator and endpoint.

## 6. How Luminous should compete

The differentiated product should be a **plasma exposure intelligence and qualification platform**, not just another handpiece.

### Layer A — device and source identity
Immutable identity for source revision, waveform family, electrode/discharge architecture, gas/lot/purity, accessories, firmware/software, calibration/service state, consumables and operator/site context.

### Layer B — exposure observability
A traceable stream of source configuration and measured interface observations with:
- calibrated sensors and methods;
- geometry/time registration;
- spatial exposure maps and active-area definitions;
- gas/UV/electrical/thermal/emission observations where applicable;
- uncertainty budgets, detection limits and missing-data states;
- raw-data manifests and cryptographic digests.

A sensor reading is an observation, not a validated biological-dose scalar. Use an analyte-specific measurement model and preserve uncertainty.

### Layer C — biological and chemical response
Separate direct CAP from PAL/PAW. Capture liquid preparation and time-dependent chemistry, assay identity, tissue/model provenance, assay controls, response timing and independent replication. Keep direct physical effects, liquid-mediated effects and assay artefacts distinguishable.

### Layer D — evidence and clinical qualification
Bind each observation to its intended indication and study scope. Clinical endpoints, standard care and adverse events live in a separate governed layer. No automatic transfer from in vitro to human benefit, or from one tissue/device/indication to another.

### Layer E — clinician-governed operation
Any eventual device should remain within a validated intended use, with human-controlled workflow, interlocks and audit logs. Symthaea may help organize evidence and identify uncertainty; this programme does not build an autonomous patient dose selector or medical actuator.

## 7. Proposed fair benchmark suite

### 7.1 Source/metrology benchmark — DIN SPEC-aligned

For each device/configuration, document the applicable DIN SPEC tests and record:
- target distance and treatment time;
- effective/active area under a specified test method and threshold;
- electrical current/energy and waveform observability;
- UV and gas-emission measurements;
- temperature, pH and TEWL as applicable;
- chemical species in liquid where relevant;
- microbial inhibition / viable count using declared organisms, matrix and controls;
- cytotoxicity and genotoxicity assays with valid positive/negative controls;
- measurement uncertainty, calibration lineage and raw-data digest.

### 7.2 Spatial/throughput benchmark

Use identical test geometry and a declared target matrix. Compare:
- ASTA and coverage fraction at declared times/distances;
- spatial 5th/50th/95th percentiles, local maxima, dead zones and overlap;
- treatment-area throughput (cm²/min) without implying biological equivalence;
- repeatability across runs, device units, consumable lots, operators and environmental conditions;
- target temperature and evaporation/hydration change;
- area coverage under both fixed-position and predeclared scanning paths.

The numerical acceptance limit should be fixed per analyte, instrument capability, device intended use and hazard analysis. Do not set a universal CV threshold before that study design exists.

### 7.3 Multijet-array benchmark

For a 1/2/4/N emitter research family, report each outlet separately as well as the combined field:
- emitter-to-emitter output variation and drift;
- spatial plume overlap, interference, blind spots and hotspot formation;
- per-outlet and total gas/energy burden;
- active-area growth as a function of geometry and time;
- target temperature distribution and accumulation during scanning;
- performance under controlled environmental perturbation;
- detection of a failed, blocked, degraded or misaligned emitter.

A higher outlet count passes only if it improves a predeclared coverage/throughput objective without worsening qualified thermal, electrical, gas, chemical or biological-risk boundaries. This is a research benchmark, not a patient-use design instruction.

### 7.4 Biological comparability benchmark

For a specific indication and model, measure both beneficial and harmful outcomes. Include appropriate tissue-relevant models, gas-only/sham controls when justified, assay-validity controls, time-resolved response, appropriate replicates and blinded analysis. Do not use conditioned PBS as a universal surrogate for direct CAP.

### 7.5 Clinical benchmark

For wound care, preserve separate outcome dimensions: complete healing, time to healing, recurrence, wound area, infection/microbial measures, pain, adverse events, rescue interventions, amputation where relevant, quality of life and cost. Preregister the primary endpoint and analysis, use best-practice standard care in comparator arms, power sample size from the actual endpoint and independently replicate. Do not optimize for a surrogate at the expense of patient-relevant outcomes.

## 8. Prospective targets that are ambitious but defensible

| Dimension | Prospective Luminous ambition |
|---|---|
| Exposure lineage | 100% of research records bind device/configuration, method, calibration state, time, geometry and provenance |
| Claim integrity | Zero synthetic false accepts across frozen modality/indication/evidence-level negative controls |
| Spatial coverage | Every coverage claim has a map, threshold, sampling plan, coverage fraction and hotspot review |
| Repeatability | Independently establish within-unit, between-unit, between-lot and between-site variance per declared measurand |
| Transferability | No cross-device, modality, tissue or indication transfer without a preregistered evidence bridge |
| PAW stability | Every formulation binds matrix, generation, container, storage, elapsed time and chemistry at application |
| Biological safety | Predeclared tissue-specific cytotoxicity/genotoxicity and temperature/emissions criteria with valid controls |
| Clinical evidence | Adequately powered, preregistered randomized comparisons with patient-relevant endpoints and transparent null results |
| External validation | At least one independent laboratory replication before any claim of reproducible preclinical performance |
| Economics | Lifecycle cost per patient reaching a prespecified clinical endpoint, only after endpoint efficacy is established |

Do not call these regulatory acceptance criteria. They are development goals. Numeric engineering limits must follow measurement-system analysis and intended-use risk assessment. Clinical sample sizes must follow power analysis.

## 9. What “better than SOTA” should mean here

Luminous should aim to outperform the field on **quality of comparison and decision evidence**, not make premature claims of better healing:
1. A substantially more complete and queryable exposure record than the minimal experimental metadata typically seen in papers.
2. An openly versioned mapping to DIN SPEC 91315 rather than a competing proprietary vocabulary.
3. Spatially resolved comparisons that reveal hotspots and blind spots, not only average power, area or treatment time.
4. Cross-device studies that separate gas-only effects, geometry and chemistry, with no false direct-CAP/PAW equivalence.
5. Evidence models that predict uncertainty and abstain on out-of-domain transfer.
6. Clinical study tracking that separates trial protocol, results, safety, efficacy, regulatory authorization and marketing authorization.
7. Independent, reproducible measurement and clinical outcomes rather than internal demo metrics.

The first defensible competitive claim would be: **Luminous can produce a complete, independently replayable exposure and evidence dossier for a specified plasma configuration and research protocol.** That capability has to be demonstrated with qualified metrology partners before it can be claimed publicly.

## 10. Immediate next work

1. Complete PLASMA-MEDICAL-002 software compile, schema validation and exact-head CI.
2. Add a versioned DIN SPEC crosswalk without copying proprietary standard text; cite the standard edition and use the licensed source for method interpretation.
3. Create an analyte-specific measurement-system analysis plan (instrument range, detection limit, calibration, uncertainty, spatial registration, replicate design).
4. Start a blind spatial-reconstruction challenge using only safe, non-clinical targets and qualified laboratory oversight.
5. Create a multijet-array challenge protocol that tests emitter interaction and coverage nonlinearity before any biological experiments.
6. Update the source register in a versioned release with the 2026 DFU RCT and November 2026 multijet safety study, preserving their distinct clinical/preclinical evidence classes.
7. Identify a biomedical/plasma physics laboratory partner and a wound-care clinical partner in parallel; do not begin patient-facing work.

## 11. Safety / regulatory status

This report is research strategy only. No patient-facing treatment settings, hardware actuation, medical treatment recommendation, human/animal experiment, clinical efficacy, device conformity or regulatory authorization is established. Device classification and market access are jurisdiction-, model-, intended-use- and date-specific. Clinical investigation requires current regulatory and ethics review and qualified institutional partners.
