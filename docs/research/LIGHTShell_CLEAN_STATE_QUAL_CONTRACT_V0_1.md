# Atelier Lightshell — Clean-State and Post-Clean Functional Qualification Contract v0.1

## Purpose

Qualify the ability of the exact robot-mounted garment article to be cleaned to an application-defined residual state while preserving its declared functions. This is not equivalent to domestic washability, visual cleanliness, microbiological disinfection or a hygienic-design certificate.

The protocol composes:
- Symthaea #12751 specimen/coupon and raw-data provenance;
- Symthaea #12759 method applicability;
- Symthaea #12761 mission-derived exposure plan;
- Symthaea #12802 as-built metrology and #12818 lifecycle change impact;
- Symthaea #12831 installed-system qualification and #12836 pre-registered trials;
- Symtropy #5868 protective system and #3628 endurance;
- Mycelix #5985/#6092 article/lot/passport lineage.

This module owns the application-specific clean-state decision and post-clean function retention only. It does not own a wash-method library, chemistry database, generic LIMS, robot controller, test executor, hygienic certification or safety authority.

## Industry baseline

ASP Protection already offers food-industry robot covers with smooth dirt-repellent surfaces, high-pressure-cleaner and strong-chemical cleanability claims, protected seams, ventilation/pressurization and repair options:
- https://www.asp-protection.eu/protective-covers/protective-cover-for-the-food-industry/
- https://www.asp-protection.eu/protective-covers/ventilation-for-protective-robot-cover/

This is the incumbent capability set we should benchmark against where the product and application are genuinely comparable. Lightshell must not claim cleanability, washdown compatibility or repairability as novel by themselves.

## Standards and literature boundaries

- ISO 6330:2021 defines domestic washing and drying procedures for textile articles. It specifies machine types, detergent references and drying procedures, and notes that parameter choices affect results. It does not define an industrial robot-cover washdown method or universal cleaning efficacy limit: https://www.iso.org/standard/75934.html
- ISO 105-C06:2010 concerns colour-fastness to laundering; it is not a soil-removal or cleaner-residue assay: https://www.iso.org/standard/51276.html
- EHEDG test methods help assess hygienic/aseptic equipment suitability, but EHEDG says method results are not application-specific and the process owner remains responsible for an appropriate cleaning regime: https://www.ehedg.org/certification-testing/useful-information/test-methods/
- ISO/DIS 14159 is a draft under development in 2026, not a published replacement for ISO 14159:2002. It is relevant design context for machinery hygiene where applicable, not a robot-garment compliance claim: https://www.iso.org/standard/88584.html
- Rotzler et al. (2021) reviewed e-textile washing practice and found substantial protocol variation and limited cross-study comparability: https://doi.org/10.1177/0040517521996727
- A 2024 study evaluated textile-integrated circuit washability up to 100 cycles and identified construction-dependent damage: https://doi.org/10.1002/adfm.202417344
- A 2026 functional smart-textile review identifies ongoing cracking, delamination and electrical continuity risks under repeated use: https://doi.org/10.1007/s43939-026-00672-w

No named standard may be used beyond its actual scope. The licensed method, its edition, apparatus, specimen preparation and lab competence must be approved by the method owner.

## Paradigm: clean-state as a measurable lifecycle transition

A cleaning claim is a state transition with multiple independent claims:

`article generation + zone + challenge + process + cycle index -> post-clean residue + retained function`

For each application-defined zone:
1. identify the challenge soil, batch, loading, area, application and dwell;
2. condition the article under the declared mission context;
3. measure pre-clean loading with a validated assay and baseline;
4. execute the declared cleaning process;
5. measure residual soil and cleaning-agent/rinse residue where applicable;
6. measure textile/coating/seam/attachment/sensor properties after the cycle;
7. inspect seams, folds, overlaps, fasteners, sensor carriers and drainage/vent locations;
8. repeat at the declared cycle indices and retain the degradation curves.

This is not one scalar "washable" label. Soil removal and functional retention are separate curves. One must not compensate for the other.

## Application and challenge design

The application owner declares:
- the cleaning requirement and acceptance-rule source;
- operating environment and intended cleaning interval;
- the exact zone and surfaces that must be cleaned;
- the relevant contamination/soil class and challenge load;
- acceptable residual soil and, where relevant, cleaning-agent residue;
- which materials, sensors, seams, attachments and interfaces must retain function.

P0 admits non-microbiological soil surrogates: particulate, oil, protein surrogate, mineral and synthetic marker. Soil types are not interchangeable; the exact challenge recipe, batch, area, loading, uncertainty, dwell and conditioning are version-bound.

P0 does not claim microbiological removal, kill, disinfection or sterilization. Those require a distinct application-specific validated microbiological method and responsible hygiene authority.

## Cleaning process record

Each repeated cycle binds:
- #12759 method-selection and applicability references;
- #12761 mission-exposure plan;
- cleaning mode: installed-in-place or removed module;
- cleaner product and batch; concentration and units;
- contact time and measured surface temperature;
- application mode and pressure/nozzle/standoff where applicable;
- rinse-water quality and volume;
- drying/conditioning method;
- robot isolation/lockout evidence for an executed in-place process;
- exact soil application, cleaning run, process receipt and raw-data digest.

For removed-module cleaning, preserve exact removal and reinstallation inspection records. A cleaning pass does not authorize motion or return-to-service. The robot owner and safety authority remain responsible for isolation and any later operation.

## Topology-linked cleanability sampling

A zone-level average is not enough to establish cleanability. Residual soil can concentrate at seam allowances, overlaps, folds, fastener pockets, sensor carriers, vent lips and drainage interfaces while the accessible panel face appears clean.

Use the existing pattern-topology (#12785), manufacturing-route (#12800) and as-built metrology (#12802) identities to bind an explicit set of sample-location references to each zone. The campaign binds the exact topology sample-map revision reference and SHA-256 digest; sample-point labels without this revisioned map identity are insufficient. Do not introduce a second geometry or sampling-location registry.

For each required claim gate, freeze its required sample-location references before the first cycle. Typical locations to consider, where physically present, include:
- accessible panel face / representative surface;
- seam edge, seam intersection, overlap, fold or gusset root;
- closure, attachment, fastener pocket or captive hardware interface;
- sensor tile/carrier, cable exit or encapsulation boundary;
- vent lip, drain path, recessed interface or other soil-retention feature.

These are examples, not a universal mandatory list. The application and design owners must select locations based on the exact assembled topology and declared soil/cleaning challenge. Every reference must resolve to the exact zone's topology-owned sample map. The external evidence layer must dereference the map and confirm that every required location is present in that frozen revision before admitting a physical observation.

For every required gate × zone × sample location × cleaning cycle, collect at least the declared replicate count. A missing location, unknown topology reference or insufficient replicate count makes that gate insufficient evidence. Not-applicable requires an owner and rationale; an unobserved hotspot cannot be silently excluded.

The validator returns a `SampleLocationAssessment` for each required gate × zone × sample location × cycle, including disposition, observation count, passing replicate count and worst-case bound. It also returns a gate-level worst-case summary across those location records. **Do not average a clean surface result with a failed seam or fastener pocket.** The purpose is to prevent a high-performing open surface from masking a contamination trap.

## Assay controls and uncertainty

Soil/cleaner-residue methods require:
- blank control and owner limit;
- positive control and owner minimum;
- recovery control and owner-defined recovery interval;
- LOD and LOQ for every measurement;
- instrument and current calibration;
- exact method, raw-data digest and evidence reference;
- expanded uncertainty.

Below-LOD, below-LOQ, above-range, stale-calibration and unquantified observations are not zero and cannot pass a gate. A quantified value below its own declared LOQ is an invalid quantification state.

### Supported comparison modes

**Residual fraction upper bound**

`R_upper = (post + U_post) / (pre - U_pre)`

Pass only when the point estimate is not above the limit and `R_upper <= R_limit`. If the pre-clean lower bound is not positive, the residual fraction is insufficient evidence.

**Absolute post-clean residue**

`P_upper = post + U_post`

Pass only when the point estimate and upper bound are within the owner-defined absolute residue limit.

**Retained functional property lower bound**

`F_lower = (post - U_post) / (pre + U_pre)`

Pass only when the point estimate is not below the owner minimum and `F_lower >= F_min`.

These conservative bounds do not assume cancellation or independence of uncertainty. A later covariance model must be owner-reviewed and evidenced.

## Functional claim gates

Declare only properties actually needed by the application:
- soil residue;
- cleaning-agent residue;
- textile-property retention;
- coating integrity;
- seam integrity;
- attachment integrity;
- sensor function.

Every gate binds one exact zone, metric family, unit, comparison mode, owner limit, method selection, method applicability, acceptance rule and replicate minimum. Each required gate is evaluated at every declared cleaning cycle. A sensor-free article may mark sensor function not applicable only with an owner and rationale. Unknown is not not-applicable.

All P0 required claim gates are non-compensable (the schema fixes `hard_gate=true`). Do not average a high soil-removal result against a failed seam, sensor, coating or attachment gate. Publish gate-specific outcomes.

## Visual/physical inspection

Every executed cycle has a documented post-clean inspection. Unresolved residue pockets, trapped liquid, delamination, coating loss, seam lifting, attachment loosening, sensor encapsulation damage, blocked drainage or other owner-defined defects prevent clean-state admission. Photographs support evidence but do not replace validated quantitative assays where quantitative acceptance is required.

## P0 adversarial controls

- pre-clean soil load at/below quantitation limit;
- blank/positive/recovery control outside owner limits;
- below-LOQ observation incorrectly represented as zero;
- residual-ratio interval overlaps limit;
- high apparent soil removal but seam/coating/sensor retention fails;
- chemical batch or concentration mismatches frozen process;
- stale/expired calibration or wrong method selection;
- missing cleaning cycle, replicate or zone/gate observation;
- unknown contamination class or an attempted microbiological claim;
- removed module reinstalled without a successor inspection record;
- robot operated or moved during an in-place cleaning test without external isolation evidence;
- cleaning chemistry or mission method changes without invalidating the dependent campaign.

## Benchmarks

Protocol integrity targets:
- 100% of physical cycles bind exact article generation, zone, material lot, cleaning chemistry batch, method and raw process evidence;
- 100% of measured gates bind current calibration, LOD/LOQ, uncertainty, controls and raw-data evidence;
- 100% of required gate/zone/topology-sample-location/cycle combinations meet the pre-registered replicate minimum;
- every output preserves per-location disposition and worst-case bound so hotspot-level nonpasses remain visible;
- 100% required gate/zone/cycle combinations meet replicate minima;
- zero false passes on assay controls, uncertainty-boundary, stale identity, missing-data and hard-gate adversarial tests;
- zero hidden non-passing claims behind averaged performance scores;
- zero false acceptance when a required seam, fastener, sensor, vent or drain hotspot is omitted or under-sampled.

Performance benchmarks to publish:
- residual-soil fraction by soil class and zone vs cycle count;
- cleaner residue by chemistry and rinse process;
- tensile/seam/attachment/coating/sensor retention vs cleaning cycle;
- cleaning water, chemistry, time and energy per accepted clean-state article;
- installation/removal/refit time and post-clean metrology delta;
- residue hotspot maps around seams, folds, overlaps, closures, sensor tiles, vents and drainage zones.

Compare uncovered baseline, genuinely comparable incumbent and Lightshell under matched challenge/process/method. Use #12836 for pre-registered randomized condition order where comparator trials are performed. Report raw results and uncertainty; do not invent universal acceptance numbers.

Stretch research:
- multi-lot, multi-article and independent-laboratory repeatability;
- predictive relation between coupon, panel and installed-article cleaning results;
- validated accelerated-cycle correlation against representative use intervals;
- lower resource use per accepted clean-state service-hour without functional degradation;
- zero undetected residue hotspots in the declared and independently inspected zone map.

## Claim ceiling

An admitted record supports only application-specific non-microbiological soil-removal and post-clean functional claims for the exact article, zone, soil, cleaning chemistry, process, cycle count, method and environment. It does not establish sterility, disinfection, antimicrobial efficacy, food/pharmaceutical/cleanroom suitability, worker safety, chemical regulatory compliance, ingress rating, PPE protection, robot functional safety, lifetime reliability or permission to operate.

## Proposed fixture status

The accompanying fixture is synthetic and proposed-only. It declares a required cycle horizon of three, defines three planned cleaning cycles, two required claim gates and three topology-linked sample locations (surface center, seam edge and fastener pocket), and contains zero executed runs, inspections or measurements. All identities, chemistry, limits and method references are synthetic placeholders. They are not product requirements or physical results.
