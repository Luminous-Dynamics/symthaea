# PLASMA-CFRP-METAL-001 — Plasma pretreatment for conductive metal coatings on CFRP

**Date:** 13 November 2026  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11838  
**Parent:** MICROPLASMA-001 #11780; PLASMA-SURFACE-001 #11795; PLASMA-VV-001 #7588  
**Status:** literature reconstruction + benchmark contract  
**Claim ceiling:** no Luminous physical reproduction, LSP qualification, aircraft certification, or hardware authorization.

## Executive thesis

The July 2026 CFRP/zinc paper suggests a second, distinctly valuable plasma surface-processing application: atmospheric plasma pretreatment before thermal-spray metallization of carbon-fibre-reinforced polymer.

The service outcome is not "higher contact angle reduction" or "adhesive bond strength". It is a **durable, electrically functional conductive metal coating on a composite substrate without unacceptable damage to the laminate**.

This is separate from polymer/adhesive bondability. Do not pool the results with ISO 4587 polymer adhesive lap-shear data.

## 1. Source-backed starting point

Kim et al., *Effect of atmospheric pressure plasma treatments for enhanced adhesion of arc-sprayed zinc coatings on CFRP substrates*, Composites Part A: Applied Science and Manufacturing, Volume 206 (July 2026), article 109797, DOI: https://doi.org/10.1016/j.compositesa.2026.109797.

The indexed abstract and university publication record report:
- Atmospheric-pressure air and argon plasma pretreatments applied to CFRP before arc-sprayed zinc.
- Argon treatment ablated some surface epoxy and exposed carbon fibres, supporting mechanical interlocking between the substrate and incoming zinc.
- Air plasma added oxygen-containing –OH / –COOH surface functional groups, increased polar surface energy (reported as 72.08 mJ/m²) and was interpreted as supporting hydrogen bonding with ZnO species.
- Maximum pull-off adhesion of 4.21 ± 0.12 MPa for the most intense air-plasma condition, an 84% increase over the untreated CFRP comparator.
- A shift from adhesive toward predominantly cohesive failure.
- Cohesive-zone modelling was used to investigate interfacial response and delamination resistance.

The paper describes the work as a pathway relevant to CFRP structures requiring conductive metallization for lightning-strike protection (LSP). This is motivation, not evidence of LSP compliance or aircraft qualification.

Source records:
- Publisher abstract: https://www.sciencedirect.com/science/article/abs/pii/S1359835X26002447
- University of Jeonbuk publication record: https://pure.jbnu.ac.kr/en/publications/effect-of-atmospheric-pressure-plasma-treatments-for-enhanced-adh/
- DOI: https://doi.org/10.1016/j.compositesa.2026.109797

A second 2026 paper examines atmospheric plasma spray metallization of thermoset and thermoplastic polymer substrates with zinc, aluminium and copper. Its abstract emphasizes differences in substrate thermal characteristics, pre-treatment and spray conditions, and states that further optimization is needed for long-term durability. DOI: https://doi.org/10.1177/02670844261423805. It is adjacent evidence, not a matched CFRP/zinc baseline.

## 2. Critical unknowns — do not invent them

The accessible abstract / publication metadata establish the headline result and mechanisms but do not fully establish every item required for independent reproduction. The following remain to be extracted from the full methods and supplementary data before protocol equivalence can be claimed:

- Exact CFRP manufacturer, fibre/resin system, layup, fibre orientation, cure, thickness, surface finish, material lot and preconditioning.
- Exact abrasive / mechanical pretreatment (if used), and sequence relative to plasma.
- Exact plasma article, process gas/purity, nozzle, treatment standoff, traverse, pass count, power measurement boundary, waveform, and treatment-to-spray delay.
- Zinc wire/feedstock composition and lot, arc-spray equipment identity, spray distance, deposition conditions, traverse, coating thickness and post-treatment.
- Pull-off test standard, dolly geometry, fixture adhesive, cure, loading rate, specimen dimensions, replicate count, censoring / invalid specimens and failure-area classification.
- Raw replicate-level adhesion values and uncertainty construction.
- Grid-to-coating energy, gas, extraction, cooling and consumables.
- Environmental / thermal / mechanical ageing and electrical continuity after ageing.

Do not assume the article used ASTM C633 or ASTM D4541 until the actual methods confirm it. The test method is a first-class part of source identity.

## 3. Paradigm shift: co-design the interface and service

The optimal pretreatment must be evaluated together with the composite, the thermal spray process, coating microstructure, electrical joints and final service environment.

Identity key:

CFRP layup/resin/cure/surface × preparation × plasma article/regime × spray article/feedstock × deposition geometry × coating thickness/microstructure × joints/post-treatment × service profile.

The target is not the strongest coupon at any cost. Excessive resin removal may raise initial coating adhesion while weakening the laminate, exposing fibre ends, changing moisture pathways or reducing residual structural properties.

Therefore the optimization is constrained:
- maximize coating adhesion and electrical function;
- minimize laminate damage and coating defects;
- retain conductivity through ageing and mechanical/thermal cycling;
- minimize full-system cost, energy, scrap and maintenance.

## 4. Matched comparison arms

For a frozen CFRP/zinc service profile compare:
1. As-received / untreated CFRP.
2. Best applicable conventional preparation (only the source's documented preparation or a separately justified industrial incumbent).
3. Air-plasma pretreatment.
4. Argon-plasma pretreatment.
5. Conventional preparation plus plasma, only as a separately identified combination.
6. Over-treatment / resin recession / substrate-damage control.

All comparison arms share a material lot, coating feedstock lot, deposition configuration, coating thickness target, post-treatment and test method unless the design explicitly studies one of those dimensions.

A different resin, fibre architecture, zinc feedstock, coating thickness, surface preparation or pull-off protocol is a different comparison stratum. Do not pool it into the same leaderboard.

## 5. Measurement and standards anchors

### Coating adhesion

ASTM C633-24 covers tensile-normal-to-surface adhesion/cohesion testing for thermal-spray coatings. ASTM states it is useful for quality control, acceptance, equipment/procedure qualification and comparing similar thermal-spray materials. It also cautions that the result is not an intrinsic value that can directly predict service-stress survival; component geometry and residual stresses matter.

Source: https://store.astm.org/standards/c633

Use C633 only when the specimen, coating and substrate fixture are appropriate. Preserve the source paper's actual pull-off method separately if it differs. Report the failure locus and invalid/glue failures; peak stress without failure-mode classification is insufficient.

### Electrical function

Record sheet/surface resistance maps, coating continuity, contact resistance at joints, current distribution and temperature rise under the declared service current profile. The relevant standard and measurement geometry must be selected for the actual coating / component; do not invent a universal acceptance threshold.

### Substrate preservation

Measure resin recession, fibre exposure/damage, surface topography/chemistry, interlaminar damage, laminate residual properties and application-relevant moisture/thermal response.

### Manufacturing and lifecycle

Include grid-side electricity, source conversion losses, gas/compressed-air, extraction/filtration, cooling, motion, controls, zinc overspray/recovery, consumables, rejected panels, rework, maintenance, nozzle/electrode life and downtime.

## 6. Metrics

**Interface/coating:** pull-off adhesion distribution; adhesive/cohesive failure fractions; coating thickness and spatial variation; porosity, cracks, delamination and accepted area.

**Electrical:** sheet resistance distribution; continuity fraction; contact resistance; resistance drift after thermal/moisture/mechanical ageing; current-induced temperature rise.

**CFRP preservation:** resin recession; fibre exposure and damage; residual interlaminar / flexural properties where relevant; moisture response; surface morphology and chemistry.

**Manufacturing:** accepted conductive area per hour; first-pass yield; scrap/rework; energy per accepted conductive area; gas, coating feedstock and consumables per service unit; cell/line availability.

**Reliability:** batch/day variation; aged adhesion and electrical continuity; repairability; inspection intervals; failure distribution; maintenance and replacement cost.

## 7. Proposed research targets — not current results or standards

After a matched incumbent baseline and uncertainty budget are frozen:
- Seek at least 25% reduction in complete-system energy or lifecycle cost per accepted conductive area without loss of adhesion, conductivity or laminate-preservation requirements.
- Set spatial uniformity requirements for adhesion, thickness and sheet resistance from application need and measurement uncertainty.
- Confirm on a held-out CFRP batch and independent production day.
- Preserve the application-specific electrical continuity and adhesion floor after relevant environmental and mechanical ageing.
- Reject any process that improves coating adhesion by unacceptable resin recession, fibre damage or residual-laminate degradation.
- Report full distributions, failure modes, invalids, exclusions and uncertainty.

These are falsifiable goals, not performance claims.

## 8. LSP boundary

Conductive zinc coating adhesion is not lightning-strike protection qualification. LSP requires a separate system/component-level performance programme covering the complete current path, edges, joints, fasteners, repair zones, current/thermal transient, delamination, damage tolerance and residual structural strength. No aircraft qualification or certification claim is made here.

## 9. S0 offline work

Before any facility access:
- retain the source-specific headline observations and unknowns in the machine-readable literature register;
- define CFRP, coating, plasma, spray, specimen, measurement and ageing identities;
- define a synthetic evidence contract for matched arms, failure-mode completeness, sheet-resistance maps and substrate-preservation gates;
- reject contact-angle-only rankings, pooling of different CFRP grades, method substitution, LSP inference from adhesion alone and omission of grid-side utilities;
- create a held-out material-lot verification plan.

Synthetic contracts validate data/evaluator behavior only. They do not predict adhesion or authorize plasma/spray operation.

## 10. Ownership

- MICROPLASMA-001 #11780: low-temperature plasma cell architecture and device benchmark.
- PLASMA-SURFACE-001 #11795: polymer adhesive bondability, a different service endpoint.
- PLASMA-CFRP-METAL-001 #11838: conductive coating / CFRP interface qualification.
- PLASMA-VV-001 #7588: source-pinned model credibility.
- PLASMA-ENERGY-001 #11718 / PLASMA-THERMAL-METROLOGY-001 #11758: full-system energy and metrology.
- Existing manufacturing, materials, observation and experiment owners retain authority. No duplicate calibration registry, energy ledger, materials authority or hardware-control path is introduced.

## References

- Kim et al. (2026), CFRP / arc-sprayed zinc plasma pretreatment: https://doi.org/10.1016/j.compositesa.2026.109797
- Jorde et al. (2026), plasma-sprayed coatings on polymeric substrates: https://doi.org/10.1177/02670844261423805
- ASTM C633-24: https://store.astm.org/standards/c633
