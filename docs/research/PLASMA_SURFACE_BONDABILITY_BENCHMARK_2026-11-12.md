# PLASMA-SURFACE-001 — Durable polymer bondability benchmark

**Parent:** https://github.com/Luminous-Dynamics/symthaea/issues/11780  
**Programme issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11795  
**Date:** 12 November 2026  
**Status:** research contract + synthetic fixture; no physical reproduction  
**Claim ceiling:** no released device, process recipe, safety case, or manufacturing qualification.

## Executive thesis

The first non-clinical application target for a modular plasma cell is durable surface activation for polymer bonding or coating preparation.

The product must not be ranked by minimum water contact angle. It must be ranked by an application-specific, durable bond outcome, accepted bonded area per complete system resource boundary, and preservation of the substrate's other required functional properties.

Research question:

> Can a compact modular plasma process deliver a specified bond-performance floor over a useful treatment-to-bond window, at production throughput, with lower complete-system energy and lifecycle cost than the best applicable incumbent—without damaging other required substrate properties?

The objective changes from maximizing apparent activation to maximizing accepted durable bond service under constraints.

## Why this benchmark now

Three findings directly change the benchmark design.

1. **Polymer response is material- and adhesive-specific.** Péczi-Kovács and Kókai (published 20 October 2026) compare atmospheric-pressure argon and nitrogen plasma for PP, HIPS, PVC, PC and PET under identical processing conditions. Both gases improved wettability and adhesive performance relative to untreated material. Nitrogen generally gave lower water contact angles; argon produced a wider processing window without visible surface degradation and achieved comparable or higher bond strength for several polymer–adhesive systems. The lowest contact angle did not consistently produce the highest bond strength. DOI: https://doi.org/10.1007/s44245-026-00396-4.

2. **Activation can damage another valuable property.** Spanou et al. (first published 7 February 2026) studied argon cold-plasma modification of LDPE and HDPE films. Wettability improved, but water-vapour and oxygen permeability increased by up to 60% and 45%, respectively, in the reported study. This makes barrier retention a mandatory application-specific constraint for packaging films; it is not safe to assume activation preserves bulk or barrier performance. DOI: https://doi.org/10.1002/app.70533.

3. **Large-scale plasma deployment depends on process-level performance.** The 2026 review *Large-Scale Applications of Plasma Technologies: Progresses and Challenges* identifies application-relevant controllability, grid-to-outcome efficiency, long-term reliability and harmonised outcome diagnostics as major scale-up constraints. https://doi.org/10.1049/hve2.70241.

The 2024 review by Primc and Mozetič also reports that adhesion does not necessarily follow wettability because of hydrophobic recovery and weakly bonded low-molecular-weight interlayers. https://doi.org/10.3390/ma17071494.

## Paradigm shifts

### 1. Optimize bondability lifetime, not initial wettability

Define a bondability window for an exact material / adhesive / process profile: the elapsed time after treatment during which the predeclared fraction of specimens still satisfies the declared bond-performance floor under the specified storage condition. In the first offline evaluator this is the greatest sampled delay in a continuous sequence of passing timepoints; it is not interpolated between samples and is not proof of continuous performance between those observations.

Measure immediate bonding and predeclared storage delays. Never infer durability from a single immediate measurement.

### 2. Optimize a constrained outcome vector, not a universal activation score

Primary outputs are bond-strength distribution, acceptance fraction, bondability window, durability, accepted area per hour, and complete-system resource use. Constraints include substrate damage, barrier retention where applicable, dimensional integrity, residue, and other application-specific functional properties.

A low contact angle is a diagnostic. It is not a product outcome.

### 3. Make material + adhesive + storage a first-class process identity

Never pool PP, HIPS, PVC, PC, PET, LDPE or HDPE into one process recipe or one headline score. Never compare different adhesive systems, cure histories, storage environments, test geometries or material lots as though they were matched.

### 4. Make inline diagnostics accountable to downstream bond results

Instrumented cells can eventually estimate whether a run remains inside a prequalified process window. But the diagnostic model must be trained against actual downstream bond outcomes, validated on held-out lots and days, and forced to return HOLD / UNKNOWN outside its validated domain. It must not self-authorize high-voltage, RF, gas or motion commands.

### 5. Co-design scale-out and the production line

Compare not only one plasma cell against one incumbent, but also line speed, treated width, overlap, array-cell uniformity, material handling, gas / air distribution, extraction, cooling, planned stops, cell replacement and yield. A small head with costly external utilities is not a compact production system.

## Frozen first comparison

Use one exact rigid-plastic adherend / adhesive pair for the first lap-shear comparison. For direct reconstruction of Péczi-Kovács and Kókai (2026), preserve their PET-G / adhesive and PP / adhesive material identities and use **ISO 4587:2003**, the method used in their study. ISO lists this edition as published/current and states that it specifies tensile lap-shear strength for rigid-to-rigid bonded assemblies; it does not provide design information. The standard was confirmed in 2021 and entered systematic review in July 2026. ASTM D3163-01(2023) may be selected for a separate protocol where appropriate, but results from different standards or specimen preparation must remain distinct.

The source paper's experimental protocol used PET-G sheet (rather than an unspecified PET grade), a 12.5 mm single-lap overlap, ISO 4587:2003, a 10 mm/s crosshead speed, and three specimens per treatment category. The joints were made immediately after treatment to minimize time-dependent surface recovery; exact bondline thickness was not measured. Reproduction must preserve these conditions and must not describe this small immediate-bond dataset as a bondability-window or lifetime study.

Freeze these arms before tuning:
- untreated control;
- the best applicable incumbent for the exact process context (corona, flame, primer/cleaning, or another justified incumbent);
- a documented commercial atmospheric-plasma reference when configuration and service boundary can be established;
- candidate modular plasma configuration;
- a deliberately over-treated / process-window negative control.

For flexible films or peel-dominant geometries, select a method appropriate to that specimen instead of reusing a rigid-plastic lap-shear test. ASTM D903-98(2025) provides comparative peel / stripping testing under defined specimen and test conditions. Do not mix peel and lap-shear values.

For corona-treated polymer film contact angle, ASTM D5946-24 explicitly states that contact angle alone is not a completely acceptable measure of adhesion and requires multiple measurements to capture spatial variation. Establish application-specific acceptance through capability studies.

## Required matched identity

Every physical result must bind to exact:
- substrate manufacturer, grade, lot, thickness, finish, conditioning and history;
- adhesive / coating product and lot, mix, cure, bondline and assembly details;
- plasma article, cell and array identity, geometry, electrode/dielectric/nozzle revision, consumable state;
- gas species / purity / flow or compressed-air supply, pressure, extraction and exhaust boundary;
- measured electrical waveform and power boundary, including supply, compressor, cooling, extraction, controls and standby;
- traverse speed, standoff, overlap, treatment width, number of passes and exposure geometry;
- treatment-to-bond delay, storage temperature / humidity / packaging;
- test standard edition, specimen geometry, test-machine calibration, conditioning and test rate;
- all failures, invalids, exclusions and missing measurements.

A comparison is NOT_COMPARABLE if the substrate/adhesive/test profile or total-system energy boundary differs. Missing auxiliary energy is UNKNOWN, never zero.

## Metrics

**Primary:** bond-strength distribution; fraction meeting the predeclared acceptance threshold; bondability window; retained strength after relevant aging; accepted bonded area per unit time; total-system kWh per accepted bonded area; and lifecycle cost per accepted bonded area.

**Substrate preservation:** barrier retention for packaging films; tensile / elongation / dimensional change where application-relevant; visible / chemical damage; contamination / residue; and process-induced embrittlement or degradation.

**Diagnostics only:** contact angle, wetting tension, surface-energy estimates, XPS / FTIR, roughness, optical emission and electrical signals. They can support explanation or process monitoring, but they cannot replace the downstream bond endpoint.

**Array performance:** per-cell and spatial output variation, cell failure rate, cross-talk, flow/power distribution, fault isolation, accepted throughput with failed cells disabled, changeover time and serviceability.

**Full system boundary:** wall-side electricity, power conversion, compressed air / process gas, pumping, cooling, extraction, exhaust treatment, standby, start-up, replacement cartridges, maintenance, failed runs, scrap and rework.

## Prospective research targets — not achieved results or standards

Once a matched baseline and uncertainty budget exist, test these hypotheses:
- at least 25% reduction in complete-system energy or lifecycle cost per accepted bonded area without deterioration in bond strength, bondability window, durability or substrate-preservation constraints;
- no more than 5% coefficient of variation in the predeclared spatial process-output metric at the frozen service point, only where metrology uncertainty is materially below that limit;
- held-out confirmation on a new material lot and production day;
- report the full distribution, failure modes, invalid trials and confidence intervals; never select only the best run;
- for a chosen array design, graceful degradation under a predeclared cell-failure topology without hiding disabled cells or silently lowering acceptance criteria.

These objectives may fail. A null result is useful if it identifies the energy, material, throughput or durability bottleneck.

## Evidence stages

- S0 — synthetic contract: offline schema, fixture and negative controls; proves only accounting and evaluator behavior.
- S1 — literature reconstruction: source-specific data and uncertainty extracted where raw data and rights permit; reported summaries stay distinct from raw observations.
- S2 — matched lab campaign: exact article identity, calibrated metrology, preregistered protocol, matched baseline and independently reviewable results.
- S3 — held-out process qualification: new lots / days / environment and fixed process window.
- S4 — production-line evaluation: sustained throughput, maintenance, yield, safety, cost and environmental boundary.

Synthetic data never promote to measured data. A model or inline sensor is not a physical bond test.

## Offline negative controls

The synthetic evaluator must reject:
1. lowest contact angle wins despite inferior bond performance;
2. different polymer grade, lot, adhesive or test geometry is pooled;
3. contact-angle-only result is claimed as adhesion;
4. missing compressor, gas, cooling or extraction energy is silently set to zero;
5. candidate and baseline use different energy boundary IDs;
6. favorable specimens are retained while invalid / failed trials disappear;
7. treatment-to-bond delay is selected after results are viewed;
8. substrate barrier / mechanical property is degraded past the declared limit;
9. over-treatment is ranked as superior because activation diagnostics look stronger;
10. synthetic fixture or surrogate output is presented as a physical measurement.

## Ownership

- Symthaea MICROPLASMA-001 #11780: miniature low-temperature process plasma and array benchmark.
- Symthaea PLASMA-SURFACE-001 #11795: durable bondability application and matched service contract.
- Symthaea PLASMA-ENERGY-001 #11718 and PLASMA-THERMAL-METROLOGY-001 #11758: energy and observation owners.
- Symthaea PLASMA-ENG-000 #5669 / PLASMA-VV-001 #7588: regime and model credibility.
- Symtropy PLASMA-TWIN-001 #1685: deterministic synthetic scenario projection / replay only.
- Existing FIELD / SE-OBS / ROB-INSTR #4866 / ROB-EXP #4800: physical observations and experiment evidence.

No duplicate energy ledger, calibration registry, plasma solver, sensor authority or hardware-control path is introduced.

## Research references

- Péczi-Kovács & Kókai (2026-10-20), *Effect of argon and nitrogen plasma surface treatment on wettability and adhesive bond strength of PP, HIPS, PVC, PC and PET-G*. https://doi.org/10.1007/s44245-026-00396-4
- Spanou et al. (2026-02-07), *Argon Cold Plasma Modification of Polyethylene Films to Tailor Surface and Barrier Properties*. https://doi.org/10.1002/app.70533
- Primc & Mozetič (2024), *Surface Modification of Polymers by Plasma Treatment for Appropriate Adhesion of Coatings*. https://doi.org/10.3390/ma17071494
- *Large-Scale Applications of Plasma Technologies: Progresses and Challenges* (2026). https://doi.org/10.1049/hve2.70241
- ISO 4587:2003, tensile lap-shear strength of rigid-to-rigid assemblies (source paper method): https://www.iso.org/standard/34852.html
- ASTM D3163-01(2023), a separate rigid-plastic lap-shear method where applicable: https://store.astm.org/standards/d3163
- ASTM D5946-24, contact angle on corona-treated polymer films: https://store.astm.org/standards/d5946
- ASTM D903-98(2025), comparative peel / stripping strength: https://store.astm.org/standards/d903

## Execution note

The first artefact tranche intentionally contains one synthetic fixture and this benchmark contract only. No physical experiment is claimed. Rust compilation, tests and CI have not been run or inferred from this document.

## Offline evaluator

The S0 fixture is consumed by the Rust-native xtask command:

    cargo run -p xtask -- plasma-surface-bondability-audit
    cargo run -p xtask -- plasma-surface-bondability-audit --json

The evaluator checks:
- explicit SYNTHETIC evidence class and false physical-reproduction flag;
- exact service-profile and energy-boundary identities;
- required subsystem completeness and subsystem-to-total energy closure;
- complete declared delay groups and at least three replicates per delay;
- bond acceptance fraction at each delay and the continuous bondability window;
- substrate-property retention gates;
- expected arm / pair dispositions and fail-closed comparability;
- contact angle retained as a diagnostic only; it is not part of winner selection.

Eleven Rust unit tests are authored for fixture replay, bondability-window calculation, untreated controls, over-treatment rejection, contact-angle invariance and optionality, missing loads, energy mismatch, insufficient replicates, and service / energy boundary mismatch.

**Execution status:** the exact branch files and fixture are fetched back and the JSON is parsed / structurally checked. This execution environment does not contain cargo, rustc or rustfmt. The Rust module and ten tests are authored but are not compiled or executed; no CI result or physical validation is claimed.


## Machine-readable literature register

The source-specific polymer observations and their claim ceilings are recorded separately from the synthetic evaluator fixture in `data/research/plasma_surface_reference_cases_v0_1.json`.

The register contains four source records and eight negative controls. It preserves the 2026 PET-G / PP / HIPS / PVC argon–nitrogen comparison, its ISO 4587 method and small immediate-bond sample limits; the 2026 LDPE/HDPE barrier-property trade-off; and review-level context on adhesion aging and industrial scale-up. Every source case explicitly records `luminous_reproduction=false`. Reported literature results are not copied into the synthetic fixture and cannot be mistaken for Luminous measurements.
