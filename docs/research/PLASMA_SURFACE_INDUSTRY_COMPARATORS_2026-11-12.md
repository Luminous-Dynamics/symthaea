# PLASMA-SURFACE-001 — Industry comparators and SOTA map

**Date:** 12 November 2026  
**Programme:** Symthaea PLASMA-SURFACE-001 #11795 / MICROPLASMA-001 #11780  
**Status:** source-backed comparator map; public vendor data are not matched independent measurements.

## Executive assessment

Commercial atmospheric-plasma surface treatment is already an established industrial category. The opportunity is not to claim that Luminous invented inline plasma treatment, or that a small laboratory jet can beat a high-throughput industrial installation on line speed.

The competitive opportunity is to qualify a **right-sized, material-aware, modular process** for applications where localized treatment, flexible geometry, process-specific quality, lower changeover cost, serviceability, resource use or smaller production cells matter more than maximum web speed.

The benchmark winner must be determined by an application-specific service contract and a complete cost / energy boundary. Vendor brochure values and laboratory research values are not directly rankable.

## Industrial references

### Plasmatreat Openair-Plasma

The manufacturer describes atmospheric-pressure, compressed-air plasma intended for direct inline integration. The FAQ says that a pretreatment installation requires public mains current (230/400 V), compressed-air supply and extraction. It reports that speeds up to 400 m/min can be achieved in some applications, while example EPDM/TPE profiles operate around 2–30 m/min. These figures are vendor guidance and are explicitly application-dependent; they are not independent performance measurements for every material or geometry.

Sources:
- https://www.plasmatreat.com/en/what-is-plasma/plasma-surface-treatment
- https://www.plasmatreat.com/en/what-is-plasma/faq

Competitive implication:
- Do not use a single laboratory jet as a direct speed competitor to a production web-treatment system.
- Compare within matched substrate, treatment width, geometry, acceptance, line speed, and full utility boundary.
- Compressed air may avoid noble-gas logistics for suitable applications, but the required compressor, extraction and conditioning loads must be included.
- A localized modular system may win in cell manufacturing, complex parts, robotic treatment, low-volume/high-mix lines, maintenance access or treatment of selected regions, even when it does not win maximum line speed.

### DIENER PlasmaBeam reference listing

The Université de Sherbrooke equipment listing reports:
- treatment width: 8–12 mm;
- plasma beam diameter: up to 32 mm;
- beam length: 210 mm;
- processing speed: 100–1,000 mm/s;
- treatment while moving is required for the desired surface properties;
- nozzle distance and traverse speed materially affect the preprocessing result.

This is an equipment listing, not a full manufacturer datasheet or independent throughput qualification. It still supplies an important reference for compact atmospheric jet geometry and process motion.

Source:
https://www.usherbrooke.ca/3it/en/technological-ecosystem/microtechnologies/equipment/assembly-encapsulation-and-prototyping/diener-plasmabeam

## Recent research reference points

| Reference | Demonstrated / reported result | What it does not establish |
|---|---|---|
| Péczi-Kovács & Kókai, 20 Oct 2026, DOI 10.1007/s44245-026-00396-4 | Argon and nitrogen atmospheric plasma compared on PP/HIPS/PVC/PC/PET; lap-shear response depends on substrate and adhesive; lower contact angle does not always give higher strength | Long-duration industrial reliability, universal recipe, total-system energy, production-line economics |
| Spanou et al., 7 Feb 2026, DOI 10.1002/app.70533 | Argon CAP improves PE wettability; reported water-vapour permeability rises up to 60% and oxygen permeability up to 45% | That plasma is appropriate for every packaging-film application; a universal damage penalty |
| Liu et al., 2025, DOI 10.1039/D4LC00960F | Membrane-stabilized plasma-liquid microfluidic interface; gold nanoparticle size 7.31–11.32 nm with reported SD 1.8 nm | Industrial throughput, lifecycle cost, complete energy or gas boundary |
| Martinet et al., 2023, DOI 10.1002/ppap.202200131 | Five independently ignited / quenched RF microplasma cells | A qualified large array, cross-talk limits or long-term cell uniformity |
| Bhattacharya et al., 2026 issue / 2025 DOI 10.1109/TPS.2025.3646554 | 43-element LTCC cold atmospheric plasma array over 10 × 10 cm; paper-specific coupon result | Linear throughput scaling, field reliability, cross-application biological efficacy |

A recent 2026 roadmap emphasizes that scale-up is governed by process controllability, grid-to-outcome energy, long-term reliability and harmonised output diagnostics. It explicitly describes metre-scale surfaces, long-run industrial service, numbering-up and plug-to-product TEA/LCA as distinct scale-up dimensions.

Source: Attri et al., *Large-Scale Applications of Plasma Technologies: Progresses and Challenges*, High Voltage (published online 30 September 2026), DOI https://doi.org/10.1049/hve2.70241.

## Where Luminous should compete

### Not the first competitive claim

Do not claim:
- a 100-g power supply makes a complete portable product;
- a lower contact angle means better adhesion;
- a small cell means low energy or low lifecycle cost;
- a five-cell array establishes 100-cell performance;
- a synthetic twin establishes physical uniformity or reliability;
- one coupon test generalizes to all polymer families or adhesives;
- low-temperature plasma results transfer to thermal plasma, propulsion or fusion.

### Better initial market wedge

Focus on compact inline or robotic cells for a single, difficult, material / adhesive pair where an incumbent process has a measurable drawback:
- selective activation of complex or three-dimensional parts;
- high-mix / low-volume production with frequent part and recipe changes;
- local treatment where full-width corona or chamber processing creates waste;
- applications where a primer or solvent step may be avoided, but only after bond durability and material compatibility are proven;
- modular servicing / replacement where line downtime is expensive;
- substrates that need a wider, less damaging process window.

The candidate advantage is not assumed. It must be measured against the best applicable incumbent, including corona, flame, primer/cleaning, conventional plasma or no treatment as appropriate.

## Required competitive test

For each selected use case freeze one exact substrate grade/lot, adhesive/coating system, joint geometry, acceptance threshold, aging environment and test method.

Then compare:
1. untreated control;
2. best applicable incumbent;
3. documented commercial atmospheric-plasma reference;
4. Luminous candidate;
5. over-treatment / damage control.

Report:
- full bond-strength distribution and failure mode;
- treatment-to-bond delay curve and bondability window;
- retained bond performance after application-relevant aging;
- substrate functional-property retention;
- accepted area per hour and defect / rework rate;
- complete grid-side energy, gas/air, cooling, extraction, controls, standby and start-up;
- complete lifecycle cost per accepted bonded area;
- spatial uniformity, cell-level fault rate and service interruption;
- uncertainty, rejected runs, calibration, specimen lineage and exact configuration.

A different test method, adhesive, substrate, energy boundary or treatment geometry creates a different comparison. Do not normalize away these differences with a blended score.

## Benchmark ambition

After the exact baseline is reconstructed, test:
- 25% reduction in total-system energy or lifecycle cost per accepted bonded area, preserving all bond and substrate constraints;
- at most 5% coefficient of variation in the predeclared spatial process-output metric, when metrology uncertainty is sufficiently lower;
- at least one held-out material lot and independent production day;
- no degradation of barrier / mechanical properties beyond the application-specific limit;
- production-relevant cell replacement and recovery time;
- transparent multi-objective Pareto results for cost, quality, throughput, material damage and reliability.

The 25% and 5% values are Luminous stretch targets, not published industry standards or achieved performance.

## Adjacent microfluidic strategy

A 2025 review of microfluidic nanoparticle synthesis reports that parallelized microfluidic approaches can scale throughput while preserving product properties; one cited 128-channel lipid nanoparticle array achieved 100-fold throughput versus a single channel in that particular system. This is not a plasma-assisted benchmark result and must not be transferred to plasma devices as evidence.

It does strengthen the architectural hypothesis: for the plasma-liquid chip, preserve validated microchannel physics and number-up with balanced manifolds, local sensing, clogging / leakage detection and replaceable cartridges before simply enlarging channels.

Source: https://pubs.rsc.org/en/content/articlehtml/2025/lc/d5lc00194c

## Sources

- Plasmatreat Openair-Plasma surface treatment: https://www.plasmatreat.com/en/what-is-plasma/plasma-surface-treatment
- Plasmatreat FAQ, power / compressed-air / extraction and speed: https://www.plasmatreat.com/en/what-is-plasma/faq
- DIENER PlasmaBeam listing, Université de Sherbrooke: https://www.usherbrooke.ca/3it/en/technological-ecosystem/microtechnologies/equipment/assembly-encapsulation-and-prototyping/diener-plasmabeam
- Péczi-Kovács & Kókai (2026): https://doi.org/10.1007/s44245-026-00396-4
- Spanou et al. (2026): https://doi.org/10.1002/app.70533
- Liu et al. (2025): https://doi.org/10.1039/D4LC00960F
- Martinet et al. (2023): https://doi.org/10.1002/ppap.202200131
- Bhattacharya et al. (2026 publication record): https://doi.org/10.1109/TPS.2025.3646554
- Attri et al. (2026 roadmap): https://doi.org/10.1049/hve2.70241
- RSC microfluidic nanoparticle synthesis review (2025): https://pubs.rsc.org/en/content/articlehtml/2025/lc/d5lc00194c


## High-value 2026 polymer result detail

The 20 October 2026 Péczi-Kovács and Kókai article provides particularly useful material/adhesive counterexamples:

| Substrate + adhesive | Argon plasma result | Nitrogen plasma result | Interpretation |
|---|---:|---:|---|
| PET + Loctite 4860 | 2.4 ± 0.5 MPa; WCA 31° | 2.0 ± 0.1 MPa; WCA 13° | Argon gave stronger lap-shear despite the higher contact angle |
| PP + Soudal Fix All Flexi | 0.8 MPa; WCA 60° | 0.3 ± 0.05 MPa; WCA 17° | Argon gave stronger bond despite the much higher contact angle |
| HIPS + Soudal Fix All X-treme Power Express | 4.9 MPa; WCA 25° | 4.9 MPa; WCA 0° | Dramatically lower nitrogen contact angle did not improve the reported joint strength |
| PVC + Loctite EA 9466 | — | 12.8 ± 2.3 MPa vs untreated 2.1 ± 0.3 MPa | Nitrogen was highly effective for this specific pairing; failure occurred in the polymer substrate |

These results do not establish that argon is generally superior. They establish that gas selection is conditional on polymer chemistry, adhesive chemistry, treatment window, and the relevant failure mode. The paper also reports visible degradation / burning after prolonged nitrogen treatment for some polymers, while no visible damage was observed over its investigated argon treatment range. The latter statement is bounded to that study's geometry and parameter range.

The study's test outcomes are literature observations, not Luminous measurements. Its selected lap-shear results do not establish long-term bondability, production duty-cycle reliability, full-system energy consumption, or the superiority of either gas for a different adhesive or application.

Source: Péczi-Kovács & Kókai, published 20 October 2026, https://doi.org/10.1007/s44245-026-00396-4.
