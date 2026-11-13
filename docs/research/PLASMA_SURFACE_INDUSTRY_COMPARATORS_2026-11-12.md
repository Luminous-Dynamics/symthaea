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
| PET-G + Loctite 4860 | 2.4 ± 0.5 MPa; WCA 31° | 2.0 ± 0.1 MPa; WCA 13° | Argon gave stronger lap-shear despite the higher contact angle |
| PP + Soudal Fix All Flexi | 0.8 MPa; WCA 60° | 0.3 ± 0.05 MPa; WCA 17° | Argon gave stronger bond despite the much higher contact angle |
| HIPS + Soudal Fix All X-treme Power Express | 4.9 MPa; WCA 25° | 4.9 MPa; WCA 0° | Dramatically lower nitrogen contact angle did not improve the reported joint strength |
| PVC + Loctite EA 9466 | — | 12.8 ± 2.3 MPa vs untreated 2.1 ± 0.3 MPa | Nitrogen was highly effective for this specific pairing; failure occurred in the polymer substrate |

These results do not establish that argon is generally superior. They establish that gas selection is conditional on polymer chemistry, adhesive chemistry, treatment window, and the relevant failure mode. The paper also reports visible degradation / burning after prolonged nitrogen treatment for some polymers, while no visible damage was observed over its investigated argon treatment range. The latter statement is bounded to that study's geometry and parameter range.

The study's test outcomes are literature observations, not Luminous measurements. Its selected lap-shear results do not establish long-term bondability, production duty-cycle reliability, full-system energy consumption, or the superiority of either gas for a different adhesive or application.

Source: Péczi-Kovács & Kókai, published 20 October 2026, https://doi.org/10.1007/s44245-026-00396-4.


## Method fidelity note for the 2026 polymer comparison

The study used PET-G sheet (not an unspecified PET grade), and PP, HIPS, PVC and PC sheet. It prepared single-lap adhesive joints with a 12.5 mm overlap in accordance with ISO 4587:2003, and tested them at 10 mm/s. Three specimens were tested per treatment category. Bonding was carried out immediately after plasma treatment to minimize hydrophobic recovery; bondline thickness was not measured directly. Therefore its lap-shear values are useful matched literature observations, but they do not supply delayed-bonding data, a bondability-window estimate, a large-sample reliability distribution or production-energy boundary.

For our first reconstruction, use the exact substrate grade / adhesive pairing and ISO 4587:2003 method as published. If another standard or geometry is chosen for a production-specific qualification, start a separate protocol stratum rather than pooling the values.

ISO status reference: https://www.iso.org/standard/34852.html (ISO 4587:2003 remains published/current on the ISO page; confirmed 2021; systematic review stage began 2026-07-15).


## Additional industrial comparators — product architecture and utility boundary

The industrial market is broader than Plasmatreat and DIENER. These vendor references should be treated as manufacturer or equipment-listing claims, not independent, matched performance data.

### AcXys — application-matched production integration

AcXys states that selected atmospheric plasma applications can exceed 100 m/min, and provides its own process examples:
- automotive dashboard treatment before bonding: 750 parts/day, 40 s per cycle;
- PP tube/vial treatment before screen printing: 2,000 parts/hour.

Its FAQ identifies compressed air or nitrogen, 230/400 V mains, extraction, and controlled relative motion / torch standoff as context-dependent integration requirements. It explicitly notes that manual treatment is generally not recommended for industrial applications because treatment distance and processing speed need to remain controlled. It also states that plasma activation lifetime depends on material and storage conditions (dust, moisture, UV).

Source: https://www.acxys.com/faq/

**Competitive lesson:** process speed alone is not a transferable number. Part geometry, treatment track, activation delay, line takt, quality and storage must be captured. The storage-dependent activation lifetime reinforces the bondability-window metric.

### Tantec SpinTEC — rotating atmospheric treatment

Tantec describes the SpinTEC as a high-voltage DC atmospheric-air plasma system, suitable for robotic cells, standalone use or integration into production lines. It uses two inward-facing PlasmaTEC-X nozzles on a rotating head, with stated treatment area 40–150 mm, rotation up to 1,000 rpm and processing up to 10 m/min.

Source: https://tantec.com/atmospheric-plasma-treater-spintec/

**Competitive lesson:** rotating and multi-nozzle heads are already established ways to improve circumferential coverage. Luminous must benchmark spatial uniformity and accepted part throughput, not claim that multiple plasma cells or nozzles are novel by themselves.

### Yamato PSA family — useful system-level power and gas reference

The Yamato PSA-series brochure reports:
- 900 W output per plasma head;
- compressed air at 60 psi and 30 standard L/min per head;
- treatment widths of 38–152 mm;
- 208 V three-phase / 60 Hz / 20 A or 220 V three-phase / 50 Hz / 20 A mains options;
- one- to four-head configurations (PSA110–PSA410);
- total listed equipment mass from 160 to 265 lb across those configurations.

Source: https://www.yamato-usa.com/web/content/52905

These are vendor-listed specifications, not measured wall-plug energy per accepted bond. They provide a concrete reminder that multi-head capacity can bring substantial compressed-air, electrical and equipment-mass requirements.

### relyon plasma plasmacell P300 — integrated but modular utilities

The Relyon operating manual describes the plasmacell P300 workstation for atmospheric-pressure activation and cleaning. The base module includes plasma generator, nozzle, high-voltage controller, power source and XYZ portal. Integrated extraction and compressor are listed as optional modules; an appropriate extraction system must always be available.

Source: https://www.relyon-plasma.com/wp-content/uploads/2021/03/F0354301_BA_PLASMACELL_EN_copyright-2021.pdf

**Competitive lesson:** the product system is the cell + extraction + compressed-air + motion + controls + enclosure, not the plasma source alone. A Luminous modular design should expose these utility dependencies as replaceable, metered and independently serviceable modules.

### Market implication

The current industrial market already offers:
- inline high-speed atmospheric plasma;
- multi-head, rotating and robotic treatment;
- compressed-air plasma;
- modular generator/nozzle systems;
- integrated workstations and extraction packages;
- application-specific equipment for profiles, tubes, vials and complex three-dimensional parts.

The defensible Luminous wedge is therefore not "first modular plasma tool". It is a **measured, outcome-qualified and serviceable process cell** for a narrow material / adhesive / geometry where the complete lifecycle and quality trade-off is demonstrably better. Plausible differentiators to test are:
1. lower total cost per accepted bonded area on complex or high-mix parts;
2. longer, more predictable bondability window after activation;
3. reduced spatial variation and lower over-treatment damage;
4. smaller/replaceable process cartridges and fast changeover;
5. traceable calibration, run provenance and predictable maintenance;
6. verified reduction of primer/solvent use while preserving bond durability and substrate function.

Do not claim these advantages until each has passed a matched baseline and held-out verification.


## Process monitoring is already commercial — do not claim it as a novelty

A competitive review of process monitoring materially changes our differentiation thesis.

### Plasmatreat PCU — inline monitored process variables

Plasmatreat's current Openair-Plasma PCU documentation describes:
- continuous optical monitoring of plasma emission / spectral amplitude;
- power and voltage measurement at the generator/jet;
- process-gas flow control and measurement;
- back-pressure monitoring;
- rotation-speed monitoring;
- recipe storage, multi-level user management and remote maintenance;
- EtherCAT / CANopen connections and modular PCUs that separately control jets;
- optional multi-point / triple process monitoring.

The manufacturer reports availability above 99% for its Openair systems. Treat that as a vendor claim, not independent reliability benchmarking.

Sources:
- https://www.plasmatreat.com/en/products-and-services/openair-plasmar-process-monitoring
- https://www.plasmatreat.com/en/products-and-services/process-control
- https://www.plasmatreat.com/en/news-and-stories/news-and-press/detail/pr-comprehensive-process-control-for-openair-plasma-systems

### Surfx SPM — integrated RF head monitoring

A Surfx SPM atmospheric-plasma cleaning workstation listing describes a 27.12 MHz RF source, automatic matching, closed-loop coolant control and real-time monitoring of forward/reflected power, two process-gas flow rates and head temperature, with Ethernet/RS232 connectivity.

Source: https://www.dkshsce.com/Products-en/Surfx-SPM

This is an equipment distributor/product listing, not a matched independent performance trial.

### 2026 research context: ML process diagnostics

A June 2026 review of machine-learning-enabled in-situ diagnostics for plasma-based semiconductor manufacturing describes anomaly detection, plasma diagnostics, virtual metrology, predictive maintenance, multimodal sensor fusion and process optimization. It identifies limited labeled data, interpretability, domain transfer and the lab-to-fab gap as outstanding barriers.

Source: https://doi.org/10.1021/acsami.6c04768

This review concerns plasma-based semiconductor manufacturing (e.g. PECVD, etch and sputter), not atmospheric polymer surface activation. It is an adjacent capability reference, not direct proof that the same predictive models transfer to plasma surface treatment.

## Revised Luminous differentiation hypothesis

Do **not** position "sensorized plasma", "closed-loop power/flow monitoring", "optical emission monitoring", "recipe logging", "predictive maintenance", "multi-head modular control" or "Industry 4.0 connectivity" as novel by themselves. Commercial systems already describe these capabilities.

A potentially defensible next step is to qualify **the link between upstream plasma-state telemetry and downstream material/service outcomes**:
- calibrate and retain sensor signals against actual bond, coating or product outcomes;
- identify when identical plasma telemetry produces different results because material lot, contamination, storage, adhesive, coating or geometry changed;
- predict the distribution of downstream outcomes and uncertainty, not only whether the discharge looks normal;
- validate on held-out substrate lots, devices, factories and days;
- refuse out-of-domain configurations instead of issuing an unqualified process pass;
- quantify avoided scrap/rework, reduced destructive testing, energy and lifecycle cost against incumbent quality-control practice.

The differentiator is therefore not "more sensors". It is a validated, material-aware process-quality model whose predictions are audited against actual downstream product outcomes and whose economic value is demonstrated under production conditions. Even this is a hypothesis until compared against the vendor's existing process-control and quality systems.
