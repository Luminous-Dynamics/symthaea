# Energy-storage discovery and manufacturing frontier — 13 November 2026

Status: research strategy and benchmark map. External values are source-reported comparators. No Luminous physical performance, manufacturing, safety or cost result is claimed.

## Executive thesis

Luminous should build an **evidence-qualified energy-storage invention and manufacturing capability** spanning batteries, electrochemical capacitors, hybrid devices and storage systems.

The competitive object is not a chemistry name or a record material property. It is a mission-conditioned, manufacturable device and system with measured lifetime service, power, safety, cost, yield, repairability and end-of-life outcomes.

The common engineering chain is:

`Mission → materials → interfaces → electrode architecture → process window → device → module → operating policy → field service`

Each link has a versioned identity and evidence boundary. Predictions, calculations, literature values, vendor claims, physical measurements and independent-lab results remain distinct.

## 1. Where the current work already stands

Compose with the existing owners; do not build duplicate solvers or data systems.

| Capability | Existing owner / work | What it contributes |
|---|---|---|
| Mission-conditioned battery inverse design | [#11671](https://github.com/Luminous-Dynamics/symthaea/issues/11671), [draft PR #11672](https://github.com/Luminous-Dynamics/symthaea/pull/11672) | Material + electrode + process + cell + policy portfolio; evidence-first Pareto and experiment selection |
| Practical all-solid-state translation | [#11759](https://github.com/Luminous-Dynamics/symthaea/issues/11759) | Complete-cell, pressure, loading, inactive mass and manufacturing constraints |
| Decision-value experiment selection | [#11722](https://github.com/Luminous-Dynamics/symthaea/issues/11722) | Decision regret / experiment-value comparisons at equal budgets |
| Materials discovery / qualified MLIP | MAT-RD-001 [#4997](https://github.com/Luminous-Dynamics/symthaea/issues/4997), MAT-MLIP-001 [#4992](https://github.com/Luminous-Dynamics/symthaea/issues/4992) | Candidate generation and qualified atomistic-model evidence |
| Ion transport / solid electrolytes | ION-001 [#4996](https://github.com/Luminous-Dynamics/symthaea/issues/4996), SSE-001 [#4993](https://github.com/Luminous-Dynamics/symthaea/issues/4993) | Transport and electrolyte evidence |
| Process–structure–property–performance | MAT-PSPP [#6224](https://github.com/Luminous-Dynamics/symthaea/issues/6224) | Manufacturing lineage and process-window evidence |
| Interface identity / uncertainty | MAT-INTERFACE [#6227](https://github.com/Luminous-Dynamics/symthaea/issues/6227), MAT-UQ [#6225](https://github.com/Luminous-Dynamics/symthaea/issues/6225) | Interface-conditioned observations and uncertainty |
| Capacitor device qualification | [CAPACITOR-DEVICE-DISCOVERY-001 #11841](https://github.com/Luminous-Dynamics/symthaea/issues/11841) | New complete-device EDLC / zinc-ion hybrid-capacitor lane |
| Aircraft energy-system integration | Symtropy [AERO-ENERGY-CO-DESIGN-001 #3885](https://github.com/Luminous-Dynamics/symtropy/issues/3885) | Existing mission-level comparison includes battery + supercapacitor hybrid storage; consume this owner rather than creating a competing system benchmark |

The battery programme also has separate thermal-propagation, practical solid-state and interlaboratory evidence slices. Their synthetic software qualification must not be described as physical battery qualification.

## 2. Current industrial and research frontier

### 2.1 Commercial battery manufacturing

The IEA's *Global EV Outlook 2026* reports about 1.2 TWh of EV battery deployment in 2025, nearly 30% above 2024; global lithium-ion manufacturing capacity exceeded 4 TWh, with China accounting for more than 80%. It reports latest cell energy-density values up to approximately 265 Wh/kg for NMC, 205 Wh/kg for LFP and 175 Wh/kg for sodium-ion. These are market-reported upper values, not a matched independent comparison or full-pack values.

Strategic implication: do not attempt to outbuild CATL, BYD or other incumbents in commodity-cell gigafactory scale. Compete first in specialised devices, evidence-qualified engineering, process transfer, application integration and designs where safety, power, temperature, local supply or serviceability matter more than commodity cell cost.

### 2.2 Solid-state batteries: interfaces and practical cell translation

The 26 October 2026 JACS paper *Amorphous Halide Cathodes Enable 75C All-Solid-State Lithium Batteries* is an important new comparator. It reports that amorphous LiVCl4 lowered grain-boundary-associated resistance relative to its crystalline counterpart, with a 90 wt% active-material cathode reaching 74.3 mAh/g at 5 mA/cm²; the reported high-rate configuration coupled amorphous LiTaOCl4 and reached approximately 75C, with 82.3% retention after 16,000 cycles. These results are tied to the reported research architecture and conditions; they are not automatically automotive complete-cell or pack results.

The useful paradigm is not “bulk conductivity no longer matters.” It is: **bulk conductivity is insufficient; effective transport depends on grain boundaries, conformal contact, tortuosity, interface chemistry and mechanical state.** This should become a crystalline/amorphous and interface-ablation study within SSE-001 and MAT-INTERFACE, not a duplicate materials engine.

The 7 November 2026 low-pressure sulfide report cited in [#11759](https://github.com/Luminous-Dynamics/symthaea/issues/11759) is a separate practical-cell comparator: ≤0.3 MPa operating pressure, 2.4 mAh/cm² and 0.1C at room temperature in its reported configuration. Preserve its loading, pressure, rate, cell format and maturity boundary. Do not combine it with high-rate halide or high-energy pouch-cell claims.

### 2.3 Capacitors and zinc-ion hybrid devices

Commercial EDLCs already have strong power and lifetime attributes. Eaton's XLR-48 is one reference; Skeleton's 54 V / 188 F SkelMod datasheet lists 76.14 Wh maximum stored energy, 16 kg mass, 13.5 L volume and 3.0 mΩ / 3.4 mΩ ESR at 10 ms / 1 s. These correspond to approximately 4.76 Wh/kg and 5.64 Wh/L when dividing the vendor's maximum stored energy by its module mass and volume—vendor-specification proxies, not independently measured usable energy. The same module integrates balancing, temperature sensing and CAN communication. Its datasheet states IP65 with IP covers fitted, while noting that covers are separately supplied. The EDLC benchmark therefore has to include module electronics, enclosure and field-interface requirements, not only the electrochemical stack. Eaton's XLR-48 vendor datasheet specifies a 48.6 V maximum working voltage, 166 F, 54 Wh stored energy, 5 mΩ maximum initial ESR and a 1,000,000-cycle specification under stated test conditions. This is a **module-level vendor comparator**, not an independent cell result and not a direct chemistry-level target.

The standards baseline for EDLC work includes IEC 62391-1:2022 (generic measurement / quality assessment) and IEC 62391-2:2025 (power-application EDLCs). UL 810A Second Edition was published 3 February 2026 and covers electrochemical capacitors in equipment such as UPS, emergency lighting, engine starting and power equipment. Applicability must be confirmed for the actual product and jurisdiction.

Zinc-ion hybrid supercapacitors are attractive because they can combine zinc-based Faradaic storage with capacitive/pseudocapacitive electrodes. The 2026 literature still identifies zinc instability, parasitic reactions, kinetic mismatch, electrolyte trade-offs and architecture scale-up as unresolved issues. A particularly useful 27 July 2026 *Advanced Science* comparator used a 10 mg cm⁻² carbon cathode and N/P ≈16.4 in a ZIHC pouch cell, reporting 38.3 mAh g⁻¹ at 10 A g⁻¹ after 3,000 cycles and 2.82 Wh kg⁻¹ at a 0.746 g total-device denominator. The authors improved thick-electrode transport using directional long-range pores plus CNT conductive networks. This is a stronger matched engineering target than an active-material-only headline; see [the paper](https://doi.org/10.1002/advs.76823). A 2026 multifunctional-ZIC review warns that >100 Wh/kg reports commonly use cathode mass at low loading, while practical areal/volumetric performance and working-state durability remain weak.

A 13 November 2026 review of aqueous zinc-ion batteries quantifies why mass boundaries matter: one representative value changes from 119.8 Wh/kg on an active-material basis to 33.6 Wh/kg on a whole-cell basis. It also reports that practical 1.0–2.7 Ah pouch devices commonly occupy roughly 68–79.5 Wh/L and 120–390 cycles under the reviewed material constraints. These are zinc-battery references, **not direct ZIHSC SOTA values**.

A useful longer-range parameter envelope comes from Wu et al., *Advanced Science* (6 November 2024): a representative ZIHC model indicates that >30 Wh kg⁻¹ cell energy density requires simultaneous ranges such as 10–20 mg cm⁻² carbon loading, >100 mAh g⁻¹ carbon specific capacity, N/P <20 and electrolyte-to-carbon mass ratio E/C <5. **The model explicitly excludes stainless casing, springs and current-collecting plates**, so >30 Wh kg⁻¹ is not a measured complete packaged-device result. We use these values to shape a long-range design space only; all omitted component mass/volume must be restored in physical comparisons. [Source](https://doi.org/10.1002/advs.202408997).

### 2.4 Experiment automation

The 10 November 2026 DOE/Lawrence Berkeley report on AutoBot describes exploring more than 5,000 halide-perovskite synthesis combinations while physically testing fewer than 1% and selecting informative experiments. This is a method comparator from perovskite materials, not a battery result.

A July 2026 *Nature Reviews Chemistry* review identifies scalability, generalizability and provenance-complete experimentation as central challenges for self-driving laboratories. Luminous should therefore build a human-authorized, evidence-complete experiment loop before attempting autonomous execution.

## 3. Paradigm shifts to make defining principles

1. **Material property → complete-device service.** Report active material, electrode, device, module and installed-system boundaries separately. Include all electrolyte, separator, current collectors, binder, conductive additive, casing, terminals and thermal/control hardware in the appropriate boundary.
2. **Bulk-property leaderboard → transport-network engineering.** Optimise grain boundaries, pore architecture, tortuosity, interfaces, wetting, contact retention and stress distribution as coupled variables.
3. **Chemistry-first → mission-conditioned inverse design.** Co-design composition, electrode, process window, cell architecture, pack topology and operating policy for an explicit duty cycle and environment. Preserve Pareto choices rather than declaring a universal winner.
4. **Maximum performance → robust manufacturable performance.** Score the width of a process window, yield, scrap, variability, process energy, material utilisation and cost per accepted device.
5. **More experiments → decision-value experiments.** Choose tests that can change an admissible design decision; compare random, conventional DOE, space-filling, Bayesian optimisation, uncertainty/diversity and decision-focused policies under the same frozen budgets. STOP / NO_EXPERIMENT must remain valid.
6. **Point prediction → calibrated uncertainty and abstention.** Hold out complete chemistries, designs, batches, lines and labs. Report worst-group performance, calibration, selective risk, false-feasible decisions and OOD abstention.
7. **Digital twin → evidence-bearing digital thread.** Bind every observation and prediction to immutable material lots, device revisions, process conditions, lab, fixture, calibration, protocol, raw-data digest and failure/censoring state.
8. **Single-purpose storage → multifunctional and reconfigurable systems.** Explore hybrid battery-capacitor systems, flow architectures, structural storage and repairable modules where the application values them. Keep the new physics and safety boundaries explicit.
9. **Prototype success → lifecycle economics.** Measure accepted-good-device cost, service energy, efficiency, degradation, repair, recycling, critical-material exposure and end-of-life pathways.

## 4. Benchmark architecture

Keep four benchmark tracks distinct but composable.

### A. Materials and interface discovery

- Reproduce published conductivity, transport and interface results on exact conditions before claiming parity.
- Separate bulk, grain-boundary, electrode/electrolyte interface and complete-stack impedances.
- Measure phase, composition, microstructure, pore distribution, wetting/contact, mechanical evolution and degradation where appropriate.
- Use chemistry-family, structure-family, synthesis-route and lab holdouts.

### B. Device performance

For batteries: usable Wh/kg and Wh/L at cell and pack boundaries; areal capacity/loading; charge rate; operating pressure for ASSBs; temperature window; cycle/calendar life; impedance growth; safety; full-cell mass and volume.

For capacitors: device/module Wh/kg and Wh/L; usable energy over an explicit voltage window; peak and continuous W/kg or W/L; ESR and frequency-dependent impedance; leakage/self-discharge; coulombic and energy efficiency; capacitance/ESR retention; cycle count and cumulative energy throughput; temperature behavior.

For ZIHSC: also report zinc utilization, N/P ratio, electrolyte-to-capacity ratio, loading, areal capacity, current density, corrosion/gas loss and full inactive mass.

For flow batteries: separate stack power from electrolyte energy capacity; report electrolyte concentration/volume, membrane/crossover, pumping parasitics, round-trip efficiency, stack life, balance-of-plant, footprint and installed $/kWh at a declared duration.

For structural storage: report electrochemical performance and structural properties together—specific stiffness/strength, fatigue, damage tolerance, moisture/temperature durability and retained storage after mechanical cycling. Never compare composite material metrics with cell-only Wh/kg.

### C. Manufacturing and lifecycle

- First-pass and accepted-good-device yield by batch, line and recipe revision.
- Within-roll / within-batch spatial variability and between-batch variability.
- Scrap, rework, process energy, solvent/water use, material utilisation and takt time.
- Cost per accepted device and cost per lifetime-delivered kWh or kW service.
- Process-window width and sensitivity to measured variation.
- Repairability, disassembly, recycling and critical-material exposure.

### D. Discovery policy and decision quality

- Reproduce the Discovery Learning 7.2% cycle-life test-error comparator only on an exact matched task, dataset, cutoff, feature set and split.
- Then target <5% test error on that same task as a stretch research objective, with calibrated prediction intervals and worst-design performance. Neither number is a Luminous result.
- Compare random / DOE / space-filling / uncertainty / diversity / Bayesian optimisation / decision-focused selection at equal experiment, material, time and compute budgets.
- Measure experiments to a validated feasible design, prospective top-K enrichment, normalized decision regret, false-feasible and missed-feasible rates, resource-to-decision and model abstention.
- A model is not superior because it lowers average error if it makes worse physical experiment or engineering choices.

## 5. Initial prototype portfolio and sequencing

### First: reference EDLC and battery-capacitor buffer

Use a commercial EDLC module and a well-characterised laboratory EDLC as reference controls. Build a modest carbon-carbon EDLC prototype with controlled loading, assembly and measurement. Separately, integrate a commercial battery and capacitor module in a low-voltage, current-limited power-buffer demonstrator. Measure whether peak battery current, voltage sag or thermal load improves after accounting for converter loss and added mass/cost.

This gives the programme an achievable physical engineering target without pretending that a new chemistry already exists.

### Second: aqueous zinc-ion hybrid capacitor

Treat this as a research device, not a commercial-ready chemistry. Establish the device boundary, measure all inactive mass/volume and compare against matched EDLC and relevant literature designs. Screen electrode architecture/loading and electrolyte formulation with a preregistered, batch-blocked DOE. Preserve failures, corrosion, gas evolution, self-discharge and low-yield conditions.

### Third: continue advanced battery invention

- Solid-state: crystalline/amorphous halide comparison, sulfide low-pressure translation, oxide/polymer/composite lanes kept distinct.
- Sodium-ion: focus on cold-weather, stationary, local supply and application-specific niches rather than trying to immediately exceed leading LFP cell energy density.
- Zinc/iron/organic flow and aqueous batteries: target safe stationary use and full installed-system economics, not just active-material capacity.
- Structural batteries: long-horizon research with coupled mechanical/electrochemical qualification.

### System integration owner

The battery + supercapacitor hybrid system comparison already has a system-level home in Symtropy [AERO-ENERGY-CO-DESIGN-001 #3885](https://github.com/Luminous-Dynamics/symtropy/issues/3885), which compares battery + supercapacitor storage as one aircraft energy architecture among several alternatives. The new Symthaea capacitor work owns device/material/process evidence; the existing Symtropy work owns mission-level integration, power topology and system service. Do not duplicate that aircraft benchmark.

### Fourth: pilot manufacturing capability

Start with outsourced synthesis and specialist characterisation. Add in-house electrode preparation, controlled assembly, electrochemical testing and environmental chambers only when experiment bottlenecks and partner access justify each capital purchase. A pilot line is a later gate after repeatable lab cells and process windows exist.

## 6. South African collaboration pathway

Potential collaborators and facility routes identified from public capability descriptions:

- [CSIR Energy Storage Testbed](https://www.csir.co.za/what-we-do/natural-environment/energy/energy-storage-testbed): independent battery testing and validation, local standards and recycling protocols, battery cycling and temperature testing. Confirm capacitor-specific scope and service availability before relying on it.
- [Wits Materials for Energy Research Group](https://www.wits.ac.za/physics/research-areas/condensed-matter-experiment-and-material-science/materials-for-energy-research-groupmerg/): materials modelling/design, synthesis/processing, characterisation, ion storage and supercapacitors.
- [University of Pretoria Carbon-Based Nano-Materials Research Group](https://www.up.ac.za/physics/carbon-based-nano-materials-research-group): graphene, 3D graphene foam, porous activated carbon, supercapacitor research, electrochemical testing and electrode coating equipment.

These are candidate collaboration routes, not confirmed partnerships. A partner must agree on access, IP/background rights, raw-data access, instrument calibration, safety review, publication and independent replication before work is described as partner-backed.

## 7. Standards and qualification map

Use standards as a product-specific applicability map, not as generic badges:
- EDLC: IEC 62391-1:2022 and IEC 62391-2:2025; UL 810A Second Edition (2026) where applicable.
- Industrial lithium cells/batteries: IEC 62619:2022.
- Grid-integrated electrochemical energy storage system: IEC 62933-5-2:2025.
- Vehicle, aerospace, marine, stationary and hazardous-location applications each require a separate jurisdiction / product / system assessment.

Testing against a standard is not equivalent to certification. Certification and conformity decisions belong to the appropriate accredited bodies and authorities.

## 8. Execution gates and claim discipline

| Gate | Evidence needed | What can be claimed |
|---|---|---|
| G0 — research contract | Source register, schema, synthetic negative controls, reproducible offline audit | Software contract only |
| G1 — reference reproduction | Exact reference cell/module, calibrated methods, repeatable baseline | Method reproduction on stated device |
| G2 — lab prototype | Immutable device and process identity, full-device mass/volume, measured performance, failures retained | Prototype result under stated conditions |
| G3 — independent replication | Independent lab, fixtures, calibrations, unique devices, common protocol | Reproducibility within tested domain |
| G4 — pilot process | Multiple batches, process capability, yield, cost and quality records | Pilot manufacturability in stated envelope |
| G5 — product qualification | Product-specific safety, durability, conformity and field evidence | Only the qualified claims and intended use |

For cross-laboratory variance analysis, require at least three independent laboratories and three unique physical devices per laboratory as a minimum entry condition; use a power analysis to choose the actual sample size. Do not treat repeated cycles as independent devices.

## 9. Immediate runner-independent work

1. Maintain the public programme map and source register.
2. Validate JSON schemas/fixtures locally and independently; a hosted CI queue is not a reason to stop research.
3. Add a dependency-free Rust audit command after this data contract is reviewed.
4. Prepare a partner-ready one-page experiment brief and data/IP terms.
5. Secure a rights-cleared reference dataset or a physical partner baseline.
6. Freeze a low-voltage EDLC reference protocol and hybrid-buffer duty cycle before selecting prototype materials.
7. Keep physical safety review and human authorization as mandatory gates for all laboratory execution.

### 2.4 Spatially programmed zinc-ion hybrid capacitor electrodes

A further high-loading comparator appeared on 5 October 2026: Meng et al., *Advanced Functional Materials*, “3D-Printed Dual-Gradient Gel Electrodes Synchronize Ion Transport and Active-Site Utilization for High-Energy Zinc-Ion Hybrid Supercapacitors” (DOI: 10.1002/adfm.78804; https://doi.org/10.1002/adfm.78804). The abstract reports a dual-gradient MXene/polypyrrole@CNT gel electrode made by stepwise multi-ink 3D printing: a macroscopic microlattice grid-spacing gradient combined with an intra-filament pore-structure gradient. It reports micro-CT verification, 71.1% capacitance retention across a 40-fold current-density increase, 98.8% retention after 15,000 cycles, and pouch-device metrics of 6.2 F cm⁻² and 0.86 mWh cm⁻² at 42.7 mg cm⁻² loading.

These are source-reported areal / retention metrics for that research device. The electrolyte-facing region uses larger microlattice openings, lower micron-scale porosity and higher specific surface area; the current-collector side uses smaller openings, higher micron-scale porosity and coarser transport pores. Micro-CT verifies the geometry, while COMSOL simulations—not direct local-current measurements—are reported to show suppressed concentration polarization and a more homogeneous reaction-current distribution. The authors state that supporting data are available from the corresponding author on reasonable request. The abstract alone does not establish a complete-device gravimetric or volumetric denominator, all inactive mass, package boundary, manufacturing yield or independent-laboratory reproducibility. Do not infer Wh kg⁻¹ or Wh L⁻¹. Do not merge this gel-electrode device with the separate July 2026 carbon/Zn pouch-cell result (10 mg cm⁻² carbon cathode, N/P ≈16.4, 2.82 Wh kg⁻¹ reported device boundary).

The new research question is whether an **impedance-programmed electrode** can outperform a fixed dual-gradient architecture: infer the local ionic/electronic transport and reaction-demand field, then design a spatial distribution of lattice spacing, pore connectivity, conductive network, active-site density and mechanical compliance. The gradient need not be monotonic. The field-conditioned design must be frozen before fabrication and evaluated on held-out devices/lots; unknown model domains require abstention.

A focused experiment contract is [ZIHC-IMPEDANCE-FIELD-001 #11875](https://github.com/Luminous-Dynamics/symthaea/issues/11875). It complements, rather than replaces, [ZIHSC-TRANSPORT-ABLATION-001 #11853](https://github.com/Luminous-Dynamics/symthaea/issues/11853): the latter isolates electron-network and long-range pore-channel effects in carbon cathodes; the new issue evaluates spatially varying, multi-scale gradient architecture and field-conditioned inverse design. Exact numerical reproduction of the October paper is blocked until its supplementary protocol and relevant data/rights are obtained.

## Sources

- IEA, *Global EV Outlook 2026*, battery deployment and chemistry context: https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries
- Duan et al., *Amorphous Halide Cathodes Enable 75C All-Solid-State Lithium Batteries*, JACS (online 26 October 2026): https://pubs.acs.org/doi/10.1021/jacs.6c16957
- Lee et al., low-pressure sulfide composite ASSB, *Nature Communications* (7 November 2026): https://www.nature.com/articles/s41467-026-79393-w
- Eaton, XLR-48 EDLC module datasheet: https://www.eaton.com/content/dam/eaton/products/electronic-components/resources/data-sheet/eaton-xlr-48-supercapacitor-module-data-sheet.pdf
- IEC 62391-1:2022: https://webstore.iec.ch/en/publication/66557
- IEC 62391-2:2025: https://webstore.iec.ch/en/publication/82213
- UL 810A, Second Edition (3 February 2026): https://shopulstandards.com/ProductDetail.aspx?productId=UL810A_2_S_20260203
- 2026 ZIHSC perspective: https://onlinelibrary.wiley.com/doi/full/10.1002/ente.70641
- Multifunctional zinc-ion capacitors review: https://www.nature.com/articles/s43246-026-01152-7
- Aqueous zinc-ion battery review (first published November 2026): https://cje.ustb.edu.cn/en/article/doi/10.13374/j.issn2095-9389.2026.08.21.006
- DOE / Berkeley Lab AutoBot (10 November 2026): https://www.energy.gov/science/bes/articles/accelerating-experiment-design-through-super-intelligence-driven-robotics
- Self-driving laboratories review, *Nature Reviews Chemistry* (31 July 2026): https://www.nature.com/articles/s41570-026-00847-2
- Discovery Learning, *Nature* (2026): https://www.nature.com/articles/s41586-025-09951-7
- CSIR Energy Storage Testbed: https://www.csir.co.za/what-we-do/natural-environment/energy/energy-storage-testbed
- Wits MERG: https://www.wits.ac.za/physics/research-areas/condensed-matter-experiment-and-material-science/materials-for-energy-research-groupmerg/
- UP CBNM: https://www.up.ac.za/physics/carbon-based-nano-materials-research-group
