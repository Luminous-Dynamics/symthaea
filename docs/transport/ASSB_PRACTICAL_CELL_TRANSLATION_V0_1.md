# ASSB practical-cell translation benchmark v0.1

Status: evidence-contract and synthetic audit scaffold. Not an electrochemical model, cell design recommendation, manufacturing process recipe, safety evaluation or product qualification.

## Purpose

Evaluate whether solid-electrolyte and interface advances survive translation to complete cells under explicitly bounded fabrication and operating pressure, realistic areal loading and complete-cell mass accounting.

This composes existing material discovery, ion transport, interface, process–structure–property–performance, uncertainty and sample-lineage owners. It does not create a new electrolyte predictor, ion-transport solver, interface model, degradation model or experiment authority.

## Recent benchmark anchors

- Lee et al., *Nature Communications* (published 7 November 2026), reported a sulfide composite-electrolyte/current-collector design operating at room temperature under **≤0.3 MPa stack pressure**, at **2.4 mAh/cm² areal loading** with NMC811 at 0.1C. The mechanism couples a self-polymerizable thioctic-acid-modified composite electrolyte with a porous carbon current collector. This is a study-specific cell demonstration, not proof of EV fast charging, factory yield, long-life qualification or pack-level performance. DOI: 10.1038/s41467-026-79393-w. https://www.nature.com/articles/s41467-026-79393-w
- Zang and Qin, *Interdisciplinary Materials* (1 July 2026), review low-stack-pressure silicon ASSBs and report that many state-of-the-art demonstrations depend on pressures above 50 MPa, compared with practical cell manufacture/operation goals at or below 2 MPa. They organize the problem across electrode design, interface engineering, structural optimization and assembly. DOI: 10.1002/idm2.70068. https://doi.org/10.1002/idm2.70068
- Cui and Kong, *ECS Journal of Solid State Science and Technology* (30 March 2026), review the transition from laboratory ASSB fabrication to industrial scale-up. DOI: 10.1149/2162-8777/ae5451. https://doi.org/10.1149/2162-8777/ae5451
- The 2026 sulfide electrolyte review identifies moisture degradation/H₂S release, interface reactions, raw-material cost, solvent recycling, thin membranes, continuous manufacturing and low-pressure operation as translation constraints. https://www.sciencedirect.com/science/article/pii/S2352152X26043392

These sources concern different chemistries, architectures and test boundaries. The audit does not pool their data or transform any result into a universal target.

## Paradigm shift

Compare the complete mechanical/electrochemical stack rather than ranking one material property:

`electrolyte | cathode/SSE interface | cathode composite | anode/SSE interface | anode/current collector | pressure/containment architecture`

**Fabrication pressure and operating pressure are distinct variables.** A high densification pressure during manufacturing is not evidence that a cell requires high stack pressure during cycling; conversely, a low assembly pressure does not prove stable operation at low pressure. Both must have separate measurements, methods, uncertainty and evidence.

## Technology lanes — do not pool

- Sulfide / lithium metal ASSB
- Halide / compatible-anode ASSB
- Oxide / garnet / NASICON ASSB
- Polymer and composite solid electrolyte
- Silicon-based ASSB
- Hybrid solid–liquid / quasi-solid cell

The hybrid lane cannot be promoted to an all-solid-state claim. Silicon-based anode results cannot be used as lithium-metal results.

## Matched comparison protocol

Every comparison declares:
- exact design and revision;
- chemistry lane, electrolyte/cathode/anode family and cell format;
- test level (full cell, symmetric cell or half cell);
- areal capacity, cathode loading, electrolyte thickness;
- temperature, current density, voltage window and cycle protocol;
- separate fabrication and operating pressure;
- evidence record, source digest and metrology.

A pressure comparison matches the non-pressure axes and intentionally varies operating pressure. Full-cell outcomes require complete-cell mass accounting. Symmetric-cell and half-cell observations remain useful mechanistic evidence but cannot support full-cell energy or lifetime claims.

## Metrics to report

Electrochemical: conductivity, activation energy, interface resistance and growth, critical current density under the exact protocol, areal capacity, current density, coulombic efficiency, retention, energy throughput and failure/censoring.

Mechanical: fabrication pressure, operating pressure, pressure distribution, pressure relaxation, contact retention, swelling, creep, voids, fracture and post-test structure.

Cell engineering: complete-cell Wh/kg and Wh/L (when raw measured energy is available), areal energy, electrolyte thickness, lithium excess, inactive mass, pressure hardware and thermal/power constraints.

Manufacturing: membrane thickness/porosity/density distribution, process window, humidity sensitivity, yield, scrap/rework, process energy, solvent/water use, throughput and cost per accepted cell.

These remain separate dimensions. A weighted score may not hide an infeasible or unsupported condition.

## Baselines

- High-pressure ASSB control with exact measured pressure (not a vague “high-pressure” category).
- Low-pressure / near-ambient candidate, including the ≤0.3 MPa literature anchor when exact methods/data permit.
- Liquid-electrolyte Li-ion reference under matched complete-cell test boundary.
- Component ablations for interface, current collector, electrolyte and pressure architecture.
- Random, DOE/factorial, uncertainty/diversity and decision-value experiment selection baselines when designing a prospective campaign.

## Initial numeric targets — research targets, not achievements

1. Reproduce the 7 November 2026 near-ambient sulfide full-cell study on its exact design, protocol and measurement boundary where source data and rights permit.
2. Map pressure response with fabrication and operating pressure separated; include near-ambient and ≤2 MPa operating conditions.
3. Demonstrate complete-cell operation at realistic areal capacity under low pressure, with energy, impedance, pressure and lifetime measured together.
4. Reduce electrolyte thickness, lithium excess and inactive mass without sacrificing interfacial stability or manufacturing yield.
5. Confirm any pressure/interface advantage on independent fabrication batches.
6. Measure yield and accepted-cell cost before claiming scale-up.
7. Set numeric life and charge-rate thresholds only after reproducing the reference and measuring repeatability/power.

## Industry comparator and practical translation targets

The USABC October 21, 2025 solid-electrolyte RFPI is a particularly useful automotive comparator because it separates **material-level goals** from **cell-level goals**, calls for 5 Ah or larger cells, and supplies test methods rather than a single marketing figure.

| USABC level | Parameter | Goal | Method / boundary |
|---|---|---:|---|
| Material | Ionic conductivity | >10 mS/cm at 30°C; >1 mS/cm at 0°C | Pellet |
| Material | Electronic conductivity | <1×10⁻⁸ mS/cm | Pellet |
| Material | Critical current density | >13 mA/cm² | Pellet |
| Material | Thermal stability | >250°C | DSC, <1 J/g, 5°C/min |
| Material | Water stability | <10% ionic-conductivity loss | 72 h, −40°C dew point |
| Material | Cost | <$10/kg | 75 kt/year scale assumption |
| Cell | Stack pressure | <2 MPa | USABC EV Test Manual |
| Cell | Low-temperature usable energy | >70% of 30°C usable energy | −20°C, C/3 discharge, unassisted |
| Cell | Electrolyte thickness | <20 μm | Sheet |
| Cell | Calendar life | >15 years | EV goals |
| Cell | Cycle life | >1,000 cycles | Dynamic stress test |

These are aspirational development goals, not current market averages or independently achieved Luminous results. The 5 Ah minimum applies to the cell-level RFPI context. Read the official appendix and associated test manual before using these numbers in a formal comparison:
https://uscar.org/wp-content/uploads/2025/12/USABC-Electrolyte-RFPI.pdf

A new sulfide full-cell paper published 7 November 2026 reports room-temperature operation at ≤0.3 MPa and 2.4 mAh/cm² at 0.1C. This is a valuable research frontier because it is far below the USABC <2 MPa target, but it does not by itself establish >5 Ah format, fast charge, >1,000 cycles, calendar life, complete-cell energy density or production yield:
https://www.nature.com/articles/s41467-026-79393-w

The industry scale-up bar is also manufacturing, not just electrochemistry. IEA reports >4 TWh of global Li-ion nameplate capacity by end-2025, more than 80% of that capacity in China, and says >90% average manufacturing yield is required to compete in today's market; it also notes that new plants can take more than five years to approach nominal output. These are industry context/competitive thresholds, not ASSB-specific demonstrated yield:
https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries

### Our matched evaluation ladder

1. **Material:** reproduce ionic/electronic conductivity, critical current, stability and cost at the exact test temperature, method, density and sample preparation.
2. **Electrode:** require practical areal loading/capacity, composite inactive fractions, current-collector and lithium-inventory accounting.
3. **Cell:** move from symmetric/half cell to full cell, ≥5 Ah, then larger-format samples; retain independent physical replicates and censored outcomes.
4. **Pressure:** vary operating pressure while holding fabrication pressure, design revision, production batch, line, process revision and measurement method fixed. Any process-pressure change becomes a separate DOE factor, not a pressure-only comparison.
5. **Manufacturing:** qualify thickness/density/porosity distributions, yield, scrap/rework, cycle time, moisture control and cost per accepted cell across multiple lots.
6. **Application:** compare useful energy, power, cold performance, life and safety at the same test boundary and cell format.

A practical target progression is: exact-paper reproduction → ≤2 MPa cell-level operation → ≤0.3 MPa replication → ≥5 Ah matched full cells → >1,000 DST cycles and >15-year calendar-life evidence under the applicable methods → repeatable manufacturing yield above 90%. Do not assume that progress on one rung implies the next.

## First software slice

The Rust audit validates only:
- fabrication vs operating pressure as separate fields;
- pressure units, nonnegative finite values, measurement method and uncertainty;
- exact matched non-pressure conditions;
- cell test-level identity;
- complete-cell mass closure and declared boundary for full-cell records;
- chemistry-lane separation and evidence admission;
- distinct observed-EOL, right-censored, failed, invalid, incomplete and synthetic statuses;
- deterministic reports and adversarial negative controls.

All examples are synthetic. Physical experiments remain under qualified laboratories and institutional safety authorities. Sulfide handling, moisture control, H₂S hazards, cell assembly and pressurized testing are not specified or authorized by this work.

## Claim ceiling

A passing software audit establishes only identity, comparison, pressure separation, mass-boundary and evidence semantics on the authored corpus. It does not prove conductivity, dendrite suppression, cycle life, low-pressure operation, safety, manufacturability, cost, pack performance or commercial readiness.
