# Plasma Energy Systems — Source and Boundary Register

**Programme:** PLASMA-ENERGY-001  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11718  
**Status:** Research design / source register only  
**Date:** 2026-11-12  
**Claim ceiling:** No new simulation, experiment, prototype, physical measurement, energy yield, economic result, or SOTA improvement is claimed in this document.

## 1. Decision summary

Plasma is not a general-purpose energy-storage medium. Maintaining an ionized state generally requires a continuous or pulsed energy supply; the stored energy that can be usefully recovered belongs to the surrounding magnetic field, thermal mass, chemical product, or another explicitly identified storage element.

The research programme should separate four roles:

| Role | What plasma does | Actual storage / output | Priority |
|---|---|---|---|
| Plasma-assisted chemical conversion | Activates species or drives reaction pathways | Hydrogen, ammonia, carbon or another product whose chemical energy can be stored | Medium; high scientific interest, uncertain full-chain economics |
| Plasma-charged thermal storage | Transfers electrical energy into a high-temperature store | Refractory solids, particles, molten salts or phase-change media | High for bounded industrial-heat comparisons; no assumed plasma advantage |
| Fusion | Supplies energy through nuclear reactions in a confined or compressed plasma | Heat and, after balance-of-plant conversion, electricity | High long-horizon research; compose the existing fusion programme |
| MHD / pulsed direct conversion | Couples conducting fluid or plasma motion to electromagnetic fields | Electrical output or a recoverable magnetic-energy pulse | Targeted subproblem only; do not mistake blanket MHD for a standalone generator |

**Recommended execution order**

1. Start with the system-level accounting and thermal-store comparison, because the output boundary is measurable and the work composes existing thermal engineering.
2. Run a source-pinned ammonia-cracking comparison as a focused challenge case. The present literature gap is large enough to make it falsifiable.
3. Continue fusion through the existing PROCESS / architecture / availability work rather than creating another fusion systems code.
4. Treat standalone MHD generation as a lower-priority option unless a specific application has a defensible efficiency, materials and lifecycle case.

## 2. Source register

### 2.1 Plasma ammonia cracking to hydrogen

**Primary experimental reference**

- Article: *Hydrogen production from ammonia using a microwave plasma torch at atmospheric pressure*.
- Journal: International Journal of Hydrogen Energy, 2025.
- DOI: https://doi.org/10.1016/j.ijhydene.2025.151121
- Reported result: ammonia conversions above 99%; lowest reported energy cost 25 kWh/kg H2 for the stated 90 vol.% NH3 in N2 feed condition; outlet hydrogen concentration around 69 vol.% at that operating point.
- Evidence class: published laboratory experiment, as reported by the paper. Not an independent reproduction by Luminous Dynamics.
- Interpretation: high conversion alone does not establish low energy consumption, hydrogen purity after separation, or system-level economic competitiveness.
- Missing boundary details to extract before comparison: exact electrical metering point; microwave source and matching-network efficiency; gas/feed preparation; cooling; separation and compression; startup and standby; uncertainty and duration/stability; catalyst/electrode or torch consumables; product purification and pressure.

**Techno-economic comparator**

- Institution: Lawrence Berkeley National Laboratory.
- Article: *Benchmarking plasma and electrolysis decomposition technologies for ammonia to power generation*.
- Journal: Energy Conversion and Management, 2023.
- DOI: https://doi.org/10.1016/j.enconman.2023.117166
- Reported conclusion: the assessment estimated plasma-based ammonia decomposition would need energy efficiency equivalent to 10 kWh/kg H2 or lower to reach cost parity with the thermal-chemical cracking options in the study.
- Evidence class: published techno-economic analysis, not an experimentally demonstrated plasma performance target.
- Interpretation: use ≤10 kWh/kg H2 as a source-defined economic challenge threshold for that study's assumptions. Do not call it a universal break-even value, and do not compare it directly with the 2025 25 kWh/kg result until input/output boundaries and price assumptions are reconciled.

**Initial benchmark definition**

- Functional output: one kilogram of hydrogen delivered at a frozen purity, pressure, temperature and delivery boundary.
- Input boundary: total wall-plug electricity plus separately accounted feedstock and consumables.
- Required sub-ledgers: microwave/RF source; matching and power electronics; plasma coupling; feed gas; cooling; vacuum or compression if present; product separation; hydrogen compression; startup and standby.
- Required physical outputs: NH3 conversion; H2 yield; nitrogen-containing by-products; outlet composition; product purity; flow rate; temperature; pressure; run duration; uncertainty.
- Comparator: conventional thermal-catalytic ammonia cracking and direct-ammonia power conversion, using the same product boundary.
- Primary metrics: kWh/kg H2 delivered, hydrogen recovery fraction, purity, pressure, throughput, startup time, degradation, availability and cost range.
- Falsifier: if the complete wall-plug energy use remains materially above the source-defined parity threshold and no independently justified system benefit offsets it, do not promote the plasma route as a competitive storage pathway.

### 2.2 Plasma-assisted ammonia synthesis

**Source A**

- Article: *Sparks to synthesis: A roadmap to feasible ammonia production via plasma catalysis*.
- Journal: Energy Conversion and Management, 2025.
- DOI: https://doi.org/10.1016/j.enconman.2025.119802
- Research themes: non-thermal plasma, pulsed power, catalyst integration and small-scale electrified synthesis.
- Evidence class: review / roadmap.

**Source B**

- Article: *Research Progress in Novel Non-Haber-Bosch Ammonia Synthesis Technologies*.
- Journal: Green Energy and Resources, available online 2026-10-06.
- DOI: https://doi.org/10.1016/j.gerr.2026.100219
- Reported comparison: compares electrocatalytic, plasma-catalytic, photocatalytic and chemical-looping routes on nitrogen activation, selectivity, productivity, energy efficiency, stability, feedstocks and scale-up. It identifies plasma selectivity and electrical efficiency as bottlenecks; plasma-NOx/electrochemical coupling is described as promising but not yet evidence of full-chain superiority over green-H2 Haber-Bosch.
- Evidence class: comparative review, not a single matched full-plant experiment.

**Source C**

- Article: *Sustainable ammonia synthesis: An in-depth review of non-thermal plasma technologies*.
- Journal: Journal of Energy Chemistry, 2024.
- DOI: https://doi.org/10.1016/j.jechem.2024.04.018
- Interpretation: plasma synthesis should be compared at the plant boundary with a renewable-electricity Haber-Bosch route, including hydrogen production, nitrogen supply, compression, recycle, separation and catalyst lifetime.

**Initial benchmark definition**

- Functional output: one kilogram of specification-grade NH3 at an agreed pressure and purity.
- Metrics: kWh/kg NH3, g NH3/kWh, nitrogen conversion, NH3 selectivity, single-pass yield, nitrogen utilization, by-product distribution, separation/recycle load, catalyst life, startup/ramp behaviour and cost.
- Separate process energy from upstream green-hydrogen energy. Do not omit the hydrogen source in a full-chain claim.
- Compare steady-state and intermittent operation as different scenarios, not as a single pooled average.
- Falsifier: no superiority claim where only reactor-level selectivity or yield is reported and the full upstream electricity and separation boundary is absent.

### 2.3 Plasma-assisted methane pyrolysis

**Source A**

- Article: *Towards plasma-assisted methane pyrolysis for hydrogen production: A review*.
- Journal: Next Energy, 2026.
- DOI: https://doi.org/10.1016/j.nxener.2026.100700
- Reported status: plasma-assisted methane pyrolysis is a low-emissions hydrogen pathway with solid carbon co-product potential, but current work remains largely laboratory-scale and scale-up requires improved hybrid plasma-catalytic systems and carbon valorization.
- Evidence class: review.

**Source B**

- Article: *A comprehensive review of direct-heating modalities for low-emission hydrogen production via methane pyrolysis*.
- Journal: 2026.
- DOI: https://www.sciencedirect.com/science/article/pii/S016523702600183X
- Study scope: techno-economic and performance assessment over 156 experimental data sets from 26 peer-reviewed sources, comparing microwave, plasma, resistive/combustion and shock-wave heating.
- Reported finding: in the study's multi-objective comparison, plasma and resistive/combustion configurations were displaced from the Pareto front by higher energy intensity and/or lower conversion; microwave and shock-wave cases were more competitive under the analyzed weighting scenarios.
- Evidence class: review/data synthesis and modeled Pareto comparison. Do not transfer its rank to another reactor or boundary without reproducing the underlying data and assumptions.

**Initial benchmark definition**

- Functional outputs: hydrogen at frozen purity/pressure and solid carbon meeting a declared quality/offtake condition.
- Count methane feedstock, plasma electricity, gas handling, separation, compression, thermal losses and carbon finishing/handling.
- Metrics: kg H2/kg CH4, kg solid carbon/kg CH4, carbon morphology/purity where relevant, kWh/kg H2 delivered, conversion, selectivity, carbon recovery, uptime, fouling, erosion and whole-process cost/emissions.
- Falsifier: no low-emissions or economically superior label without carbon fate, methane leakage and full energy boundary.

### 2.4 Fusion generation and plant integration

**DOE Fusion Science and Technology Roadmap**

- Finalized: 2026-06-09.
- Source: https://www.energy.gov/articles/energy-department-releases-finalized-fusion-science-and-technology-roadmap-accelerate
- Key gaps relevant to this programme: fusion-facing materials and plasma-facing components; high-heat-flux and transient loads; tritium fuel-cycle closure; blanket heat extraction; MHD pressure drop and flow/heat-transfer coupling in liquid breeder systems; qualified coatings/insulators; coupled neutronics, MHD, thermofluids, tritium transport and structural response.
- Interpretation: the blanket is a critical conversion and fuel-cycle subsystem; plasma performance alone does not make a power plant.

**National Ignition Facility record**

- Source: LLNL FY2025 annual report: https://annual.llnl.gov/fy-2025/national-ignition-facility-2025
- 2025-04-07 result: 8.6 MJ fusion yield; 2.08 MJ laser energy delivered to target; target gain 4.13.
- Denominator: laser energy delivered to the target, not total electrical energy drawn by the facility.
- Interpretation: this is a major inertial-fusion physics result. It is not wall-plug breakeven, net electricity, or a grid-connected power demonstration.

**Concept-neutral milestone framework**

- Article: *Beyond power gain: Toward a comprehensive milestone framework for all fusion energy concepts*.
- Physics of Plasmas, 2025.
- DOI: https://doi.org/10.1063/5.0287340
- Key lesson: benchmark physics performance, engineering maturity and whole-system economics concurrently; include recirculating power, availability, maintainability, fuel-cycle closure, replacement and grid delivery.

**Existing implementation owner**

Compose Symthaea FUS-SYS-001 (#7509), FUS-OPS-001 (#7510) and FUS-ARCH-001 (#7511). This document does not define a new fusion systems code or claim a reactor concept is qualified.

### 2.5 Liquid-metal MHD in fusion blankets

- DOE roadmap source: https://www.energy.gov/documents/fusion-science-and-technology-roadmap
- Related research overview: https://fusion.columbia.edu/research-projects/breeding-blanket
- Research issue: electrically conducting liquid breeder flow interacts with strong magnetic fields, modifying pressure drop, flow distribution, turbulence and heat transfer. Insulating coatings and channel/manifold designs are not yet fully validated under relevant integrated conditions.
- Interpretation: this is primarily a blanket-flow and thermal-extraction challenge in the fusion context. It must not be relabelled as evidence for an efficient standalone MHD generator.
- Initial metrics: pressure-drop prediction error; velocity and temperature field discrepancy; flow distribution; heat-transfer coefficient; current density; coating/insulator condition; corrosion and compatibility; transient response; pumping power; net thermal energy delivered.
- Falsifier: no blanket design improvement claim from an analytical MHD result without a source-pinned geometry, material state, boundary condition and matched validation reference.

## 3. Thermal storage as the near-term applied-physics opportunity

Plasma can, in principle, be used as an electrical-to-thermal charging process. The storage medium—not the plasma—holds the energy. The relevant comparison is therefore not plasma temperature versus resistance-heater temperature. It is wall-plug electricity to useful stored heat, at a specified temperature and duty cycle.

Candidate stores:
- refractory solid blocks or structured ceramic media;
- particle beds;
- molten salts or other qualified high-temperature fluids;
- phase-change materials where material compatibility and cycling data support the operating range.

Matched heating baselines:
- resistance heating;
- induction heating;
- microwave heating;
- heat pumps where the required temperature lift is within the candidate technology's qualified range;
- combustion or waste-heat integration only with a consistent emissions and primary-energy boundary.

Required metrics:
- wall-plug-to-store efficiency;
- charge rate and controllability;
- temperature uniformity and spatial gradients;
- usable thermal energy at the customer temperature;
- exergy fraction, not only total joules;
- storage loss over time and standby draw;
- discharge power and response;
- electricity reconversion efficiency when electricity, rather than process heat, is the delivered service;
- thermal-cycle lifetime, corrosion/oxidation, refractory cracking, erosion and replacement;
- cost per delivered MWh-thermal and delivered service availability.

**Prospective research gate:** a plasma charging system must show a repeatable system-level advantage over the simplest suitable heater/store combination. A higher plasma temperature is not a win if wall-plug efficiency, heat-transfer losses, electrode/nozzle lifetime or maintenance are worse.

## 4. Whole-system accounting rules

Every candidate gets an immutable boundary map and a native energy/resource vector. A normalized metric may be computed only after all conversions have explicit provenance.

Required identities:
- source energy input;
- conversion-stage input and output;
- storage charge, state and discharge;
- useful output at the declared service boundary;
- auxiliary and standby consumption;
- losses and unaccounted residual;
- material and consumable flows;
- time basis and operating availability.

Required dimensional distinctions:
- kWh-electric versus kWh-thermal;
- HHV versus LHV;
- primary energy versus final energy;
- energy quantity versus power/rate;
- energy versus exergy;
- plasma-source input versus delivered/coupled process energy.

Conservation and comparison gates:
1. Never count a converter's electrical input and its derived optical/thermal/plasma output as two independent upstream inputs.
2. Never omit source, vacuum, pumping, cooling, separation, compression, feedstock conditioning or standby loads from a claimed wall-plug efficiency.
3. Never infer full-system round-trip efficiency from one reactor stage.
4. Never compare unlike output quality, pressure, temperature or time service without an explicit normalization model.
5. Missing auxiliary loads remain unknown and block a superiority claim; they are not set to zero.
6. Every reported number retains source, date, evidence class, uncertainty, unit, denominator and boundary.
7. Literature-reported, synthetic, solver-predicted and physically observed values remain different evidence types.

## 5. Benchmark and experiment-selection strategy

### Level 0 — Source and boundary closure
Extract each paper's operating point, denominator, meter locations, reported uncertainty, omitted loads, product quality, duration, source rights and raw-data availability. No cross-paper ranking is allowed yet.

### Level 1 — Accounting fixtures
Build small synthetic cases for:
- exact energy conservation;
- double-counted conversion input/output;
- omitted vacuum/cooling/separation load;
- HHV/LHV mismatch;
- kW versus kWh confusion;
- thermal/electric mix-up;
- missing or unknown loads;
- output below required purity or pressure;
- storage loss and standby draw;
- failed/non-converged solver run.

### Level 2 — Reproduction
Reproduce one source-pinned published case with its own declared boundary and metrics. A literature number in a manifest is not a reproduced model result.

### Level 3 — Matched comparator
Only compare candidates after the same input basis, product quality, process duration, service profile and system boundary are satisfied.

### Level 4 — Robust Pareto search
Report energy, exergy, cost, lifecycle, throughput, availability, material degradation and uncertainty as separate axes. Reject candidates that violate hard constraints; do not conceal failures inside a weighted average.

### Level 5 — Independent validation
Compare prediction with calibrated observation on a held-out run, operating condition or facility. Keep the measurement and simulation provenance distinct.

## 6. Prospective targets and claim boundaries

| Lane | First quantitative objective | Stretch objective | Claim ceiling before physical validation |
|---|---|---|---|
| NH3 cracking | Reproduce the reported 25 kWh/kg H2 point and close omitted-load accounting | ≤10 kWh/kg H2 at a matched product and system boundary | Source reproduction / model comparison only |
| NH3 synthesis | Establish common-basis yield, selectivity, power, recycle and lifetime metrics | System-level advantage for a defined intermittent/distributed service | Review comparison / solver result only |
| Methane pyrolysis | Rebuild published Pareto inputs and check data and boundary consistency | Plasma configuration enters a matched Pareto frontier with validated carbon handling | Literature synthesis / model comparison only |
| Thermal storage | Compare wall-plug electricity to useful heat for one declared store and duty cycle | Durable advantage in temperature, response, exergy or lifecycle service | Model and materials-coupon design only |
| Fusion | Close energy denominator, recirculating loads and balance-of-plant assumptions | Robust Pareto improvements in whole-plant net electricity and availability | Systems-code candidate only |
| Liquid-metal MHD | Reproduce a source-pinned flow/pressure-drop/heat-transfer case | Better qualified blanket flow and thermal extraction trade-off | Numerical benchmark only |

The ≤10 kWh/kg H2 cracking figure is a study-specific techno-economic threshold from LBNL, not a universally valid industry standard. All other stretch objectives require a common baseline before hard numeric thresholds are selected.

## 7. Symthaea / Symtropy division of responsibility

**Symthaea**
- maintain the source/evidence graph and design candidate identity;
- select models according to regime and intended use;
- represent uncertainty, model discrepancy, and out-of-domain status;
- select the next most informative computational or measurement action;
- compare candidates on declared energy/service boundaries;
- generate proposals, not physical operating commands.

**Symtropy**
- own deterministic scenario execution and replay for supported synthetic subsystems;
- consume exact solver/model outputs under explicit capability and run receipts;
- exercise thermal networks, buffer dynamics and balance-of-plant scenarios only where its declared models support them;
- preserve synthetic state as synthetic and reject promotion to physical observation.

**External specialist solvers**
- remain authoritative for their supported equations and numerical implementation;
- must be version-pinned with assumptions, validity envelopes and raw result artifacts;
- do not become physical truth merely because a run converged or two solvers agree.

## 8. Next offline deliverables

1. Complete the source/boundary extraction table for the ammonia-cracking papers.
2. Add source-pinned data rows for ammonia synthesis, methane pyrolysis and at least one thermal-store comparator.
3. Create a machine-readable energy-flow manifest with separate input, conversion, storage, loss and useful-output nodes.
4. Define conservation and comparability negative controls.
5. Reuse existing Rust quantity, evidence, resource-ledger and deterministic-replay code; do not introduce a second unit system.
6. Identify one solver-free evaluator slice that can be tested locally without WarpX, Zapdos, PROCESS or hosted runners.
7. Keep an explicit status ledger: source inspected, data admitted, model reproduced, numerical verification, external validation and physical experiment are different milestones.

## References

- Gharahshiran & Zheng (2024), non-thermal plasma ammonia synthesis review: https://doi.org/10.1016/j.jechem.2024.04.018
- Plasma-catalysis ammonia roadmap (2025): https://doi.org/10.1016/j.enconman.2025.119802
- Non-Haber-Bosch ammonia review (online 2026-10-06): https://doi.org/10.1016/j.gerr.2026.100219
- Ammonia decomposition by microwave plasma torch (2025): https://doi.org/10.1016/j.ijhydene.2025.151121
- LBNL ammonia decomposition techno-economic benchmark (2023): https://doi.org/10.1016/j.enconman.2023.117166
- Plasma-assisted methane pyrolysis review (2026): https://doi.org/10.1016/j.nxener.2026.100700
- Fusion Science and Technology Roadmap (DOE, 2026): https://www.energy.gov/articles/energy-department-releases-finalized-fusion-science-and-technology-roadmap-accelerate
- NIF FY2025 annual report (LLNL): https://annual.llnl.gov/fy-2025/national-ignition-facility-2025
- Comprehensive fusion milestone framework (2025): https://doi.org/10.1063/5.0287340


## 9. Evidence update — 2026-11-12

This update adds important newer comparators and changes the recommended research ranking. Source-reported figures below are not Luminous measurements or reproductions.

### 9.1 Ammonia cracking: conversion is not energy efficiency

A second 2025 microwave-plasma study examined pure ammonia at reduced pressure (typically 50–500 mbar):

- DOI: https://doi.org/10.1016/j.ijhydene.2025.03.118
- It reports a lowest energy cost of 33 kWh/kg H2 at 28% dissociation (reported energy efficiency 13%) and 50 kWh/kg H2 for full dissociation.
- It discusses both electron-impact and thermal dissociation, surface reactions and strong spatial gradients.

This is a distinct reactor, pressure and operating condition from the atmospheric-pressure 2025 torch above. Do not pool the 25, 33 and 50 kWh/kg values as a single trend line. Their value is to establish a source-specific challenge corpus and reveal the trade-off among conversion, output composition and energy consumption.

A 2026 experimental/kinetic study of non-equilibrium plasma-driven NH3 decomposition (Fuel, Volume 405, DOI https://doi.org/10.1016/j.fuel.2025.136427) reports a modified low-temperature kinetic model with 53.4% better predictions for the stated task and identifies excited Ar* pathways as important in the proposed reaction mechanism. The next benchmark should compare the published chemistry/kinetics against a deliberately simpler baseline on held-out operating conditions; the paper's reported prediction improvement is not transferable as a universal model-accuracy claim.

### 9.2 Ammonia synthesis: vibrational-state coupling is a high-value hypothesis

A 2026 plasma-electrocatalytic nitrogen-reduction study (JACS, DOI https://doi.org/10.1021/jacs.6c04148) reports an integrated plasma-electrolysis configuration. Among 14 screened metals, Ag is reported to achieve an NH3 production rate of 7.7 ± 0.8 nmol s-1 cm-2 and electrochemical Faradaic efficiency of 86 ± 14% under the selected conditions. Plasma/electrolysis on-off controls and 15N2 isotope labelling support the supplied N2 as the nitrogen source.

This points to a more interesting paradigm than simply maximize plasma power: deliberately engineer long-lived vibrationally excited nitrogen and couple it to a downstream catalyst/electrode interface. However, reported area productivity and electrochemical Faradaic efficiency are not full-chain kWh/kg NH3, industrial throughput, lifetime or plant economics. Require all of those before claiming competitive ammonia production.

The 6 October 2026 review of non-Haber-Bosch ammonia technologies (DOI https://doi.org/10.1016/j.gerr.2026.100219) specifically warns that ampere-scale plasma-NOx/electrochemical demonstrations do not yet establish full-system superiority over green-hydrogen Haber-Bosch. It should be a source for the comparator matrix, not a shortcut to a superiority claim.

### 9.3 Methane pyrolysis: deprioritize plasma unless a distinct use case survives

Azadmehr et al., *A comprehensive review of direct-heating modalities for low-emission hydrogen production via methane pyrolysis*, Journal of Analytical and Applied Pyrolysis, June 2026, DOI https://doi.org/10.1016/j.jaap.2026.107776, curates 156 experimental data sets from 26 peer-reviewed sources and compares microwave, plasma, resistive/combustion and shock-wave heating.

Its reported Pareto analysis favours shock-wave heating for its cost/conversion weighting (approximately 92% methane conversion at approximately USD 0.70/kg H2), while microwave heating is favoured when selectivity is included (approximately 91% conversion, 99% selectivity and approximately USD 1.09/kg H2). Plasma and resistive/combustion configurations are reported as displaced from the Pareto front in the studied data/assumptions.

**Decision:** do not prioritize plasma-assisted methane pyrolysis as a lead Luminous project. First reproduce the review's curated data and objective/boundary choices; only reopen plasma if a specific duty, feedstock, carbon-product requirement or integration case changes the Pareto result.

### 9.4 High-temperature thermal storage: industrial incumbent is already strong

Rondo reported that its 100 MWh heat battery entered commercial operation on 16 October 2025, delivering continuous steam from off-grid solar. Rondo reports operation above 1000 °C and over 97% round-trip thermal efficiency after a 10-week initial operation period:
https://www.rondo.com/news-press/rondo-powers-up-worlds-largest-industrial-heat-battery

These are **manufacturer-reported operational claims**, not independently audited time-series results in this source. The exact meter and efficiency boundary must be established before using the >97% number as a matched benchmark. It is not electricity-to-electricity round-trip efficiency.

Sandia documents a two-tank particle thermal-storage system up to 6 MWh and 800 °C, as well as a radial packed-bed facility up to 800 °C:
https://energy.sandia.gov/programs/renewable-energy/concentrating-solar-thermal-technologies/thermal-energy-storage-tes/

Riahi et al. (Journal of Energy Storage, July 2026, DOI https://doi.org/10.1016/j.est.2026.122533) experimentally investigates an electrically charged packed-bed store operating over 25–700 °C with air-based discharge. The study reports that charge rate and the resulting temperature profile materially affect discharge performance.

The August 2026 HT-TES review (DOI https://doi.org/10.1016/j.applthermaleng.2026.132532) identifies sensible solid media and molten salts as more mature than latent/thermochemical alternatives for industrial heat above 500 °C.

**Decision:** the first thermal-store benchmark must use resistance/Joule heating as the incumbent and a source-pinned experimental packed-bed or particle store. Plasma is an optional challenger, not the reference. A higher temperature is not a performance win unless it produces more useful service under a matched duty.

### 9.5 Fusion blanket MHD: a particularly strong model-qualification target

A September 2026 paper (Hua et al., DOI https://doi.org/10.1016/j.fusengdes.2026.115888) reports reduced-order MHD modelling in SAM and high-fidelity MHD in NekRS for fusion liquid-metal blanket applications, with analytical/cross-code comparisons and validation against historical Argonne Liquid Metal Experiment data.

A 2025 KIT/MEKKA WCLL mock-up study (DOI https://doi.org/10.1016/j.fusengdes.2025.115216) reports measured pressure distributions and a component-resolved total pressure-drop correlation for a scaled test-blanket manifold.

This is a more immediate and falsifiable work package than attempting a new general MHD code. It is now tracked separately as Symthaea [PLASMA-MHD-BENCH-001 #11733](https://github.com/Luminous-Dynamics/symthaea/issues/11733).

### 9.6 New executable/benchmark workstreams

- [PLASMA-ENERGY-001 #11718](https://github.com/Luminous-Dynamics/symthaea/issues/11718): whole-system plasma energy conversion, storage and dispatch.
- [THERMAL-STORE-INDUSTRIAL-001 #11734](https://github.com/Luminous-Dynamics/symthaea/issues/11734): matched industrial high-temperature store and heater comparison.
- [PLASMA-MHD-BENCH-001 #11733](https://github.com/Luminous-Dynamics/symthaea/issues/11733): liquid-metal blanket MHD solver and experiment qualification.

The new source-and-boundary work does not change the claim ceiling: source inspection, literature comparison, synthetic accounting, solver output, measured observation and physical qualification remain distinct stages.


### 9.7 Stoichiometric reaction-enthalpy sanity check

The atmospheric-pressure microwave-plasma article quotes the ammonia decomposition reaction:

`2 NH3 -> N2 + 3 H2`, with reaction enthalpy `+91.88 kJ` per stoichiometric reaction as written.

Using `M(H2) = 2.01588 g/mol`, the reaction produces approximately `0.00604764 kg H2`, so the standard-state reaction enthalpy is:

`91.88 kJ / 0.00604764 kg / 3600 = 4.220 kWh/kg H2`.

The paper's reported 25 kWh/kg H2 net-microwave-energy point is therefore about 5.92 times this reaction-enthalpy reference.

**This ratio is not an efficiency.** It is a derived thermodynamic sanity check, not a complete process balance. The process may also heat the feed/products, transfer energy imperfectly, incur wall and coupling losses, require gas handling and purification, and consume auxiliary power. A proper system efficiency needs a frozen wall-plug boundary and product quality. The calculation and its inputs are stored as `NH3-STOICHIOMETRIC-REACTION-ENTHALPY-001` in the machine-readable reference corpus.


### 9.8 Fuel-coupled direct conversion: D–T versus D–³He

A key DEC design variable is the **fuel-specific charged-particle population**. The MIT PSFC MFE Formulary table (based on evaluated ENDF/B-VII reaction data) gives:
- D–T: alpha 3.52 MeV + neutron 14.07 MeV, approximately 20% primary reaction energy in charged particles.
- D–³He: alpha 3.67 MeV + proton 14.68 MeV, 100% of the primary reaction energy in charged particles.

Reference: https://www-internal.psfc.mit.edu/research/MFEFormulary/files/MFEFormulary.pdf.

But a D–³He *fuel mixture* is not neutron-free. Morandi et al. (Fusion Engineering and Design 222, January 2026, DOI https://doi.org/10.1016/j.fusengdes.2025.115491) model a simplified 50%D/50%³He ARC-class tokamak and include D–³He, both D–D branches and secondary D–T reactions from tritium generated by D–D. At a source-reported 525 MW comparison, the model reports:
- D–T: 15.3 keV, electron density 7×10¹³ cm⁻³, 17.5 MeV average energy per reaction, 80.06% neutron energy.
- D–³He: 69 keV, electron density 1.3×10¹⁴ cm⁻³, 16.4 MeV average energy per reaction, 2.9% neutron energy.

Derived from those source-specific mixture values, charged-product energy share is 19.94% for D–T versus 97.1% for the D–³He mixture. That is approximately 4.87× in fractional charged-energy availability, or 4.56× in charged MeV per source-reported average reaction. **These ratios are not converter efficiencies or net plant electricity.** The same study's higher D–³He temperature/density and simplified ignition/stability scope must remain visible.

A 2026 reactivity study reports a ballpark bremsstrahlung crossover temperature of 31 keV for D–³He versus 4.4 keV for D–T under equal-ion/electron-temperature assumptions; it explicitly describes the values as estimates and notes that colder electrons alter them. DOI https://doi.org/10.1088/1361-6587/ae3b71.

The design question is now fuel × confinement × converter co-design: what fraction of each reaction-origin population is actually transported through the mirror/expander, captured, electrically recovered and conditioned, after plasma sustainment, magnet/cryogenic loads, neutron blanket heat and auxiliaries are included? Do not infer an all-charged primary reaction means all plant power is recoverable.


### 9.9 Miniaturized and numbered-up low-temperature plasma devices

Four new source cases were admitted to the machine-readable corpus under MICROPLASMA-001 (#11780):

**Mini atmospheric-pressure helium plasma jet (Mao et al., 2026).** *Environmental Technology & Innovation* 42, 104852. DOI https://doi.org/10.1016/j.eti.2026.104852. The source reports a custom power supply approximately 100 g and 15 × 10 × 2 cm, a 54 mm glass tube, 2 L/min helium flow and total component cost below USD 45. Its in-vitro antimicrobial results are source-specific. The paper reports >20 h of power-supply operation before tests but does not quantify long-term electrode wear / treatment drift; a limited treatment area is a practical limitation. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13290277/.

**Membrane-stabilized microfluidic DBD chip (Liu et al., 2025).** *Lab on a Chip* 25(12), 2807–2815. DOI https://doi.org/10.1039/D4LC00960F. A hydrophobic porous membrane passes plasma while blocking liquid, stabilizing the gas/liquid interface. Gold nanoparticle synthesis reports 7.31–11.32 nm particle sizes and 1.8 nm standard deviation. Throughput, complete grid-to-product energy, membrane life and industrial cost remain separate questions. PubMed: https://pubmed.ncbi.nlm.nih.gov/40406968/.

**43-element LTCC cold atmospheric plasma array (Bhattacharya et al., published 30 December 2025; issue record 2026).** *IEEE Transactions on Plasma Science* 54(2), 496–502. DOI https://doi.org/10.1109/TPS.2025.3646554. The array provides a 10 × 10 cm treatment area from 43 linear discharge elements; the source reports a 91% CFU decrease for a specified *Pseudomonas fluorescens* biofilm coupon after 150 s at a 1.5 cm standoff. It is an application-specific array result, not universal antimicrobial performance. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12923079/.

**Individually ignitable RF microplasma array (Martinet et al., 2023).** *Plasma Processes and Polymers* 20, e2200131. DOI https://doi.org/10.1002/ppap.202200131. Five cells are demonstrated with individual ignition and quenching. The authors discuss scaling to tens or hundreds of cells; this is a proposed direction, not a demonstrated large array.

The key paradigm is **numbering-up**: preserve useful per-cell physics and replicate independently controlled modules, then prove flow/power distribution, cell-to-cell uniformity, cross-talk, cell-fault isolation, serviceability and full-system energy. Miniature cell success does not imply array success. A microfluidic reaction-volume advantage does not imply industrial throughput. Low source component cost does not imply low lifecycle cost.


**Portable argon MobiPlas (Bansemer et al., 2024).** *Plasma Medicine* 14(4), 39–51, DOI https://doi.org/10.1615/PlasmaMed.2025057641. The device integrates a refillable steel argon tank, battery powerbank and Raspberry Pi Pico control. The authors report approximately 10 minutes of operation per gas fill, a maximum patient leakage-current reading of 11 μA and a treatment-tip temperature up to 40°C; they also report occasional current fluctuations due to heating in the supply current-limitation circuit. INPTDAT's device profile assigns TRL 4; that is recorded only as institute-catalogue metadata, not independent regulatory or engineering qualification. Article PDF: https://www.dl.begellhouse.com/download/article/46f5a11a2aff6c76/PMED-57641.pdf. The reference highlights why autonomy per gas fill, current drift and thermal stability belong in portable-device benchmarks.


### 9.10 Ambient-air DBD and outcome-qualified water treatment

**Grid-like ambient-air DBD for bioaerosol treatment (Scaltriti et al., 2026).** *Plasma Processes and Polymers* 23(8), e70226, first published 4 August 2026. DOI https://doi.org/10.1002/ppap.70226. The paper reports up to 4.5-log single-pass reduction of *E. coli* MG1655 bioaerosols at a 12.5 L/min air-flow condition, with approximately 0.65 W average power delivered to the discharge and a source-reported EEO of 2×10⁻⁴ kWh·m⁻³·order⁻¹. Crucial boundary: the EEO uses discharge power and explicitly neglects power-supply losses. Preliminary FTIR measured about 9 ppm residual ozone at the outlet; the authors say an integrated ozone-removal stage and dedicated realistic ventilation/occupancy safety assessment are needed before practical indoor use. At 50% active discharge coverage, the paper reports around 2-log reduction versus over 4-log under full coverage. This source is a promising low-power architecture, not a wall-plug energy or occupied-air safety result.

**Compact ambient-air surface DBD (Choudhury et al., 2024).** *Frontiers in Bioengineering and Biotechnology*, DOI https://doi.org/10.3389/fbioe.2024.1325336. The paper describes an approximately 48 cm³, 55 g compact surface-DBD reactor using surrounding atmospheric air without a separate process-gas supply. This is a useful size / feed-gas architecture reference, but it is a surface-treatment device—not an air-cleaning CADR result.

**Flexible coplanar DBD water treatment (Zhong et al., 2026).** *Plasma Processes and Polymers* 23(4), e70174, first published 28 April 2026; correction 15 September 2026. DOI https://doi.org/10.1002/ppap.70174. A flexible coplanar DBD plus bipolar nanosecond pulse generator transfers reactive species to water as microbubbles. The paper reports >5-log *E. coli* reduction in 800 mL within 5 minutes and a source-defined bacterial inactivation energy efficiency of 3.64×10⁶ CFU/J. The raw data are available upon reasonable request. CFU/J is not equivalent to EEO, and this bacterial endpoint does not establish drinking-water quality, chemical contaminant mineralization or by-product safety.

**Plasma-liquid catalysis review (Yuan et al., October 2026).** *Journal of Water Process Engineering*, article 110831, DOI https://doi.org/10.1016/j.jwpe.2026.110831. The review identifies incomplete mineralization, transformation-product toxicity, catalyst deactivation/leaching, matrix effects, inconsistent performance metrics, energy and reactor scale-up as unresolved challenges. It advocates linking plasma-liquid interfacial physics, RONS diagnostics, catalyst pathways and reactor engineering.

**Multiphase reactor intensification review (Chen et al., 2026).** *Water* 18(6), 649, DOI https://doi.org/10.3390/w18060649. The review highlights gas-liquid transfer limits for short-lived reactive species and specific energy consumption. It discusses microbubble-assisted plasma, hydraulic cavitation, piezoelectric coupling and plasma-Fenton intensification, but retains by-product/toxicity, hydrodynamic scale-up, fouling and long-term reliability as challenges.

**Cross-study AOP comparison (RSC Advances, 2026).** The review *Advanced oxidation processes (AOPs) for drinking water treatment: a state-of-the-art review on applications, efficacy, and implementation challenges* reports median EEO values of about 3.3 kWh·m⁻³·order⁻¹ for plasma, 0.15 for ozonation and 0.7 for UV/H₂O₂. These are cross-study review medians, not matched trial results. Their use here is a screening warning that a plasma water-treatment candidate must justify its full-system energy and application-specific advantage; they are not a direct performance ranking. https://pubs.rsc.org/en/content/articlehtml/2026/ra/d6ra00697c.

The new air and water source cases are tracked separately in PLASMA-AIR-DBD-001 (#11804) and PLASMA-WATER-QUAL-001 (#11805). For air, the governing service metric is post-abatement clean-air delivery with wall-plug energy, pressure drop and ozone/NOx accounted for. For water, bacterial inactivation, chemical oxidation, mineralization and plasma-catalyst synergy are separate lanes; parent disappearance, CFU/J or review-level EEO cannot substitute for effluent-quality, transformation-product and toxicity evidence.
