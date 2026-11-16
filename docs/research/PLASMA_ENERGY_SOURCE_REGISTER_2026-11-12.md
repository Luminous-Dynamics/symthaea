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


## 9.8 Direct energy conversion from charged plasma and fusion products

A new research direction is now tracked as [FUS-DEC-001 #11755](https://github.com/Luminous-Dynamics/symthaea/issues/11755). Direct energy conversion (DEC) is an energy-conversion path for charged-particle kinetic energy; it is not a new energy source.

### ENN electrostatic fundamental-unit experiment (2026)

Kong et al., *Experimental study on electrostatic direct conversion of kinetic energy of charged energetic particles to electric energy*, Plasma Science and Technology 28(5), 054001, published 11 May 2026.

- DOI: https://doi.org/10.1088/2058-6272/ae445a
- Article: https://pst.hfcas.ac.cn/en/article/doi/10.1088/2058-6272/ae445a
- The paper reports low-parameter experiments using a fundamental electrostatic DEC unit structure and energy-conversion efficiency exceeding 80%.
- The abstract explicitly says that these tests preceded observation of fusion alpha particles in the experiments.

This is a published physical experiment on a fundamental charged-particle DEC structure. It is **not** a demonstration of fusion-born alpha-particle conversion, net electricity or a fusion plant. The denominator, incident species/distribution, exact voltage/current values, source power, grid/interception losses and uncertainty must be extracted from the full paper before reproduction.

### Realta / WHAM electrostatic DEC prototype (2026)

Realta announced its WHAM direct-energy-conversion prototype on 30 June 2026. The unit uses a single-stage three-grid electrostatic arrangement (grounded grid, electron-repulsion grid and ion collector) at a magnetic-mirror end ring.

- Company announcement: https://realtafusion.com/insight/a-fusion-first-realta-demos-direct-energy-conversion/
- Independent trade coverage: American Nuclear Society Nuclear Newswire, 6 July 2026: https://www.ans.org/news/article-8179/realta-fusion-makes-electricity-but-not-through-heat-conversion/

The company reports multiple amperes at around 100 V and output in the hundreds of watts. The ANS article confirms the three-grid demonstration and makes the important claim boundary explicit: most recovered energy in this D-only WHAM prototype came from externally supplied plasma heating/sustainment, not fusion-born energy. Both the company and ANS state that the demonstration was not net electricity or large-scale conversion of fusion-born power.

Keep this case separate from the ENN fundamental-unit experiment and separate from a D-T fusion plant result. Future multi-stage scale-up and >90% target figures are not prototype measurements.

### D-T energy origin partition

For the D-T reaction, approximately 17.6 MeV is divided between a charged alpha particle (about 3.5 MeV) and a neutron (about 14.1 MeV).

- National Academies: https://nap.nationalacademies.org/read/13438/chapter/6
- JET D-T energy partition discussion: https://doi.org/10.1088/1741-4326/ad5c81

The alpha component is about 19.89% of the primary reaction energy; the neutron component is about 80.11%. This is a nuclear reaction partition, not a DEC capture fraction. Alpha energy may first heat the plasma; only the fraction reaching a qualified converter can be recovered electrically. Neutron energy is a separate neutral-particle blanket/thermal path.

**Critical accounting rule:** energy recovered from externally supplied neutral-beam, RF, microwave, ohmic or other plasma-heating/current-drive inputs is not fusion-born energy. Do not count it as fusion gain. Preserve particle origin through the source, expander, converter, collector, power conditioning and plant boundaries.

## 9.9 Stronger industrial thermal-storage incumbent: refractory RTES

Saini, Lidor & Hammond, *Refractory-based thermal energy storage for industrial process heat: one-dimensional modeling, control, and optimization*, Applied Thermal Engineering 303 (August 2026), Article 132078.

- DOI: https://doi.org/10.1016/j.applthermaleng.2026.132078
- Institutional record: https://research-hub.nlr.gov/en/publications/refractory-based-thermal-energy-storage-for-industrial-process-he/

The peer-reviewed study models electrically heated dense ceramic refractory storage over 950–1,800 °C, storage capacities below 50 MWh and discharge powers of 5–30 MW. It includes transient gas-solid heat transfer, fan-assisted discharge, bypass control, reheating, fan power, insulation losses and genetic-algorithm optimization of brick-channel geometry and insulation. It reports indicative modelled cost maps of USD 1.6–2.0/kWh; the precise cost basis must be extracted before comparison.

This is a **numerical study of a resistance/Joule-heated incumbent**, not a measured plant and not a plasma demonstration. It strengthens the first thermal-store benchmark: plasma must compete against a well-designed, controlled refractory store at matched temperature, discharge service, hold time and meter boundary—not against a simplistic heater-only baseline.

The source case is in the machine-readable corpus as HTTES-REFRACTORY-950-1800C-MODELLING-2026.

## 9.10 Current machine-readable portfolio

The plasma energy corpus now contains 16 source cases and 16 negative controls, including:
- ENN electrostatic DEC fundamental-unit experiment;
- Realta WHAM DEC prototype with origin and net-power caveats;
- a D-T alpha/neutron energy-partition derived check;
- a 950–1,800 °C refractory thermal-storage model as a strong resistive incumbent.

The corresponding benchmark programmes are [PLASMA-ENERGY-001 #11718](https://github.com/Luminous-Dynamics/symthaea/issues/11718), [THERMAL-STORE-INDUSTRIAL-001 #11734](https://github.com/Luminous-Dynamics/symthaea/issues/11734), [PLASMA-MHD-BENCH-001 #11733](https://github.com/Luminous-Dynamics/symthaea/issues/11733), and [FUS-DEC-001 #11755](https://github.com/Luminous-Dynamics/symthaea/issues/11755).

All remain source-register and research-design evidence. No Luminous reproduction, physical measurement or whole-plant superiority is claimed.


### Historical SOTA: LLNL electrostatic direct-converter experiments

The new evidence register now includes two highly relevant historical physical benchmarks:

**Barr, Moir & Kinney (1977)**, *Experimental and computational results on direct energy conversion for mirror fusion reactors*, Nuclear Fusion 17, DOI https://doi.org/10.1088/0029-5515/17/5/012.

- A 22-stage electrostatic periodic-focusing converter with magnetic expander and hydrogen plasma source.
- Reported low-beam-density measured average efficiency: 86.5 ± 1.5%; corresponding model prediction 88.6 ± 1.5%.
- Efficiency fell as beam density and space-charge effects increased.
- The paper's scaling/economic discussion suggested a mean ion energy above 500 keV for the studied recovery system to be likely economical.

This is a strong historical electrostatic converter reference, not a plant result. Match the beam distribution, ion energy, density, stage geometry and meter denominator before comparing it with modern prototypes.

**Barr & Moir (1983)**, *Test Results on Plasma Direct Converters*, Fusion Science and Technology 3(1), DOI https://doi.org/10.13182/FST83-A20820.

- Single-stage and two-stage Venetian-blind bench units were tested up to 100 kV and 6 kW for approximately 80 hours, with power density up to 70 W/cm².
- A separate single-stage unit mounted at the Tandem Mirror Experiment (TMX) end wall intercepted 138 W of end-loss plasma; gross recovered power was 79 W and suppressor-grid consumption was 12 W.
- The paper reports 48% net efficiency. Direct arithmetic on the rounded source values gives 79 − 12 = 67 W net, or 48.55% of 138 W, consistent with a rounded 48% report.

The 6 kW bench test and the 138 W TMX test are distinct test configurations. Preserve them separately. The TMX case is not a D-T fusion-born alpha-particle or whole-plant electricity result.

**Scale-up lessons:** the 1983 paper reports background-gas ionization and secondary-electron emission becoming important as power density increased, and identifies high-voltage breakdown as a concern for large converters. The 1977 work shows space charge and ion-energy distribution can materially alter efficiency. These phenomena must become explicit benchmark axes.

### Alternate industry architecture: Helion pulsed inductive energy recovery

Helion describes a different architecture: high-beta field-reversed-configuration (FRC) plasmas expand against the machine magnetic field, inducing current through Faraday's law that is recaptured by the pulsed electrical system.

- Technology: https://www.helionenergy.com/technology
- Polaris prototype: https://www.helionenergy.com/polaris
- FAQ / net-electricity boundary: https://www.helionenergy.com/faq

Helion identifies inductive energy recovery as a core design feature and reports measured high-efficiency magnetic energy recovery, but its public material does not provide a source-matched numerical efficiency dataset suitable for independent ranking. Its FAQ clarifies that the company now prioritizes demonstrating electricity from fusion because its earlier use of the term “net electricity” was insufficiently defined.

Keep pulsed inductive FRC recovery separate from electrostatic mirror loss-cone conversion. Compare only after the input/output boundary, pulse energy, magnetic field/current trajectories, storage/conditioning, auxiliary power and fusion-origin contribution are specified.

### Revised direct-conversion benchmark ladder

The machine-readable corpus now spans:
1. 1977 multi-stage electrostatic converter and its space-charge scaling evidence;
2. 1983 single-/two-stage bench and TMX end-loss tests with explicit parasitic suppressor power;
3. ENN's 2026 low-parameter electrostatic fundamental-unit experiment;
4. Realta/WHAM's 2026 three-grid D-only prototype;
5. Helion's 2026 manufacturer-described pulsed-inductive recovery architecture.

The source corpus separately includes a derived TMX net-power check (67 W, approximately 48.55% of the rounded 138 W incident figure). This validates arithmetic against the source values only; it does not turn the paper into a modern plant benchmark or independent reproduction.


## 9.11 Plasma high-temperature processing and electrified thermal-storage SOTA

### Electrified thermal energy storage review (2026)

Huang et al., *Electrified thermal energy storage*, Nature Reviews Clean Technology 2, 89–106, published 9 January 2026.

- DOI: https://doi.org/10.1038/s44359-025-00131-4
- The review states that commercially available ETES uses resistive heating with sensible or latent storage; it describes market viability around USD 50/kW but notes low energy density and thermal losses limit many systems to short durations (days/weeks).
- Emerging non-contact/volumetric microwave and inductive heating are described as faster and capable of very high temperatures (up to 3,000°C in reviewed systems). These are not necessarily plasma technologies.
- Thermochemical storage may increase duration/energy density but remains lower TRL.

This changes our baseline selection: compare plasma not only with resistance but with microwave and induction heating as independent alternatives. Do not relabel the review's microwave/induction temperature range as a plasma capability.

### Plasma scale-up roadmap (2026)

Attri et al., *Large-Scale Applications of Plasma Technologies: Progresses and Challenges*, High Voltage 11(5), published 30 September 2026.

- DOI: https://doi.org/10.1049/hve2.70241
- The review identifies four cross-domain deployment constraints: controllability of plasma parameters and discharge modes during scale-up; **grid-to-outcome** system energy efficiency; long-term reliability/maintainability; and outcome-relevant diagnostics/harmonized benchmarking.
- It specifically warns that high plasma-to-chemistry or plasma-to-heat efficiency does not guarantee grid-to-outcome efficiency once power conversion, coupling, cooling, feed, separation and other auxiliaries are included.

This is a useful top-level comparator for the Luminous evidence architecture: every plasma lane should measure from grid input to the actual industrial outcome, rather than stop at reactor or torch terminals.

### Microwave-plasma pilot furnace for MgO refractories (2025)

Oppelt et al., *Thermal treatment of MgO refractories in a pilot furnace heated with microwave plasma burner*, Journal of the European Ceramic Society, May 2025.

- DOI: https://doi.org/10.1016/j.jeurceramsoc.2024.117146
- A microwave plasma burner heats air as a process gas, which is then blown into a pilot furnace and transfers energy to MgO refractory specimens.
- The study compares MWP treatment with a conventional electric resistance furnace and combinations of the two.
- It reports no adverse effects on density, microstructure or flexural strength and no phase-composition change in the specimens studied.

This is a physical pilot-process reference for plasma-assisted refractory treatment. It is not a thermal battery efficiency result. Its value for our programme is the possibility of plasma as a high-temperature gas/heat-transfer process in an existing furnace architecture; its missing wall-plug and furnace heat balance is part of the benchmark opportunity.

### Microwave-induced plasma firing versus resistance and gas firing (2026)

Schneider et al., *Comparative effects of microwave-induced plasma and conventional firing on the properties of silicate ceramics: A statistical and validated study*, Ceramics International, available online 30 July 2026.

- DOI: https://doi.org/10.1016/j.ceramint.2026.07.435
- The study uses a 2.45 GHz / 6 kW microwave-induced plasma burner at 1,000°C and compares natural gas, electric resistance and plasma firing with a two-factor design (dried or pre-fired specimens).
- For dried samples, reported MWP results are comparable to resistance for water absorption (21.21% vs 21.25%), Young's modulus (20.05 vs 21.34 GPa) and biaxial strength (30.38 vs 32.94 MPa); the study reports higher modulus and strength for gas/reference firing in this specimen condition.
- For pre-fired specimens, MWP has the highest reported biaxial strength of the compared pre-fired groups (36.73 MPa).

This is a useful controlled material-quality comparison, but it does **not** establish wall-plug energy efficiency, throughput superiority or storage performance. Specimen history, firing atmosphere and thermal profile must remain matched/stratified; material properties cannot stand in for an energy balance.

### Benchmark implication

The first physical thermal-store/heat-treatment comparison should include at least:
- resistance/Joule heating;
- microwave heating without plasma where applicable;
- induction heating where the store/material supports it;
- microwave-plasma burner heating.

Freeze material batch, geometry, mass, moisture/pre-firing history, target temperature profile, hold/discharge duty, atmosphere and outlet service. Meter grid-side electricity, generator/rectifier/magnetron and matching losses, auxiliaries, process-gas heating, cooling, fan/pump loads, furnace/store energy change, hold loss and useful delivered heat. Include material quality and lifetime as separate outcome axes.

The new source cases are in the machine-readable register as:
- HTTES-ELECTRIFIED-TES-REVIEW-2026;
- PLASMA-SCALEUP-ROADMAP-2026;
- PLASMA-MW-MGO-REFRACTORY-PILOT-2025;
- PLASMA-MW-CERAMIC-FIRING-2026.


## 9.12 Grid-to-outcome thermal metrology contract

The matched measurement work is now tracked as [PLASMA-THERMAL-METROLOGY-001 #11758](https://github.com/Luminous-Dynamics/symthaea/issues/11758). Its machine-readable contract is:

`data/research/plasma_thermal_metrology_contract_v0_1.json`

It separates two service modes that must not be conflated:
- `THERMAL_STORE_CHARGE_HOLD_DISCHARGE`: electricity → stored heat → hold → useful delivered heat;
- `THERMAL_MATERIAL_PROCESSING`: electricity/process-gas/plasma → material treatment → conforming product.

The contract defines six measurement layers: grid-side input, power conversion/coupling, process gas/plasma coupling, material/store control volume, useful heat delivery, and product quality/throughput. Each layer binds meter/instrument identity, calibration, uncertainty, clock/time basis, raw artifact digest and applicable boundary.

The source crosswalk deliberately marks current literature gaps:
- 2025 MgO pilot furnace: material-quality outcomes are reported, but grid-side input, process-gas enthalpy and complete auxiliaries remain unknown until full-paper/source-data extraction.
- 2026 silicate-ceramic MWP study: material outcomes and 6 kW burner specification are reported, but full grid-to-outcome energy boundary is not established.
- 2026 refractory RTES: numerical/modelled energy boundary, not physical metrology.
- Sandia particle-store webpage: facility capability, not a performance dataset.

Ten frozen negative controls block nameplate-as-meter, absent auxiliary-as-zero, setpoint-as-sample-temperature, dried/pre-fired pooling, material-quality-as-efficiency, and other boundary errors. The JSON round-trip and identity checks passed at authoring time; this is contract validation, not instrument validation or physical experiment.

This adds a practical next-step question to the science programme: what additional calibrated measurements would make a source case comparable, and which missing measurement changes the design decision? It does not authorize a high-temperature or plasma experiment.


## 9.13 Updated ammonia-plasma frontier — 2026-11-15

Two September 2026 publications materially improve the ammonia cracking comparison.

### Microwave decomposition: conversion and specific-energy Pareto points

Quiroz Marnef et al., *Microwave plasma-based NH3 decomposition: Reaching optimum performance by understanding the mechanisms*, International Journal of Hydrogen Energy 265 (2026), DOI https://doi.org/10.1016/j.ijhydene.2026.156912.

The paper reports:
- near-complete conversion (>99%) at a hydrogen energy yield of 45 g H2/kWh and a production rate of 150 g/h;
- the highest reported energy yield of 51 g H2/kWh at a separate 76% NH3 conversion operating point;
- a 2D-axisymmetric model coupling gas dynamics, heat transfer, electromagnetic fields and chemical reactions.

Reciprocal energy-yield checks give approximately 22.22 kWh/kg H2 at the >99% conversion point and 19.61 kWh/kg H2 at the 76% conversion point. These are two operating points, not one performance point. The higher energy yield at 76% conversion must not be combined with the >99% conversion figure.

The complete wall-plug boundary, H2 separation/purification, compression, cooling, auxiliaries and lifetime must be independently reconciled before system-level ranking. The feed is pure NH3 at atmospheric pressure, which differs from the 2025 atmospheric NH3/N2 mixture and reduced-pressure pure-NH3 cases.

### Turbulent transport: a counter-intuitive model-form result

Maerivoet et al., *Effect of turbulent transport on plasma-based NH3 cracking into H2: Insights from a RANS computational fluid dynamics model*, Energy Conversion and Management 364 (2026), DOI https://doi.org/10.1016/j.enconman.2026.121741.

The source reports a coupled 2D-axisymmetric flow/heat/chemistry/species/current model compared with experiments over flow-rate and current ranges. In the studied warm-plasma reactor, increased turbulence reduced NH3 conversion and increased energy cost because it suppressed the hot afterglow that supports additional conversion.

This is an important model-discrimination challenge: do not assume that more turbulence or mixing universally improves plasma conversion. The effect is chemistry- and regime-specific. The result is not a universal law and is not directly transferable to microwave NH3 cracking, RF cold plasma, CO2 splitting or fusion.

### Benchmark update

The ammonia lane should now retain at least these separate Pareto dimensions:
- NH3 conversion per pass;
- H2 energy yield (g/kWh) and reciprocal specific energy (kWh/kg H2);
- H2 production rate;
- exact feed composition, pressure, reactor/source family and operating point;
- full grid-side energy boundary and accepted H2 purity/pressure;
- separation/recycle energy and unconverted NH3 recovery;
- uptime, start/stop response, maintenance and consumables.

Compare complete process chains rather than isolated conversion figures. A lower single-pass conversion can be preferable only if separation and recycle produce lower total energy and cost per kg of delivered accepted H2. That proposition remains a hypothesis to benchmark, not a claimed result.


## 9.13 Updated ammonia-plasma frontier — 2026-11-15

Two September 2026 publications materially improve the ammonia cracking benchmark.

### Microwave decomposition: conversion–energy Pareto points

Quiroz Marnef et al., *Microwave plasma-based NH3 decomposition: Reaching optimum performance by understanding the mechanisms*, International Journal of Hydrogen Energy 265 (2026), DOI https://doi.org/10.1016/j.ijhydene.2026.156912.

The paper reports three separate operating points:
- >99% conversion at 45 g H2/kWh and 242 kJ/mol;
- 150 g H2/h at 92.4% conversion, 50 g H2/kWh and 217 kJ/mol;
- best energy yield of 51 g H2/kWh and lowest energy cost of 210 kJ/mol at 76% conversion.
- a 2D-axisymmetric model coupling gas dynamics, heat transfer, electromagnetic waves and chemical reactions.

Reciprocal energy-yield arithmetic gives approximately 22.22 kWh/kg H2 at >99% conversion, 20.00 kWh/kg H2 at the 92.4% / 150 g/h point, and 19.61 kWh/kg H2 at 76% conversion. These are three operating points, not one combined performance claim. The 150 g/h result must not inherit the >99% conversion value, and the energy-yield optimum must not inherit the high-conversion result.

The reported yield basis must be checked against complete wall-plug input, hydrogen separation/purification, compression, cooling, auxiliaries, startup, downtime and lifetime before system-level ranking. This atmospheric pure-NH3 case differs from the 2025 atmospheric NH3/N2 torch and reduced-pressure pure-NH3 study.

### Turbulent transport: chemistry-specific model-form result

Maerivoet et al., *Effect of turbulent transport on plasma-based NH3 cracking into H2: Insights from a RANS computational fluid dynamics model*, Energy Conversion and Management 364 (2026), DOI https://doi.org/10.1016/j.enconman.2026.121741.

The source describes a coupled 2D-axisymmetric gas-flow, heating, chemical-kinetics, species-transport and current model, with reported agreement against experiments across flow-rate and current ranges. For the studied warm-plasma process, increased turbulence reduced NH3 conversion and increased energy cost by suppressing the hot afterglow that supports additional conversion.

This is a valuable counterexample to the assumption that more turbulence or mixing universally improves plasma conversion. It is specific to the studied reactor/chemistry and must not be transferred to microwave NH3, CO2 splitting, NOx formation, RF cold plasma or fusion plasma without validation.

### Benchmark implication

Keep these as independent Pareto dimensions:
- NH3 conversion per pass;
- H2 energy yield (g/kWh) and reciprocal specific energy (kWh/kg H2);
- production rate;
- feed composition, pressure, reactor/source family and exact operating point;
- complete grid-side energy and accepted H2 purity/pressure;
- separation/recycle energy and recovery of unconverted NH3;
- duty-cycle, start/stop response, maintenance and consumables.

A lower single-pass conversion could be preferable only if the full separation/recycle chain achieves lower total energy and cost per kg of accepted delivered H2. That is a falsifiable benchmark hypothesis, not an established result.


## 9.15 Industrial plasma Power-to-X TEA comparator — 2026-11-15

Kaufmann et al., *Electrification of the chemical industry via plasma power to X – A techno economic view*, Energy 358 (2026), DOI https://doi.org/10.1016/j.energy.2026.141338.

The source models plasma-based routes from biogas to syngas and downstream Fischer–Tropsch fuels or methanol. It reports a leading biogas-reforming route at approximately EUR 840/t syngas and about 44% plant efficiency for a 300 MW scale. Electricity accounts for about 50–60% of total syngas cost. Under favourable renewable-electricity assumptions and reduced full-load hours, the study models syngas costs below EUR 600/t.

This is a plant-level techno-economic model, not a physical demonstration or observed product price. Its 44% metric is not plasma coupling efficiency. The favourable-cost result depends on source-specific power price, location, capacity-factor and process assumptions.

Benchmark consequence: preserve complete syngas quality, biogas/feed boundary, grid-side electricity, power conversion, reactor/plant auxiliaries, capacity factor, renewable supply profile, CAPEX/OPEX, maintenance, coproducts and downstream integration. A plasma route only beats the incumbent if it produces the same accepted syngas service at lower risk-adjusted lifecycle cost under matched assumptions.


## 9.16 Liquid-metal fusion-blanket MHD: new SAM/NekRS reference — 2026-11-15

Hua et al., *MHD capabilities in SAM and NekRS for fusion liquid metal blanket applications*, Fusion Engineering and Design (September 2026), DOI https://doi.org/10.1016/j.fusengdes.2026.115888.

The paper presents:
- reduced-order high-Hartmann / high-interaction-parameter MHD modelling in system code SAM;
- full MHD CFD capability in NekRS;
- analytical verification for conducting square ducts in uniform magnetic fields;
- code-to-code comparisons;
- validation against historical ALEX measurements for square and round ducts under applied/non-uniform magnetic fields.

The paper states that insulating-duct capability and validation against ALEX insulating round-duct experiments remain future work. Therefore this is a valuable source-pinned duct V&V baseline, not a blanket qualification or a validated insulating-wall result.

### Benchmark implication

Reproduce each source case with exact geometry, electrical wall condition, conductivity ratio, Hartmann / Reynolds / interaction parameters, applied-field profile, boundary conditions, model closure, solver revision and experimental uncertainty. Compare pressure drop, flow distribution and thermal observables separately. Hold out at least one field profile or geometry for transfer testing. Keep SAM reduced-order output, NekRS high-fidelity output and ALEX measurements as three distinct evidence records.

The source register adds `FUSION-BLANKET-MHD-SAM-NEKRS-ALEX-2026` plus negative controls against model/evidence substitution and extrapolation to insulating/full-blanket designs.


## 9.17 Warm-plasma CO2 quenching frontier — 2026-11-15

The new [PLASMA-QUENCH-001 #12342](https://github.com/Luminous-Dynamics/symthaea/issues/12342) separates post-plasma chemistry preservation from in-plasma dissociation. Four source cases have been added:

- **Counterflow experiment:** Chen et al., Journal of Environmental Chemical Engineering 14 (2026), DOI 10.1016/j.jece.2026.122322. The MAGD-stabilized 4 slm / 792 W point reports 23.6% conversion and 25.1% efficiency. A separate 302 W highlight reports 46.2% efficiency; its conversion is not inferred. The source reports 207.3% / 203.6% relative improvements at the 2.5 counterflow ratio / 30 mm distance parameter point. These relative and absolute metrics remain distinct.
- **Particle-assisted catalytic quench model:** Albrechts et al., Journal of CO2 Utilization 112 (2026), DOI 10.1016/j.jcou.2026.103579. A kinetic model predicts 49% / 29% baseline to 79% / 47% with Al2O3-particle-assisted cooling and O-O recombination; SO2-assisted model results are 72% / 43%. All are model predictions.
- **Multidimensional arc quenching model:** Albrechts et al., Chemical Engineering Journal (2026), DOI 10.1016/j.cej.2026.176010. The 2D model is compared with nozzle and heat-exchanger experiments. It predicts around 20% conversion / 30% efficiency under source-comparable stable-takeover cases; >50% conversion is conditional on perfect wall cooling and must not be represented as demonstrated.
- **Thermal-management measurements/model:** Kaufmann et al., Results in Chemistry 27 (2026), DOI 10.1016/j.rechem.2026.103513. The source reports radiation up to 70% of plasma power in its reactor and source-specific thermal gradients / catalyst-zone temperature results; these are not universal constants or a complete product efficiency.

The benchmark treats product-species preservation, cooling trajectory, discharge stability, heat recovery, pressure drop, quench auxiliaries, separation, uptime and maintenance as coupled but separately measured dimensions. The corpus now includes four quench source cases and eight adversarial controls against operating-point mixing, relative/absolute metric confusion, model-to-experiment promotion, ideal-wall extrapolation, chemistry transfer and omitted quench loads.


## 9.18 CO2 quench benchmark and point identity — 2026-11-15

The new scientific workstream is [PLASMA-QUENCH-001 #12342](https://github.com/Luminous-Dynamics/symthaea/issues/12342), with Symtropy scenario-replay integration at [#5391](https://github.com/Luminous-Dynamics/symtropy/issues/5391).

Four 2026 source cases were added for counterflow quenching, particle-assisted catalytic kinetic quenching, multidimensional arc quenching CFD, and thermal-management measurements/modeling. Eight negative controls prevent mixing operating points, confusing relative with absolute gains, promoting model predictions into experimental evidence, applying ideal-wall assumptions as measured results, transferring chemistry regimes and omitting quench energy/auxiliary burdens.

The counterflow paper's 792 W / 4 slm MAGD point (23.6% conversion / 25.1% efficiency) is separate from its 302 W point (46.2% efficiency; conversion not specified in that highlight). The 207.3% / 203.6% values at a 2.5 ratio and 30 mm distance are relative improvements against a paper-defined baseline.


## 9.19 Corpus snapshot — 2026-11-15

The source-register branch contains 31 unique source cases, 40 unique negative controls and 4 derived checks. The separate offline-audit branch merges the source-register content with six additional microplasma cases, for 37 cases / 46 controls / 5 derived checks. These counts refer to different branch snapshots.


## 9.20 Separate DBD syngas comparator — 2026-11-15

A further 2026 competitor has been added as a different product lane: *Direct synthesis of syngas from humidified CO2 using a coaxial dielectric barrier discharge plasma reactor*, RSC Advances 16 (2026), DOI https://doi.org/10.1039/D5YA00293A.

The experimental study reports energy-efficiency values from 4.5% to 23.3% and H2/CO molar ratios from 0.1 to 0.2 across tested humidified CO2/H2O conditions. A pure-CO2 feed comparator is separately reported at 13.3% energy efficiency with predominantly CO product.

This is a useful DBD technology comparator but not a direct leaderboard point against dry warm-arc CO2-to-CO quenching: feed, product, and possibly energy denominators differ. H2/CO ratio is not CO2 conversion or CO selectivity. The manifest keeps the pure-feed point separate from the humidified syngas ranges and adds three negative controls against feed/product and operating-point mixing.


## 9.21 DBD humidified-CO2 syngas comparator — 2026-11-15

A 2026 coaxial dielectric-barrier-discharge experiment was added as a separate feed/product lane: *Direct synthesis of syngas from humidified CO2 using a coaxial dielectric barrier discharge plasma reactor*, RSC Advances 16 (2026), DOI https://doi.org/10.1039/D5YA00293A.

The source reports 4.5–23.3% energy efficiency and H2/CO molar ratios of 0.1–0.2 across humidified CO2/H2O conditions. A pure-CO2 condition is separately reported at 13.3% energy efficiency with predominantly CO. These are not warm-arc dry CO-only quench points. H2/CO ratio is not CO2 conversion or CO selectivity. The case is retained as NOT_COMPARABLE until feed, product and energy boundaries are harmonized.


## 9.16 Methane-pyrolysis challenge set — 2026-11-15

Three recent sources now form a deliberately adversarial benchmark rather than a pro-plasma narrative.

### 2026 direct-heating multi-objective review

Azadmehr et al., *A comprehensive review of direct-heating modalities for low-emission hydrogen production via methane pyrolysis*, Journal of Analytical and Applied Pyrolysis 196 (June 2026), DOI https://doi.org/10.1016/j.jaap.2026.107776.

This review curates 156 experimental data sets from 26 peer-reviewed sources and compares microwave, plasma, resistive/combustion and shock-wave heating across LCOH, methane conversion and hydrogen selectivity. The paper reports shock-wave heating near 92% conversion at about USD 0.70/kg H2 when cost and conversion are prioritized; microwave near 91% conversion, 99% selectivity and USD 1.09/kg H2 when selectivity is also prioritized. Plasma and resistive/combustion configurations are reported outside the Pareto front in the studied corpus/weightings.

This is a review-level, dataset-derived Pareto result, not a theorem about every plasma reactor or future process. The underlying 156-row data and inclusion/normalization choices must be reconstructed before claiming an independent replication.

### Arc-plasma methane pyrolysis process TEA

Xu, Sun and Elgowainy, *Modeling and techno-economic analysis of methane pyrolysis via arc plasma for hydrogen and carbon black production*, International Journal of Hydrogen Energy 221 (27 March 2026), DOI https://doi.org/10.1016/j.ijhydene.2026.154181.

The Aspen Plus / discounted-cash-flow model uses approximately 52 kMT H2/year and nearly USD 1 billion total capital. It reports 39 kWh/kg H2 without on-site power generation and 32 kWh/kg H2 with on-site power generation. Its modeled minimum selling price is USD 1.10–1.36/kg H2 (2023 dollars), with black-carbon sales offsetting electricity costs.

This is a process model and TEA, not a commissioned plant or observed product price. The carbon-black market price and grade are major assumptions. The on-site generation case cannot be compared directly with a purchased-grid or reactor-only meter.

### Harmonized methane-pyrolysis LCA and total cost

*When does methane pyrolysis become competitive? A life-cycle and total cost assessment of hydrogen pathways*, International Journal of Hydrogen Energy 229 (28 April 2026), DOI https://doi.org/10.1016/j.ijhydene.2026.154768.

The study harmonizes a 1 kg H2 plant-gate functional unit at 200 bar, cradle-to-gate supply/conversion/PSA/compression, ReCiPe 2016 impacts and monetized environmental endpoints. It reports median total costs of USD 6.6/kg for SMR, USD 10.8/kg for methane pyrolysis without carbon credit and USD 4.8/kg with avoided-product carbon-black credit. The reported methane-pyrolysis GWP is about 12.6 kg CO2e/kg H2 without credit and approaches 5.1 kg CO2e/kg with credit. The modeled carbon-credit advantage erodes below about 50–55% methane conversion.

This "total cost" includes monetized damages and is not plain LCOH or the same metric as the arc-plasma paper's minimum selling price. Avoided-product credit is a counterfactual displacement assumption, not automatic revenue.

### Paradigm shift: coproduct evidence before coproduct economics

A carbon coproduct only counts when the benchmark establishes, in order:
1. gross theoretical carbon yield from stoichiometry;
2. captured and weighed carbon mass with uncertainty;
3. carbon grade/phase/morphology/ash/surface properties and contaminant limits;
4. accepted product specification and independent qualification;
5. buyer/offtake and realistic market-capacity envelope;
6. actual displaced product and allocation/avoided-product counterfactual;
7. net value after collection, finishing, packaging, transport, QA and sales costs.

Until those gates pass, the evaluator must show a no-credit case and may not let speculative carbon-black value make hydrogen appear cost-competitive.

The benchmark must preserve distinct quantities: reactor energy, wall-plug energy, on-site power-generation boundary, PSA/compression load, methane conversion, H2 selectivity, carbon collection yield, carbon grade and product-level accepted output. Pareto comparisons should be repeated under no-credit, allocation and qualified avoided-product-credit policies, with sensitivity intervals for electricity, methane leakage and carbon price.


## 9.17 Machine-readable research targets — 2026-11-15

The source register now includes `benchmark_targets`, each bound to exact source-case IDs and marked `RESEARCH_TARGET_NOT_RESULT`.

- `NH3-DELIVERED-H2-PARETO-001`: stretch target >99% NH3 conversion, >=51 g H2/kWh and >=150 g H2/h **at the same operating point**, with full wall-plug-to-accepted-H2 accounting and a matched catalytic/thermal baseline.
- `CH4-PYROLYSIS-PLASMA-PARETO-CHALLENGE-001`: combined challenge envelope >=92% methane conversion, >=99% H2 selectivity and <=USD 0.70/kg H2 LCOH, all at the same operating point and matched boundary. Show no-credit, allocation and avoided-product-credit cases independently. Carbon credit requires qualified product grade, market capacity and offtake.

The goals intentionally combine advantages that current published points report separately. They are falsifiable targets, not results, standards or assumed feasible operating points.


## 9.18 Plasma-assisted ammonia synthesis — separate product and energy lane

The 7 July 2026 JACS study, *Plasma-Electrocatalytic Nitrogen Reduction for Ammonia Synthesis* (DOI https://doi.org/10.1021/jacs.6c04148), reports an integrated nonthermal-plasma/electrolysis membrane-electrode assembly. Among 14 screened metals, Ag gave 7.7 ± 0.8 nmol NH3 s−1 cm−2 and 86 ± 14% electrochemical Faradaic efficiency at −0.54 V vs RHE, room temperature and atmospheric pressure. Plasma/electrolysis on-off controls and 15N2 labeling support nitrogen-source attribution; modeling and Raman results support a vibrationally excited N2 pathway.

These are valuable reactor-level mechanism and selectivity results, not a complete plant energy metric. The 2026 industrial roadmap separately reports plasma-ammonia synthesis energy yields around 0.3–3 g NH3/kWh and about 127 g NH3/kWh for Haber–Bosch, but explicitly notes that the stated comparison excludes hydrogen-production energy. Neither the JACS area-normalized rate nor its electrochemical Faradaic efficiency can be promoted into full-chain grid-to-ammonia efficiency.

A third machine-readable target, `NH3-PLASMA-SYNTHESIS-FULL-CHAIN-001`, requires at least 3 g accepted NH3/kWh after including hydrogen generation, plasma and electrolysis power, gas/feed handling and product separation, with a matched Haber–Bosch baseline under the same H2-energy boundary. This is a research target, not an achieved result.


## 9.19 Power-to-X biogas reforming stretch target

`BIOGAS-PLASMA-SYNGAS-COST-001` binds the September 2026 biogas-to-syngas source case to a falsifiable challenge: <= EUR 600/t accepted syngas while maintaining >=44% plant efficiency at the same operating point. The target requires matched syngas composition/quality, complete grid-side plant energy, electricity price and capacity-factor scenario, downstream product service and no speculative coproduct credit. It is a research goal, not a result or a direct reuse of the source's separate favourable-cost and baseline-efficiency numbers.


## 9.22 Ammonia plasma thermal integration challenge — 2026-11-15

The September 2026 microwave NH3 cracking study (Quiroz Marnef et al., DOI 10.1016/j.ijhydene.2026.156912) reports a coupled fluid / heat-transfer / electromagnetic / chemistry model and identifies heat-balance losses and an insulating sleeve of cold NH3 as an improvement pathway. Its three measured operating points remain distinct:
- >99% conversion at 45 g H2/kWh, 242 kJ/mol;
- 92.4% conversion at 150 g H2/h, 50 g/kWh, 217 kJ/mol;
- 76% conversion at 51 g/kWh, 210 kJ/mol.

### New research target: NH3-THERMAL-INTEGRATION-FULL-CHAIN-001

Target: at least 15% lower **net grid-to-accepted-H2 specific energy** than a paired no-recovery baseline, while achieving >99% NH3 conversion and at least 150 g H2/h at the same operating point, with heat-balance residual no greater than 5% of the declared thermal input or an appropriately tighter uncertainty-supported tolerance.

The target requires matched feed/composition/pressure, product purity and pressure, steady-state duration and duty cycle; measured useful heat at the declared process interface; insulation/recuperator pressure drop, thermal storage, cooling, pumps and all auxiliaries included; no double counting of recirculated heat or reaction enthalpy; and uncertainty/repeat evidence.

This is an intentionally difficult Pareto challenge. The published 150 g/h point is at 92.4% conversion, not >99%; the benchmark refuses to combine those attributes. The target is a research objective, not published performance or a Luminous result.

Four new controls refuse omitted thermal auxiliaries, unmeasured recoverable enthalpy, operating-point mismatch and non-closing/double-counted thermal balances.


## 9.24 Plasma-to-electrocatalyst N2 vibrational-state transfer — 2026-11-15

The JACS 2026 plasma-electrocatalytic nitrogen reduction source (DOI 10.1021/jacs.6c04148) is now represented with an explicit state-transfer evidence chain. It keeps three physical observation classes separate:
- 15N2 isotope attribution and plasma/electrolysis on-off controls;
- in-situ Raman observation of N-N-containing surface intermediates;
- measured NH3 area production rate and electrochemical Faradaic efficiency.

It separately records model-derived estimates: effective N2 vibrational-state temperature near 4,300 K; estimated 11.8% population in v=4–8; and modelled persistence toward the cathode. Those are not directly measured catalyst-interface populations or fluxes.

### New target: NH3-PLASMA-N2V-TRANSFER-CAUSAL-001

The benchmark requires state-resolved N2(v) at the catalyst interface or an independently validated measurement operator, uncertainty in discharge-to-interface transfer, isotope-attributed product, matched thermal/electrical controls and full grid-to-accepted-NH3 accounting. It explicitly refuses:
- model-derived state estimates promoted to physical interface measurements;
- discharge-exit population substituted for catalyst-interface population;
- Raman intermediate signatures substituted for quantitative flux;
- isotope attribution or Faradaic efficiency promoted to a causal mechanism or full-system energy claim.

This is a mechanism/evidence qualification target, not a yield target or achieved result. The physical interface-resolved population/flux and its uncertainty remain an open research gap.


## 9.25 Gas–liquid plasma-electrocatalytic ammonia comparator

Zhu et al., *One-step ammonia synthesis via a gas–liquid plasma-electrocatalytic tandem reactor with effective proton utilization*, AIChE Journal 72(8), e70407 (first published 20 April 2026), DOI https://doi.org/10.1002/aic.70407.

The source reports:
- 150.2 μmol NH3/h at about 20 mA/cm²;
- 1,161 kWh/kg NH3 specific energy consumption;
- a gas–liquid non-thermal plasma + electrocatalytic tandem architecture using plasma-activated nitrogen components and H2O-derived adsorbed H on Cu;
- 0.861 g NH3/kWh by reciprocal arithmetic on the source-reported system-specific-energy figure.

The absolute 150.2 μmol/h rate is not directly comparable with the JACS 7.7 nmol/s/cm² area-normalized rate without actual active area and equivalent product basis. The 1,161 kWh/kg value is a source-specific tandem-system metric, not automatically plant-gate accepted NH3 energy: Table S6 separates NTP and ERR input accounting; product isolation/purification and all auxiliaries still need reconciliation.

### Reproduction opportunity

The publisher identifies numerical data for Figures 1–5 and supplementary figures, Ansys Fluent H5 CFD input cases, and supplementary Tables S5–S7 covering economic assumptions and separate NTP/ERR energy inputs. Triplicate independent measurements are reported for selected figures; other results are single-run data. This provides a concrete source-reconstruction path, with repeatability limits preserved.

The benchmark should independently reconstruct the combined NTP+ERR energy denominator, then extend the boundary to accepted NH3 product recovery. Do not borrow the JACS area-normalized product metric, Faradaic efficiency or N2(v) mechanism into this different reactor.


## 9.26 Full-chain ammonia competitiveness: integrated SOEC–Haber–Bosch baseline — 2026-11-16

Díaz-Motta et al., *Modeling and performance analysis of an integrated SOEC-Haber-Bosch system for offshore ammonia production*, International Journal of Hydrogen Energy 236 (22 May 2026), DOI https://doi.org/10.1016/j.ijhydene.2026.155184, provides a stronger incumbent reference.

The published steady-state plant model reports 3,037 t NH3/day at 1,030.2 MW total input, with 943.8 MW SOEC, 67.6 MW Haber–Bosch and 18.8 MW air separation. It reports 8.14 kWh/kg NH3 and 63% LHV-NH3 efficiency. This is a modeled plant, not a commissioned plant. Its stated analysis excludes offshore wind farm, electrical/hydrogen storage, ammonia storage and downstream transport. The paper discusses seawater desalination/MVC; its treatment in the headline power total needs reconciliation before treating it as a universal plant-gate baseline.

The IEA *Global Hydrogen Review 2026: Assumptions Annex* separately reports 2.22 GJ/t NH3 electricity for Haber–Bosch including ASU and 0.18 t H2/t NH3 consumption. That is a component baseline only: hydrogen-production energy and electrolysis plant CAPEX are outside this HB+ASU figure. It must not be presented as total ammonia energy.

### The actual competitiveness gap

The gas–liquid plasma-electrocatalytic tandem study (Zhu et al., AIChE Journal 72(8), e70407, DOI 10.1002/aic.70407) reports 1,161 kWh/kg NH3, equivalent by reciprocal arithmetic to about 0.861 g NH3/kWh. The separate SOEC–HB model reports 8.14 kWh/kg NH3, equivalent to 122.85 g/kWh. The source metrics have different system boundaries and **are not a matched comparison**, but they expose the scale of the challenge: their reported energy intensities differ by roughly 142.6x.

The earlier >=3 g NH3/kWh research target is an intermediate demonstrator target, not industrial parity. It corresponds to 333.3 kWh/kg NH3, still about 41x the 8.14 kWh/kg integrated reference. The new competitiveness target is at least 10% better than the 8.14 kWh/kg modeled reference, i.e. <=7.32 kWh/kg accepted NH3, after matching NH3 purity/pressure, hydrogen generation, nitrogen supply, plant auxiliaries, water/desalination, availability and all declared exclusions. This is a stretch research target, not an achieved result.

### Area-scale implication

The JACS 2026 source's area-normalized experimental rate of 7.7 nmol NH3 s-1 cm-2 corresponds by unit conversion to approximately 0.1133 kg NH3/m2/day. A purely geometric extrapolation gives about 8,826 m2 active area for 1 tonne/day at that rate. This is not a scalability prediction: packing, current distribution, mass transfer, catalyst loading, degradation, downtime and balance-of-plant are omitted.

### New source controls and derived check

The source manifest now includes:
- NH3-SOEC-HB-OFFSHORE-2026: full integrated model baseline;
- NH3-IEA-HB-ASU-2026: IEA component electricity and H2-feed baseline;
- NH3-PLASMA-TO-SOEC-HB-ENERGY-GAP-2026: derived source-metric and geometric area arithmetic, with explicit NOT_COMPARABLE boundary status;
- NH3-PLASMA-SYNTHESIS-SOEC-HB-COMPETITIVENESS-001: <=7.32 kWh/kg accepted NH3 target, at least 10% below the 8.14 kWh/kg modeled integrated reference.

Six new negative controls reject boundary mixing, false industrial-parity claims from the 3 g/kWh intermediate target, area-rate/absolute-rate conflation, promotion of IEA HB+ASU electricity to full-chain energy, and false inclusion of storage/wind/transport in the integrated plant model.
