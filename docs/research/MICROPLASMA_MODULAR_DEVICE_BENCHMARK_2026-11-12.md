# MICROPLASMA-001 — Miniaturized and modular plasma device benchmark

**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11780  
**Parent / owner composition:** PLASMA-ENG-000 #5669; PLASMA-VV-001 #7588; PLASMA-ENERGY-001 #11718; PLASMA-THERMAL-METROLOGY-001 #11758; ROB-INSTR #4866; ROB-EXP #4800; Symtropy PLASMA-TWIN #1685 and ENG-SYNTH-TWIN #1467.  
**Date:** 12 November 2026  
**Status:** source register and benchmark design only  
**Claim ceiling:** no Luminous reproduction, hardware design release, high-voltage operating procedure, medical efficacy, water-treatment qualification, or physical energization authority.

## 1. Executive thesis

Miniaturization is practical for selected low-temperature plasma devices. The next step is not merely smaller electrodes or a shorter plasma tube: it is a complete miniature system whose power conversion, gas/liquid handling, thermal management, diagnostics, useful process output and replaceable service interfaces are co-designed.

The most promising paradigm shift is:

`one large source -> many independently qualified microcells`

or, for microfluidics:

`one precision microreactor -> parallel numbered-up microreactors`

But numbering-up only works if cell-level behavior remains controlled and measured. An array must not hide failed cells, uneven gas flow, current-sharing, cross-talk or output drift.

This programme explicitly separates:
- atmospheric-pressure nonthermal plasma jets;
- microfluidic plasma/liquid reactors;
- RF / DBD microdischarge cells and arrays;
- larger-area LTCC discharge arrays;
- thermal plasma and high-temperature/fusion devices.

No generic “microplasma” score should combine these different physical regimes.

## 2. Source-backed industry reference points

### 2.1 Low-cost miniature helium APPJ — 2026

Mao et al., *A miniaturized and low-cost atmospheric pressure helium plasma jet device with high antimicrobial efficiency*, Environmental Technology & Innovation 42, 104852 (9 March 2026), DOI 10.1016/j.eti.2026.104852.

Reported source details:
- custom supply around 100 g and 15 × 10 × 2 cm;
- output approximately 900 V RMS at 83.6 kHz from a reported 10 V / 0.5 A DC input;
- 54 mm glass tube, 1.0 mm inlet ID, 0.6 mm outlet, 3 mm selected ring-electrode spacing;
- helium flow 2 L/min;
- total component cost below USD 45;
- five planktonic strains completely inactivated within 40 s under source test conditions;
- >20 h supply operation before biological tests.

Interpretation limits:
- the 10 V × 0.5 A figure is the reported DC input condition, not a complete independently metered wall-plug/service balance;
- gas consumption has economic and logistical consequences;
- the source notes limited treatment area / uneven surface coverage;
- long-run electrode wear and drift were not quantified in the paper;
- in-vitro results do not establish clinical efficacy or medical-device qualification.

Full source: https://pmc.ncbi.nlm.nih.gov/articles/PMC13290277/.

### 2.2 Membrane-stabilized microfluidic DBD — 2025

Liu et al., *Semi-permeable membrane stabilized microfluidic plasma chip for continuous, tunable synthesis of sub-10 nm nanoparticles*, Lab on a Chip 25(12), 2807–2815, DOI 10.1039/D4LC00960F.

A hydrophobic porous membrane blocks liquid while allowing plasma to pass, preserving a controlled gas/liquid interface. Gold nanoparticle sizes of 7.31–11.32 nm with 1.8 nm standard deviation are reported.

This is a strong microscale interface / product-control reference. It is not yet an industrial throughput or lifecycle-cost reference. The full energy, gas, liquid-pumping, membrane replacement and downstream separation boundaries must be extracted before comparisons.

Source: https://pubmed.ncbi.nlm.nih.gov/40406968/.

### 2.3 Individually controlled RF microplasma array — 2023

Martinet et al., *Individual ignition of RF microplasma array at atmospheric pressure*, Plasma Processes and Polymers 20, e2200131, DOI 10.1002/ppap.202200131.

The paper demonstrates five identical cells, individually ignitable and quenchable, and discusses possible extension to tens or hundreds of cells.

This is a useful modular-control direction. The larger array is not demonstrated by the five-cell result; distribution, thermal/electrical interaction, uniformity and fault isolation remain required qualification questions.

Source: https://doi.org/10.1002/ppap.202200131.

### 2.4 43-element LTCC CAP array — 2025 publication

Bhattacharya et al., *Fabrication and Characterization of a 10 × 10 cm Cold Atmospheric Pressure Plasma Array*, IEEE Transactions on Plasma Science 54(2), 496–502, DOI 10.1109/TPS.2025.3646554.

The device stacks 43 linear-discharge elements into a 10 × 10 cm treatment area using low-temperature co-fired ceramic (LTCC). The article reports 91% reduction in colony-forming units for a specified Pseudomonas fluorescens biofilm coupon treatment after 150 s at a 1.5 cm standoff.

This provides a useful numbering-up reference and a larger-area comparator. The endpoint is organism-, material-, geometry- and protocol-specific, not a universal sterilization result.

Full source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12923079/.

## 3. Why shrinking a discharge is not geometrically similar

Miniaturization changes dominant physical effects:
- sheath and wall interactions occupy a larger fraction of the active volume;
- surface-to-volume ratio increases, changing heat transfer, radical loss, quenching and material interactions;
- electrode-edge fields and breakdown paths become more influential;
- parasitic surface conduction / dielectric charging / contamination may become proportionally larger;
- gas residence time and flow distribution change;
- thermal removal and local hot spots become harder to infer from bulk temperature;
- source impedance, parasitic capacitance, waveform distortion and local cell coupling change;
- microfluidic gas/liquid interface stability becomes a first-order design concern;
- electrode/dielectric/membrane wear can dominate lifecycle performance.

Relevant dimensionless / regime descriptors must be preserved per architecture, not treated as one universal similarity rule: pressure × gap, reduced electric field, gas composition, electron-energy/transport model, residence time, flow regime, surface condition, discharge frequency and local heat-removal characteristics.

A smaller physical size is not evidence of lower energy per useful output.

## 4. The paradigm shift: a plasma fabric, not a plasma pencil

Build a library of independently characterized cell modules:
1. plasma source cell;
2. gas/liquid/material interface cell;
3. per-cell energy and status observation;
4. local thermal/material interface;
5. independently replaceable service cartridge.

Then compose them into:
- individual tools;
- linear arrays;
- planar arrays;
- stacked ceramic arrays;
- parallel microfluidic chips;
- maintainable field modules.

Each cell should have exact geometry/configuration identity. An array record should preserve its cell map, flow and power topology, disabled/failed cells, per-cell observation and aggregate service output.

Avoid common-cause failures: shared power source overload, gas-manifold maldistribution, thermal coupling, controller cross-talk, shared contamination and a single unobservable dead channel.

## 5. Competitive benchmark design

The first tournament will have four lanes:

| Lane | Main comparator | Primary output |
|---|---|---|
| Miniature APPJ | 2026 helium APPJ and commercial portable jets | useful treated area / surface endpoint per total-system energy |
| Microfluidic plasma chip | 2025 membrane-stabilized DBD chip and non-plasma microfluidic control | conforming product mass, particle-size distribution, yield and throughput |
| Numbered-up cell array | five-cell RF array and 43-element LTCC array | uniformity, independent control, fault isolation and output per footprint |
| Modular material-processing source | resistance / microwave / induction / conventional plasma treatment | conforming material outcome per total-system energy and cost |

Propulsion and fusion are explicitly separate programmes. Electrospray thrusters are not automatically plasma devices; compact fusion is not a miniature version of a cold atmospheric plasma jet.

## 6. Benchmark scorecard

**Physical size and deployability**
- total device mass / volume / footprint;
- source electronics mass/volume separately from active discharge;
- utilities, gas bottle / gas generator, pumps, cooling and exhaust equipment included;
- BOM, assembly, yield, repair and cartridge replacement cost;
- portability, service interval and field robustness.

**Energy and resource efficiency**
- grid-side energy per useful delivered output;
- supply conversion efficiency and idle/standby/startup energy;
- gas or liquid consumption per useful output;
- cooling, pumping, vacuum, controls and exhaust treatment;
- use phase and lifecycle materials.

**Plasma/process quality**
- discharge stability and spatial uniformity;
- electron/heavy-species temperature distinctness;
- species yield with calibrated diagnostics;
- dose/fluence delivered at the actual product boundary;
- process outcome quality and repeatability;
- uncertainty and run failure accounting.

**Array quality**
- per-cell output distribution and coefficient of variation;
- independent ignition, quench and status observability;
- electrical, thermal and fluid cross-talk;
- flow and power distribution;
- failed-cell isolation and graceful degradation;
- throughput per unit footprint and system power.

**Reliability**
- run hours and start/stop cycles;
- electrode, dielectric, membrane and nozzle wear;
- contamination/fouling/clogging;
- calibration/diagnostic drift;
- failed-run rate;
- replacement time and total lifecycle cost.

For water treatment, the relevant comparison must be contaminant-specific and use realistic water matrices, declared effluent endpoints and a full energy boundary (including gas, pumping and post-treatment). Report kWh/m³ and energy per order only where the underlying measurement definitions are satisfied. Compare against suitable non-plasma alternatives; do not give plasma a pass for an endpoint that UV, ozone, filtration or electrochemical methods achieve better.

## 7. Prospective stretch goals

These are research targets, not achieved results or regulatory standards:
- at least 25% improvement in useful output per total-system energy or lifecycle cost against the best matched baseline, with matched service and quality;
- array cell-to-cell output coefficient of variation ≤5% at a frozen operating/service point;
- independently observable and isolatable failed cells;
- no hidden loss of process quality as cell count or array footprint grows;
- demonstrate sustained drift and wear across repeated runs before claiming lifetime;
- preserve at least 90% of baseline per-cell useful-output performance when a designed fraction of cells is disabled, subject to a predeclared fault topology;
- at least 3 matched gas / chemistry / material conditions held out from tuning.

The initial source results are not directly comparable, so the first task is to reconstruct each source boundary rather than declaring a winner.

## 8. Luminous architecture and ownership

Symthaea:
- device identity, source register, architecture constraints, model applicability, evidence chain, benchmark and design-space exploration;
- PLASMA-ENG-000 / PLASMA-VV-001 own regime and model qualification;
- FIELD / SE-OBS / SE-SEM and ROB-INSTR / ROB-EXP own observations and experimental evidence;
- PLASMA-ENERGY-001 / PLASMA-THERMAL-METROLOGY-001 own useful service and metrology.

Symtropy:
- deterministic synthetic scenario / array topology / fault replay through #1685 and #1467;
- no plasma PDE solver, physical observation authority or hardware actuation.

Mycelix:
- later manufacturing, sourcing, maintenance, calibration chain and service-lifecycle provenance if a real product is selected.

Safety/authority:
- no direct design-to-voltage/current/RF/gas-flow command path;
- any physical test requires exact article identity, bounded engineering envelope, independent facility/operator approval, protective interlocks, calibrated instrumentation, safety case and preregistered protocol.

## 9. Next execution order

1. Keep the five published references and their claim ceilings in the machine-readable plasma source corpus.
2. Extract missing raw time-series, current/voltage waveforms, gas cost/flow and per-cell uniformity wherever source data are available.
3. Build an offline synthetic schema/validator for cell identity, array topology, output uniformity, complete utility boundaries, failed-cell accounting and matched service.
4. Reconstruct the 2026 miniature APPJ and 2025 microfluidic/array reference results only where raw inputs are available and rights permit.
5. Select one first application after boundary analysis. The current best candidates are non-clinical surface/material processing and a low-flow microfluidic chemistry test; water treatment remains a useful later service target because it has strong realistic-matrix and by-product requirements.
6. Only after model/data qualification, design any physical pilot under independent facility and safety approvals.

## 10. Negative controls

- miniature power supply size treated as total system mass/volume;
- parts cost treated as lifecycle cost;
- supply operation hours treated as electrode lifetime;
- one sample / five strains treated as universal treatment efficacy;
- one 43-element array result treated as a universal 43× scale-up factor;
- individually ignitable five-cell proof treated as hundred-cell validation;
- synthetic array replay treated as physical uniformity;
- missing gas or cooling energy treated as zero;
- a visually bright / long plasma plume treated as useful output;
- microfluidic particle size treated as industrial throughput;
- cold microplasma evidence transferred to thermal arc, MHD or fusion regime.

**Final invariant:**

`miniature cell != useful miniature system != reliable modular array != qualified industrial product`

Every transition needs its own matched evidence.


### 2.5 Integrated portable argon MobiPlas — 2024

Bansemer et al., *On the Development and Characterization of the Novel Portable Plasma Device MobiPlas*, Plasma Medicine 14(4), 39–51, DOI 10.1615/PlasmaMed.2025057641.

The device integrates a refillable steel argon tank, mobile powerbank and Raspberry Pi Pico controller. The paper reports approximately 10 minutes of operation per gas fill, maximum patient leakage current of 11 μA and a tip temperature reaching 40°C. It also documents occasional leakage-current fluctuation caused by heating in the high-voltage generator's current-limitation circuit. The INPTDAT institutional catalogue labels the device TRL 4; this is catalog metadata, not independent qualification.

The comparison lesson is that a portable plasma device must be scored on **autonomous useful service per gas fill and battery charge**, refill/recharge downtime, drift over the duty cycle and thermal/electrical stability—not just mass or absence of mains connections.

Sources: https://doi.org/10.1615/PlasmaMed.2025057641 and https://www.inptdat.de/mobiplas.


The 2026 low-cost helium APPJ paper annotates 2 L/min helium use at approximately £0.2/min under its stated supply-cost basis—roughly £12 per operating hour if sustained. This figure is source-specific, not a universal gas price, but it demonstrates why a low component BOM does not mean a low-cost portable system. The scorecard therefore treats gas consumed per useful output and autonomous runtime as first-class metrics.


## 11. Application down-selection: ambient-air DBD before noble-gas portability

New literature changes the first process-plasma application priority. Ambient-air DBD should be the leading low-cost miniaturization candidate where the required chemistry permits it; a helium/argon APPJ remains useful for localized surface or material processing where a directed jet is required.

The 2026 GAPS grid-like DBD study reports up to 4.5-log single-pass *E. coli* MG1655 bioaerosol reduction at 12.5 L/min and approximately 0.65 W discharge power. Its EEO of 2×10⁻⁴ kWh·m⁻³·order⁻¹ excludes power-supply losses. Preliminary outlet ozone is approximately 9 ppm; the authors state that downstream ozone removal and a realistic safety assessment are required. At half discharge coverage the reported reduction falls to around 2 log. These details make the source a valuable architecture reference, not a complete occupied-air cleaner.

A separate ambient-air surface-DBD paper describes an approximately 48 cm³, 55 g portable reactor without a separate process-gas feed. Its surface-treatment endpoint cannot be translated into air-cleaning CADR.

For an air-treatment pilot, the primary service is **post-abatement clean-air delivery**. Include grid-side supply, fans/compressors, filters, sensors and ozone/NOx abatement; pressure drop; humidity; particle-size distribution; bypass; noise; lifetime and maintenance. Compare to HEPA and UVGI under the same flow, removal, safety and installed-system boundary. The target is not simply maximum log reduction.

See Symthaea PLASMA-AIR-DBD-001 (#11804).

## 12. Water treatment is a separate process qualification lane

The 2026 flexible coplanar DBD water-treatment paper reports >5-log *E. coli* reduction in 800 mL within 5 minutes and 3.64×10⁶ CFU/J bacterial inactivation efficiency. That is a batch microbial endpoint, not EEO, drinking-water compliance or mineralization.

The 2026 plasma-liquid catalysis review identifies incomplete mineralization, toxic transformation products, catalyst deactivation/leaching, realistic-matrix effects, energy and scale-up as unresolved issues. A cross-study AOP review reports substantially higher median EEO for plasma than ozonation or UV/H₂O₂, but those are heterogeneous review medians—not matched trials. Any proposed water application must therefore demonstrate a contaminant-specific advantage after complete energy and by-product accounting.

Keep microbial disinfection, refractory pollutant oxidation, plasma-catalyst synergy and pre-treatment as separate benchmark lanes. Include parent conversion, TOC/COD and carbon balance, transformation products, toxicity, residual oxidants, gas-liquid transfer, catalyst stability, energy and lifecycle cost. Stop if a matched incumbent achieves the target more safely or cheaply.

See Symthaea PLASMA-WATER-QUAL-001 (#11805).
