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

1. Keep the four published references and their claim ceilings in the machine-readable plasma source corpus.
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
