# LUNAR-PLASMA-ISRU-001 — fluorinated plasma regolith pathway benchmark v0.1

**Status:** Source-grounded research protocol and evidence register. No Luminous plasma result, process design, operational recipe, or physical qualification is claimed.  
**Parent:** [Symthaea issue #12294](https://github.com/Luminous-Dynamics/symthaea/issues/12294)  
**Symtropy companion:** [LUNAR-PLASMA-PROCESS-TWIN-001 #5343](https://github.com/Luminous-Dynamics/symtropy/issues/5343)  
**Source register:** [lunar-plasma-isru-sources-v0.json](../../data/research/lunar-plasma-isru-sources-v0.json)

## Executive finding

Two peer-reviewed 2026 studies provide an early experimental basis for applying fluorinated dry-plasma processing to lunar-regolith simulants. This is a credible new research lane, but it is not yet an established oxygen-production pathway.

The important change in research question is:

> Do not optimize plasma exposure or detected oxygen-bearing species. Qualify the complete conversion of characterized regolith into accepted, purified oxygen and other qualified products, while closing the fluorine-bearing reagent loop, energy boundary, residues, containment, reliability, and maintenance requirements.

The first paper uses optical-emission spectroscopy (OES) and X-ray photoelectron spectroscopy (XPS) to report oxygen/carbon-monoxide signatures and fluorination effects after treating EAC-1a and LHS simulants. The second investigates measurable O2 / CO / CO2 extraction signatures and mineral-specific solid-phase changes using EDS, XRD, and XPS. These are valuable experimental observations, but do not establish a production-rate or whole-system advantage.

The literature therefore opens an opportunity while also defining the first hard gates: quantitative product recovery, reagent inventory closure, mass and energy balances, product purity, and feedstock transfer.

## Source-backed state of the art

| Pathway | Source-reported result | Evidence boundary | Important missing comparison terms |
|---|---|---|---|
| Fluorinated plasma, EAC-1a / LHS | O and CO spectral signatures; fluorine detected on processed surfaces; separate SF6 and C4F8 source cases | Published experimental qualitative demonstration | Accepted O2 mass rate, full gas composition, purity, grid-to-product energy, reagent recovery, lifetime |
| Fluorinated plasma, mare simulant | Study reports measurable O2 / CO / CO2 gases; SF6 and C4F8 show different mineral response | Published experimental investigation of gas and solid-phase changes | Common-basis product yield, gas purification, fluorine and oxygen closure, campaign-scale repeatability |
| Molten regolith electrolysis (MRE) | NASA reports 25 kg simulant processed and average O2 production of 0.07 kg/h during a 9-hour vacuum demonstration | NASA-reported physical demonstration | Matched feedstock composition, product purity, complete energy and auxiliary boundary, lifetime |
| Solar-vacuum pyrolysis | 1.05% O2 mass yield from EAC-1 simulant; 2.47% of total simulant oxygen content recovered | Published experimental research | Matched product purity, system energy, equipment mass, throughput, recovery and maintenance |

These four rows must not be ranked directly. The evidence classes, feedstocks, process boundaries, observables, product definitions and measurement methods differ. A comparison becomes scoreable only after a common product contract and matched boundary are defined.

### Primary sources

1. Cowley et al., *Fluorine plasma processing of lunar regolith simulant for lunar in-situ resource production*, Planetary and Space Science 282 (2026), published 1 November 2026. [DOI 10.1016/j.pss.2026.106337](https://doi.org/10.1016/j.pss.2026.106337). The authors describe a first qualitative experimental demonstration using EAC-1a and LHS simulants, with OES and XPS evidence.
2. Jaggernauth et al., *Impact of oxygen extraction from lunar regolith simulant using fluorinated plasma processes*, Scientific Reports (2026), published 2 September 2026. [DOI 10.1038/s41598-026-67891-2](https://doi.org/10.1038/s41598-026-67891-2). The study reports O2 / CO / CO2 gas extraction evidence and mineral-dependent reactions in mare simulant.
3. NASA, *Large-Scale Demonstration of Molten Regolith Electrolysis for Oxygen Production on a Relevant Environment*. [NASA NTRS 20260006167](https://ntrs.nasa.gov/citations/20260006167). NASA reports 25 kg of simulant and 0.07 kg/h average O2 production over a 9-hour sustained vacuum test.
4. Robinot et al., *Quantification of oxygen production from solar pyrolysis of lunar regolith*, Advances in Space Research 77 (2026). [DOI 10.1016/j.asr.2026.02.003](https://doi.org/10.1016/j.asr.2026.02.003). The source reports 1.05% O2 extraction mass yield from EAC-1 and recovery of 2.47% of the simulant's oxygen mass content.

Source identities and evidence classes are machine-readable in `data/research/lunar-plasma-isru-sources-v0.json`. The JSON is a source register, not the papers' raw datasets.

## Paradigm shifts

### 1. Qualify a product, not a signal

Keep the following states distinct:

`detected species → quantified gas stream → separated O2 → purified O2 → accepted stored/delivered O2`

OES signatures are not a calibrated mass-flow measurement by themselves. CO and CO2 are not O2. A product contract must specify purity, pressure, delivery interval, storage compatibility, contaminants, and accepted mass before a candidate can score.

### 2. Close the reagent loop before claiming lunar advantage

Fluorine-bearing process material must be represented as a circulating inventory with explicitly tracked inputs, recovered streams, makeup, purge, residue binding, contamination, leakage, regeneration loss, and waste disposition. Missing reagent terms block a closed-loop or low-resupply claim. A dry process is not reagent-free.

### 3. Optimize mission service, not reactor temperature

The reported low process temperature is a useful research signal, not an energy-efficiency result. Account for wall-plug power conversion, plasma source, vacuum, feed preparation, reagent circulation, separation, purification, compression, thermal management, instrumentation, standby, maintenance, and waste handling.

Primary system metric:

`E_grid / m_O2,accepted`

Keep peak power, process throughput, reagent makeup per accepted oxygen, thermal rejection, equipment/support mass, and availability as separate constraints. Do not collapse unlike quantities into a synthetic scalar score.

### 4. Treat regolith as a mineral assemblage

Mare and highland simulants, mineral phases, particle size, conditioning, and batch provenance remain separate strata. The 2026 work reports different behaviour across mineral components and process gases. The benchmark must preserve those differences rather than assume one universal regolith response.

### 5. Treat coproducts as conditional options

Metal-bearing or silicon-bearing streams, salts, fluorinated solids, and residual material are not automatically useful products. Record composition, purity, recovery, physical form, and downstream process requirements. Credit a coproduct only after its receiving process and product qualification are evidenced.

### 6. Design for repair and closure from the beginning

A lunar process must account for replacement parts, seals, pumps, sensors, gas capture, contaminated surfaces, residue handling, calibration, and repair. The best process may be the one with lower imported critical-path mass and easier maintenance, even if its reactor-level conversion figure is not the highest.

## Benchmark protocol

### A. Source-faithful reproduction

- Pin exact article versions, supplementary materials, source rights, and data availability.
- Preserve article-specific sample, diagnostic, instrument, and processing identities where reported.
- Reproduce reported qualitative findings only when source data and methods are accessible.
- Record absent raw traces, calibration, detection limits, sample counts, uncertainty, and unreported settings as explicit unknowns.
- Never infer numeric oxygen yield from qualitative OES / XPS statements.

### B. Feedstock transfer

Separate EAC-1a, LHS, mare, highland, and any later simulant / lunar-return feedstock. Bind each specimen to lot, mineralogy, composition, particle-size distribution, conditioning, provenance, and uncertainty. Hold out complete material lots for transfer evaluation.

### C. Product and mass balance

Track dry feed, retained solids, gas products by species, reagent inventory, recovered reagent, makeup, purge, residue-bound reagent, and rejected/unknown material. O2 mass must be independently quantified and traceable to a calibrated measurement and stated separation boundary.

### D. Energy and exergy boundary

Use complete, non-overlapping boundaries. Distinguish grid electricity, plasma terminal power, vacuum, gas handling, separation, purification, compression, thermal management, standby, and maintenance. State the energy basis and time interval. Heat recovery is a separately measured useful service, not an assumed credit.

### E. Reagent containment and circularity

The research contract must include inventory reconciliation, recovery yield, impurity accumulation, regeneration demand, and waste disposition. The work does not provide operational gas recipes or authorize hazardous process operation. Containment and environmental review are gating evidence, not optional score bonuses.

### F. Reliability and supportability

Use per-article exposure, starts, accepted batches, rejected batches, downtime, drift, maintenance events, consumables, and component replacement. Program-level totals cannot be misrepresented as per-unit lifetime.

### G. Matched comparison

Compare plasma with MRE, solar-vacuum pyrolysis, and other routes only for the same feedstock class, accepted oxygen specification, delivery boundary, duty cycle, and included subsystems. Until matched, set comparison status to `NOT_COMPARABLE`.

## Required metrics

| Dimension | Metric | Scoring rule |
|---|---|---|
| Product | kg accepted O2 / kg dry feed | Requires calibrated, source-traceable product mass |
| Resource | fraction of feedstock oxygen recovered as purified O2 | Requires feed assay and complete product recovery |
| Product quality | purity, pressure, contaminants, storage compatibility | Must satisfy frozen customer contract |
| Chemistry | O2, CO, CO2 and other species separately | Never pool unlike species |
| Reagent loop | makeup mass / accepted kg O2; recovered fraction | Unknown streams block closed-loop claim |
| Mass closure | input = products + residues + losses + inventory change | Include uncertainty; do not force unknown to zero |
| Energy | grid kWh / accepted kg O2 | Requires complete source-to-product boundary |
| Power | peak kW, time profile, duty cycle | Separate from cumulative energy |
| Thermal | heat rejected and useful heat recovered by grade | Credit only measured/qualified recovery |
| Reliability | accepted batches / attempted batches; uptime and drift | Keep failed starts and rejected batches |
| Supportability | imported spares, replacement mass, repair time | Bind to exact article and campaign |
| Generalization | held-out feedstock lot / mineral class | Report each stratum; no pooled headline alone |
| Decision quality | regret, false-feasible rate, abstention and information gain | Fixed budgets, sealed holdouts, paired uncertainty |

## Proposed gates (future physical campaign; not achieved results)

- 100% of mass streams, energy inputs, reagent streams, residues and waste streams have explicit measured / modelled / unknown status.
- Every accepted O2 claim binds to a defined purification and delivery contract.
- Mass and energy balance residuals close within a preregistered uncertainty-aware tolerance.
- No closed-loop claim with any unknown inventory, recovery, purge, residue or leakage term.
- All attempted batches, failed starts, rejected products and maintenance intervals remain visible in denominators.
- Product yield generalizes to held-out feedstock lots before claiming broad lunar-regolith transfer.
- A candidate is preferred only when it is Pareto-better on a mission-relevant dimension without violating product, containment, mass-closure, energy-availability, or supportability constraints.
- If the measurement system cannot resolve a decision margin, the result is `BLOCKED` or `NOT_SCOREABLE`, not a pass.

## Implementation sequence (runner-independent first)

1. **Source register:** freeze the four literature / NASA records and missing-field inventory.
2. **Process graph:** define species- and inventory-specific feed/product/reagent/residue edges.
3. **Synthetic accounting fixture:** add hand-checkable positive, blocked, and rejected cases. All values must be visibly synthetic and not interpreted as operating guidance or physical results.
4. **Independent evaluator:** add a small Rust-native, solver-free evaluator that reuses existing quantity, unit, evidence, and mass-balance contracts.
5. **Adversarial suite:** mutate reagent inventory, species identity, meter boundary, calibration, lot identity, and unknown streams; verify fail-closed dispositions.
6. **External solver preflight:** only after the published experimental case, solver capability, inputs, and diagnostic observables are source-pinned.
7. **Physical programme:** requires separate facility, metrology, environmental, containment, safety, and campaign authorization. This document does not authorize it.

## Existing Luminous owners

Compose, do not duplicate:

- Symthaea #5669 (regime-explicit plasma engineering), #7588 (plasma V&V), #7600 (plasma credibility audit), #7685 (decision-aware multifidelity experiment selection).
- Symthaea #11718 / #11751 (plasma energy boundary and audit), #1604 (Planetary Industrial Ecology).
- Symtropy #1685 / #1703 (plasma scenario receipts and fail-closed solver capability), #1803 (resource-to-product mass closure), #1970 (lunar energy dispatch), #5343 (this process-twin integration).

## Explicit non-claims

This benchmark plan does not establish Luminous oxygen yield, reagent recycling capability, lunar industrial feasibility, cost advantage, safe operating envelope, environmental acceptability, product qualification, or flight readiness. A synthetic accounting pass is not a plasma simulation or physical measurement. Hash integrity is not scientific validation. No process recipe, gas-handling procedure, hardware actuation, or operational authority is introduced.
