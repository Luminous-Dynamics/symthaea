# Sodium-battery mission benchmark v0.1

Status: benchmark contract and synthetic reference manifest. No Luminous material, cell, manufacturing or performance claim.

## Mission

Compare sodium-ion and zero-excess sodium-metal concepts against LFP and NMC on matched stationary, cold-climate and selected mobility missions. The objective is not a technology leaderboard; it is to discover where each architecture improves delivered service, cost, safety, supply resilience and lifecycle performance under comparable boundaries.

## Technology lanes are distinct

- `commercial_sodium_ion`: hard carbon + Prussian White/PBA, hard carbon + layered oxide, and hard carbon + polyanionic cathodes.
- `zero_excess_sodium_metal`: cathode-provided finite sodium inventory, metal plating/stripping with no excess sodium foil reservoir. This is a separate post-lithium technology with different failure modes and no right to inherit commercial sodium-ion baselines.
- `lithium_iron_phosphate`: primary market baseline for stationary and cost-sensitive mobility.
- `nickel_manganese_cobalt`: higher-energy-density baseline where mass/volume constraints justify its cost and supply trade-offs.

## Evidence anchors and why boundaries matter

IEA Global EV Outlook 2026 reports commercial sodium-ion cells up to approximately 175 Wh/kg, latest LFP up to 205 Wh/kg and NMC up to 265 Wh/kg. It notes sodium-ion's low-temperature advantage, but also that sodium-ion manufacturing capacity remains only just over 1% of lithium-ion cell capacity and hard-carbon supply is poorly developed and concentrated. These are external reported/market figures, not a matched Luminous cell comparison.

Pilot-scale aqueous HC|Prussian White: Maddar et al. report ≈1 Ah pouch cells at ≈120 Wh/kg practical full-cell specific energy; approximately 91.9% capacity retention after 120 cycles at C/5, above 250 cycles before reaching 80% at 1C, and 80% at 237 cycles at 2C in their three-cell-averaged results. The paper shows moisture/dehydration and process-window sensitivity. It is a pre-pilot study, not a commercial-cell benchmark.

PNNL's localized high-concentration electrolyte work reports HC||PB full cells retaining 80% capacity beyond 1,000 cycles at 2C and multilayer pouch cells retaining 82% after 500 cycles. This is a specific electrolyte/system result; composition, loading, cell geometry, electrolyte-to-capacity ratio, temperature and mass boundary must be matched before comparison.

Zero-excess sodium-metal review (11 November 2026) summarizes demonstrations above 200 Wh/kg at the cell level. It identifies finite sodium-inventory loss and non-uniform plating/SEI/electrolyte consumption as central life constraints; 1,000 cycles at 80% retention requires average Coulombic efficiency of at least 99.98% under stated high-loading, lean-electrolyte conditions. This is a research frontier, not a commercial sodium-ion score.

References:
- IEA Global EV Outlook 2026: https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries
- Maddar et al., Advanced Energy Materials, DOI 10.1002/aenm.202505344: https://doi.org/10.1002/aenm.202505344
- PNNL localized high-concentration electrolyte study: https://www.pnnl.gov/publications/tailoring-electrolyte-solvation-structure-enhance-rate-capability-cycle-life-and
- Zhao et al., Nature Reviews Clean Technology (11 Nov 2026): https://www.nature.com/articles/s44359-026-00234-6

## Composition with existing owners

- Symthaea MAT-RD #4997, MAT-MLIP #4992, ION #4996, MAT-013 #4344, MAT-PSPP #6224, MAT-UQ #6225 and MAT-INTERFACE #6227.
- Symthaea BATTERY-INVERSE-DESIGN #11671, dynamic cell model #11485, pack capability #11271 and mission feasibility #11589.
- Symtropy EV benchmark #2164, battery circularity #2155 and vehicle portfolio #2112.

This benchmark consumes those owners. It does not create a separate material ontology, ion-transport model, sample/process lineage, uncertainty system, or pack simulator.

## Required comparison identity

Every result must bind:
- technology lane, exact cell design/configuration revision and physical specimen identity;
- chemistry, cathode/anode/electrolyte/interphase, cell format and construction;
- cell mass and volume; complete-cell inactive-component accounting (current collectors, separator, electrolyte, tabs and pouch/can);
- areal loading, N/P balance, electrolyte-to-capacity ratio, voltage window, formation protocol and stack pressure;
- temperature and ambient/coolant state, SOC range, C-rate, rest schedule, cycling protocol and end-of-life definition;
- process/sample lineage, test instrument/calibration, raw-data digest and evidence class;
- censoring/competing-event status and repeat-batch grouping.

An active-material or cathode-only Wh/kg figure cannot be promoted to complete-cell, pack or system performance. A coin cell, 1 Ah pouch and commercial-format prismatic cell are separate evidence levels.

## Matched mission scenarios

1. Cold-climate stationary storage: delivered kWh and kW at −20 °C / −40 °C, including preheating energy, power derating, thermal management, reserve and lifetime throughput.
2. Remote/off-grid stationary storage: lifetime delivered kWh, accepted-good system cost, service intervals, failure response, recyclability and supply-chain resilience.
3. Industrial equipment / urban fleet: charge windows, high-power duty, daily energy throughput, cold start, fast charge, and mass/volume envelope.
4. Hybrid sodium-ion + LFP: compare split-chemistry architecture against LFP-only under equal pack mass, volume, cost and useful service; include extra BMS/thermal/interconnect overhead and control complexity.

## Benchmark metrics

- Practical complete-cell and pack specific/volumetric energy.
- Usable discharge energy and power at each temperature, including preheat/thermal overhead.
- Capacity retention, cycle life, calendar life, Coulombic efficiency, impedance rise, self-discharge and failure mode.
- Charge acceptance, 2C/5C power, pulse-duration envelope and thermal derating.
- First-pass yield, accepted-cell yield, within/between-batch variability, scrap and process-window width.
- Cost per accepted kWh and per lifetime delivered kWh; water/solvent/process energy and material utilization.
- Lifecycle impacts and end-of-life material recovery under declared boundaries.
- OOD support, measurement uncertainty, censoring rate and independent specimen count.

## Evaluation discipline

- Freeze dataset release, source/license, test protocol, feature availability and physical group splits before model development.
- Split by physical cell, design combination and manufacturing batch. Do not place sibling cells from the same batch in train and test when claiming new-batch transfer.
- Keep observed end-of-life events, right-censored observations, competing failures and unknown endpoints distinct.
- Report all candidate and failed runs, not just the best cell.
- Compare DOE, random, uncertainty, diversity and mission-conditioned experiment-selection baselines on equal budgets.
- Require repeat production batches and independent test review before making manufacturing-capability or commercial claims.

## v0.1 software checks

The manifest schema requires each comparison to identify the technology lane, cell boundary, mass accounting, exact test condition, mission, reference evidence and source rights state. The first validator should reject: mixed sodium-ion/zero-excess-metal lane scores; active-material mass presented as full-cell mass; mismatched C-rate/temperature/voltage/loading; unknown or inadmissible data rights; duplicate physical specimens; and censored lifetimes treated as observed point targets.

## Claim ceiling

A passing manifest/contract audit demonstrates comparison structure and evidence-reference consistency only. It does not establish that sodium chemistry is safer, cheaper, more durable, greener or more manufacturable than LFP/NMC. Those conclusions require matched, replicated physical cells and complete pack/system evaluation.