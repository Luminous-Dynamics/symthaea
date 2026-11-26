# InstantClean thermal-storage and drying architecture benchmark

Status: research programme, 3 November 2026. No thermal architecture is selected and no performance result is claimed for InstantClean.

## Why this research exists

A June 2026 Applied Thermal Engineering paper studies a dishwasher integrating thermoelectric heating/cooling with thermal energy storage (TEHC-DW). The paper reports, for its own reference setup:

- 8.7% reduction in total energy;
- 40% lower internal water-heating energy;
- 42% shorter drying time at the same remaining moisture content;
- a resistance-capacitance model that predicts system temperatures within ±1 K;
- 0.952 kWh/cycle for the modified setup, as summarized in the associated research record.

These are study-specific prototype results. The 0.952 kWh/cycle number is not directly comparable with current EU Eco-label products around 0.39 kWh/cycle; the reference machine, load, test programme and boundary differ. Obtain and reproduce the paper's complete protocol before drawing market conclusions.

Paper: https://doi.org/10.1016/j.applthermaleng.2026.130695
ORNL research record: https://impact.ornl.gov/en/publications/thermoelectric-heating-and-coolingintegrated-dishwasher-with-ther/

## Architecture options to compare

1. **Resistance heating + conventional/passive drying.** Simple baseline, but high peak power and direct heat input.
2. **Heat-pump heating/drying.** Existing V-ZUG AdoraDish V6000 heat-pump models are incumbent references; exact model, programme and label must be retained.
3. **Zeolite/desiccant drying.** Bosch PerfectDry Zeolith is an existing commercial mechanism. Confirm exact SKU and region because not all PerfectDry models use Zeolith.
4. **AutoOpen / airflow / fan-assisted hot-air drying.** Miele already sells AutoOpen on current product families and announced variable hot-air drying in its G8 Diamond.
5. **TEHC + thermal storage.** Reproduce the 2026 paper's thermoelectric heating/cooling and storage system.
6. **Low-complexity thermal recovery.** Heat exchanger, retained hot water, sensible/phase-change storage or recovered drain heat without thermoelectric modules.
7. **Hybrid sequencing.** Use thermal storage to move heat across wash/rinse/dry stages, but only where total cycle and standby energy fall.

References:
- Bosch PerfectDry: https://www.bosch-home.co.uk/en/c/spotlight/blt50ac03612c881a86
- V-ZUG heat pump / OptiStart: https://www.vzug.com/gb/en/gb/en/products/kitchen/dishwashers/ultra-efficient/adoradish-v6000-with-heat-pump--p4112100010
- Miele G8 Diamond (camera / variable hot-air drying): https://www.miele.de/en/m/world-premiere-for-the-g8-diamond-8453.htm
- Miele AutoOpen / drying options: https://www.miele.co.za/domestic/1534.htm?info=200163058-ZPV

## Core hypotheses

### H1 — thermal storage reduces peak power, but may not reduce total energy

Measure charging energy, thermal retention, idle losses, discharge efficiency and cycling. The storage subsystem only wins if total energy at the appliance wall falls for the declared use profile. Peak power reduction is a separate benefit, not an energy saving.

### H2 — staged heating/cooling can improve drying energy and duration

Measure water heating, load thermal mass, hot humid air removal, condensation/dehumidification, fan power and residual moisture. Drying must be compared at the same moisture endpoint and material mix, particularly plastics and recessed cups.

### H3 — simple heat recovery may outperform TEHC on lifecycle cost

Compare thermal exchangers, drain-water recovery, sensible/phase-change stores and thermoelectric modules by energy, volume, mass, capital cost, service burden, reliability and control complexity. Do not choose TEHC because it is technologically novel.

### H4 — thermal control and cleaning optimization must remain coupled but separable

Pre-soak and enzyme/soil-specific kinetics may allow lower wash temperature; thermal storage may shift heat between stages. But temperature histories must remain within the validated soil-specific cleaning and hygiene envelope. A thermal controller cannot relax minimum rinse/sanitization constraints.

## Experimental design

### Physical reference set

Use at least:
- one current efficient heat-pump dishwasher;
- one current zeolite/desiccant-capable exact model;
- one AutoOpen / hot-air model;
- one ordinary resistance-heated model;
- the TEHC test rig or a fully documented research prototype.

Obtain exact SKU, market, firmware, options, programme, capacity and published energy/water values. Do not compare one model's best energy label with another model's best drying claim as if they were the same configuration.

### Controlled variables

Hold load configuration, soil, soil aging, detergent batch/dose, water hardness, inlet temperature, ambient temperature/humidity, programme, cycle start state and target residual moisture constant. Randomize run order and hold out full days/batches.

### Instrumentation

- calibrated wall real-power/energy analyser;
- DC bus/module power measurement for thermoelectric modules;
- water inlet and drain volume;
- upstream hot-water energy where applicable;
- temperature at inlet, sump, representative dish/load surfaces, storage and air paths;
- humidity / dew point in drying loop;
- storage state-of-charge proxy validated against measured stored energy;
- fan, pump and actuator power;
- final residual moisture by gravimetric or standardized method;
- noise, leakage and fault telemetry.

### Negative controls

- TEHC module off / bypassed;
- storage bypassed;
- storage precharged but idle for different durations;
- fan disabled;
- temperature or humidity sensor disconnected;
- deliberate thermal leakage;
- identical cycle with and without recovered drain heat.

## Metrics

Primary:
- total wall kWh per successful clean + rinse + dry;
- internal water-heating kWh;
- drying kWh and time at equal residual moisture;
- total water and cycle duration;
- cleaning/rinse/drying pass rates.

Secondary:
- peak electrical demand;
- thermal storage charge/discharge efficiency;
- storage idle loss over 1/4/8/24 hours;
- temperature-trajectory error on held-out runs;
- module mass/volume, cost, serviceability, failure rate and acoustic emissions.

Always report cold-start, warm-start and delayed-start separately. Include standby and thermal leakage. A system with lower peak draw but higher total kWh is not an energy improvement.

## Aspirational gates

- Reproduce the study's reference result before changing its topology.
- Achieve held-out thermal trajectory MAE <=1 K without hiding peak errors or unstable cases.
- At a matched wash/rinse/dry endpoint, require >=5% lower total wall energy than the best matched conventional/heat-pump/desiccant reference before pursuing product integration.
- Require a lifecycle/service case for extra thermoelectric modules and storage hardware.
- If TEHC only beats the paper's conventional baseline but loses against current heat-pump/zeolite products, reject or narrow it to peak-load shifting or a specific drying sub-process.

These are research gates, not predicted or achieved outcomes.

## Link to broader InstantClean architecture

Thermal storage and drying are separate from QuickCell topology and the scheduler. Do not integrate additional hardware into a full prototype until the topology bake-off (issue #10161) establishes whether a dedicated cell or isolated micro-zone is warranted. Thermal modelling should report exact configuration scope into Symthaea, with Symtropy only where geometry/flow interactions require validated physics.

Research issue: https://github.com/Luminous-Dynamics/symthaea/issues/10197
