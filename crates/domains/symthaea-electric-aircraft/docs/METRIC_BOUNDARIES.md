# Electric aircraft metric and measurement boundaries

**Purpose:** prevent false comparisons between cell, module, pack, installed-system, propulsion and aircraft-level figures.

**Rule:** every benchmark number must include its numerator, denominator, configuration, operating point, environmental conditions, uncertainty, evidence class and source revision. A number without its boundary is not an engineering benchmark.

## Evidence classes

| Class | Meaning | May be called independently verified? |
|---|---|---|
| Type-certified | Approved type design / component certificate within its stated scope | Only for the approved scope and configuration |
| Physical measurement | Raw observations from a documented, instrumented test | Only after calibration, method, traceability and uncertainty review |
| Company-reported | Manufacturer specification, announcement or internal test statement | No; attribute to manufacturer |
| Published research | Scientific paper, technical report or programme result | Depends on peer review, data availability and replication |
| Simulation | Numerical model result | No physical validation implied |
| Derived | Arithmetic using other disclosed data | Must retain and cite all source values and denominator assumptions |
| Target | Programme or project objective | Never present as achieved performance |

A certificate for an engine, motor or battery installation is not an aircraft type certificate. A flight-test authorisation or market-survey certificate is not a type certificate.

## Energy storage

Always distinguish these measurement boundaries:

1. **Cell specific energy:** measured discharge energy divided by active cell mass, with cell format, chemistry, discharge rate, voltage cut-offs, temperature, and test method stated.
2. **Module specific energy:** usable module-terminal energy divided by complete module mass, including mechanical enclosure, electrical interconnects, sensing and module-level protection actually installed.
3. **Pack specific energy:** usable pack-terminal energy divided by complete pack mass, including BMS, contactors, fuses, sensing, busbars, enclosure, thermal hardware, internal coolant and pack-level protection.
4. **Installed energy-system specific energy:** usable aircraft-installed energy divided by all installed energy-system mass, including mounting, cooling loop/pumps/heat exchangers, wiring, connectors, isolation devices, crash/fire/thermal containment, venting and installation structure.
5. **End-of-life usable specific energy:** same boundary as above, but at the stated retirement/end-of-life criterion and after the stated ageing profile.

Always state:
- beginning-of-life (BOL) or end-of-life (EOL);
- nominal/rated versus usable energy;
- SOC window and voltage limits;
- C-rate / current profile;
- cell and pack temperatures;
- state of health and ageing history;
- charge/discharge direction and rest time;
- containment and thermal-management included/excluded;
- uncertainty and repeatability.

Do not compare a cell Wh/kg with a pack or installed-system Wh/kg. Do not treat rated energy as usable mission energy. A programme target of 1,000 Wh/kg at EOL net-system level is not equivalent to a 1,000 Wh/kg cell target.

## Electric propulsion

### Power density

Report at least:
- peak/short-duration power and duration;
- takeoff power and permitted duty cycle;
- maximum continuous shaft power;
- continuous power after thermal equilibrium at a stated altitude and ambient condition;
- mass boundary (motor alone; motor plus inverter; complete propulsor including controller, cooling and mounts);
- redundancy and number of active/available channels;
- voltage, RPM, torque, coolant, inlet conditions and temperature limits.

Never calculate kW/kg by dividing a published peak power by a mass figure from an ambiguous or differently configured table. If the manufacturer's stated maximum power, specific power and installed mass do not reconcile, mark the comparison as unresolved and request the exact configuration/denominator.

### Efficiency

Report separate maps:
- inverter DC input to motor electrical output;
- motor electrical input to shaft output;
- propeller shaft input to measured useful thrust power;
- integrated battery-terminal-to-propulsive-useful-power.

Every map must identify RPM, torque/load, DC bus voltage/current, switching mode, power factor where relevant, propeller diameter/pitch, airspeed, density, advance ratio, Reynolds/Mach regime, thermal state, cooling parasitics and measurement uncertainty.

A single 97.5% peak or internal-test point is not mission-average powertrain efficiency. Integrate measured or validated map values over the actual mission profile.

## Aircraft performance

Range and payload claims must bind to:
- exact aircraft configuration and revision;
- payload mass, crew, baggage/cargo volume and centre of gravity;
- takeoff and landing mass;
- route length and reserve policy;
- takeoff/landing runway and obstacle assumptions;
- airspeed, altitude and climb schedule;
- wind, temperature, pressure, density altitude and icing assumptions;
- usable battery energy, SOC window, SOH and temperature;
- diversion/alternate and contingency energy;
- charging losses if reporting energy from the grid.

Report passenger-km and payload-km separately. Cargo volume and cargo mass are not interchangeable. A proof-of-concept flight with different payload/reserve conditions is not automatically a certified mission range.

## Thermal management

Report heat generation at each conversion stage and its duty-cycle profile. Cooling mass must include pumps, coolant, plumbing, radiators/heat exchangers, fans, mounts and protective structure. Cooling parasitic power must be included in mission energy.

Test hot-soak, cold-start, repeated climb/takeoff, rapid turnaround, degraded cooling and high-altitude conditions. A single lumped pack temperature is useful for screening but cannot establish cell hot spots or thermal propagation safety.

## Lifecycle and operations

Use:
- equivalent full cycles and energy throughput;
- cycle/calendar age at a defined capacity or resistance retirement criterion;
- battery replacement interval and cost;
- energy cost from grid input through charger to battery;
- charge time over a specified SOC window and temperature;
- dispatch reliability (with cause categories);
- maintenance labor and downtime per flight hour;
- cost per payload-kilometre and passenger-kilometre;
- charging-site, grid-connection and site-utilisation costs.

Do not compare calendar life or cycle life without matching depth of discharge, C-rate, temperature, storage SOC, ageing profile and end-of-life criterion.

## Fault tolerance and safety

Redundancy count alone is not a safety metric. State the failure domains, electrical/mechanical independence, shared cooling/control/sensing/power paths, common-cause assumptions, fault detection and isolation times, latent-fault exposure, continued safe-flight capability, and verified failure-response envelope.

A component fault-injection result is not equivalent to aircraft-level safety qualification. The safety case must cover hazards, installation, propagation/containment, structure, software/hardware assurance, human factors and continued airworthiness as applicable to the agreed certification basis.

## Comparison receipt

Each published Luminous/industry comparison should include:
- immutable input dataset identifiers and hashes;
- aircraft/component revision and serial/configuration identity;
- test procedure and instrument calibration identifiers;
- model/solver and source revision;
- operating-point and environmental manifest;
- all excluded and missing values;
- unit conversion and derivation steps;
- uncertainty and confidence/calibration method;
- failed/incomplete tests;
- independent reviewer/authority and date.

Unknown denominator or missing operating conditions means Incomplete, not comparable.
