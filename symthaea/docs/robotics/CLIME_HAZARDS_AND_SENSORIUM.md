# Symthaea Clime: Hazards, Sensorium, Modes, and Homeostasis Logic

`symthaea-clime` is the atmospheric and circadian homeostasis platform for shared indoor space. It should not behave like a thermostat with extra fields. Its job is to maintain breathable, comfortable, biologically coherent habitat while respecting safety, energy limits, occupancy patterns, and public-health constraints.

## Operating environment

- indoor civic and residential spaces: nexus hubs, corridors, homes, clinics, workspaces
- dynamic occupancy with shifting thermal and air-quality loads
- interaction with utility routing (`symthaea-plexus`), adaptive habitat (`symthaea-hearth`), biosensing (`symthaea-vita`), and broader civic safety systems
- competing objectives: comfort, health, quiet operation, and energy efficiency

## Primary hazards

### Atmospheric hazards
- CO2 buildup and oxygen imbalance
- particulate or smoke intrusion
- volatile chemical contamination
- humidity extremes causing discomfort, mold, or respiratory stress

### Thermal hazards
- heat stress in crowded or sun-loaded spaces
- cold stress in low-occupancy or underpowered zones
- thermal stratification creating uneven habitat quality
- equipment overheating or condensation risk

### Circadian / neurological hazards
- inappropriate color temperature or brightness for time of day
- sleep disruption from lighting or HVAC noise
- glare or flicker that increases agitation or fatigue

### Operational hazards
- occupancy misestimation
- over-optimization for efficiency at the expense of comfort
- stale sensor data
- degraded utility supply from upstream infrastructure

## Required sensorium

- temperature across occupied zones
- humidity
- CO2 and basic air-quality / particulate estimate
- occupancy density / motion estimate
- ambient light level and fixture output feedback
- noise proxy from environmental systems
- utility availability / power budget telemetry

## Optional sensorium

- volatile organic compound sensing
- pathogen / filtration proxy
- window / aperture state integration
- circadian preference schedule by zone
- local thermal-comfort feedback from inhabitants

## Mission variables

- breathable-air integrity
- thermal comfort across occupied zones
- circadian coherence
- noise-aware environmental stability
- energy-aware habitat continuity

## Failure variables

- air-quality degradation
- thermal discomfort
- humidity instability
- lighting mismatch
- occupancy-model error
- low utility reserve

## External risk variables

- public-health hazard from smoke, contamination, or poor ventilation
- compounding thermal stress on vulnerable occupants
- sleep disruption across residential zones
- unsafe coupling with fire, medical, or utility incidents

## Recovery variables

- ventilation authority
- cooling / heating recovery margin
- filtration effectiveness
- zone-isolation confidence
- occupant-safe fallback profile

## Operating modes

- `BalancedHabitat`: normal homeostasis across air, temperature, and light
- `AirRecovery`: prioritize ventilation and filtration under air-quality degradation
- `ThermalRecovery`: prioritize cooling/heating rebalance under heat or cold stress
- `CircadianSupport`: bias light and noise profiles toward biological rhythm support
- `QuietNight`: low-noise, low-glare nighttime protection mode
- `UtilityConstrained`: preserve minimum safe habitat under energy or water limits
- `IsolationMode`: zone-level containment for smoke, contamination, or public-health events

## Homeostasis logic

`symthaea-clime` should optimize for **safe, lived biological comfort**, not maximal efficiency.

- if air quality degrades, ventilation and filtration outrank comfort tuning
- if thermal stress rises, protect vulnerable occupied zones first
- if nighttime conditions apply, reduce glare, color-temperature disruption, and mechanical noise
- if utility supply is constrained, preserve minimum breathable / thermally safe conditions before non-essential comfort enhancements

## Abort / degraded-mode logic

- if sensor confidence collapses, fall back to conservative safe ventilation and neutral lighting
- if a zone becomes unsafe, isolate the zone rather than spreading contamination or thermal failure
- if utility limits bite, degrade gracefully into `UtilityConstrained` rather than oscillating between aggressive setpoints
- if an intervention increases occupant distress proxies, reduce actuation aggressiveness and stabilize

## First implementation targets

1. Define state channels for air quality, CO2 load, humidity stress, thermal stress, circadian mismatch, occupancy confidence, utility reserve, and zone-isolation confidence.
2. Infer operating mode from environmental and civic pressure rather than explicit manual mode selection.
3. Add scenario tests for:
   - crowded room CO2 spike triggering `AirRecovery`
   - hot afternoon load triggering `ThermalRecovery`
   - nighttime residential zone shifting into `QuietNight`
   - smoke / contamination event entering `IsolationMode`
   - low utility reserve falling back to `UtilityConstrained`
