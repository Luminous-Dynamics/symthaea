# symthaea-scavenger: Hazards, Sensorium, Modes, and Abort Logic

## Purpose

`symthaea-scavenger` models a disassembly and recovery platform operating in dirty,
mixed-material, human-adjacent environments. It should behave like a machine that
must separate salvage from contamination while avoiding jams, dust, unsafe cuts,
and human harm.

## Mission Variables

- `salvage_value_rate`: useful recovered value per unit time
- `scene_coverage`: fraction of scene processed
- `classification_confidence`: confidence in material separation
- `mission_progress`: cumulative teardown / recovery progress

## Failure Variables

- `hopper_fill`: material backlog in the intake chain
- `compactor_load`: downstream compression load
- `dust_level`: airborne particulate burden
- `thermal_load`: cutter / drive heat burden
- `tool_wear`: cutter degradation
- `jam_risk`: risk of line stoppage from geometry or overload
- `contamination_risk`: risk of mixing hazardous and salvageable streams
- `blade_bind_risk`: risk of cutter seizure or kickback
- `sorter_clog_risk`: risk of separator fouling

## External Risk Variables

- `hazard_fraction`: hazardous material concentration in the scene
- `human_proximity`: nearby worker / bystander proximity
- `incident_risk`: fused safety risk
- `toxic_dust_risk`: airborne hazard escalation
- `containment_breach_risk`: risk of unsafe escape from the waste stream
- `quarantine_load_ratio`: fraction of intake being held aside as unsafe

## Recovery Variables

- `battery_ratio`: available energy
- `chassis_stability`: motion and manipulation stability
- `payload_mass_norm`: current carried / buffered mass
- `salvage_purity`: quality of recovered stream

## Sensorium

### Required Sensor Bundle

- cutter current / RPM sensing
- hopper and compactor load sensing
- dust / particulate sensing
- thermal probes
- material classification signals
- human proximity sensing
- chassis stability / tilt sensing

### Strong Next Sensors

- hazardous chemical / radiation proxy
- line vision for contamination detection
- acoustic jam or blade-bind sensing
- sealed-container / battery identification sensing

## Operating Modes

- `Recovery`: normal salvage mode
- `DustControl`: prioritize suppression and slow cutting
- `JamRecovery`: reduce feed and clear the line
- `Quarantine`: isolate hazardous material from the salvage stream
- `HumanSafe`: suppress aggressive tools near people
- `EmergencyStop`: maximal shutdown / containment behavior

## Escalation Logic

Escalation should emerge from:

- high `jam_risk`
- high `contamination_risk`
- high `toxic_dust_risk`
- high `containment_breach_risk`
- high `human_proximity`
- low `classification_confidence`

## First Implementation Targets

1. Add state channels for contamination, toxic dust, quarantine, blade bind,
   sorter clog, containment breach, and salvage purity.
2. Infer operating mode from state instead of treating every situation as generic
   cutting + sorting.
3. Degrade recovery throughput under dust, jams, contamination, and human presence.
4. Add regression tests for:
   - hazardous feed entering quarantine mode
   - unsuppressed dust raising toxic dust risk
   - jam conditions forcing jam recovery
   - nearby humans pushing the platform into safe mode
