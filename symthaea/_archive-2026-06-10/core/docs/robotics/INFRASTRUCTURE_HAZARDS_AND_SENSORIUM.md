# symthaea-infrastructure: Hazards, Sensorium, Modes, and Abort Logic

## Purpose

`symthaea-infrastructure` models a stationary or semi-stationary civic node:
microgrid controller, storage hub, thermal plant, relay hub, or utility router.
It should behave like a system that must preserve voltage, queue flow, and public
service continuity under overload and partial failure.

## Mission Variables

- `storage_ratio`: energy reserve
- `mission_progress`: sustained civic uptime / useful service completion
- `forecast_confidence`: confidence in near-term demand and load forecasts
- `community_demand`: current demand pressure from dependents

## Failure Variables

- `thermal_load`: accumulated heat stress
- `coolant_temp_c`: coolant loop state
- `queue_depth`: unresolved routing / service backlog
- `grid_stress`: aggregate stress from demand, storage scarcity, and congestion
- `brownout_risk`: likelihood of degraded power quality
- `thermal_runaway_risk`: escalation risk if cooling cannot recover
- `switchgear_wear`: degradation from repeated routing and shedding
- `deadlock_risk`: routing contention / queue lock condition
- `islanding_risk`: risk of isolation from the wider network

## External Risk Variables

- `service_integrity`: quality of downstream civic service
- `critical_load_fraction`: fraction of dependent load that is safety-critical
- `unserved_demand_ratio`: demand currently unmet
- `shed_load_ratio`: fraction of load deliberately dropped to preserve the node
- `incident_risk`: fused operational danger estimate

## Recovery Variables

- `relay_health`: health of switching / relay layer
- `voltage_stability`: electrical stability margin
- `recovery_margin`: how much headroom remains for recovery actions
- `islanding_capability`: ability to survive disconnected from the wider grid
- `maintenance_backlog`: unresolved maintenance debt

## Sensorium

### Required Sensor Bundle

- voltage and current sensing
- storage state-of-charge
- coolant loop temperature and flow
- switching / relay health telemetry
- queue / request backlog depth
- branch flow sensing for north/south/east/west routing
- thermal cabinet / enclosure probes

### Strong Next Sensors

- transformer vibration or failure proxy
- weather / external ambient coupling
- generator or feeder quality telemetry
- critical-load identification and priority sensing
- anomaly / cyber-fault detection feed

## Operating Modes

- `Balanced`: normal dispatch with healthy reserve and low incident pressure
- `LoadShedding`: deliberate service reduction to preserve stability
- `CoolingRecovery`: thermal-first behavior to prevent runaway
- `Islanding`: preserve local service while network links are degraded
- `DeadlockRecovery`: reduce routing contention and clear queue lock
- `Emergency`: severe incident mode prioritizing critical loads only

## Abort / Escalation Logic

Infrastructure does not “abort” like a rover. It escalates operational posture.
Escalation should emerge from:

- high `brownout_risk`
- high `thermal_runaway_risk`
- high `deadlock_risk`
- low `relay_health`
- low `voltage_stability`
- high `unserved_demand_ratio`
- low `recovery_margin`

## First Implementation Targets

1. Add hazard-bearing channels for brownout, deadlock, islanding, shedding,
   unserved demand, recovery margin, and thermal runaway.
2. Infer operating mode from state rather than explicit mode commands.
3. Bias routing and discharge effectiveness downward under overload.
4. Add regression tests for:
   - overload leading to load shedding
   - cooling failure raising thermal runaway risk
   - routing contention increasing deadlock risk
   - islanding degrading service integrity but preserving uptime
