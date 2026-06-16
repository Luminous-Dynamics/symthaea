# symthaea-subterranean: Hazards, Sensorium, Modes, and Abort Logic

## Purpose

`symthaea-subterranean` models a hazard-dense underground scout / boring platform.
It should not behave like a generic rover with a depth counter. The platform exists
to reason under occlusion, heat, water ingress, gas risk, localization drift, and
communications degradation.

## Mission Variables

- `depth_m`: excavation / probe progress into the subsurface
- `mission_progress`: cumulative mission completion progress
- `vein_signal`: geological feature or resource detection proxy
- `mapping_confidence`: confidence in the local model of the tunnel / seam

## Failure Variables

- `cutter_temp_c`: active boring thermal load
- `motor_temp_c`: drive and auger thermal load
- `spoil_buffer_fill`: local excavation spoil saturation
- `slip_ratio`: traction loss and motion uncertainty
- `tool_wear`: boring head degradation
- `hull_stress`: mechanical stress from wedging, loading, and uneven strata
- `water_ingress_ratio`: detected fluid intrusion into the tunnel or hull boundary
- `slurry_load`: combined water + spoil drag that impairs escape and digging
- `seal_integrity`: the platform's ability to isolate itself from fluid ingress
- `roof_stability`: risk proxy for roof / ceiling collapse above the agent

## External Risk Variables

- `aquifer_risk`: risk of breaching a water-bearing layer
- `gas_risk`: risk from methane or other trapped gases
- `obstacle_proximity`: trapped geometry / jam / wedging hazard
- `abort_recommendation`: fused danger estimate indicating the machine should stop
  digging and stabilize or retreat

## Recovery Variables

- `comm_signal`: raw communications viability at depth
- `relay_link_quality`: effective connection quality after moisture / geometry losses
- `return_path_confidence`: current confidence in following the reverse route out
- `escape_confidence`: current confidence the platform can retreat safely
- `localization_confidence`: confidence in dead reckoning + sensor fusion state

## Sensorium

### Required Sensor Bundle

- IMU / odometry
- cutter current / motor current sensing
- hull strain / stress sensing
- cutter and motor thermal probes
- slip estimation
- proximity / contact sensing
- moisture ingress detection
- gas detection
- depth / pressure estimate

### Strong Next Sensors

- subsurface imaging proxy / simplified GPR
- acoustic / seismic anomaly sensing
- conductivity / salinity sensing
- relay beacon ranging
- geological signature / vein estimator

## Operating Modes

- `Dig`: normal boring while hazards remain bounded
- `Probe`: low-aggression surveying / sensing mode
- `Stabilize`: reduce heat, settle spoil, preserve roof and hull
- `Retreat`: active reverse motion to restore escape margin
- `Surface`: mission abort and return to surface
- `BlackoutAutonomy`: local autonomy under weak relay / comm signal
- `FloodResponse`: aquifer / slurry response emphasizing seal preservation and retreat

## Abort Logic

Abort should not be a binary panic bit. It should emerge from weighted hazard fusion:

- high `water_ingress_ratio`
- high `gas_risk`
- low `roof_stability`
- low `seal_integrity`
- low `escape_confidence`
- extreme `cutter_temp_c`
- extreme `hull_stress`

When `abort_recommendation` crosses a high threshold, the simulator should bias
away from digging and toward stabilization / retreat.

## First Implementation Targets

1. Add hazard-bearing state channels for ingress, aquifer, gas, roof, localization,
   relay, seal, slurry, escape, and abort.
2. Infer operating mode from state rather than forcing every mode through explicit
   commands.
3. Gate effective digging aggressiveness by hazard state.
4. Add regression tests for:
   - aquifer breach / flood response
   - gas spike suppressing digging
   - roof instability raising abort recommendation
   - blackout autonomy under comm degradation
   - spoil jam reducing escape confidence
