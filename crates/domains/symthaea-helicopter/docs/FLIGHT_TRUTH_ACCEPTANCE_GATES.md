# Helicopter Flight-Truth Acceptance Gates

This crate is an experimental simulation and control research scaffold. It is
not an airworthy flight-control system. Promotion claims must remain bounded by
the gates below.

## Physics truth

- Default hover trim holds altitude within 0.75 m for five simulated seconds in calm air.
- Main-rotor trim thrust agrees with declared vehicle weight within 5 N after RPM convergence.
- Neutral tail-rotor anti-torque balances main-rotor reaction within 10 N·m at trim.
- Pedal authority is produced through tail-rotor pitch/RPM, not a direct yaw-moment shortcut.
- Requested controls pass through bounded servo/governor lag and slew limits.
- Declared engine, rotor, tail-rotor, payload, and crosswind perturbations alter simulator dynamics.
- Payload changes update total mass, center of gravity, trim moment, and diagonal inertia through a declared station model.
- First ground contact is classified and latched as safe touchdown, hard landing, or crash.
- Autorotation exposes rotor kinetic energy and collective-dependent energy extraction.
- The reduced-order rotor model emits induced velocity, effective translational lift, and vortex-ring exposure as explicit regimes.
- Local pressure, temperature, density, speed of sound, and density altitude come from a bounded deterministic atmosphere; rotor thrust and anti-torque scale with local density.
- Rotor power consumes finite fuel, obeys power/thermal limits, and protects a return reserve.
- Quaternion and every feedback channel remain finite and uprightness stays in [-1, 1].

## Control truth

- Every mission command is conditioned on an explicit local-frame reference.
- HDC-LTC output is a bounded adaptive residual over a deterministic guidance backbone.
- Fault-aware allocation reports saturated actuators, degraded axes, and unrealized virtual-control residuals.
- Debounced fault diagnosis converts rotor, servo, power, and yaw evidence into conservative actuator health.
- Derivative HDC channels are rates computed with an explicit sample interval.
- Invalid cadence, network, learning, or noise configuration fails before simulation.
- Yellow/Orange restrictions preserve lift while bounding maneuver authority.
- A final rotor-regime guard emits explicit interventions for overspeed, low RPM, depleted autorotation energy, vortex-ring exposure, and degraded tail authority.
- Emergency fallback transitions are time-based rather than scheduler-cycle-based.
- One deterministic mission supervisor resolves authority, navigation, fuel, terrain, weather, operator, and critical-fault contingencies by documented precedence.
- A replayable runtime monitor checks state/command invariants and bounded-response obligations for critical faults and post-landing disarm.

## Benchmark truth

- Phi is not multiplied into actuator commands in scientific benchmarks.
- Crosswind recovery includes horizontal displacement, altitude error, and angular motion.
- Missing preregistered windows produce incomplete/NaN evidence, never a synthetic zero.
- Crashes are explicit outcomes and cannot be reported as successful recovery.
- Correlation claims require multiple seeds, matched controls, effect sizes, and confidence intervals.
- Fixed-authority negative controls emit a versioned manifest, samples, correlation, and seed-cluster bootstrap interval.
- Every evidence run records requested/applied commands, state, faults, fuel, rotor energy, events, and a deterministic replay digest.

## Navigation truth

- Navigation is unavailable before a validated timestamped measurement.
- Stale or excessively uncertain navigation fails closed at the time of use.
- Non-finite values, negative covariance, and backward timestamps fail closed.
- Multi-source navigation rejects implausible innovations and grows uncertainty during dead reckoning.
- Repeated source-specific failures quarantine only the offending source, and authority may require fresh independent absolute sources.
- Navigation health also fails closed on covariance collapse/inflation, sustained high normalized innovations, or an excessive rolling rejection fraction.
- Mission routes and projected trajectories fail closed on unknown terrain, clearance loss, or geofence exit.
- Emergency landing-zone selection rejects unknown terrain, unsafe slope or roughness, blocked approaches, geofence margins, and excessive crosswind.

## Remaining non-claims

The reduced-order simulator does not establish airworthiness or certification.
It still lacks validated blade-element or free-wake aerodynamics, retreating-
blade stall, loss-of-tail-rotor-effectiveness aerodynamics, structural/fatigue
loads, redundant certified sensor fusion and flight computers, and physical
actuator/hardware timing qualification. Its induced-flow and vortex-ring
regimes are bounded research approximations, not validated flight envelopes.

## Hardware boundary

- Simulation-only backends cannot be constructed as physical hardware bridges.
- Physical output requires fresh sequenced sensors, mission-bound authority, and a command watchdog.
- Any clock, sequence, freshness, authority, or I/O fault disarms outputs and clears authority.

## Release qualification truth

- Every required scenario declares minimum distinct seeds, required exercised faults, and explicit metric thresholds.
- Missing seeds, metrics, completion, fault coverage, log digests, or verified replay chains are `Incomplete`, never pass.
- Any observed gate violation remains `Fail` even when other evidence is incomplete.

## Rotor-hub state

Cyclic input is no longer converted directly into an instantaneous body
moment. The reduced-order rotor hub carries longitudinal/lateral disk flapping,
thrust-dependent coning, advance ratio, and RPM-dependent cyclic authority.
Qualification evidence must retain these states when evaluating low-RPM control
or forward-flight asymmetry. This remains a bounded hub model, not blade-resolved
CFD or a certification-quality comprehensive rotor code.

## Parameter provenance and uncertainty

A numerical match to an expected trajectory is not sufficient when the model
constants are untraceable. `FlightModelCalibration` records units, standard
uncertainty, validity bounds, source class, and an evidence identifier for each
parameter. Missing or invalid required rotor parameters reduce readiness to
research-only; explicit assumptions remain visible as
`TraceableWithAssumptions`. Only a fully traceable set may support claims about
a named physical airframe, and even that does not establish airworthiness.

## Claim-scope gate

Every published result should name a `ClaimLedger` claim and requested assurance
level. Evidence below the declared level is incomplete; a requested level above
the claim ceiling is refused. In particular, deterministic simulation, HIL, and
traceable calibration artifacts cannot be relabeled as airworthiness or
certification evidence.


## Fault-model and deployment-extension truth

- Sensor qualification campaigns apply scheduled, deterministic fault models rather than feeding perfect truth directly into the estimator.
- Plant-side actuator faults create measured command/realization divergence and conservative actuator-health evidence.
- Engine spool, governor, transmission efficiency, torque limits, and freewheel disengagement are explicit transient states.
- Competing command producers are lease-bound and arbitrated per channel; ordinary learned or guidance sources cannot disarm both rotors.
- Every declared contingency case has a capability-feasible abstract path to a safe terminal within its deadline, or qualification remains incomplete.
- Rollback is restricted to qualified, compatible deployment ancestors within a bounded depth.
- Icing, precipitation, dust, salt aerosol, visibility, and anti-ice effects accumulate as environmental exposure evidence.
- Release evidence is assessed as one campaign/deployment bundle; missing required kinds are incomplete and malformed or unauthenticated physical evidence is rejected.
