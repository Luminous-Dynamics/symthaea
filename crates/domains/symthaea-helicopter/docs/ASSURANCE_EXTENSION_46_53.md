# Assurance Extension 46–53

This series adds explicit campaign fault models and deployment assurance
contracts. The additions remain research and verification infrastructure; they
do not establish airworthiness.

## Added truth boundaries

- Sensor truth can be transformed through scheduled bias, scale, stuck, dropout,
  white-noise, and random-walk models with deterministic replay.
- Actuator requests can diverge from realized positions through gain loss,
  jams, deadband, runaway, and intermittent operation after command arbitration.
- Engine power reaches the rotor through spool, governor, torque, transmission,
  and freewheel transients rather than an instantaneous rigid shaft.
- Commands from learned, guidance, operator, envelope, emergency, and watchdog
  sources are lease-bound and arbitrated per channel with fixed precedence.
- Abstract contingency cases must retain a capability-feasible path to a safe
  terminal within their declared deadline.
- Rollback targets must be qualified, compatible ancestors of the running
  deployment and remain within a bounded rollback depth.
- Icing, precipitation, dust, salt aerosol, and visibility degradation are
  accumulated as stateful environmental exposure rather than one-step flags.
- Qualification artifacts are bound into one campaign/deployment bundle with
  completeness, duplicate, digest-shape, and authenticity-reference checks.

## Remaining non-claims

The stochastic sensor models are deterministic test stimuli, not certified
sensor error distributions. Actuator and drivetrain models are reduced-order.
Safe-state reachability is finite-state analysis and does not prove continuous
vehicle dynamics. FNV bundle digests provide deterministic identity only;
cryptographic authenticity still requires an external signature provider.
