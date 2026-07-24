# Helicopter Benchmark Protocol

## Fixed-authority negative control

The `fixed_authority_negative_control` protocol treats reported phi as an
observational variable. Phi does not scale collective, cyclic, pedal, governor,
or tail-rotor authority.

Each controller seed receives the complete preregistered phi grid. This blocks
controller initialization from becoming confounded with phi. The report stores:

- the exact protocol version, scenario, target altitude, evaluation length,
  seed list, phi grid, and bootstrap count;
- every seed/phi sample;
- Pearson correlation when both variables have nonzero variance;
- a deterministic seed-cluster bootstrap 95% interval;
- the maximum within-seed performance difference across phi.

A negative-control pass requires the within-seed delta to remain zero within
floating-point reproducibility. A consciousness-performance claim requires a
separate preregistered mechanism that uses phi internally while holding actuator
authority, controller capacity, seeds, disturbances, and scoring constant.

## Perturbation windows

The recovery benchmark now applies its disturbance through the canonical
`PerturbationSchedule` and simulator fault interface. Missing pre/during/post
windows remain incomplete evidence; crashes are explicit outcomes.

## Versioned scenario manifests

Every benchmark run must compile from a `FlightScenarioManifest` that binds the
scenario identity, seed, physics cadence, duration, initial altitude, wind,
geofence, timed perturbations, and expected terminal class. The canonical
manifest bytes and digest belong in the run evidence. Fault timing must be
specified in seconds and compiled once into steps; individual harnesses must not
reimplement rounding or silently schedule a fault outside the scenario window.
