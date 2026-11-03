# Paired closed-loop lane-keeping rollout v0.1

**Status:** simulator-only implementation; no controller performance results reported  
**Implementation:** `symthaea_vehicle::paired_rollout`  
**Baseline options:** existing PD road teacher, versioned Stanley-style road tracker, or versioned Pure Pursuit road tracker  
**Advanced subject:** frozen HDC-LTC weights with evolving recurrent state

## Experiment design

The runner clones one `BicycleModelSimulator` instance into two independent plants. Both begin from the same vehicle state, chassis parameters, friction state and pending external force. It applies the same `PerturbationSchedule` to both branches at every step. Each controller reads its own plant's current road-relative observation, issues a proposal, and advances only its own plant.

This is a paired, closed-loop comparison—not same-state open-loop replay after the first action. The trajectories can diverge, as they should when actions differ.

The HDC-LTC controller is reset at the start and its recurrent state evolves as it receives observations. No optimizer/training updates are performed inside the rollout. The existing PD baseline is recomputed from the baseline plant's current state.

## Frozen experiment identity

The profile digest binds:
- caller-supplied candidate and road-profile labels;
- selected classical policy ID and parameter digest (PD, Stanley or Pure Pursuit);
- selected classical baseline policy ID and its policy-parameter digest;
- digest of actual resolved road segment geometry, projection look-ahead and closed-loop flag;
- caller-supplied advanced controller digest;
- exact PD policy ID and gain-profile digest;
- simulator chassis/friction/external-force profile digest;
- initial state digest;
- perturbation schedule digest;
- target speed, timestep and step budget.

The output digest binds the full retained result, including frame-level pre/post plant states, both observation states, commands and trajectory summaries.

These BLAKE3 values are integrity fingerprints, not signatures, authentication, binary attestation or proof of executable provenance. The advanced controller digest and human-readable road label remain caller-supplied. They must be replaced by build-artifact-derived identities in a later evidence pipeline.

## Recorded metrics

Each branch reports:
- mean absolute and RMS lateral offset;
- maximum absolute lateral offset;
- mean absolute speed error;
- mean command effort proxy;
- maximum observed yaw-rate magnitude;
- projected road progress;
- final speed, lateral offset and heading error.

The result also includes signed advanced-minus-baseline differences. Lower tracking error and effort are not automatically preferable if progress or safety constraints worsen. These metrics do not include tire-temperature, tire-force envelope, collision, actor interaction, rule compliance, or energy-model validity.

## Failure and validity semantics

The runner rejects malformed profile identities, invalid or non-finite chassis parameters, invalid road geometry, invalid initial state, malformed perturbation schedules, invalid commands, non-finite or out-of-envelope post-step states and simulator timestamp increments inconsistent with the requested timestep. Clearing a tire-blowout schedule restores both global and per-axle grip multipliers. It caps the step budget at 100,000.

The road profile digest is derived from the resolved segment geometry rather than merely trusting the supplied road label. The dynamics profile digest includes chassis parameters, friction multipliers, initial speed and pending external force.

## Capability ceiling

The current `BicycleModelSimulator` has no pedestrian, cyclist or general dynamic-traffic actors, and no intersection collision observable. The result deliberately records these unsupported capabilities. It therefore cannot evaluate the seven-factor pedestrian-crossing scenarios from `SYNTHETIC_INTERSECTION_FIXTURE_V0_1.md`. Those remain assigned to Symtropy's actor-capable execution/replay work (#3387).

The model itself is a simplified bicycle model with a simplified tire-force law. A result is evidence about this named simulation implementation only—not about a production vehicle, real tire physics, public roads, Level 4 capability or safety certification.

## Verification status

Ten Rust test declarations cover cloned initial conditions, identical disturbance application to both branches, deterministic replay receipts, PD/Stanley/Pure Pursuit policy identity binding, road/profile identity tampering, output-trace tampering and invalid-profile rejection. The classical-baseline module adds nine unit tests for Stanley/Pure Pursuit steering direction, centered behavior, curvature/feed-forward response, parameter identity and invalid configuration. One additional perturbation-scheduler regression test checks that clearing tire blowout restores per-axle grip. One additional perturbation-scheduler regression test checks that clearing tire blowout restores per-axle grip. They have not been compiled or executed in this session. The environment has no Rust toolchain; current CI workflows on this branch were reported as `skipped`, not passed.

## Next progression

1. Run exact-head focused unit tests, formatting and Clippy on a Rust-capable host.
2. Add constrained MPC with frozen horizon, constraints, solver and infeasibility semantics; retain PD, Stanley and Pure Pursuit as distinct classical references.
3. Add RTA in shadow mode and compare intervention/fallback proposals without giving the monitor real actuator authority.
4. Bind controller identities to measured build artifacts and record hardware/runtime budgets.
5. Move pedestrian/intersection execution to Symtropy; add explicit simulator capability and semantic-loss records.
6. Build independent dynamics cross-checks (another implementation, parameter sweeps and model-discrepancy envelopes) before interpreting simulator outcomes as physical safety evidence.
