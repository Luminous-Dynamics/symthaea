# Same-state shadow comparison v0.1

**Status:** implemented proposal-comparison primitive; no closed-loop performance result  
**Code:** `symthaea_vehicle::shadow_evaluation`  
**Baseline:** existing `pd_cruise_baseline_with_road` teacher in `types.rs`

## Purpose

The current training loop blends a PD teacher command into the HDC-LTC proposal while training the HDC-LTC controller against PD targets. Evaluating the trained model inside that curriculum is not a fair frozen-controller comparison.

This module creates a separate *same-observation* comparison primitive. Given a frozen advanced command and one exact `VehicleState`, it recomputes the existing PD road-baseline command from that same state and records both proposals.

It does not advance either controller, mutate the state, step the simulator, select an actuator command or establish a safety result.

## Exact comparison contract

Every input frame binds:

- scenario candidate ID;
- contiguous frame index and strictly increasing simulation timestamp;
- complete finite 20-channel state and timestamp;
- road heading and non-negative target speed;
- advanced-controller 64-hex identity and advanced proposal;
- internal PD road-baseline policy identity and its deterministic profile digest.

The output binds:
- the same scenario ID, frame index, timestamp and state digest;
- both controller identities;
- the exact advanced and baseline commands;
- per-actuator absolute deltas and L1 delta;
- whether each command simultaneously requests meaningful throttle and brake.

The advanced controller digest is caller-supplied. The baseline digest binds the versioned policy ID and fixed gain bit patterns. Neither field authenticates executable bytes, source origin or build provenance. The state digest binds the serialized state representation in this Rust implementation; it is not a cross-language canonical format.

## Why reuse the existing PD baseline?

The current vehicle project already contains a road-relative PD controller used as the imitation/training teacher. Creating a second PD or Stanley implementation before measuring the existing one would make the first comparison harder to interpret.

The PD teacher is therefore the first reference baseline. It is not claimed to be industry-best, independently safety-qualified, or architecturally independent from the training setup. Future comparison tracks should add:
- a versioned Stanley-style road-tracking baseline and a versioned Pure Pursuit baseline are now available in `classical_baselines.rs`, each with fixed parameters and a deterministic policy digest;
- a documented constrained MPC controller with fixed horizon, model, solver budget and infeasibility semantics remains the next classical baseline.
- frozen HDC-LTC without RTA;
- the same frozen HDC-LTC with RTA applied in shadow mode.

## What the report measures

The report preserves all per-frame records and summarizes:
- mean absolute difference for steering, throttle and brake;
- maximum per-axis command difference;
- p50/p95/p99 L1 proposal disagreement;
- simultaneous throttle/brake overlap counts for each proposal.

These values describe controller-output divergence only. High disagreement does not establish that either controller is wrong. Low disagreement does not establish safety or task performance. There are no collision, route-completion, constraint-satisfaction, comfort or energy claims in this module.

## Missing closed-loop step

Same-state command comparison is not yet a paired trajectory benchmark. A proper closed-loop paired test needs two independent simulator instances initialized from the same exact state/configuration, identical exogenous disturbance schedules, frozen controller instances, and independently retained trajectories. Each branch must then evolve on its own state; the second controller must not be fed the first controller's trajectory.

That work must compose the vehicle-dynamics owner #6247 and Symtropy's execution/replay owner #3387 / #1172. In particular, the current vehicle bicycle simulator does not contain pedestrian actors or the sensor/occlusion semantics of the new intersection candidate contract. Do not report the intersection candidate generator as evaluated against pedestrians until Symtropy executes those actors and observables with explicit capability/unsupported outcomes.

## Acceptance and claim ceiling

Unit tests cover identity validation, input rejection, temporal ordering, same-state PD proposals, retained trace and disagreement distribution. They are authored only until the exact PR head is compiled/formatted/tested in a Rust-capable environment.

A passing test would establish only that the proposal-comparison primitive records the named same-state inputs and outputs. It does not establish controller superiority, closed-loop behavior, dynamics credibility, collision avoidability, safety, roadworthiness, legal compliance or SAE Level 4/5 capability.


## Paired closed-loop implementation update

The follow-on module `symthaea_vehicle::paired_rollout` now supplies the first paired closed-loop lane-keeping runner for the existing bicycle-model simulator. It clones the same initial simulator into independent advanced and classical plants, supports the existing PD teacher, a versioned Stanley-style baseline or a versioned Pure Pursuit baseline, applies the same external perturbation schedule, evolves each plant on its own state, and retains both complete frame traces. Its profile binds actual resolved road geometry, simulator/dynamics profile, initial state, controller identities, perturbation schedule, target speed, timestep and step budget; its output digest binds the trace and summaries.

The runner checks command ranges, finite states before/after each step, simulator profile validity and expected simulation-clock increments. It remains a lane-keeping test harness only. The bicycle model has no pedestrian, cyclist, general traffic actors or intersection collision observable, so this does not execute or score the intersection candidate generator.

See [PAIRED_CLOSED_LOOP_ROLLOUT_V0_1.md](./PAIRED_CLOSED_LOOP_ROLLOUT_V0_1.md). The earlier text about paired rollouts being a missing next step is superseded for this limited bicycle-model lane-keeping scope.