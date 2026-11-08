# Road autonomy runtime assurance v0.1

**Status:** simulation-only research contract  
**Owner:** `symthaea-vehicle`  
**Parent programmes:** [ROAD-AUTONOMY-RTA-001](https://github.com/Luminous-Dynamics/symthaea/issues/10012), [TRANSPORT-ROAD-BENCH-001](https://github.com/Luminous-Dynamics/symthaea/issues/6256), [SE-RTA](https://github.com/Luminous-Dynamics/symthaea/issues/3779), [TRANSPORT-ROAD-001](https://github.com/Luminous-Dynamics/symthaea/issues/6247)

## Purpose

This module establishes a software boundary between an adaptive driving controller and any downstream command consumer. It is a first research step toward a Simplex-style road autonomy architecture. It does not provide a vehicle actuator interface, certify a fallback controller, validate the vehicle dynamics, or establish road-driving safety.

The governing arrangement is:

```text
perception + state estimation
        |
        v
Symthaea advanced controller -----> proposed command
        |                                  |
        |                         independent RTA monitor
        |                                  |
qualified reversionary controller ------> selected command or no command
```

The monitor receives both commands and evidence through an explicit observation contract. It does not call the learned controller, infer safety from Phi/free-energy values, or use the learned model to certify itself.

## Implemented semantics

- Advanced commands are checked for finite values and strict actuator ranges. Clamping is not considered validation.
- Every field in the current 20-channel vehicle state plus timestamp is checked for finiteness; selected state bounds are evaluated against the named profile.
- Observation timestamps must be finite, non-negative and strictly increasing. Excessive gaps and stale/future state timestamps trigger a finding.
- Estimator, real-time, actuator, model-applicability, envelope-margin and controllability-margin gates are explicit.
- Disagreement between advanced and baseline commands is a monitor finding.
- A baseline command is selectable only when it is well-formed, its profile is explicitly declared admissible, its own identity is valid, a non-empty baseline-independence evidence reference is supplied, and separate caller-supplied health declarations cover baseline inputs, baseline controller, baseline execution path, monitor timing and the shared actuator path. If the advanced identity is malformed, advanced authority is denied and this valid baseline may still be selected. A malformed baseline identity, aliased identities or missing independence reference yields `NoAdmissibleFallback`.
- If the fallback is unavailable or malformed, the monitor returns `NoAdmissibleFallback` and `selected_command = None`.
- Any monitor finding immediately selects the admissible baseline; a clean dwell and a bounded linear transition are required before advanced authority is restored.
- Each decision retains a compact witness, exact profile ID, a BLAKE3 digest of the `serde_json` serialization of the monitor configuration, mode and findings. The witness also records 64-hex advanced/baseline controller identifiers, a non-empty baseline-independence evidence reference, and a BLAKE3 observation fingerprint when the full observation is serializable. These are identity/replay fields, not signatures or canonical cross-language encodings.

The baseline-admissibility and health booleans remain caller assertions. A malformed advanced identity denies advanced authority but can still allow an independently identified, admissible baseline to take over. A malformed baseline identity, valid-but-aliased controller identities or absent baseline-independence evidence reference prevents fallback selection. A non-empty reference is not verified, distinct hashes do not prove architectural independence, and this module still does not authenticate the controller binaries, source provenance, sensors or evidence. A no-command result is not an emergency-stop maneuver; any future physical system needs a separately engineered and qualified local fail-safe path.

## Assumptions and current limitations

1. The input state and command values are supplied by a caller; sensor acquisition, sensor authentication, estimator implementation and actuator feedback are outside this crate.
2. The envelope and margin values are caller-supplied. The module does not calculate reachability, tire forces, stopping distance, collision probability or controllability.
3. Default thresholds in `VehicleRtaConfig` are provisional software-test values. They are not calibrated automotive limits, a valid ODD or evidence of operational capability.
4. The transition is a command-space interpolation, not a verified bumpless transfer for a real vehicle.
5. Time is represented as floating-point seconds in the simulation. A deployed real-time implementation must define clock source, resolution, rollover, synchronization, deadline behavior and restart semantics.
6. The monitor's own process failure, memory corruption, shared-sensor common cause, power loss and command-path faults are not covered by this module.
7. The profile digest identifies the serialized monitor configuration only. It does not bind the executable, vehicle configuration, sensor calibration, learned model, baseline implementation, source evidence or runtime measurement. Serialization does not authenticate the producer or the evidence. Content identity, source authenticity and trusted measurement are separate obligations.
8. Advanced and baseline controller digests are format-checked and required to differ; they are not recomputed from executable bytes. The baseline-independence evidence ID is required to be present but is not dereferenced or authenticated. The integration must independently bind each digest to measured artifacts and review shared sensors, clocks, memory, power, compiler/runtime, scheduler, actuator and failure-containment paths. A hash difference alone is not an independence proof.
9. The observation fingerprint hashes the complete serde_json-serializable observation. It is absent when serialization rejects non-finite values. It supports same-version replay correlation; it is not an authenticated event receipt and should not be described as cross-language canonical identity.

## Benchmark protocol

Use the portfolio in #6256 rather than inventing a single autonomy score. Compare at least:

- PD / deterministic conventional baseline;
- model-predictive controller with a frozen model and solver profile;
- frozen HDC-LTC controller without RTA;
- HDC-LTC plus this RTA adapter;
- baseline-only / RTA-only control;
- where useful, a benchmark-native reference controller.

Use identical initial states, scenario seeds, road/traffic inputs, sensing availability and compute budgets. Freeze train/dev/test scenario families before evaluation. Hold out complete scenario families, geography and environment combinations rather than randomly splitting nearly identical frames.

Report independently:
- collisions and collision severity as defined by the simulator;
- vulnerable-road-user contact and minimum separation;
- off-road / drivable-area violations;
- traffic-rule infractions and failed maneuvers;
- minimum TTC and stopping-distance margin where the simulator exposes defensible inputs;
- route completion and progress;
- intervention, fallback and no-admissible-fallback rates;
- time from injected fault to monitor finding and selected fallback;
- duration spent in degraded mode and false interventions on positive controls;
- lateral/longitudinal envelope violations;
- comfort and energy use;
- monitor/control-loop p50, p95 and p99 latency and deadline misses;
- ODD slice, failure family, exact code/config/model identity, simulator build and scenario-corpus digest.

Do not aggregate safety-critical failures into an average reward. Preserve denominators and exposure. Publish confidence intervals and the full failure inventory, not only best-seed results.

## Initial target gates (proposed, not achieved)

- 100% rejection of the deterministic frozen malformed, stale, inadmissible-fallback and prohibited-command corpus.
- Zero positive-control failures: a valid baseline/advanced pair must remain operable, preventing “reject everything” from passing.
- Immediate selection of an admissible baseline in the same monitor evaluation when a finding is present.
- No re-entry before the configured clean dwell completes; any fault during transition returns to baseline.
- Every decision binds the profile ID and monitor-configuration digest and contains the observed inputs / reasons required for replay.
- No unexplained non-finite output and no stale approval reuse after profile/configuration change.
- Performance is reported as a Pareto frontier across safety, completion, comfort, energy, compute and intervention rate. No single weighted pass score.

These are software-contract goals for a deterministic test corpus. They are not acceptable substitutes for formal analysis, hardware-in-the-loop, physical validation, or an independent safety case.

## Comparison and external reference tracks

Follow #6256 for exact source admission and versioning. Candidate external tracks include Waymo Open Dataset perception/motion tasks, nuScenes, Argoverse 2, CommonRoad, nuPlan, CARLA / CARLA Leaderboard, Waymax, and ASAM OpenSCENARIO. Pin each selected release, data split, metric and access/license state before use.

Open-loop prediction, replay-based planning and reactive closed-loop simulation are different evaluation problems. Keep them separate. CARLA/Waymax/nuPlan simulation results are not road exposure. Operational comparisons such as published fleet safety studies are only comparable when ODD, geography, exposure, outcome definitions, reporting and statistical methods align.

## Next steps

1. Run exact-head formatting, focused crate tests and Clippy on a Rust-capable host; retain all diagnostics. Queued jobs are not passes.
2. Add canonical scenario/decision serialization and a replay digest using the repository's existing evidence/provenance owner rather than creating a second ledger.
3. Add monitor fault injection and mutation tests, including common-cause estimator/monitor corruption and missing baseline. Bind controller digests to independently measured artifacts and resolve the baseline-independence evidence reference before making any independence claim.
4. Connect the selected command to the existing simulator harness only; add controller ablations and per-scenario measures.
5. Extend #6256 with measured RTA overhead, intervention quality and held-out scenario transfer.
6. Keep hardware work behind #6254's low-consequence physical qualification ladder and separate regulatory review.

## Claim ceiling

A passing software test may establish that this exact source revision behaves according to this finite contract on the named test corpus. It does not establish a validated vehicle controller, safe emergency maneuver, collision-risk reduction, roadworthiness, functional-safety certification, SAE Level 4 or Level 5 capability.
