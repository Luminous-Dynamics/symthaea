# MANIP-HLB-BIO M1: deterministic contract fixtures v0.1

**Status:** research/contract scaffold. These fixtures are not yet consumed by a Rust test harness and do not establish a code, simulator, HIL, or physical qualification pass.

**Owner:** [Symthaea #12735](https://github.com/Luminous-Dynamics/symthaea/issues/12735)

**Architecture dependencies:** [#1274 UEC v2](https://github.com/Luminous-Dynamics/symthaea/issues/1274), [#1276 cognitive readiness vs plant safety](https://github.com/Luminous-Dynamics/symthaea/issues/1276), [#1281 metric semantics](https://github.com/Luminous-Dynamics/symthaea/issues/1281), [#1284 cognitive vs plant time](https://github.com/Luminous-Dynamics/symthaea/issues/1284), [#1272 typed prediction residuals](https://github.com/Luminous-Dynamics/symthaea/issues/1272), and [#1269 recurrent snapshot/restore](https://github.com/Luminous-Dynamics/symthaea/issues/1269).

## 1. Why this tranche exists

M1 starts with deterministic semantic and authority-boundary cases, not a new controller. It tests whether the next HLB experiment can distinguish:

- physical contact from missing contact data;
- an observed physical state from a latent HDC representation;
- a fresh observation from a stale or out-of-order sample;
- a local reflex proposal from a global skill/mission proposal;
- a prediction hypothesis from a measured outcome;
- a compatible experience from a superficially similar but incompatible one;
- cognitive readiness from physical plant-safety evidence;
- cognitive time from physical plant time;
- a requested action from a projected, applied, observed, and independently accepted result.

The core biological hypothesis is **distributed sensing with hierarchical coordination**. Local sensing and bounded reflexes may reduce response latency and communication load; a slower supervisory layer may coordinate task-level behavior. We do not assume that an octopus, insect, or vertebrate circuit can be copied directly into a robot.

## 2. Source audit: exact facts to preserve

Source was fetched from the public `main` branch on 18 November 2026. These SHAs identify the inspected blobs, not a claim that the runtime is qualified.

| Source | Blob SHA | Audited fact |
|---|---|---|
| `crates/domains/symthaea-manipulator/src/sensorimotor.rs` | `40f8a4fd4fbdfb868b9cd4840d6d7be929f6bf0c` | `ControllerInputSchema::default()` is `LegacyDirectV0`; role-bound thought/body fusion is an explicit schema. |
| `crates/domains/symthaea-manipulator/src/controller.rs` | `eb7a81daa75aa1298de1c7e2139992dc79844980` | HDC-LTC output is projected to seven normalized joint torque outputs and one gripper output. |
| `crates/domains/symthaea-manipulator/src/predictor.rs` | `c9943f13f12e46442543b0d434688f3efa2de29f` | Current state-prediction comparison is a fixed-scale normalized RMS scalar; it is not a calibrated predictive distribution. |
| `crates/domains/symthaea-manipulator/src/embodiment.rs` | `c56a5fc5ad8ff83f02534aaec10db15ef8c84ce3` | The current bridge fuses thought/body HVs, obtains a learned command, applies admittance, and routes it through the stateful safety supervisor before stepping the simulator. The legacy `phi` argument is mapped to a `MotorSafetyLevel` and max-composed with other restrictions. |
| `crates/domains/symthaea-manipulator/src/safety_supervisor.rs` | `da7c25b79d662a13674853d5746132d6e1229702` | The manipulator supervisor declares final authority over nominal commands before the physics/hardware backend. |
| `src/cognitive_loop/accessors/consciousness.rs` | `5558d956e6831c05d939a591e37fc8ca1200afe5` | `safety_gate_phi()` returns a stored compatibility scalar with two unrelated writer formulas and records writer/cycle provenance separately. The accessor documentation warns that interleaving entry points makes the value order-dependent and unattributed. |

The manipulator is therefore a useful research bridge, but it is not yet the proposed HLB-only cognitive proposal architecture. The existing HDC-LTC controller participates in the control path, albeit behind the platform's independent supervisor. New high-level HLB planning must not be added as another direct motor-command path.

The `ControllerWeights` compatibility detail is also intentional: `ManipulatorController::export_weights()` labels the default schema `LegacyDirectV0`, while the trainer's export wrapper explicitly changes it to `SensorimotorBoundV1`; `ManipulatorEmbodiment::install_weights()` accepts only the latter. Future manifests must bind the schema actually used.

## 3. Biological evidence: useful mechanism, bounded claim

Del Dottore et al., *Peripheral control enabled by distributed sensing in an octopus-inspired soft robotic arm for autonomous underwater grasping*, Nature Machine Intelligence 8, 708–721 (published 12 May 2026), DOI [10.1038/s42256-026-01230-y](https://doi.org/10.1038/s42256-026-01230-y), is a concrete engineering reference.

The paper reports a single 410 mm soft arm with ten sensorized suction cups. Each cup integrates optical mechanosensing; the reported mean force-signal slope is about 394.9 ± 21 mV/N, directional error is below 18 degrees, and the tested 0–2 N range gives approximately 0.1 N accuracy. Local contact circuits trigger suction reflexes, while an arm-level peripheral controller coordinates grasp strategy. The reported task demonstrations use a restricted object set in still water; the prototype does not implement a central nervous system or inter-arm coordination.

**Do not compare those sensor numbers directly with a rigid seven-joint manipulator's force sensor or claim equivalent task performance.** Use the paper to motivate a mechanism ablation: local reflex vs centralized-only processing vs hybrid local/global coordination. For a fair comparison, match sensing budget, plant, contact scenario, communication assumptions, and physical task.

## 4. Fixture envelope

Every fixture binds, or explicitly declares unavailable:

- embodiment/profile and instance identity;
- morphology, tool/gripper, actuator map, and calibration identity;
- physical role, unit, frame kind, realized frame/transform identity;
- sample source, sequence, source timestamp, clock domain, and freshness policy;
- model/checkpoint identity and prediction horizon;
- action request, projection, admission, application, observation, and acceptance stages;
- memory provenance and the exact compatibility predicate;
- independent safety policy and required evidence set.

Missing, stale, invalid, uncalibrated, wrong-frame, or wrong-configuration evidence is never silently converted to a measured zero, a healthy value, a current value, or a successful action.

## 5. Known-answer cases

The companion `fixtures.json` contains the normative M1 cases. The first implementation should cover:

1. **Missing vs zero:** absent tactile coverage is not a measured zero-force/no-contact sample.
2. **Stale observation:** expired tactile/proprioceptive data cannot support a current contact or grasp assertion.
3. **Frame mismatch:** equal numeric vectors in different realized frames are not directly comparable.
4. **Duplicate/out-of-order samples:** replay or stale sequence identity cannot manufacture a new observation.
5. **Wrong-configuration memory:** an episode from another gripper/tool/calibration profile is diagnostic-only unless an explicit transfer predicate passes.
6. **Slip + local reflex:** local bounded stabilization proposal is emitted from measured slip/contact evidence; it does not grant mission-level authority.
7. **Prediction mismatch:** repeated typed residuals may invalidate a model/trigger re-observation; a heuristic scalar cannot become calibrated uncertainty or physical safety.
8. **HLB timeout:** stale/late cognition is discarded while the already-qualified local controller/fallback continues.
9. **Independent veto:** a physical hard constraint vetoes a cognitive proposal regardless of cognitive readiness, HDC similarity, or predicted reward.
10. **Partial sensor/network loss:** missing local nodes are explicit; the global layer must not infer full coverage from the remaining sensor subset.
11. **Plant/cognition time separation:** changing cognitive cadence cannot change physical elapsed time or plant integration.
12. **Replay determinism:** same frozen inputs, identities and event ordering reproduce the same fixture disposition and canonical digest.

## 6. Proposed M1 decision vocabulary

These are fixture-level dispositions, not production API names:

- `ACCEPT_FOR_EVALUATION`: complete, fresh, compatible evidence may be evaluated further.
- `RESTRICTED`: a declared restriction applies; only the independently qualified restricted path remains.
- `ABSTAIN`: required evidence is unavailable or ambiguous.
- `QUARANTINE`: a sample, memory, or model artifact is retained for diagnosis but cannot support action.
- `REJECT_STALE`: expiry/freshness condition failed.
- `REJECT_IDENTITY`: embodiment, tool, calibration, model, frame, or contract identity mismatched.
- `VETO_PHYSICAL`: independent plant-safety policy rejected the proposed physical action.
- `FALLBACK_LOCAL`: the platform's separately qualified local fallback owns the immediate response.

A fixture disposition is not an actuator command. The test harness must separately assert that no proposal or evidence disposition can bypass action projection, plant admission, and platform safety supervision.

## 7. Biology-to-engineering ablation plan

After M1 fixtures are implemented, run a minimal, matched manipulation study with the same robot, sensors, task family, initial states, compute and physical limits:

| Arm | Mechanism |
|---|---|
| A | Centralized tactile processing; no local reflex |
| B | Local contact reflex only; no HLB integration |
| C | Local reflex + centralized task coordination |
| D | C + typed HDC semantic representation |
| E | D + CfC/LTC action-conditioned prediction |
| F | E + episodic memory with configuration-bound retrieval |
| G | F + active sensing / information-seeking proposals |

Keep a strong deterministic/impedance baseline and a matched recurrent baseline. Add a tactile modality/availability ablation and a communication-delay/loss stratum. Compare the cognitive components only after the local controller and physical plant are fixed and independently measured.

Measure separately:

- contact detection and direction/force error against independent reference instrumentation;
- sensor-to-local-reflex latency and full sensor-to-controller latency;
- communication bytes, dropouts, and local operation under network partition;
- grasp retention, slip, damage, task success and partial progress;
- recovery latency, retries, human intervention and preserved recovery options;
- prediction error and interval coverage by physical role/regime;
- memory retrieval precision, false analogue rate and invalidation after configuration changes;
- CPU/GPU, memory, sensor, actuator and network energy with explicit system boundaries;
- p50/p95/p99/worst latency and hard deadline misses.

Do not collapse these into one weighted score. Report paired mission/episode results and worst-condition strata.

## 8. Promotion gates

- M1 fixture implementation and known-answer tests pass independently of HLB.
- Existing controller/supervisor behavior is unchanged by this research scaffold.
- All HLB proposals remain shadow-only during the first comparative experiment.
- #1276 cognitive-readiness/plant-safety separation and #1284 plant-time semantics are respected; no new path treats the legacy `phi` argument as a physical measurement.
- #1272 typed residual semantics and #1281 metric-kind distinctions are used where their prerequisites exist; otherwise record explicit unavailability.
- Memory/model promotion remains quarantined until independent held-out evaluation and rollback evidence are present.
- No HLB advantage is claimed unless matched ablations show positive held-out task/decision benefit without increasing hard physical-envelope violations.

## 9. Claim ceiling

This contract scaffold is not a test pass, an implemented sensor/reflex network, a calibrated world model, a biological reproduction, a physical qualification, or evidence of SOTA superiority. A synthetic fixture pass proves only the fixture implementation for its declared cases. In-simulation results remain in-simulation results; HIL and physical evidence require separate instrumentation and qualification.
