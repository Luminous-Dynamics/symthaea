# SAM Follow Evidence Boundary v0.1

Status: simulation/research API; not a production perception interface.

## Purpose

Prevent three different information domains from silently becoming the same lead-vehicle measurement:

1. **Simulator oracle state** — exact peer state available to a synthetic scenario engine.
2. **Local tracked-actor observation** — metric estimate produced by a local perception / tracking adapter under a named scene, frame, time, source and uncertainty contract.
3. **Remote cooperative advisory** — a V2V/V2I claim from another system. Authentication of a sender does not establish physical truth or local observation.

The legacy 20D `VehicleState` remains unchanged for checkpoint/channel-layout compatibility. Its peer fields are legacy simulation fields, not a local perception API.

## Typed contract

`FollowConsumerProfile` declares the consumer:
- `SimulationBenchmark`: exact scenario, frame and maximum observation age.
- `LocalPerception`: exact scene snapshot, frame, maximum age and maximum admitted absolute distance-error bound.
- `Deployment`: separate profile with the same evidence obligations; simulator truth cannot satisfy it by relabelling.

`FollowTargetEvidence` keeps the evidence domains distinct:
- `SimulatorOracle`: scenario/actor/frame, observation age, longitudinal gap and relative speed.
- `LocalTrackedActor`: scene snapshot, track lineage, source IDs, frame, age, longitudinal gap, relative speed and explicit distance-error bound.
- `CooperativeAdvisory`: claim identity only; this variant is explicitly rejected as a Follow target.

`admit_follow_target` validates profile/evidence compatibility, context identity, frame, numeric ranges, freshness, non-empty unique local source IDs and the declared error bound. Local gap is reduced to a conservative lower bound before it is consumed.

**Important limitation:** the current enum/API is a semantic admission boundary, not cryptographic attestation. Public enum construction does not prove that a caller's source IDs, scene snapshot, calibration, frame, or error bound are authentic. A future production adapter must only create the local-track variant from the qualified SAM scene-evidence admission path and independently reviewed calibration/tracking outputs. The current local-track tests are synthetic conformance vectors, not physical perception tests.

## HDC and FEP policy treatment

The 20-channel layout is retained, but:
- Default `VehicleHdcEncoder::encode` neutralizes legacy peer distance/speed and mesh brake/confidence channels before both position and derivative encoding.
- `encode_simulator_oracle` is a separately named synthetic-only method that admits peer distance/speed. It still masks aggregate brake density and claimed confidence.
- HDC channel weights for raw mesh brake-density and claimed-confidence remain zero.
- FEP observation channel 9 remains a neutral compatibility slot. Raw mesh brake-density cannot select a policy action. The legacy `IncreaseMeshTrust` action identifier remains for index/checkpoint compatibility but has no modulation effect.

The FEP policy does not yet consume a qualified local lead-track object. That is intentional: the current controller is simulation-only, and an untrusted peer claim must not be substituted as local perception.

## Follow training and reward

The synthetic swarm adapter:
- returns a scenario-bound `SimulatorOracle` target;
- reports the actual age of the broadcast sample selected from its latency buffer (including startup);
- selects only actors ahead of the ego;
- measures longitudinal road-frame separation, not Euclidean distance including lateral displacement.

Training:
- uses the simulator-oracle encoder only when the target passes the benchmark profile;
- supplies the same admitted target to the Follow PD teacher and gap reward;
- skips Follow teacher-labelled training and replay insertion when evidence is missing or rejected;
- records `follow_evidence_unavailable_steps` per episode;
- does not infer a 200m "clear" gap from the legacy default field.

The controller still produces its learned policy output if no Follow evidence is available; this branch does not implement a vehicle-level abstention/RTA. Command admission belongs to the existing #10012 RTA work and is not qualified by this API.

## Falsification and regression tests

Positive controls:
- current scenario-matched oracle target is accepted for simulation benchmark;
- current local synthetic track is accepted under the local/deployment profile;
- the conservative lower-bound gap is used;
- explicitly admitted simulator oracle can alter the simulator-mode HDC representation.

Negative controls:
- no target cannot use a naked `nearest_peer_distance`;
- simulator oracle cannot satisfy local-perception/deployment;
- remote V2X advisory cannot satisfy any Follow target profile;
- stale/future, wrong scenario/frame, empty/duplicate sources and excessive error bound are rejected;
- unknown target increments unavailable-step accounting and carries no numeric fallback;
- default encoder output is invariant to all legacy peer-channel changes, including derivative history;
- simulator-oracle mode still ignores aggregate brake-density and claimed-confidence fields;
- FEP observation and rule-based action are invariant to unqualified mesh aggregates.

## Benchmark plan

Report these as separate outcomes; do not collapse them into one reward:
- oracle-follow gap error vs local-track-follow gap error (never compare them as sensor-equivalent);
- Follow evidence availability/stale/unknown rate and age distribution;
- PD teacher and controller action differences under identical local state / evidence cases;
- longitudinal gap error, relative-speed error, TTC and stopping margin where the simulator semantics support them;
- intervention, abstention, completion, false interventions and progress;
- p50/p95/p99 end-to-end observation-to-command latency and deadline misses;
- performance under actor speed changes, sensor delay/dropout, occlusion, wrong frame, stale scene, and ODD/profile mismatch;
- exact scenario, actor policy, source/track, map, model, config and replay identities.

A useful comparison ladder is:
1. legacy naked peer scalar (retained only as an explicitly labelled historical/negative-control baseline);
2. scenario-oracle Follow target;
3. local metric tracked actor with an explicit error bound;
4. no-target/unknown evidence case;
5. remote cooperative advisory attack / stale / replay case.

The oracle arm is not an attainable perception baseline and must not appear on a sensor-only leaderboard.

## Claim ceiling

The work is a typed research boundary and synthetic training integration. It does not implement camera/LiDAR/radar tracking, production sensor calibration, V2X security, independent RTA, CAN/actuator integration, vehicle safety qualification, public-road testing, or SAE Level 4/5 readiness.
