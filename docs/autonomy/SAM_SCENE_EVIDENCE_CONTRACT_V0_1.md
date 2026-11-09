# SAM Scene Evidence Contract v0.1

Status: research interface / synthetic conformance contract  
Machine-readable structural schema: `SAM_SCENE_EVIDENCE_SCHEMA_V0_1.json`  
Canonical synthetic fixture: `fixtures/SAM_SCENE_EVIDENCE_CONTRACT_V0_1.json`  
Owner: Symthaea SAM perception-to-planning boundary  
Composes: SAM-SCENE-ADMISSION-001 (#11109)

## 1. Purpose

This contract defines the minimum information SAM needs to turn asynchronous sensor evidence into a road-scene representation that a planner can reason over without confusing:

- pixel-space features with metric geometry;
- an object's appearance with its semantic class;
- an observation with an estimate or prediction;
- no detection with observed free space;
- a single preferred track association with a uniquely identified physical actor;
- a numeric confidence with calibrated uncertainty;
- a valid data structure with admissible evidence.

This is not a camera/LiDAR/radar driver, object detector, sensor-fusion engine, map provider, safety case or actuator interface. It defines an integration boundary for existing or external sensing and estimation components.

## 2. Current implementation boundary

The active `symthaea-vehicle` crate is a simulation-only controller. Its `VehicleState` contains 20 compact channels: 14 simulated vehicle/proprioceptive values, two road-context values, and four synthetic peer/mesh values. It does not consume raw camera, LiDAR or radar data.

The active `symthaea-vision-manifold` provides useful HDC video representations, temporal object memory, grid-based track velocity and relational scene-graph edges. Its current `TrackedObject` data uses image-grid centroids and grid-cell-per-observed-frame velocities. The scene graph carries relations between visual tracks. These representations do not, by themselves, establish calibrated metric 3D geometry, physical actor classes, a road-relative coordinate frame, calibrated uncertainty or free-space coverage.

The former `symthaea-perception` crate is archived and workspace-excluded. It is not an active vehicle integration path.

**Permitted VisionManifold contribution:** appearance descriptors, temporal similarity, surprise, association proposals and qualitative visual relations. These are hypotheses / features with their own source identity and frame context.

**Not permitted from HDC-only outputs:** metric range, metric velocity, physical extent, semantic ground truth, a calibrated probability, proof of free space, or an actuator-authorizing safety fact.

## Traffic convention context

Every road-scene snapshot carries an explicit `traffic_convention_profile_id` sidecar reference. The profile separately binds traffic side (left/right), steering-wheel position, travel-frame identity, keep/kerbside and median-side signs, passing-rule profile/version and lane-ordering profile. The reference must be resolved before rule-sensitive lane selection or passing decisions. Its `travel_frame_id` must resolve to a `local_road` reference frame in the same scene snapshot and must match both the ego-state and road-geometry frame IDs. No country, GPS, pixel-side or steering-wheel-layout inference is allowed.

See `SAM_TRAFFIC_CONVENTION_METAMORPHIC_V0_1.md`, the companion profile schema and paired synthetic LHT/RHT fixtures. A mirror transform changes lateral/yaw/curvature/steering signs but does not infer steering-wheel position or replace the jurisdiction rule profile.

## 3. Evidence architecture

~~~
sensor observations
  -> timestamp / clock-transform admission
  -> calibration / extrinsic / reference-frame admission
  -> metric measurement and occupancy hypotheses
  -> association hypotheses and track lineage
  -> road-scene evidence snapshot
  -> actor prediction sets and world-model evaluation
  -> action feasibility / decision envelope
  -> choose, measure more, degrade or abstain
~~~

The scene snapshot is a bounded hot-path structure. The high-rate raw data plane stays at the edge or in a separately governed telemetry system. Mycelix may preserve selected evidence commitments, provenance and lifecycle receipts, but Holochain/DHT reconciliation is never a dependency in the hard real-time braking or steering path.

### Source-role separation

Every source declares a `source_role`, separate from its modality and epistemic/evidence kind:

- `local_sensor_observation`: physical sensor measurement candidate;
- `synthetic_sensor_observation`: synthetic sensor output with sensor-like semantics, explicitly not simulator oracle truth;
- `remote_cooperative_claim`: V2V/V2I or other remote advisory claim;
- `simulator_oracle`: privileged simulator state unavailable to a deployed vehicle;
- `map_prior`: map or road prior;
- `derived_estimate`: estimator/model output;
- `unknown`: unresolved source role.

The canonical fixture deliberately contains both a remote V2X claim and simulator oracle truth, but keeps them in separate `cooperative_advisory_refs` and `simulator_truth_refs` lists. Neither may appear in a physical local actor track's `source_refs`. Object tracks, HDC appearance proposals and observed-free regions may cite only local or explicitly synthetic sensor observations; road geometry may additionally cite a map prior.

These roles are declared metadata and semantic consistency checks, not cryptographic authentication. The contract can reject internal role/modality contradictions (for example V2X labelled as a local sensor), but it cannot prove that the source's author declared its real origin honestly. Signatures, device attestation, revocation and evidence provenance remain separate owners and admission obligations.

## 4. Canonical coordinate and time semantics

### 4.1 Coordinates

Every metric vector, geometry, covariance and trajectory must name its reference frame.

The initial synthetic profile uses a right-handed local road frame:

- x: forward along the declared local route direction;
- y: left when looking in the positive x direction;
- z: up;
- SI units: metres, seconds, metres/second, radians.

A frame identifier alone is insufficient. The snapshot references the exact transform / map / calibration profile that relates sensor, ego, road-local and world frames. An unresolved transform is an admission failure or an explicit unknown; callers must not silently assume identity.

Image coordinates remain in a separately named image frame, in pixels. They cannot be re-labelled as metres.

### 4.2 Time

Preserve acquisition/event time, receive time and processing time separately. Each timestamp must carry a clock-domain identity. Cross-domain comparisons require a referenced clock transform and its uncertainty.

A receive timestamp must not be substituted for acquisition time. A sequence number does not establish clock synchronization. If the transform is missing, expired, non-monotonic or outside its supported interval, temporal alignment remains unknown.

## 5. Evidence states and admission are orthogonal

Each scene fact carries an epistemic state:

- `observed`: a measurement was obtained under the declared sensor/evidence contract;
- `estimated`: a state inferred from admitted observations and a named estimator;
- `predicted`: a future state from a named predictive model and horizon;
- `occluded`: a relevant region or object extent is not currently observable;
- `contradicted`: admitted evidence sources disagree beyond the declared comparison rule;
- `stale`: the evidence is too old for the consumer's declared currentness profile;
- `out_of_scope`: the fact lies outside the declared model / ODD applicability;
- `unknown`: required evidence is absent, unsupported or unresolved.

Admission is a different set of checks. Keep clock, calibration, transform, numeric validity, source provenance, currentness, spatial applicability and model applicability separate. Do not compress them into a single green status or confidence score.

A structurally valid JSON object is not necessarily admissible evidence.

## 6. Snapshot envelope

A v0.1 snapshot contains:

1. **Identity:** schema version, snapshot ID, monotonic sequence and parent / predecessor reference.
2. **Scenario / ODD:** exact scenario, operational-domain, map, vehicle-configuration and policy-profile identities.
3. **Clock:** clock domain, event tick, receive tick, processing tick, transform reference and uncertainty.
4. **Reference frames:** named frames, transform identities and the declared unit/axis convention.
5. **Sources:** source ID, modality, explicit source role (local sensor / synthetic sensor / remote cooperative claim / simulator oracle / map prior / derived estimate / unknown), sensor / software identity, failure-domain lineage, evidence reference, calibration reference and currentness.
11. **Source-domain side references:** cooperative advisory claims and simulator oracle truth are retained in separate typed reference lists; they cannot be smuggled into local actor tracks.
6. **Ego state:** pose, velocity, orientation, estimator identity, uncertainty and contributing evidence references.
7. **Road geometry:** lane / boundary / drivable-area hypotheses, direction, validity interval, provenance and uncertainty.
8. **Object hypotheses:** track lineage, metric state, class hypotheses, physical extent, motion, uncertainty, visibility, time and source references.
9. **Free space:** observed-free / occupied / occluded / unknown regions, coverage geometry and sensor-health scope.
10. **Predictions:** references to action-conditioned prediction sets, horizons, model identity and applicable support profile. Prediction-set evaluation is owned by SAM #10256.
11. **Unsupported / unresolved:** explicit missing modalities, unavailable fields, approximation declarations, contradictions and incomplete outputs.

## 7. Object hypothesis contract

A physical-object hypothesis should preserve at least:

- stable hypothesis ID and track-lineage ID;
- association state: confirmed, provisional, ambiguous, split, merged, unmatched or retired;
- epistemic state and measurement time;
- class distribution with classifier identity, calibration status and applicable class vocabulary;
- position, orientation, velocity and acceleration in an explicit metric frame;
- object extent / shape with units and estimation method;
- uncertainty representation, including component order, units, coordinate frame and covariance / set semantics;
- last observed time, predicted-through time, visibility and occlusion state;
- evidence-source references and source/failure-domain dependency graph;
- estimator, association and model identities.

Do not force a unique track ID when multiple associations remain plausible. Preserve parent/child lineage during split and merge. A track prediction must not overwrite the last observed state.

Class scores from an uncalibrated model are ranking metadata, not calibrated probabilities or ground truth.

## 8. Free-space and occupancy contract

Absence of an object detection is not proof that space is clear.

Any free-space assertion must bind:
- the sensor's field of view and effective range for this frame;
- the relevant acquisition time and ego pose;
- occlusion / shadow geometry;
- sensor-health and no-return / saturation disposition;
- the region actually covered;
- the method and thresholds used to classify cells or polygons.

Use distinct labels for at least `observed_free`, `occupied`, `occluded`, `unobserved`, `out_of_range`, `sensor_invalid` and `unknown`. Missing coverage must remain unknown, not free.

## 9. Admission invariants

1. Non-finite, malformed or out-of-profile values cannot mutate the accepted snapshot.
2. A frame / extrinsic transform must be present and current before metric geometry is consumed.
3. Timestamps from different clocks cannot be compared without an admitted transform.
4. Expired calibration cannot silently inherit currentness.
5. A source count is not an independence count. Shared provider, model, clock, map or processing roots remain explicit.
6. Image-space centroids and HDC similarities cannot be converted into metric state by scaling or relabeling.
7. No detection is not free-space evidence outside admitted observation coverage.
8. Occlusion is not clear space. A stale track is not a current observation.
9. Prediction and simulation outputs remain distinct from observations.
10. A missing covariance, support envelope or model applicability declaration becomes unknown / rejected, not zero uncertainty.
11. Contradictory sources remain visible; averaging incompatible frames or sources is not a valid resolution.
12. Local object tracks cannot cite remote cooperative claims or simulator oracle truth; those sources remain in their separate typed reference collections.
13. The scene layer cannot grant vehicle, ODD or actuator authority. Symthaea's separate runtime-assurance boundary owns command admission.

## 9.5. Follow-target evidence context

A controller or teacher that needs a lead-actor gap must name the evidence domain supplying it. The typed `follow_target_context` is separate from the legacy `VehicleState` mesh channels.

- `simulation_benchmark`: may consume only a current, exact-scenario `simulator_oracle` source, linked by `target_reference_kind=simulator_truth_actor` and an exact `target_reference_id` into `simulator_truth_actor_states`. The actor-state record must share the same oracle source, scenario, frame and acquisition tick.
- `local_perception`: may use a current local sensor measurement or an explicitly synthetic sensor observation for conformance. It must use `target_reference_kind=tracked_object_hypothesis` and link `target_reference_id` to an exact object hypothesis. The context actor ID must equal that hypothesis's track-lineage ID, and the evidence source must appear in the hypothesis's own source references.
- `deployment`: requires a current, non-synthetic camera/LiDAR/radar/ultrasonic local sensor measurement, linked to a tracked object hypothesis with matching calibration/frame/time provenance. A passing contract check still does not attest the installed sensor or physical vehicle.
- `remote_cooperative_claim`: is not a valid Follow-target source in any profile. V2X provenance may enrich an advisory stream, but cannot silently substitute for a locally observed lead track.
- `unknown` / `rejected`: carries no selected source, actor reference or numeric gap fallback. The controller must expose the target as unavailable and let the separately qualified planner/RTA determine the admissible response.

The source, target actor/track and scene must agree on scenario, frame, acquisition/measurement tick, clock domain and transform. The v0.1 Follow calculation is deliberately restricted to a resolved `local_road` frame with x forward: the checker recomputes `gap_m = target.x - ego.x` and `relative_speed_mps = target.vx - ego.vx`. A caller cannot pass a plausible-looking scalar that contradicts the linked scene state. Missing, future-dated, stale, wrong-frame, unlinked, non-actor-modality or role-incompatible targets are rejected or remain unknown.

The canonical fixture includes an explicit simulator-oracle actor-state record and a separate synthetic-LiDAR local-perception track. The latter can exercise the local-perception conformance path but is intentionally rejected for the deployment profile. All fixture records are synthetic; no physical sensing is claimed.

## 10. Ownership boundaries

Compose rather than duplicate:

- Symthaea #5820 and its existing observation, clock, calibration, sensor-fusion and active-perception owners;
- Symthaea #10256 (trajectory prediction-set coverage / sharpness);
- Symthaea #10257 (horizon-conditioned world-model discrepancy);
- Symthaea #10320 (decision-sensitive measurement / value of information);
- Symthaea #10012 (road runtime assurance);
- Mycelix #413 (domain-neutral spatial truth, time, uncertainty, source correlation and privacy);
- Symtropy #3387 / #3471 / #3532 (bounded scenario execution, causal actors and reciprocal interaction);
- Sol-Atlas #585 / #636 (typed route graph and read-only spatial projection).

The scene contract is not a replacement for any of those components.

## 11. Initial conformance campaign

Use the companion fixture `fixtures/SAM_SCENE_EVIDENCE_CONTRACT_V0_1.json`. It is explicitly synthetic and demonstrates:

- two asynchronous sensor modalities;
- separate acquisition and receive times;
- metric road-local coordinates;
- a pedestrian-like actor with class ambiguity;
- an observed vehicle and a static obstacle;
- an occluded / unknown occupancy region;
- an image-grid HDC appearance proposal kept separate from metric state;
- an explicit unsupported / unknown region.

Required negative controls:

| Mutation | Required disposition |
|---|---|
| Change camera intrinsics / extrinsics but preserve image centroid | Old metric mapping becomes inapplicable; no silent coordinate reuse |
| Change clock transform | Recompute temporal alignment; reject stale transform |
| Advance receive time but not acquisition time | Does not refresh observation age |
| Remove occlusion coverage | Region becomes unknown, never clear |
| Duplicate a source with a new sender ID but same failure domain | No independent corroboration gain |
| Split one track into two plausible continuations | Preserve lineage ambiguity |
| Change HDC similarity only | No change to metric state / class truth |
| Remove covariance or transform | Reject or unknown; never assume zero uncertainty |
| Serialize a prediction into an observation field | Reject schema / evidence-state mismatch |
| Supply a sensor no-return outside valid range | Unknown / out-of-range, not free |

Synthetic conformance establishes only contract behavior for the declared fixture. It does not establish sensor accuracy, human actor realism, real-world generalization or vehicle safety.

## 12. Benchmark path

After the interface vectors are qualified, use the existing benchmark owners:

- perception / detection / tracking: exact Waymo Open Dataset or nuScenes task split and label semantics;
- motion prediction: Waymo Open Motion and Argoverse 2, preserving minADE / minFDE / miss-rate metrics separately from coverage and set size;
- interaction: WOSAC and the Symtropy reciprocal-policy harness;
- closed-loop: NAVSIM v2, CARLA Leaderboard 2.1 and Waymax only under matched sensor privilege and version;
- decision quality: NIST's Tier 1–4 measurement framing and SAM's counterfactual / discrepancy / active-measurement owners.

Do not compare scores across incompatible splits, sensor privileges, data rights or benchmark versions. Do not claim that a better image metric proves better closed-loop driving.

## 13. Claim ceiling

V0.1 is a schema and synthetic conformance contract. It is not a production perception stack, calibrated sensor-fusion system, physical sensor qualification, safety case, roadworthiness assessment, public-road authorization or SAE Level 4/5 evidence.
