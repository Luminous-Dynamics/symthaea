# SAM Scene-Derived Follow Capability v0.1

Status: research-only typed capability boundary; no qualified physical scene-admission producer exists.
Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/11280
Composes: scene evidence #11109, typed Follow #11186, simulator-oracle Follow #11213, privilege measurement #11258, RTA #10012.

## 1. Problem being closed

The legacy Follow API accepts a serializable LocalTrackedActor variant containing caller-supplied scene IDs, source IDs, frame labels, observation age, gap, relative speed and distance-error bound. Checking that strings are non-empty and that a scalar is within a configured range does not establish that those claims resolve to an admitted scene, valid object hypothesis, current sensor source, calibration, clock transform or frame transform.

This PR therefore makes the serialized/self-asserted LocalTrackedActor variant fail closed with SceneAdmissionReceiptRequired. Serialization/deserialization does not restore an admission capability.

The existing SimulatorOracle variant remains the explicit simulation/benchmark path. Remote cooperative claims remain non-authoritative. This does not introduce a physical sensor adapter or enable deployment Follow.

## 2. Opaque capability boundary

scene_follow.rs defines:

- SceneAdmissionReceipt: crate-private, non-serializable scene admission result with private fields.
- SceneSourceReceipt: source/event/failure-domain identity, source and normalized clock domains, acquisition/receive time, clock/frame transform admission, calibration/extrinsic identities, source role and currentness.
- SceneTrackReceipt: metric hypothesis state, track lineage, association state, visibility, extent, measurement time, source references and a named uncertainty semantic.
- AdmittedLocalFollowTarget: public but non-serializable capability with private fields, bound to the exact scene digest, provenance-receipt digest, schema version, sequence, track/hypothesis IDs, source IDs, frame/clock transforms, measurement time and derived gap/closing speed.

There is deliberately no public constructor for SceneAdmissionReceipt. Its only constructor is crate-private and is reserved for a future adapter that consumes a validated scene/provenance result. The current repository does not yet have that qualified physical producer. The synthetic unit-test helper only exercises the contract; it is not a production evidence source.

## 3. Derived Follow measurement

For a deployment/local-perception profile, the internal admission function:

1. Requires the exact expected scene snapshot and road frame.
2. Requires scene clock, frame transform, calibration-currentness and numeric-admission flags.
3. Requires the road-local x-axis to be explicitly forward.
4. Verifies source IDs are unique and well-formed; only sources referenced by the selected target may supply local Follow evidence.
5. Requires each selected source to be a current local physical sensor observation with admitted clock/frame transforms, current calibration and a bounded age in the common scene clock.
6. Rejects simulator-oracle, synthetic, remote V2X, map-prior and unknown roles as local Follow evidence. Such sources may coexist as unrelated scene records, but cannot appear in the local target's source set.
7. Selects the nearest positive-x in-lane object; a closer ambiguous, occluded, stale or otherwise inadmissible candidate blocks selection rather than being skipped for a convenient farther actor.
8. Requires confirmed association, a single hypothesis for the selected lineage, visible state, finite metric geometry, unique/resolved source references and named uncertainty semantics.
9. Derives longitudinal gap and relative speed from the ego and target metric states in the same scene snapshot. It never accepts a caller-supplied gap or speed.
10. Reduces the gap by the upstream-provided distance uncertainty radius. The radius and its semantics are bound into the capability; this module does not reinterpret it as a calibrated probability or claim an independent uncertainty proof.

The resulting capability is short-lived. The command consumer checks its evaluated-at timestamp against the current vehicle-state timestamp and rejects stale or future reuse.

## 4. Fail-closed negative controls

The Rust tests exercise:

- the canonical synthetic scene-to-target derivation;
- scene snapshot mismatch;
- stale scene and stale source acquisition/receive time;
- missing clock/frame transform admission and calibration currentness;
- simulator, synthetic, V2X, map-prior and unknown source-role rejection;
- unrelated V2X record coexisting with a valid local sensor track without becoming its evidence;
- V2X source substitution into the selected track;
- ambiguous lineage and partially occluded actor rejection;
- unresolved source references;
- uncertainty radius above the consumer profile;
- malformed/non-finite scene track rejection before nearest-actor selection;
- duplicate hypothesis identity rejection at the receipt boundary;
- source-role, frame, timestamp and uncertainty-semantic identity retention;
- stale/future capability reuse at the command consumer.

The tests are source-authored in this PR. Only a green exact-head SAM Follow Scene Capability workflow run will count as executed validation.

## 5. Ownership and claim ceiling

- Symthaea #11109 owns the schema and structural/semantic scene contract.
- Mycelix #413 owns spatial frames, clock/time semantics, source lineage, uncertainty and provenance semantics.
- #11186 owns the typed Follow consumer boundary; this PR makes its raw local-track claim non-authoritative.
- #11213 owns the synthetic simulator-oracle target selector and frozen/shadow-teacher telemetry.
- #11258 owns privilege-aware Follow benchmark measurement.
- #10320 owns decision-sensitive measurement / value of information.
- #10012 remains the command-admission RTA boundary.

No second tracker, sensor-fusion stack, identity service, map engine, vehicle simulator or RTA is introduced.

This PR does not qualify local sensor accuracy, covariance-to-distance-bound computation, V2X truth, a production scene producer, a physical vehicle, public-road operation, collision reduction, regulatory compliance or SAE Level 4/5 capability. The current vehicle crate remains simulation/research-only. No actuator authority is granted by a scene receipt or Follow capability.
