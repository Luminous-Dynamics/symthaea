# SAM Scene Evidence Checker v0.1

Standalone Rust validator for the SAM Scene Evidence Contract v0.1.

## Run from the repository root

```bash
cargo fmt --manifest-path tools/sam-scene-evidence-check/Cargo.toml -- --check
cargo test --manifest-path tools/sam-scene-evidence-check/Cargo.toml
cargo run --manifest-path tools/sam-scene-evidence-check/Cargo.toml -- \
  docs/autonomy/SAM_SCENE_EVIDENCE_SCHEMA_V0_1.json \
  docs/autonomy/fixtures/SAM_SCENE_EVIDENCE_CONTRACT_V0_1.json \
  docs/autonomy/SAM_TRAFFIC_CONVENTION_PROFILE_SCHEMA_V0_1.json \
  docs/autonomy/fixtures/SAM_TRAFFIC_CONVENTION_METAMORPHIC_V0_1.json
```

This is a nested Cargo workspace and is intentionally not part of the root workspace glob. It is independently buildable so that root workspace metadata failures do not prevent contract validation.

## Validation layers

1. JSON Schema Draft 2020-12 meta-schema and instance validation using the exact-pinned `jsonschema` crate.
2. Offline schema compilation: this contract uses local `$defs` references only; unexpected remote reference retrieval is disabled.
3. SAM semantic checks that JSON Schema alone cannot express:
   - source, frame and track references must resolve;
   - source IDs and hypothesis IDs must be unique; alternate hypotheses may share a track-lineage ID;
   - snapshot event/receive/processing ticks must be ordered;
   - diagonal covariance dimensions must match state ordering and units;
   - diagonal covariance entries must be finite and non-negative;
   - calibrated class probabilities must sum to one;
   - observed-free regions require source evidence;
   - prediction references must name a track in the scene;
   - HDC proposals cannot gain metric or semantic authority (also fixed by schema);
   - scene admission cannot grant actuator authority (also fixed by schema).

## Traffic convention / mirror benchmark

The same checker validates `SAM_TRAFFIC_CONVENTION_PROFILE_SCHEMA_V0_1.json` and the paired LHT/RHT fixture. It enforces explicit, versioned traffic-side and passing-rule profiles, verifies that map and scene profile IDs agree, and checks the mirror transform for lateral position/velocity, yaw, curvature, steering, lane sides, passing side, and invariant longitudinal / throttle / brake quantities. Steering-wheel position is intentionally independent of traffic side.

## Negative controls

Unit tests mutate the canonical fixture to test rejection of:
- actuator authority escalation;
- HDC metric-geometry authority escalation;
- unknown source references;
- covariance dimension mismatch;
- observed-free regions without source evidence;
- duplicate hypothesis identity (while allowing multiple hypotheses on one track lineage);
- predictions referring to unknown tracks;
- out-of-order snapshot clock fields;
- LHT/RHT profile and lane-side mismatches;
- untransformed steering under road reflection;
- map/profile identity mismatch;
- passing-side / rule-profile mismatch;
- steering-wheel-layout-only changes that must preserve road semantics.

A PASS means structural and declared SAM semantic conformance for this interface version. It does not establish calibration accuracy, source independence, physical truth, sensor validity, ODD applicability, model safety, or vehicle authority.
