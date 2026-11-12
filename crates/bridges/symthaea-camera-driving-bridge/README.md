# CAMERA-CONDITIONED-DRIVING-INPUT-001

**Status:** deterministic visual/proprioceptive policy-input bridge; no trained camera-driving policy  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11476

## Purpose

This crate composes two existing Symthaea owners:

- `symthaea-vision-manifold`: checked RGB/grayscale/RGBA frame ingestion, patch HDC encoding, CfC temporal scene state, prediction/surprise telemetry and frozen-learning support.
- `symthaea-vehicle`: the existing `VehicleState` and `VehicleHdcEncoder`.

It creates one fused HDC policy-input vector and an immutable SHA-256 receipt tying the vector to its source frame, vehicle state, capture clock, camera, episode, scenario, simulator revision and policy revision.

It deliberately does not call `VehicleController`, command an actuator, or claim that a controller trained on proprioception-only vectors can use this fused visual/proprioceptive representation. A controller must be trained/retrained and evaluated against this exact input schema.

## Input contract

- Input channels are grayscale (1), RGB (3) or RGBA (4).
- Frame dimensions, byte lengths and maximum allocation are checked before mutation.
- Capture timestamps, capture sequence numbers and vehicle-state timestamps must each strictly increase.
- The upstream renderer's RGB8 sRGB observation identity and receipt SHA-256 are recomputed and checked against the actual pixel bytes and policy-visible metadata before any temporal state advances.
- Camera/vehicle timestamp skew is bounded.
- The first timestep is supplied as a configuration constant; subsequent timesteps are derived from capture timestamps.
- A configured maximum inter-frame gap and maximum camera/vehicle timestamp skew are enforced.
- Capture-to-delivery age is bounded by `maximum_observation_delivery_age_us`; an over-age frame is rejected before Vision Manifold or vehicle temporal state advances.
- Vehicle channels and timestamp must be finite.
- Vision HDC dimension must equal the vehicle encoder dimension (16,384).
- Depth-enabled VisionConfig is rejected because this bridge has no physical depth input.
- Vision Manifold learning is frozen for the entire evaluation run.

Rejected preflight inputs do not advance frame, sequence, timestamp or encoder state.

## Fusion

The visual manifold state and vehicle-state HDC are normalized and combined with a fixed caller-declared visual weight:

`policy_input = normalize(weighted_bundle(visual_hv, proprioceptive_hv))`

The visual weight is fixed-point parts-per-million in the receipt. This is representation fusion, not a learned cross-modal alignment. The relative weight must be selected by an ablation, not tuned against held-out test outcomes.

## Evidence receipt

The receipt contains:
- raw frame SHA-256;
- visual scene-state SHA-256 and previous visual-state digest;
- proprioceptive-state SHA-256 and previous vehicle-state digest (the vehicle encoder includes derivative history);
- serialized VisionConfig SHA-256 and vehicle encoder seed-phrase SHA-256;
- HDC dimension and vehicle encoder level count;
- proprioceptive HDC SHA-256;
- fused policy-input SHA-256;
- capture sequence, timestamp and derived timestep;
- frame geometry and channel count;
- run/episode/scenario/camera/simulator/policy identities;
- frozen-learning and unchanged training-step evidence;
- selected finite Vision Manifold telemetry;
- a SHA-256 receipt commitment over all preceding receipt fields.

HDC hashes use the exact IEEE-754 f32 bit patterns serialized as little-endian bytes. Receipts do not contain raw camera pixels or the full HDC vector; the caller retains those artifacts under its data-governance policy.

## Build

From this directory:

```sh
cargo fmt -- --check
cargo test --locked
cargo clippy --locked --all-targets --all-features -- -D warnings
```

The bridge introduces no third-party dependency; it uses workspace-locked `serde` and `sha2`.

## Episode integration CLI

The crate also exposes `camera-driving-bridge`, which processes a complete episode in one process so the Vision Manifold and vehicle encoder retain their temporal state across observations.

Build and run from the repository root:

```sh
cargo run --locked \
  --manifest-path crates/bridges/symthaea-camera-driving-bridge/Cargo.toml \
  --bin camera-driving-bridge -- \
  episode-request.json receipts-output.json policy-inputs/
```

Generate the standard non-secret fixture configs with:

```sh
cargo run --locked \
  --manifest-path crates/bridges/symthaea-camera-driving-bridge/Cargo.toml \
  --example write_fixture_configs -- bridge-config.json vision-config.json
```

The helper pins the expected episode, agent, camera, 640×360 RGB geometry, one-frame-drop timing bounds, a 5,000 µs maximum capture-to-delivery age and default 16,384D no-depth VisionConfig. Its seed phrase is public fixture data, not a production secret.

The request schema is `schemas/episode-request-v1.schema.json`; the emitted fused receipt schema is `schemas/driving-observation-receipt-v1.schema.json`. Each observation references an RGB pixel artifact by a path relative to the request file. The CLI rejects absolute paths, parent-directory traversal and symlinks that resolve outside the request directory. It checks the source pixel digest against the upstream Symtropy policy receipt before advancing the Vision Manifold.

The CLI validates the complete episode before committing output files. Fused HDC vectors are staged in memory; if a later observation fails identity, receipt, timing, sequence or encoder validation, neither the receipts JSON nor any policy-input artifact is written. The receipts output path must not already exist, and the policy-input directory must be absent or empty before processing. This prevents a rejected multi-frame episode from leaving a partial set of apparently usable policy inputs. The staged binary HDC output is capped at 1 GiB per episode; the cap is checked before the commit phase.

The CLI writes:
- one serialized `DrivingObservationReceipt` per delivered observation;
- a sequential predecessor-receipt chain;
- one binary HDC policy-input artifact per sequence under the requested output directory;
- artifact metadata with sequence, dimension, byte length and SHA-256.

The HDC artifact format is an 8-byte unsigned little-endian dimension followed by 16,384 little-endian IEEE-754 f32 components. The binary file hash uses the same dimension-prefixed representation as the bridge receipt's `fused_policy_input_sha256`. The CLI checks the two digests match before writing a successful result. The receipt itself also carries the upstream observation ID, upstream policy-receipt SHA-256, action receipt identity, camera calibration digest and renderer revision.

Only delivered frames are passed through this bridge. Missing/dropped/stale observation events must remain in the episode stream and denominator; they are not replaced or synthesized by the CLI. A successful bridge receipt does not authorize a controller or actuator.

## Claim ceiling

Passing tests establish deterministic fusion and receipt construction over finite supplied frames and vehicle states. They do not establish visual semantic accuracy, depth estimation, a trained camera-conditioned policy, closed-loop driving performance, causality, operational timing, world-model validity or physical safety.


CLI tests additionally verify that a later sequence rollback does not leave partial policy-input artifacts.
