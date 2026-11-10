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
- Camera/vehicle timestamp skew is bounded.
- The first timestep is supplied as a configuration constant; subsequent timesteps are derived from capture timestamps.
- A configured maximum inter-frame gap and maximum camera/vehicle timestamp skew are enforced.
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

## Claim ceiling

Passing tests establish deterministic fusion and receipt construction over finite supplied frames and vehicle states. They do not establish visual semantic accuracy, depth estimation, a trained camera-conditioned policy, closed-loop driving performance, causality, operational timing, world-model validity or physical safety.
