# G1 UEC v2 consumer prototype

**Status:** dependent, non-actuating research prototype.  
**Parent:** HUM-G1-UEC-ADAPTER-001 #12681.  
**Canonical contract dependency:** UEC v2 #1274 / draft PR #7674.  
**Base dependency:** exact UEC consumer pilot head ad7ff8faff84164a0461d8371c7d5521ef2c339c. This package imports symthaea_core::embodiment_v2 from that branch.

## What it implements

- A read-only EmbodimentV2Adapter for the pinned HumanoidToolBench G1 Sonic profile.
- Exactly 43 source-ordered raw G1 joint_qpos values as typed UEC v2 ObservationValueV2 entries, with named addresses, radians and source-provided clock fields.
- Explicit instance, profile, morphology, schema, source and backend-code provenance.
- Monotonic source sequence/timestamp checks when those fields are actually supplied. Missing time coordinates remain missing.
- A 36-value decoupled_v1 policy-output recorder that emits only ActionStageV2::Requested, with exact named addresses and units.
- Negative/golden tests for source order, dimensions, non-finite values, clock identity, monotonicity, left-hand legacy order, base-height units, navigation turn flag and read-only stage behavior.

The upstream policy's 32-value state is a derived input, not the whole physical observation. This adapter intentionally exposes the 43 typed raw joint facts. The separate mapping audit contract is documented in the parent HUM-TOOL-SERVICE programme; unifying both code surfaces is a follow-up once both review branches are integrated. In particular, state index 31 (previous base-height command) remains command history and is not inserted as a physical sensor observation.

## Explicitly out of scope

- RGB head image transport (a separate modality/provider with dimensions, encoding, timestamps and provenance).
- Building missing root pose, base velocity, feet/hand pose, force/torque, tactile, contact or calibration values.
- Image/action policy execution, simulator stepping, controller projection, transport publication or physical actuation.
- Safety admission, watchdog, fallback implementation or HIL/physical qualification.
- Claiming Symthaea's DMC21/Dexterous53 morphology is equivalent to Unitree G1/Dex3.

No actuator capability ID is declared. The descriptor fallback profile explicitly identifies this consumer as read-only/non-actuating, not as a qualified physical fallback.

## Source identity

HumanoidToolBench source commit: 3fec9bd36933a1648ff7f95468c983f3452243a0.

Raw producer chain:
1. SonicLocoManip._get_obs materializes joint_qpos from ordered get_robot_qpos().values().
2. MujocoEngine.get_robot_qpos delegates to G1Sonic.get_robot_qpos.
3. G1Sonic.setup_control creates its joint dictionary from WHOLE_BODY_JOINTS; get_robot_qpos returns that insertion order.

This pins the source schema. It does not prove a runtime is currently executing this exact commit.

## Validation status

Seven focused Rust tests are authored. The editing environment has no Rust toolchain, so no local compile, rustfmt, Clippy or test execution is claimed. The focused workflow is intended to perform those checks on GitHub. Queued or skipped is not a pass. This consumer remains dependent on the draft UEC v2 branch and is not merge-ready to main.
