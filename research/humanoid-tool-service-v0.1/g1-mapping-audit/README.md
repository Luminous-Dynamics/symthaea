# G1 decoupled_v1 mapping audit

A dependency-free Rust reference for the pinned HumanoidToolBench G1 policy mapping.

## What it does

- Projects 43 named G1 joint positions into the exact 32-value benchmark policy state.
- Resolves joints by explicit name rather than assuming that raw array positions are identity.
- Keeps the previous base-height command at state index 31 typed as command history, not physical measurement.
- Decodes and round-trips all 36 `decoupled_v1` action values.
- Applies the documented left-hand flat/natural-order permutation exactly once.
- Keeps torso RPY, base height, navigation velocity, turn flag and absolute target yaw semantically distinct.
- Rejects wrong dimensions, duplicate/unknown joints and non-finite values.

## Run

```bash
cargo fmt --manifest-path research/humanoid-tool-service-v0.1/g1-mapping-audit/Cargo.toml -- --check
cargo test --manifest-path research/humanoid-tool-service-v0.1/g1-mapping-audit/Cargo.toml --locked
```

The crate has no third-party dependencies and includes a committed minimal lockfile. It is intentionally isolated from the main workspace.

## Exact non-goals

This is not a G1 driver, UEC v2 platform implementation, safety projector, trajectory planner, transport, physical calibration, hardware binding, or actuator command path. It does not transform position targets into Symthaea torque/PWM values and must not be routed to the current 21-joint PCA9685 HAL.

## Raw joint ordering traced to the pinned upstream producer

The 43-value order is now traced end-to-end at `SNU-PI/HumanoidToolBench@3fec9bd36933a1648ff7f95468c983f3452243a0`:

1. `SonicLocoManip._get_obs` builds `joint_qpos` from `list(self.mujoco.get_robot_qpos().values())`.
2. `MujocoEngine.get_robot_qpos` delegates to the task robot's `get_robot_qpos`.
3. `G1Sonic.get_robot_qpos` iterates `self.joints.items()`.
4. `G1Sonic.setup_control` populates `self.joints` by iterating `self.joint_names`, the exact 43-name `WHOLE_BODY_JOINTS` profile.

Python dictionary insertion order preserves this order. The mapping manifest now records all three producer files and their pinned Git blob IDs. `project_pinned_qpos` is a convenience entry point for exactly this source-pinned order; `project_policy_state` remains available when callers provide explicit name/value pairs and can safely handle reordering.

This resolves the raw-qpos order blocker for the pinned simulation source. It does **not** qualify a UEC v2 platform adapter, simulator episode, real-G1 driver, physical calibration, safety case or actuator authority. UEC v2 integration remains gated on the canonical APIs and runtime conformance.

Upstream reference: `SNU-PI/HumanoidToolBench@3fec9bd36933a1648ff7f95468c983f3452243a0`.
