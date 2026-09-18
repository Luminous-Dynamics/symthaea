# Humanoid authoritative execution pipeline

This branch establishes a single composition boundary for humanoid execution.

The target invariant is:

> No cognitive, learned-policy, training, simulation, ROS, HIL, or hardware path may directly obtain actuator authority. Every candidate humanoid command must pass through the same deterministic control and final physical safety boundaries before it is eligible for actuation.

## Execution authority

The intended execution chain is:

```text
EmbodiedGoal / HumanoidTask
        ↓
learned or cognitive proposal
        ↓
HierarchicalHumanoidController
        ↓
whole-body / terrain / contact / floating-base control
        ↓
HumanoidSafetyProjector
        ↓
backend-specific hardware interlock
        ↓
Actuation
```

Cognitive metrics such as prediction error or consciousness estimates may reduce admitted authority, but they do not grant physical authority and they do not bypass deterministic safety constraints.

## Scope of this first tranche

1. Introduce an explicit authoritative execution composition root in `symthaea-humanoid`.
2. Route the public `HumanoidEmbodiment` path through that composition root rather than applying the learned controller directly to the simulator.
3. Preserve existing deterministic hierarchy and final safety projection semantics.
4. Keep backend-specific watchdog, e-stop, over-current, calibration, and servo enforcement in `symthaea-hal` as an additional independent layer.
5. Avoid changing learning algorithms, dynamics solvers, or certification thresholds in the same change.

## Follow-on work

Subsequent changes should make the execution contract morphology-generic, add explicit physical/epistemic authority inputs, introduce state uncertainty, and make the same composition root available to HIL and physical hardware runtimes.
