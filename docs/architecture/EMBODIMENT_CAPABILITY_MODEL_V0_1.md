# Embodiment Capability Model v0.1

Status: design baseline for capability-composed embodiment.

## Problem

`EmbodimentPlatform` is useful as a stable preset identity, but its variants mix morphology, environment, mission role, and interaction surface. Adding every new machine as another enum variant does not scale and makes capability discovery difficult.

## Decision

Keep `EmbodimentPlatform` as a backward-compatible preset identity. Add a typed, descriptive `PlatformDescriptor` in `symthaea-core::embodiment` that separates morphology, mobility, operating environment, manipulation, sensing, energy, communications, mission roles, safety criticality, and autonomy class.

Descriptors are descriptive only. They do not grant actuator authority, bypass `MotorSafetyLevel`, or replace platform-specific `SafeFallback` behavior.

## Separation of concerns

- Platform: what body/profile this is.
- Role: what task it is suitable for.
- Capability: what it can physically or informationally do.
- Authority: what it is currently permitted to do.

Capability discovery must never be treated as authority.

## Compatibility

Existing `EmbodimentPlatform` variants remain intact. Each receives a canonical descriptor preset through `EmbodimentPlatform::descriptor()`.

The legacy `src/domain::PlatformCapabilityProfile` becomes a lossy compatibility projection of the typed descriptor rather than maintaining an independent hand-written platform table.

## Extension rule

Prefer composing a descriptor for a new machine. Add a new `EmbodimentPlatform` variant / platform crate only when the machine requires materially distinct physics, control, safety fallback, hardware integration, or qualification evidence.

Likely future physical substrate families include general spacecraft buses, planetary mobility, heavy construction/mining equipment, autonomous logistics carriers, landers, rail vehicles, and marine surface craft. Those should be added only as evidence-backed implementations, not as speculative enum growth.
