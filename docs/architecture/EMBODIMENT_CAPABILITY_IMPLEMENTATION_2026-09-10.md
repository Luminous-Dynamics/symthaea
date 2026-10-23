# Embodiment capability composition — implementation notes

## Implemented in this tranche

- `symthaea-core::platform_descriptor` is the canonical typed description layer.
- Existing `EmbodimentPlatform` variants remain stable presets and expose `descriptor()`.
- `PlatformDescriptor::composed()` can describe bodies with no enum variant.
- Platform, role, capability, evidence state, safety criticality, and autonomy class are represented separately.
- `PlatformRegistryDescriptorExt` adds read-only capability discovery over the existing registry without constructing bridges.
- `src/domain::PlatformCapabilityProfile` is now a compatibility projection of descriptor strengths rather than a second hand-written platform table.

## Authority boundary

Descriptors are descriptive only. They contain no motor command, actuator handle, capability grant, authorization token, or runtime execution path. `MotorSafetyLevel`, `SafeFallback`, platform-specific control, HAL, and external authorization remain unchanged.

`DescriptorEvidence::PresetHeuristic` is intentionally the evidence status of built-in presets. A preset's declared capability must not be interpreted as measured or qualified hardware performance.

## Next platform families

Only after this descriptor tranche is verified should new physical substrates be added. Proposed order:

1. general spacecraft bus;
2. planetary mobility / rover substrate;
3. heavy construction and mining equipment;
4. autonomous logistics carrier;
5. lander;
6. rail and surface-marine families.

Each should justify a dedicated runtime plugin through distinct dynamics, control, fallback, hardware integration, or qualification needs. Otherwise it should remain a composed descriptor over existing primitives.
