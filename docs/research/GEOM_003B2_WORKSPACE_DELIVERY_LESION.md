# GEOM-003B2 — Broadcast-Path-Matched Workspace Delivery Lesion

## Status

Architecture-specific adapter under #2975 / #3123, stacked on GEOM-003B1 exact head `93a5413f6053ceb3f3327f096e0d3c38070a35a8`.

This experiment separates two different interventions that would otherwise be conflated:

1. **GEOM-003B1 emission ablation**: broadcasting is disabled entirely with `enable_broadcasting=false`.
2. **GEOM-003B2 delivery ablation**: broadcasts are still constructed and routed to a registered gate wrapper, but the lesion wrapper does not forward them to its downstream consumer.

Neither intervention is defined as a consciousness manipulation.

## Why B2 exists

B1 removes multiple operations at once:

- broadcast construction;
- recipient iteration/lookup;
- registered-handler invocation;
- downstream consumer work.

Therefore a B1 effect alone cannot distinguish loss of global availability from an upstream workload/timing artifact.

B2 preserves the upstream broadcast path through gate invocation in every condition. Only downstream forwarding is removed in the lesion.

This is a stronger control, but it is **not** total compute matching: the blocked downstream consumer necessarily does less work.

## Quartet

All four conditions use an identical `WorkspaceConfig` with broadcasting enabled.

- **intact**: gate wrapper forwards;
- **lesion**: gate wrapper blocks;
- **sham**: gate wrapper forwards;
- **rescue**: gate wrapper forwards.

GEOM-003A metadata is cloned identically across the quartet.

## Behavioral invariants

For a matched one-cycle probe using the same recipient and submitted content:

- workspace-entry count must be identical across all four conditions;
- broadcast count must be identical across all four conditions and greater than zero;
- gate-wrapper invocation count must be identical across all four conditions and greater than zero;
- downstream delivery must be zero only in the lesion;
- intact, sham, and rescue downstream-delivery counts must match.

The probe uses the existing `memory` recipient because it is part of the default Global Workspace recipient set.

## Interpretation matrix

Later GEOM measurements should compare B1 and B2 rather than treating either one in isolation.

### Similar B1 and B2 effects

If emission ablation and delivery ablation produce reproducibly similar geometric/causal changes, the explanation "B1 changed only because broadcast construction disappeared" becomes less plausible.

### B1 effect but B2 null

This would suggest the B1 result may depend on upstream broadcast-path work, representation construction, timing, or another pre-delivery property rather than recipient availability itself.

### B2 effect but B1 different

This would indicate that recipient delivery has a measurable role, while the broader B1 ablation introduces additional effects that should not be collapsed into the delivery mechanism.

### Both null

Global Workspace broadcasting/delivery, under these interventions and observables, would not have established a measurable GEOM effect.

## Claim boundary

Allowed:

> Blocking recipient delivery while preserving workspace entry, broadcast emission, recipient lookup, and gate-wrapper invocation changed measured GEOM observables by X.

Not allowed:

> Recipient delivery is consciousness.

Not allowed:

> Similar B1/B2 effects prove Global Workspace Theory.

Not allowed:

> Any B1/B2 result establishes a gravity-consciousness physical connection.

## Promotion gate

B2 remains blocked on qualification of GEOM-001, GEOM-002A/B/C, GEOM-003A, and GEOM-003B1. Its own hosted CI can qualify implementation integrity, but not scientific claims.
