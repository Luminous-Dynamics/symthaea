# Nixward Episodic Action Identity v1

Status: authored; exact-head focused qualification required before PASS.

## Boundary

Planner-conditioned episodic learning uses exact `ActionCategory` identity.

Human-readable `SystemEpisode::action` remains presentation/legacy data and is not semantic action identity.

```text
Debug/display string
!= planner action identity
!= realized command
!= authorized action intent
!= execution receipt
```

## Compatibility

The public `SystemEpisode` shape is unchanged. `NixEpisodicMemory` stores typed planner identity privately alongside an episode when the caller actually possesses that semantic provenance.

Episodes entering through the compatibility `record(SystemEpisode)` path remain untyped for planner-action purposes. Their display text cannot acquire typed semantics by resemblance.

## Active inference

`NixActiveInference::learn_from_outcome` records the abstract `ActionCategory` directly in both native and non-native builds. It no longer fabricates placeholder package/channel commands or wraps otherwise-unmapped modifying categories as `Custom(..., ReadOnly)`.

`score_action` queries exact typed planner identity. The safety penalty is therefore conditioned on semantic equality, not Rust `Debug` formatting or substring coincidence.

## Nonclaims

This subject does not define command realization, target provenance, execution authority, machine identity/currentness, or production daemon authorization. Those remain separate boundaries under #5042, #5038, #4929, #4973/#4974, and #4948.
