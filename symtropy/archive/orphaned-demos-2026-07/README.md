# Archived orphaned demos (2026-07-03)

Part of the demo-crate rot triage: `crates/apps/` had 19 demo crates but the
root `Cargo.toml` only listed ~4 as real workspace members. Of the other 15,
~11 had a completely empty `[dependencies]` section despite `src/` importing
real crates — they hadn't compiled in a long time.

13 of the 15 were restored (deps rebuilt from actual `src/` imports, added
back to `[workspace] members`). These 2 were archived instead, because
fixing them would require genuinely new logic / new external dependencies,
not just dependency restoration + straightforward renames:

## `symtropy-cognitive-integration-demo`

- Depends on `symtropy_cognitive_bridge::integration::cognitive_physics_bridge_system`,
  but `symtropy-cognitive-bridge` (`crates/bridges/symtropy-cognitive-bridge/`)
  isn't a workspace member and isn't even referenced anywhere else in the
  workspace — it's itself orphaned.
- Calls `RigidBody::new_dynamic(...)`, which doesn't exist on
  `symtropy_physics::body::RigidBody` (only `RigidBody::new` and
  `RigidBody::dynamic_sphere` exist).
- Spawns a bare `BodyHandle` as a Bevy ECS component, but `BodyHandle` isn't
  `#[derive(Component)]`.
- `symtropy-cognitive-bridge`'s own `integration.rs` references
  `PhysicsWorld<2>` without importing it — the dependency crate itself
  doesn't compile as-is.

Only 23 lines of actual demo code; not worth the multi-crate rework.

## `symtropy-xr-showcase`

- Depends on `bevy_openxr`, an external crate never vendored or referenced
  anywhere else in this workspace. Given the concurrent bevy 0.18 -> 0.19
  migration in-flight elsewhere in this repo, an OpenXR integration crate
  almost certainly doesn't have a compatible release yet (XR ecosystem
  crates typically lag mainline bevy releases by months).
- Adding a brand-new, unvetted external XR dependency just to make a
  31-line demo compile is out of scope for a dependency-restoration pass.

If either of these becomes worth reviving, start by fixing the underlying
dependency crate (`symtropy-cognitive-bridge`) or confirming a bevy
0.19-compatible `bevy_openxr` release exists, respectively — not by patching
around the missing pieces in the demo itself.
