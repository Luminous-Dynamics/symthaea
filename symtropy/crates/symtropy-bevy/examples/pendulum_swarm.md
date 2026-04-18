# `pendulum_swarm` — Tier 1 showcase design

**Status:** scaffolded 2026-04-18, awaiting implementation.
**Target:** Phase 0.6 "Demo & Visibility" keystone per [ROADMAP](../../../ROADMAP.md)
and [GAME_ENGINE_COMPONENTS.md](../../../docs/GAME_ENGINE_COMPONENTS.md).

---

## What this demo shows

A 10×10 grid of pendulums. Each pendulum is a physics body hinged to a fixed
pivot point above it. **Phi coupling modulates damping** — low-Phi pendulums
are chaotic and uncoordinated; high-Phi pendulums synchronize through the
field.

Click anywhere to "shock" the nearest pendulum (inject angular velocity).
Watch coherence propagate across the grid as Phi climbs in the neighborhood
and damping drops.

Color encodes Phi per pendulum (cool → warm LUT).

**The thing a visitor should feel:** "Consciousness isn't a metaphor in this
engine — Φ literally changes how the physics behaves."

---

## Why this demo, not another

Three options considered:

| Demo | Showcases | Rejected because |
|---|---|---|
| 4D n-body gravity | ND physics differentiator | ND rendering isn't ready (Phase 2) |
| Robotics embodiment | Symthaea adapter + Phi gating | Requires live Symthaea cognitive loop; heavy deps |
| **Pendulum swarm** | **Phi ↔ physics coupling, Bevy integration, visual emergence** | **Picked: self-contained, compelling, ~300 LOC** |

The pendulum swarm uses only `symtropy-bevy` (AGPL) + `bevy`. No Symthaea, no
Mycelix, no subprocess IPC, no network. One `cargo run --example
pendulum_swarm` away from a visitor seeing what the engine does.

---

## Architecture

### Dependencies
Already in `symtropy-bevy`'s dep graph:
- `symtropy-bevy` — the plugin
- `symtropy-physics` (re-exported) — ND rigid bodies
- `symtropy-consciousness-physics` (re-exported) — `ConsciousnessField`, `ConsciousnessInputs`
- `symtropy-math` (re-exported) — `Point<D>`
- `bevy = "0.18"` — rendering, input, time

No new deps required. This matters: the example must run by cloning the repo
and running one command.

### Dimensionality
**2D (`SymtropyPhysicsPlugin::<2>`).** Reasons: grid of pendulums is inherently
planar, Bevy 2D rendering is ready today (3D scene pipeline is Phase 2), and
the visual is cleaner without perspective.

### Scene layout
- Window: 1280×720, dark background
- 10×10 grid, pendulums spaced ~64 px
- Each pendulum: a rigid body (circle, mass 1.0, radius 10 px) hinged to a
  fixed pivot 80 px above it. Gravity `[0, -9.81]` scaled by a world-units
  factor (~100 px/m).

### Core components (Bevy entities)
```rust
#[derive(Component)]
struct Pendulum {
    body: BodyHandle,
    pivot: Vec2,        // screen coords of the fixed pivot
    base_damping: f32,  // resting linear_damping when Phi is low
}

#[derive(Component)]
struct PendulumVisual;  // the sphere sprite, one per pendulum
```

### Systems (Bevy `Update` schedule)

1. **`spawn_swarm` (Startup)** — builds 100 pendulums:
   - For each (i, j) in 10×10: compute pivot position, spawn rigid body at
     `pivot + (0, -60)` (pendulum hanging), register with `ConsciousnessField`
     via `field.register(handle, max_energy=100.0, sanctuary_radius=32.0)`,
     wire a `HingeJoint` or `DistanceConstraint` from body → pivot.
   - (Note: `HingeJoint` in `symtropy-physics` constrains rotation plane.
     For 2D planar swinging we actually want `DistanceConstraint` pinning
     the body to a fixed point — simpler and gives ideal pendulum motion.)
   - Spawn matching Bevy sprite entity with `Pendulum`, `PendulumVisual`,
     and default `Transform`.

2. **`update_phi_from_neighborhood` (Update)** — for each pendulum:
   - Compute local angular-velocity variance across the 3×3 neighborhood.
     Low variance → coherent → high Phi. High variance → chaotic → low Phi.
   - Map variance to `ConsciousnessInputs` (all 8 fields set from a single
     coherence scalar; this is a deliberate simplification for demo clarity).
   - Call `field.update_entity(handle, &inputs, Point::new([pos_x, pos_y]))`.

3. **`phi_modulates_damping` (Update)** — for each pendulum:
   - Read `field.phi(handle)` → `phi`
   - Linear `damping = base_damping * (2.0 - phi)` — high-Phi pendulums
     resist stops, low-Phi damp fast. Write back to the rigid body's
     `linear_damping` field.

4. **`color_by_phi` (Update)** — for each `PendulumVisual`:
   - Map `phi ∈ [0, 1]` to a color via a simple viridis-style LUT (or
     just `Color::hsl(240.0 - phi * 240.0, 1.0, 0.5)` — blue→red).
   - Write to the sprite's material/color.

5. **`shock_on_click` (Update)** — on mouse click:
   - Raycast from cursor to 2D world space
   - Find nearest pendulum within `50 px`
   - Inject angular velocity (or linear impulse) via `body_mut`

6. **`sync_transforms`** — already provided by `SymtropyPhysicsPlugin`.

### File layout
- `examples/pendulum_swarm.rs` — single file, all of the above (~300 LOC)
- Optional `examples/pendulum_swarm.md` — this doc, becomes README after impl

---

## Implementation steps (for future-me)

Sequential, smallest-commit-per-step:

1. **Hello Bevy + physics plugin** — `SymtropyPhysicsPlugin::<2>::default()`,
   empty scene, dark background, runs at 60 fps. Verify window opens.
2. **One pendulum** — hand-compute one pivot + body + constraint, verify it
   swings under gravity. Debug gizmo for the constraint line.
3. **10×10 grid** — loop-spawn, verify they all swing independently.
4. **Phi update from variance** — compute neighborhood variance, plug into
   `ConsciousnessField`. Print one cell's Phi to stdout. Verify it changes.
5. **Phi → damping coupling** — verify visually that high-Phi pendulums
   oscillate longer.
6. **Color by Phi** — sprite tinting.
7. **Shock on click** — input + impulse.
8. **Polish** — smoother color LUT, trails (cheap), on-screen Phi counter.

Stop conditions: if any step takes >1 hour, pause and ask whether the
scope is right. Expected total: 4-6 hours of focused work.

---

## Success criteria

The demo ships when ALL of:

- [ ] `cargo run --example pendulum_swarm --release` opens a window at 60 fps
- [ ] 100 pendulums visible, swinging
- [ ] Shocking a single pendulum produces a visible propagation of high-Phi
      color through neighbors
- [ ] Screenshot captured, saved to `examples/pendulum_swarm_screenshot.png`
- [ ] README next to the example (this file, rewritten as a user-facing README)
- [ ] Book chapter `symtropy/book/src/quickstart.md` updated to link this as the
      first thing a new user should run

Nice-to-haves (not gating):
- Audio hum that rises with coherence
- Real-time Phi heatmap overlay toggle
- Adjustable shock radius via keyboard

---

## Known landmines

1. **Constraint type choice.** `HingeJoint<D>` constrains rotation to a
   bivector plane — in 2D there's only one plane so it's effectively "no
   rotation constraint." What we actually want is: a body's position
   constrained to a circle around a fixed point. `DistanceConstraint<D>`
   between the body and a static "pivot body" (mass = infinity, type =
   `BodyType::Static`) is the correct primitive. Verify before building
   the grid.

2. **Phi-neighborhood compute cost.** 100 pendulums × 9 neighbors = 900
   variance computations per frame. Trivial, but if we push to 1000
   pendulums later this becomes LBVH territory. Keep the neighborhood
   logic in a function so it can be swapped for a spatial hash.

3. **Bevy 0.18 `Message` vs `Event` API.** Per prior session memory, Bevy
   0.18 renamed `Event` → `Message` across the board. `EventWriter →
   MessageWriter`, `add_event → add_message`, etc. Input events now come
   via `MessageReader<MouseButton>`. Grep existing code for the pattern
   rather than writing from Bevy docs.

4. **`Time` unreliability under `MinimalPlugins` in tight loops** — not
   a risk here (this demo uses `DefaultPlugins`, not `MinimalPlugins`)
   but note that `bevy::DefaultPlugins` DOES advance `Time` correctly.

---

## What this unlocks

Once pendulum_swarm ships:

1. **README animated GIF.** Loom recording of the shock-propagation
   behavior. Top of symtropy's README.
2. **Book chapter.** A "your first Symtropy scene" walkthrough using this
   example.
3. **The recipes directory.** Once we have one complete example, the
   9 gap-recipe docs in `docs/recipes/` (`bevy_tnua`, `bevy_hanabi`, etc.)
   have a template to follow.
4. **Comparative benchmarks.** The 100-pendulum scene is a natural
   first benchmark vs Rapier/bevy_xpbd — "how does Phi-coupled physics
   compare to uncoupled on the same workload?"
5. **Twitter/BlueSky post.** The 30-second GIF is a legitimate first
   marketing asset.

---

## Kickstart for next session

```bash
cd /srv/luminous-dynamics/symtropy/crates/symtropy-bevy
# Implementation goes in:
$EDITOR examples/pendulum_swarm.rs

# Iterate with:
cargo run --example pendulum_swarm

# When ready to polish:
cargo run --example pendulum_swarm --release
```

Read this doc, read step 1 of "Implementation steps," write the hello-Bevy
version, commit, then step 2. Don't try to do all eight in one commit.
