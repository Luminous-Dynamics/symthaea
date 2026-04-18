# `pendulum_swarm` — Tier 1 showcase design

**Status:** implementation steps 1-7 landed 2026-04-18 (commits
`35f6537981` through `26c41f11ae`). Demo compiles and runs; visual
parameter tuning + Step 8 polish + screenshot capture remain.
**Target:** Phase 0.6 "Demo & Visibility" keystone per [ROADMAP](../../../ROADMAP.md)
and [GAME_ENGINE_COMPONENTS.md](../../../docs/GAME_ENGINE_COMPONENTS.md).

---

## What this demo shows

A 10×10 grid of pendulums. Each pendulum is a physics body hinged to a fixed
pivot point above it. **Phi controls dissipation in a positive-feedback loop**
— a shocked pendulum has high local kinetic energy, which raises Phi in its
neighborhood, which *suppresses damping* there, which lets the energy keep
oscillating, which keeps Phi high. Outside that region, damping is high and
any residual motion dies in ~1 second.

Click anywhere to "shock" the nearest pendulum (inject angular velocity).
Watch a *sustained-motion wave* spread across the grid as Phi propagates
through neighborhoods that have inherited some kinetic energy from the shock.
Behind the wavefront, pendulums fall silent again as their KE bleeds out and
Phi drops.

Color encodes Phi per pendulum (cool → warm LUT). The visible result is a
travelling band of brightly-coloured, vigorously-swinging pendulums on a
darker background of stilled ones.

**The thing a visitor should feel:** "Consciousness isn't a metaphor in this
engine — Φ literally changes how the physics behaves." The shock-spreads-as-
sustain-wave reads in 2-4 seconds and is empirically reliable (see Step 0.5
spike notes; the Kuramoto phase-lock approach we tried first did not produce
clean phase-lock with simple linear-velocity kicks on equal-frequency PBD
pendulums).

---

## Why this demo, not another

Three options considered:

| Demo | Showcases | Rejected because |
|---|---|---|
| 4D n-body gravity | ND physics differentiator | ND rendering isn't ready (Phase 2) |
| Robotics embodiment | Symthaea adapter + Phi gating | Requires live Symthaea cognitive loop; heavy deps |
| **Pendulum swarm** | **Phi ↔ physics coupling, Bevy integration, visual emergence** | **Picked: self-contained, compelling, ~330 LOC** |

The pendulum swarm uses only `symtropy-bevy` (AGPL) + `bevy`. No Symthaea, no
Mycelix, no subprocess IPC, no network. One `cargo run --example
pendulum_swarm` away from a visitor seeing what the engine does.

---

## Architecture

### Dependencies
Already in `symtropy-bevy`'s dep graph:
- `symtropy-bevy` — provides `SymtropyPhysicsPlugin<D>` and the `SymtropyPhysics<D>`
  Resource (which bundles `world: PhysicsWorld<D>` + `field: ConsciousnessField<D>`)
- `symtropy-bevy-core` — provides `PhysicsBody` Bevy `Component` (re-exported)
- `symtropy-physics` (re-exported) — `BodyHandle`, `RigidBody`, `DistanceConstraint`,
  `BodyType` (Static/Dynamic/Kinematic), joint types
- `symtropy-consciousness-physics` (re-exported) — `ConsciousnessField`, `ConsciousnessInputs`
- `symtropy-math` (re-exported) — `Point<D>`
- `bevy = "0.18"` — rendering, input, time

No new deps required. This matters: the example must run by cloning the repo
and running one command.

### Resource access pattern
The plugin owns one `SymtropyPhysics<D>` Resource. User systems take
`ResMut<SymtropyPhysics<D>>` and reach into `physics.world` and
`physics.field`. Example:

```rust
fn my_system(mut physics: ResMut<SymtropyPhysics<2>>) {
    let h = physics.world.add_sphere(Point::new([0.0, 0.0]), 10.0, 1.0);
    physics.field.register(h, 100.0, 32.0);
    let phi = physics.field.phi(h);
}
```

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
Each of the 100 pendulum entities carries THREE components:

```rust
#[derive(Component)]
struct Pendulum {
    body: BodyHandle,                // the swinging dynamic body
    pivot_body: BodyHandle,          // the static "pivot" body (mass=∞)
    pivot_pos: Vec2,                 // screen coords of the pivot, for input/visuals
    neighbors: Vec<BodyHandle>,      // up to 8 grid neighbors for Phi diffusion
}

#[derive(Component)]
struct PendulumVisual;  // marker for the sprite

// PLUS: PhysicsBody (re-exported by symtropy-bevy) — REQUIRED for sync_transforms
//       to copy world.body(handle).position() into the entity's Transform.
//       Construct with: PhysicsBody::new(handle, visual_radius)
```

### Pivot construction (no `BodyType::Static` constructor)

Static bodies are built via `RigidBody::static_body(handle, position, collider)`
then handed to `world.add_body(body)`. For a pendulum pivot:

```rust
use symtropy_math::{Point, Sphere};                       // Sphere is in symtropy_math
use symtropy_physics::{BodyHandle, RigidBody};

let pivot_body = RigidBody::<2>::static_body(
    BodyHandle(0),                                         // re-assigned by world.add_body
    Point::new([px, py]),
    Box::new(Sphere::new(Point::origin(), 1.0)),           // Sphere::new(center, radius)
);
let pivot_handle = physics.world.add_body(pivot_body);
// Optional: zero collision_mask on the pivot if you want bobs to swing
// through it without contact resolution:
//   physics.world.body_mut(pivot_handle).unwrap().collision_mask = 0;
```

### Constraint construction (plain struct, no `new()`)

```rust
use symtropy_physics::constraint::DistanceConstraint;

physics.world.add_constraint(Box::new(DistanceConstraint::<2> {
    body_a: pivot_handle,
    body_b: bob_handle,
    rest_length: 60.0,  // arm length in world units (px)
    stiffness: 1.0,
}));
```

### Systems

The plugin's physics_step + sync_transforms run on **`FixedUpdate`** (60 Hz
default). User systems split:

**FixedUpdate (deterministic, 60 Hz):**
- `update_phi_from_neighborhood` — must run BEFORE physics_step picks up
  the field state. Acceptable to run after on some ticks (one-tick delay
  invisible at 60 Hz).
- `phi_modulates_damping` — writes per-body `linear_damping` from Phi;
  same ordering note (a one-tick stale damping value is invisible at 60 Hz).

**Update (variable-rate, render-driven):**
- `color_by_phi` — visual only, can interpolate.
- `shock_on_click` — input read; impulse application via
  `ResMut<SymtropyPhysics<2>>` lands in next FixedUpdate.

Detail per system:

1. **`spawn_swarm` (Startup)** — for each (i, j) in 10×10:
   - Compute pivot position; build a static pivot body (see snippet above).
   - Spawn the dynamic bob at `pivot + (0, -60)` via `world.add_sphere(...)`.
   - `field.register(bob_handle, 100.0, 32.0)`.
   - `world.add_constraint(Box::new(DistanceConstraint { ... }))`.
   - Spawn the Bevy entity bundling `(Pendulum { body, pivot_body, pivot_pos,
     neighbors }, PendulumVisual, PhysicsBody::new(bob_handle, 10.0),
     Sprite { ... }, Transform::default())`.
   - After all 100 spawned, do a second pass to fill `neighbors` (handles
     aren't known until all bobs are added).

2. **`update_phi_from_neighborhood` (FixedUpdate)** — for each pendulum:
   - Read `physics.world.body(handle).linear_velocity` for self + neighbors.
   - Compute neighborhood variance (low variance → coherent → high Phi).
   - Map to `ConsciousnessInputs` (all 8 fields set from a single coherence
     scalar — deliberate simplification for demo clarity).
   - `physics.field.update_entity(handle, &inputs, Point::new([pos.x, pos.y]))`.

3. **`phi_modulates_damping` (FixedUpdate)** — for each pendulum:
   - `let phi = physics.field.phi(handle)` → `f64` actually in [0, ~0.314].
     The `MasterConsciousnessEquation` response to uniform unit
     `ConsciousnessInputs` is heavily compressed (see landmine
     "Phi magnitude ≠ [0, 1]"). We MUST normalize against the
     empirical maximum before mapping to damping, or the dynamic
     range collapses from 500× to ~1.5×.
   - Named constant: `const PHI_NORMALIZE: f64 = 0.314;`  // see spike
   - Map normalized Phi to per-body damping:
     ```rust
     const LOW_DAMP: f64 = 0.001;   // phi≈max → essentially conservative
     const HIGH_DAMP: f64 = 0.5;    // phi=0 → dies in ~4 sec via PBD sleep

     if let Some(body) = physics.world.body_mut(handle) {
         let phi = physics.field.phi(handle);
         let phi_norm = (phi / PHI_NORMALIZE).clamp(0.0, 1.0);
         body.linear_damping = HIGH_DAMP + (LOW_DAMP - HIGH_DAMP) * phi_norm;
     }
     ```
   - The positive-feedback loop: shocked pendulum → high KE → low
     angular-velocity variance in neighborhood → high coherence →
     high Phi → low damping → energy persists → still high KE.
     Outside the shocked region: low KE → high variance → low Phi →
     high damping → PBD sleep freezes the bob in ~4 sec.
   - Note: do NOT use anti-damping (negative values) — PBD goes unstable.
     The 500× damping ratio (after normalization) plus the shock energy
     is enough for the visual.

4. **`color_by_phi` (Update)** — query `(&Pendulum, &mut Sprite)`:
   - Read `physics.field.phi(p.body)`; map to color via
     `Color::hsl(240.0 - phi * 240.0, 1.0, 0.5)` (blue→red).
   - Write to `sprite.color`.

5. **`shock_on_click` (Update)** — on `MouseButton::Left` press:
   - Read cursor position from `Window` → world coords.
   - Find nearest pendulum within 50 px (linear scan; 100 entities is fine).
   - `physics.world.body_mut(handle).linear_velocity[0] += 50.0`
     (or apply via the impulse pattern from system 3).

6. **`sync_transforms` (FixedUpdate, plugin-owned)** — already provided by
   `SymtropyPhysicsPlugin`. Queries `(&PhysicsBody, &mut Transform)` —
   THIS is why every pendulum entity needs the `PhysicsBody` component.

### File layout
- `examples/pendulum_swarm.rs` — single file, all of the above (~330 LOC)
- Optional `examples/pendulum_swarm.md` — this doc, becomes README after impl

---

## Implementation steps (for future-me)

Sequential, smallest-commit-per-step:

0. **Constraint spike — ALREADY RUN, RESULT: PASS.** No need to redo
   unless you've changed `symtropy-physics`. Reference run committed
   only as result-summary; the spike `.rs` file was deleted.

   Setup: pivot at origin, bob at (1.0, 0.0), arm length 1.0, gravity
   (0, -9.81), `DistanceConstraint<2>` with `stiffness: 1.0`, dt=1/240,
   no damping, simulated 6 s.

   ```rust
   // Key API shapes proven by the spike:
   use symtropy_math::{Point, Sphere};                       // Sphere lives in symtropy_math
   use symtropy_physics::{BodyHandle, PhysicsWorld, RigidBody};
   use symtropy_physics::constraint::DistanceConstraint;

   let pivot = world.add_body(RigidBody::<2>::static_body(
       BodyHandle(0),                                        // re-assigned by add_body
       Point::new([0.0, 0.0]),
       Box::new(Sphere::new(Point::origin(), 0.01)),         // Sphere::new(center, radius)
   ));
   let bob = world.add_sphere(Point::new([1.0, 0.0]), 0.05, 1.0);  // (pos, radius, mass)
   world.add_constraint(Box::new(DistanceConstraint::<2> {
       body_a: pivot, body_b: bob, rest_length: 1.0, stiffness: 1.0,
   }));
   for _ in 0..(6*240) { world.step(1.0/240.0); }
   ```

   Empirical results:
   - `dist(pivot, bob) = 1.0000` for all 1,440 steps — constraint
     rigidly holds (no drift in the constrained dimension).
   - `x ∈ [-0.998, 1.000]` — full ±L swing, bob traces a clean
     arc through (0, -1).
   - Measured period **2.296 s**. The small-angle formula
     `T₀ = 2π√(L/g) = 2.006 s` is NOT what you measure at 90°
     amplitude — the exact large-angle period is `T₀ · K(sin 45°) / (π/2)
     ≈ 2.37 s`. Measured 2.30 s is within 3% of physics. This is
     correct behavior, not a bug.
   - Energy drift: amplitude shrinks ~7% over 3 periods at default
     PBD solve iterations. Fine for a demo — high-Phi pendulums use
     `linear_damping ≈ 0.001` (essentially conservative) so the demo's
     visible shock-region stays vigorous; surrounding low-Phi pendulums
     use `linear_damping ≈ 0.5` (intentionally damped fast).

   **Decision: DistanceConstraint locked in.** Skip Step 0 unless you
   changed the constraint solver.

1. **Hello Bevy + physics plugin** — `SymtropyPhysicsPlugin::<2>::default()`,
   empty scene, dark background, runs at 60 fps. Verify window opens.
2. **One pendulum** — hand-compute one pivot + body + constraint, verify it
   swings under gravity. Debug gizmo for the constraint line.
3. **10×10 grid** — loop-spawn, verify they all swing independently.
4. **Phi update from variance** — compute neighborhood variance, plug into
   `ConsciousnessField`. Print one cell's Phi to stdout. Verify it changes.
5. **Phi → damping coupling — ALREADY VALIDATED.** Spike
   `_spike_damping.rs` (deleted, not committed) tested two pendulums
   shocked equally with the proposed damping range:
   - LOW_DAMP=0.001: retains **78.7%** of initial total mechanical
     energy at t=5s. Visibly still vigorously swinging.
   - HIGH_DAMP=0.5: retains **13.0%** at t=5s. PBD sleep mechanism
     freezes the bob at ~20° around t=4s — visually a "stopped"
     pendulum hanging slightly off-center.
   - 6.1× energy ratio at t=5s, but the visual contrast is
     stronger than the number suggests because one pendulum is
     FROZEN and the other is in full ±60° swing.

   This is the visual hook empirically confirmed. Just wire the system,
   no further verification needed unless the damping range or sleep
   mechanism changes upstream.
6. **Color by Phi** — sprite tinting.
7. **Shock on click** — input + impulse.
8. **Polish** — smoother color LUT, trails (cheap), on-screen Phi counter.

Stop conditions: if any step takes >1 hour, pause and ask whether the
scope is right. Expected total: 4-6 hours of focused work.

---

## Success criteria

The demo ships when ALL of:

- [ ] `cargo run --example pendulum_swarm --release` opens a window at 60 fps
      (physics step measured at 1.4 ms / 60 ticks for 100 bodies +
      constraints on the dev machine — 8.7% of the 16.67 ms budget,
      leaving ~91% for rendering, Phi compute, and input. Guarded by
      `tests/pendulum_swarm_invariants.rs::scene_100_pendulums_fits_60hz_budget_release`.)
- [ ] 100 pendulums visible, swinging
- [ ] Shocking a single pendulum produces a visible *sustained-motion*
      band that spreads through neighbors as Phi climbs locally — the
      shocked region keeps swinging vigorously while the surrounding
      grid (low-Phi, high-damping) is visibly stilled
- [ ] Screenshot captured, saved to `examples/pendulum_swarm_screenshot.png`
- [ ] README next to the example (this file, rewritten as a user-facing README)
- [ ] Book chapter `symtropy/book/src/quickstart.md` updated to link this as the
      first thing a new user should run

Nice-to-haves (not gating):
- Audio hum that rises with coherence
- Real-time Phi heatmap overlay toggle
- Adjustable shock radius via keyboard

---

## Regression guards

Four empirical constants in this doc are CI-protected by
`tests/pendulum_swarm_invariants.rs`. Each test is named for the
section it guards. Run them anytime you change the physics engine
or consciousness equation:

```bash
# From symtropy/ directory:
just test-pendulum-swarm

# Or directly:
cargo test -p symtropy-bevy --test pendulum_swarm_invariants --release
```

The tests must run in `--release` mode — the perf guard is noise
in debug builds and skips itself there. If a test fails, its
error message points at which spike to re-run and which section
of this doc to update.

---

## Known landmines

0. **Phi magnitude is NOT in [0, 1].** The `MasterConsciousnessEquation`
   in `symtropy-consciousness-physics` 0.1.0 heavily compresses uniform
   inputs. Empirically measured response to uniform `ConsciousnessInputs`
   (all 8 fields = coherence scalar c):

   | c | phi | | c | phi |
   |---|---|---|---|---|
   | 0.00 | 0.000 | | 0.60 | 0.116 |
   | 0.25 | 0.019 | | 0.75 | 0.180 |
   | 0.50 | 0.080 | | 1.00 | **0.314** |

   Non-uniform patterns do NOT exceed this max — we tried "only phi=1"
   (phi_out=0), "only working_memory+attention=1" (phi_out=0),
   "all=1 except phi=0" (phi_out=0.131). Max achievable phi under any
   input pattern in this demo appears to be ~0.314.

   **Warmup caveat.** The 0.314 figure is STEADY-STATE. A single
   `update_entity()` call after `register()` gives ~0.256 (measured
   by the regression test in `tests/pendulum_swarm_invariants.rs`).
   `EntityConsciousness` carries memory that converges over ~60
   frames of sustained input. The demo runs at 60 Hz so this
   is invisible — by the time any visitor looks, phi has settled.
   But: if you bring up 100 pendulums with identical high inputs and
   immediately read phi, you'll see values in the low-0.2s for the
   first second. Color + damping will visibly "warm up" during that
   second. Either accept this (arguably on-theme) or fade-in the
   demo over the first second with a gated shock.

   **Consequence:** code that treats `field.phi()` as if it reached 1.0
   will see only ~30% of intended dynamic range. Normalize via
   `phi_norm = (phi / 0.314).clamp(0.0, 1.0)` before mapping to damping
   or any other physics parameter. See system 3 for the formula.

   **If `MasterConsciousnessEquation` changes upstream**, re-measure
   with the spike logic in the memory record of this session. The
   0.314 constant is a snapshot of the equation's current shape, not
   a mathematical invariant.


1. **Constraint type choice.** De-risked by Step 0 above. The doc assumes
   `DistanceConstraint<2>` between body and a static "pivot body" (mass =
   infinity, type = `BodyType::Static`) — in 2D, `HingeJoint<D>` constrains
   rotation to a bivector plane, which collapses to "no constraint" in 2D.
   But that's an analysis, not a measurement. Run Step 0 before trusting it.

   **Damping-range tuning — empirically chosen.** `LOW_DAMP=0.001 /
   HIGH_DAMP=0.5` gives a measured 6.1× ratio in retained mechanical
   energy at 5 seconds (79% vs 13%; see Step 5 spike). Don't tune
   below 0.001 — PBD solver damping dominates. Don't go above 0.5 —
   the high-damp pendulum freezes via PBD sleep mechanism within ~4s,
   which is actually GOOD for the demo (frozen vs swinging is the
   visual story) but going higher just makes the freeze faster
   without changing readability. Negative damping is forbidden — PBD
   goes unstable.

   **Sleep mechanism is a feature here.** PBD bodies sleep when their
   velocity stays below a threshold for several ticks. With
   damping=0.5, this triggers around t=4s and the bob freezes
   wherever it is in the swing (often at ~20° from straight down,
   not at the bottom). For this demo that's *desirable* —
   unambiguous visual contrast. If a future scene needs all
   low-Phi pendulums to settle to vertical instead, disable sleep
   on the bob bodies.

   **Originally tried Kuramoto coupling.** Spike (`_spike_kuramoto.rs`,
   not committed) verified that direct sin(Δθ) horizontal/tangential
   velocity kicks do NOT produce phase-lock between equal-frequency
   PBD pendulums — even at K=10 with proper tangential direction,
   coupled mean |Δθ| was 0.71 rad vs uncoupled 0.74 rad (statistical
   tie). Real coupled-oscillator demos like Huygens' clocks rely on
   shared-medium coupling (a swaying wall). For a self-contained
   demo, damping modulation is the right primitive.

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

Steps 1-7 are landed. Open the example, run it, and iterate on visual
parameters from there.

```bash
cd /srv/luminous-dynamics/symtropy
cargo run --example pendulum_swarm --release   # verify the visual
```

Most likely tuning targets after first visual run:
- `VARIANCE_SCALE` (currently 1e-4) — controls how aggressively variance
  collapses coherence. If everything stays red even after a shock, scale
  is too low.
- `SHOCK_VELOCITY` (currently 400 px/s) — if shock doesn't visibly break
  neighborhood synchrony, raise it.
- Initial conditions — all bobs released horizontally start in lockstep,
  so background variance is 0. Consider per-(i, j) angle jitter (~5°) so
  the baseline isn't perfectly synced.
- Step 8 polish: trails (Sprite::with_alpha or a TrailComponent), an
  on-screen Phi readout (egui or simple text node), smoother color LUT.

Stop conditions still apply: if a tuning attempt takes >1 hour with no
visible improvement, pause and reconsider the metric (variance vs.
mean KE vs. phase-coherence).
