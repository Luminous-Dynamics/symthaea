# `pendulum_swarm_3d` — 3D port

3D adaptation of the 2D `pendulum_swarm` demo. Same Phi-coupled physics
(neighborhood velocity variance → coherence → Phi → damping); different
rendering (Bevy `Camera3d` + `Mesh3d(Sphere)` + `MeshMaterial3d<StandardMaterial>` +
`DirectionalLight` instead of 2D sprites).

## Layout

- 10×10 = 100 pendulums in a flat XZ grid (pivot plane at y=0).
- Each pivot is a small grey sphere (radius 0.04 m).
- Each bob is a 0.10 m sphere connected to its pivot by a `DistanceConstraint`
  with `rest_length = ARM_LENGTH` (1 m).
- Bobs hang in -Y from their pivots. Per-cell jitter of ±17° applied as a
  random direction in the XZ plane × random magnitude — bobs swing in
  arbitrary 3D paths, not just one plane.
- 3D neighborhood for variance: 9 cells (self + 8 in XZ grid).

## Camera & lighting

- `Camera3d` at `(0, 1.5, 6)` looking at `(0, -1, 0)` — slight downward tilt
  reveals depth across rows.
- One `DirectionalLight` at `(-0.7, 0.5, 0.0)` Euler rotation, illuminance 8000.
- `GlobalAmbientLight { brightness: 200.0, color: srgb(0.8, 0.85, 1.0) }` —
  cool ambient so shadowed sides don't go fully black.

## Run

```bash
cargo run -p symtropy-bevy --example pendulum_swarm_3d --release
```

For headless visual verification:

```bash
PENDULUM_CAPTURE_DIR=/tmp/p3d_caps \
  cargo run -p symtropy-bevy --example pendulum_swarm_3d --release
# Inspect /tmp/p3d_caps/pswarm3d_t{1.5,4.0,7.0}.png
```

The example auto-exits at t=8.5 s when `PENDULUM_CAPTURE_DIR` is set.

## What's the same as 2D, what's different

| Aspect | 2D demo | This (3D) demo |
|---|---|---|
| Plugin | `SymtropyPhysicsPlugin::<2>` | `SymtropyPhysicsPlugin::<3>` |
| Gravity | `[0, -981]` (px/s²) | `[0, -9.81, 0]` (m/s²) |
| Bobs | `Sprite::from_color` squares | `Mesh3d(Sphere) + StandardMaterial` |
| Camera | `Camera2d` | `Camera3d` with `looking_at` |
| Click | `viewport_to_world_2d` | `viewport_to_world` ray + point-line distance |
| Arm gizmo | `gizmos.line_2d` | `gizmos.line` (Vec3) |
| Coupling | identical (variance → Phi → damping) | identical |
| Phi normalisation | `phi / 0.314` | identical |
| Initial jitter | 1D angle | 2D direction in XZ + magnitude |

## Known polish targets

- **Color reads as pink/magenta**, not the intended HSL gradient red ↔ blue.
  PBR's `StandardMaterial` interaction with `emissive` + ambient + directional
  light shifts the perceived hue. To fix: lower `emissive_mag`, set
  `metallic = 0.0`, or use `unlit = true` for cleaner color.
- **Bobs are uniform color** at equilibrium — same dynamic as 2D (variance → 0
  in lockstep settled state). Per-bob jitter mostly resolves the visual
  flatness mid-swing; equilibrium uniformity is by design.
- **No on-screen HUD.** A "Phi: ..." text overlay would help visitors
  understand what they're seeing.

These are deferred to ROADMAP Phase 2.C (debug visualisation) and Step 8
polish from the 2D demo's design doc.
