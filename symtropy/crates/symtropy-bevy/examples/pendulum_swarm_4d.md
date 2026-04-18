# `pendulum_swarm_4d` — 4D port with hyperplane slicing

4D adaptation of the pendulum_swarm series. Same Phi-coupled physics
(neighborhood velocity variance → coherence → Phi → damping) as the 2D
and 3D variants, but the world is 4D. Hyperplane slicing perpendicular
to the W axis selects which 3D cross-section is rendered.

## Layout

- 5×5 pivots in the XZ plane × 3 layers along W = 75 pendulums total.
- W-layers spaced 1 m apart, centred on `w=0`. So pivots sit at w ∈ {-1, 0, +1}.
- Each bob hangs in -Y from its pivot via `DistanceConstraint::<4>` with
  `rest_length = 1.0`. Gravity is `[0, -9.81, 0, 0]` — only Y has gravity,
  so bobs fall and oscillate in the XZW directions perpendicular to it.
- **Per-cell jitter is 4D**: random unit vector in (X, Z, W) × random
  magnitude (±17°). Bobs CAN swing along the W axis and drift between
  W-layers, so motion crosses slice planes.

## Visualisation: hyperplane slicing

A `Projector4D` resource owns `w_slice` (current slice position) and
`slice_thickness = 0.45 m`. For each bob, alpha is computed from
W-distance to the slice plane:

```
alpha = max(0, 1 - |bob.w - w_slice| / slice_thickness)
```

Bobs with `alpha == 0` are set to `Visibility::Hidden`. Visible bobs have
their `StandardMaterial.base_color.alpha` modulated by `alpha` — bobs
near the slice edge fade smoothly.

**Press `[` / `]`** to move `w_slice` by ±0.1 m per press. The HUD
top-left shows the current value. As you move the slice, one W-layer
fades in while the previous fades out — the same physics simulation seen
from a different W cross-section.

## What the 3D camera does NOT do

This demo uses **fixed-axis hyperplane slicing** (Miegakure's approach).
It does NOT have a true 4D camera (5×5 view matrix), 4D rotation of the
viewing frame (double-rotors), or stereographic projection. Those are
ROADMAP Phase 2.B work.

## Run

```bash
cargo run -p symtropy-bevy --example pendulum_swarm_4d --release
```

For headless verification (3 PNGs, one per W-layer):

```bash
PENDULUM_CAPTURE_DIR=/tmp/p4d_caps \
  cargo run -p symtropy-bevy --example pendulum_swarm_4d --release
# Inspect /tmp/p4d_caps/pswarm4d_t{2.0,4.5,7.0}_w*.png
```

The headless mode parks the slice on each W-layer in turn and screenshots.

## Known polish targets

- **HUD text shows previous frame's w_slice value** in headless captures —
  `headless_capture` updates `projector.w_slice` and queues a screenshot
  in the same frame, but `update_hud`'s text update happens too late for
  that frame's render. Screenshot caption-mismatch only; interactive use
  is fine.
- **Orange/yellow circle gizmos** around bobs are `symtropy-bevy`'s default
  debug-gizmos rendering collider outlines and safety-tier indicators.
  Disable with `default-features = false` on the symtropy-bevy dep, or
  by passing `debug_gizmos: false` via `SymtropyPhysicsPluginConfig`.
- **Pink/magenta colour** instead of the intended HSL gradient — same
  PBR-emissive interaction as the 3D demo. See `pendulum_swarm_3d.md`.
- **Slice-thickness tuning**: `0.45 m` was chosen so each W-layer (1 m
  apart) is fully visible when slice is parked on it. Bobs with W-jitter
  can extend slightly into neighbouring layers; the fade smooths that.

These are all ROADMAP Phase 2.C / 2.B deferrals. The core capability —
4D Phi-coupled physics rendered as a movable 3D cross-section — works.
