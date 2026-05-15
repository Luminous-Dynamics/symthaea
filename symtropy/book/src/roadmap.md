# Roadmap

The full, authoritative roadmap lives at `../../ROADMAP.md` at the repo root.

## Six-phase plan

- **Phase 0** — Stabilise & unblock adoption (4–6 weeks). License split, crates.io republish, this Book, CI matrix, determinism contract, Wayland. Includes the **Symtropy Mini-Jam** community stress-test.
- **Phase 1** — Integration wins (Q2 2026, 6–8 weeks). Mycelix CI harness in production, robotics bridge wired to 3 platforms, `symtropy-rapier3d-bridge`, Digital Twin bootstrapping.
- **Phase 2** — Visual & asset pipeline (Q3 2026, ~12 weeks). Flagship: **4D → 3D cross-section slicing shader**, 3D PBR, debug gizmos, scene format, glTF import.
- **Phase 3** — Animation, audio, scripting (Q4 2026, ~10 weeks). Skeletal animation + IK, state-driven audio via `symthaea-muse`, Rhai + WASM scripting.
- **Phase 4** — Networking, platforms, XR/VR (Q1 2027, ~10 weeks). Lightyear production wrap, spatial authority, Holochain persistence, WASM target, Windows/macOS verified, `bevy_openxr`.
- **Phase 5** — Ecosystem crates (ongoing from Q2 2027). GPU broadphase, soft body, mesh, terrain, fluid — community-owned optional crates.

## What stays out of core

- GPU broadphase — welcome as `symtropy-physics-gpu`
- Soft body / cloth — welcome as `symtropy-soft`
- Triangle mesh — welcome as `symtropy-mesh`
- Heightfield terrain — welcome as `symtropy-terrain`
- Bespoke editor — use `bevy_inspector_egui`
- Custom wgpu renderer — Bevy owns this

## Locked-in scoping (from roadmap review, 2026-04)

1. Dual-track framing — research hero + generalist adoption.
2. Bevy is the permanent foundation.
3. License split — core Apache-2.0 OR MIT; research layer AGPL.
4. XR/VR — Phase 4 (`bevy_openxr` + WebXR).
5. Rapier3D — Phase 1 as `symtropy-rapier3d-bridge` (opt-in).
6. Terrain + soft-body — Phase 5 ecosystem crates.
