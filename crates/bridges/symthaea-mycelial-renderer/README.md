# Symthaea Mycelial Presence — native renderer

This is an isolated Bevy 0.20.1 native 3D preview adapter.

It consumes the renderer-neutral `MycelialModel` and the bounded CPU tube tessellator from `symthaea-canvas`. It does not own cognitive state and does not infer runtime activity. The current preview uses an empty snapshot, so it shows the ambient baseline only.

## Run

```bash
cargo run --manifest-path crates/bridges/symthaea-mycelial-renderer/Cargo.toml -- --mode ambient
cargo run --manifest-path crates/bridges/symthaea-mycelial-renderer/Cargo.toml -- --mode cognitive
cargo run --manifest-path crates/bridges/symthaea-mycelial-renderer/Cargo.toml -- --mode conversational --reduced-motion
```

The crate is intentionally excluded from the root workspace. This avoids coupling the existing Bevy 0.15 dashboard or the workspace's direct `wgpu 27` dependency to Bevy 0.20's renderer dependency graph.

## Current scope

- Uploads bounded CPU-generated tube geometry as a Bevy `Mesh3d`.
- Shows a generation-tinted filament canopy, smooth junction bulbs, terminal spores, emissive nucleus, translucent membrane, and orbital cells.
- Uses Bevy's natural HDR bloom preset with filmic tonemapping for restrained light scattering.
- Offers the three shared Mycelial Presence presentation modes.
- `--reduced-motion` disables renderer-local nucleus breathing and orbital movement.

This is a native raster prototype, not a browser renderer. It currently uses standard PBR materials and does not yet include custom WESL shaders, volumetrics, GPU-generated geometry, live runtime streaming, or performance claims. The adapter has not yet been compiled or visually validated in this environment.

The browser delivery strategy remains feature-detected WebGPU, WebGL 2 fallback, and SVG fallback. A browser implementation should use conventional vertex/fragment pipelines for cross-platform parity; Bevy's mesh-shader path is not web-supported in its current release.
