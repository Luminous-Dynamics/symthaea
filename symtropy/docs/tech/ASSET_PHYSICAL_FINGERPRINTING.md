# Asset Physical Fingerprinting Protocol

> **Code status (2026-07-02 review, corrected 2026-07-02):** Original review only checked `symtropy/crates`/`symtropy/src` (Rust) and missed `tools/symtropy_assets/` (Python). Corrected verdict: PARTIAL. The pipeline is real and wired end-to-end — `cli.py`'s `run_conversion()` invokes Blender headless on `converters/normalize_glb_basic.py`, which calls `calculate_physical_properties()` per mesh and writes mass/COM/inertia into the `assets.sqlite` registry. But `calculate_physical_properties()` itself is currently a stub — its own comment says "Bypass for now due to API changes in Blender 5.1" — and returns hardcoded placeholder values (mass=1.0, COM=origin, identity inertia tensor) rather than computing them from mesh geometry.

## Overview
Physical Fingerprinting is an automated process within the Symtropy Asset Foundry that extracts intrinsic physical properties from 3D models during the audit/normalization phase. By moving physics configuration from engine-level manual tuning to asset-level metadata extraction, we ensure that objects behave realistically (e.g., mass distribution, balance) immediately upon being added to the simulation.

## The Protocol

### 1. Blender Extraction (Foundry Side)
During `tools/symtropy_assets/converters/normalize_glb_basic.py` execution, the Foundry will calculate:
- **Volume**: Based on the mesh manifold.
- **Center of Mass (COM)**: Local space offset from origin.
- **Inertia Tensor**: Derived from the mass distribution of the mesh, assuming uniform density for non-structural objects or provided density parameters for specific material families.

### 2. Registry Schema (DB Side)
The `assets` registry will be extended with physical metadata:
- `mass_kg`: float
- `com_x`, `com_y`, `com_z`: float
- `inertia_ixx`, `inertia_iyy`, `inertia_izz`, ... : float (Inertia tensor)

### 3. Engine Ingestion (FoundryPlugin Side)
When `symtropy-foundry` loads an asset, it will look for this metadata.
- If present, it overrides standard sphere/box mass calculations with the stored physical properties.
- **Workflow**: `FoundryPlugin` looks for a `PhysicalProperties` component in the registry cache, applying it to the `RigidBody` component if the entity is registered as a Foundry-managed asset.

## Roadmap
1. **Scripting**: Enhance `normalize_glb_basic.py` with `mathutils` mass property calculation.
2. **Persistence**: Migrate/Extend `assets.sqlite` schema to include these fields.
3. **Engine-side**: Update `symtropy-foundry` to apply these properties during entity spawning.
