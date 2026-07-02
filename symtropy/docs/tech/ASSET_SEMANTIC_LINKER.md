# Symtropy Foundry Semantic Linker (Behavioral Injection)

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates/bridges/symtropy-foundry` or elsewhere. Design/vision document only.

## Overview
The Semantic Linker extends the Foundry from an "Asset Pipeline" to a "Behavioral Architecture Pipeline." It allows assets to carry *behavioral signatures*—instructions on how they should be integrated into the Symtropy meta-simulation (e.g., as a network node, a biome effector, or a consciousness bridge).

## Manifest Schema Update (`manifest.yaml`)
We will add a `behaviors` key:
```yaml
behaviors:
  - role: "network_node"
    parameters:
      bandwidth: 100.0
      phi_coupling: 0.8
  - role: "biome_effector"
    parameters:
      radius: 50.0
      signal_type: "mycelial_growth"
```

## Implementation Workflow
1.  **Registry Expansion**: Add a `behaviors` table to `assets.sqlite` to link asset IDs to specific domain-logic behaviors.
2.  **Export Compiler**: Update `export_pack.py` to bake these behaviors into an `asset_behaviors.json` file exported alongside the assets.
3.  **Engine Ingestion**: Update `symtropy-foundry` to parse `asset_behaviors.json` and automatically attach domain-specific components (e.g., `NetworkNode`, `BiomeEffector`) to entities at runtime.
