# Attribution

All textures in this directory are CC0-1.0 (no attribution legally required —
credited here anyway as good practice, matching the Foundry pipeline's own
auto-generated ATTRIBUTION.md convention).

- **old_concrete_\*.jpg** — ambientCG "Concrete034" — https://ambientcg.com/a/Concrete034
- **wet_concrete_\*.jpg** — ambientCG "Concrete048" — https://ambientcg.com/a/Concrete048
- **painted_steel_\*.jpg** — ambientCG "PaintedMetal004" — https://ambientcg.com/a/PaintedMetal004

License: https://creativecommons.org/publicdomain/zero/1.0/

`_roughness.jpg` files were downloaded alongside `_color.jpg`/`_normal.jpg`
but are not currently wired into any material (see the comment on
`waterworks_textured_material` in `src/systems/rendering_3d.rs` — a plain
grayscale roughness map can't be safely packed into Bevy's combined
metallic-roughness texture slot without also providing a real metallic
channel). Kept for a future proper ORM-texture pass.
