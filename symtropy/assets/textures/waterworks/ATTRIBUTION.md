# Attribution

All textures in this directory are CC0-1.0 (no attribution legally required —
credited here anyway as good practice, matching the Foundry pipeline's own
auto-generated ATTRIBUTION.md convention).

- **old_concrete_\*.png** — ambientCG "Concrete034" — https://ambientcg.com/a/Concrete034
- **wet_concrete_\*.png** — ambientCG "Concrete048" — https://ambientcg.com/a/Concrete048
- **painted_steel_\*.png** — ambientCG "PaintedMetal004" — https://ambientcg.com/a/PaintedMetal004

License: https://creativecommons.org/publicdomain/zero/1.0/

Converted from the original ambientCG `.jpg` downloads to `.png` because the
project's Bevy build doesn't enable the `jpeg` image-loader feature (loading
a `.jpg` fails with "invalid image extension: jpg" at runtime).

`_roughness.png` files were downloaded alongside `_color.png`/`_normal.png`
but are not currently wired into any material (see the comment on
`waterworks_textured_material` in `src/systems/rendering_3d.rs` — a plain
grayscale roughness map can't be safely packed into Bevy's combined
metallic-roughness texture slot without also providing a real metallic
channel). Kept for a future proper ORM-texture pass.
