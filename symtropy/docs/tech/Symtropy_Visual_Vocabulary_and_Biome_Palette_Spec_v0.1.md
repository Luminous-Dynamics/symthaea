# Symtropy Visual Vocabulary and Biome Palette Spec v0.1

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates/domains/symtropy-terrain` or elsewhere. Design/vision document only.

**Project:** Symtropy / Luminous Dynamics  
**Date:** 2026-06-15  
**Status:** v0.1 art-direction ground truth  
**Purpose:** Define the palette, material, emissive, surface, and style-pass rules that turn public/open assets into coherent Symtropy assets.

---

## 1. Executive Purpose

The Asset Foundry can automate ingestion, license review, conversion, tagging, thumbnailing, and Bevy export. It cannot decide what Symtropy *feels like* without a visual vocabulary.

This document is the missing bridge between:

```text
Public/open source asset
        ↓
legal + provenance approval
        ↓
technical conversion
        ↓
Symtropy style pass
        ↓
Bevy-ready world asset
```

The goal is to prevent Symtropy from becoming a collage of asset packs. Every approved asset should carry:

```yaml
visual_spec_version: symtropy_visual_vocab_v0.1
palette_id: wetland_mycelial
material_family: biospheric_tissue
surface_archetype: bark_skin
emissive_role: biospheric_signal
style_status: style_approved
```

---

## 2. Core Visual Principles

1. **Ecology is legible.**  
   Materials should reveal flows: water, repair, decay, bloom, memory, energy, and governance.

2. **Civic systems are warm, not sterile.**  
   Public infrastructure should look cared for, patched, audited, and lived in.

3. **Biology is patterned, not random.**  
   Fungal, coral, wetland, bark, membrane, and microbial surfaces should use controlled pattern families.

4. **Emissive color encodes meaning.**  
   Glow is language: consent, trust, warning, ritual, machine state, memory, or biospheric communication.

5. **Public spaces remain readable.**  
   Atmosphere, fog, particles, and bloom should never hide gameplay-relevant information.

6. **No asset-pack soup.**  
   A public asset must pass legal review, technical review, and style review before becoming a Symtropy asset.

---

## 3. Global Color Grammar

Use HSL ranges as art-direction constraints, not rigid law. Each palette can use accent exceptions when narrative meaning requires it.

| Semantic role | Hue range | Saturation | Lightness | Meaning |
|---|---:|---:|---:|---|
| Care / repair | 145-175 | 35-75 | 45-75 | health, restoration, community maintenance |
| Consent / permission | 185-205 | 40-80 | 55-80 | safe passage, voluntary participation, verified access |
| Trust / civic legitimacy | 205-235 | 30-65 | 50-75 | public records, institutional clarity, auditability |
| Memory / ancestry | 260-290 | 25-55 | 40-70 | memorial systems, history, dreams, lineage |
| Biospheric signal | 95-155 | 45-90 | 45-75 | ecological communication, growth, adaptation |
| Machine active | 185-225 | 55-85 | 50-75 | sensors, robotics, diagnostics |
| Warning / boundary | 25-55 | 70-95 | 45-65 | caution, heat, transition zones |
| Quarantine / hazard | 345-20 | 65-95 | 35-65 | Red Bloom, contamination, forbidden absorption |
| Grief / solemn ritual | 230-270 | 15-40 | 20-45 | mourning, silence, historic harm |
| Celebration / festival | 280-330 plus warm accents | 60-95 | 50-75 | public joy, dance, abundance, night festivals |

### Emissive intensity rule

- Ambient world glow: subtle, readable, low bloom.
- Signal glow: localized and semantically colored.
- Hazard glow: high contrast but not eye-searing.
- Ritual glow: soft gradients, strong silhouettes.
- UI/hologram glow: crisp edge, limited bloom, accessibility-tested.

---

## 4. Canonical Biome Palettes

### 4.1 Wetland Mycelial

**Use:** living wetlands, flood-regulating biospheres, mycelial root bridges, damp civic edges.

| Token | HSL range | Notes |
|---|---|---|
| base_water_green | H 145-175, S 25-55, L 25-55 | dark reed water, wet moss |
| mycelial_thread | H 80-110, S 25-60, L 55-80 | root/fungal highlights |
| civic_blue_mist | H 185-205, S 25-50, L 55-80 | consent paths, public markers |
| decay_brown | H 25-45, S 25-55, L 25-45 | mud, old wood, compost |
| bloom_accent | H 110-145, S 55-90, L 50-75 | biospheric signal only |

Material personality:

```yaml
roughness: 0.65-0.95
metallic: 0.0-0.15
emissive: low_to_medium, localized veins
alpha: occasional membranes, reeds, water films
```

Avoid: neon swamp, horror sludge, generic jungle green.

### 4.2 Mist Forest

**Use:** cloud-root ecologies, soft recovery districts, sensory refuge spaces.

| Token | HSL range | Notes |
|---|---|---|
| fog_leaf | H 120-160, S 15-40, L 45-75 | soft vegetation |
| pale_air | H 180-210, S 10-35, L 70-90 | fog and sky |
| bark_gray_violet | H 250-285, S 10-30, L 30-55 | trunks, memory surfaces |
| recovery_gold | H 40-55, S 35-70, L 55-80 | care markers, wayfinding |

Material personality:

```yaml
roughness: 0.75-0.95
metallic: 0.0-0.05
emissive: very low, mostly ritual or wayfinding
alpha: fog layers, leaves, breath clouds
```

Avoid: fantasy elven forest, oversaturated green, horror fog with no legibility.

### 4.3 Desert Spore

**Use:** desert spore networks, heat-stressed settlements, fungal dunes, solar public works.

| Token | HSL range | Notes |
|---|---|---|
| sand_ochre | H 35-55, S 35-70, L 45-75 | dunes, plaster, dust |
| sun_bleached | H 45-60, S 10-35, L 75-92 | old civic surfaces |
| spore_rust | H 15-35, S 45-80, L 35-60 | fungal crusts, oxidized metal |
| cool_consent | H 185-205, S 35-65, L 55-75 | readable access/safety markings |
| deep_shadow_blue | H 215-240, S 20-45, L 20-40 | heat contrast and interior shade |

Material personality:

```yaml
roughness: 0.8-1.0
metallic: 0.0-0.35
emissive: low except warning, night markets, and spore signals
alpha: dust/fog particles, heat shimmer
```

Avoid: generic post-apocalypse brown, Mad-Max-only language, unreadable dust storms.

### 4.4 Ocean Reef Mind

**Use:** ocean-mind reefs, aquatic habitats, coral logic, submerged civic archives.

| Token | HSL range | Notes |
|---|---|---|
| deep_reef_blue | H 195-225, S 40-80, L 20-45 | water volume, distant forms |
| coral_memory | H 320-15, S 45-85, L 45-70 | living coral logic and memory marks |
| plankton_glow | H 160-190, S 65-95, L 55-80 | biospheric signals |
| shell_ivory | H 35-60, S 15-40, L 75-92 | shells, calcium, civic interiors |
| abyss_violet | H 250-285, S 25-60, L 15-40 | depth, unknown, ritual silence |

Material personality:

```yaml
roughness: 0.35-0.85
metallic: 0.0-0.2
emissive: medium, plankton/coral localized
alpha: water membranes, bubbles, soft caustic overlays
```

Avoid: aquarium kitsch, random rainbow coral everywhere, murky unreadability.

### 4.5 Orbital Commons

**Use:** orbital habitats, public docking rings, station gardens, low-g civic spaces.

| Token | HSL range | Notes |
|---|---|---|
| warm_white_panel | H 35-60, S 5-20, L 75-92 | human warmth, not sterile gray |
| civic_blue_line | H 200-225, S 35-70, L 45-75 | navigation, ledgers, trust systems |
| garden_green | H 120-155, S 25-60, L 35-65 | agriculture and bioregenerative loops |
| shadow_navy | H 220-245, S 20-50, L 12-35 | space-facing depth |
| ritual_violet | H 265-295, S 30-65, L 45-70 | low-g dance, memory, ceremony |

Material personality:

```yaml
roughness: 0.45-0.85
metallic: 0.15-0.65
emissive: medium, thin civic/UI lines
alpha: glass, garden membranes, holographic panels
```

Avoid: generic sterile white space station, pure chrome, corporate lobby sci-fi.

### 4.6 Subterranean Archive

**Use:** underground memory vaults, seed banks, mycelial data archives, old infrastructure.

| Token | HSL range | Notes |
|---|---|---|
| archive_earth | H 25-45, S 25-55, L 20-45 | soil, clay, old stone |
| fungal_lamplight | H 65-95, S 35-75, L 45-70 | archive fungus, gentle visibility |
| memory_violet | H 260-285, S 20-55, L 35-65 | knowledge, ancestral records |
| water_black | H 190-220, S 15-45, L 8-25 | cisterns, depth, drainage |
| chalk_mark | H 40-60, S 5-20, L 70-90 | labels, inscriptions, public records |

Material personality:

```yaml
roughness: 0.85-1.0
metallic: 0.0-0.25
emissive: low, fungus lamps and memory inscriptions
alpha: dust motes, condensation, water films
```

Avoid: dungeon horror, unreadable caves, skull-and-torch fantasy shorthand.

### 4.7 Civic Commons

**Use:** public infrastructure spaces, councils, clinics, care halls, schools, markets.

| Token | HSL range | Notes |
|---|---|---|
| public_warmth | H 35-55, S 20-50, L 65-85 | welcoming material base |
| trust_blue | H 205-230, S 30-65, L 45-70 | ledgers, signage, public dashboards |
| care_green | H 145-170, S 35-70, L 45-70 | repair, health, mutual aid |
| repair_orange | H 25-45, S 55-85, L 45-65 | active maintenance, work-in-progress |
| soft_shadow | H 220-250, S 10-25, L 20-45 | depth without grimness |

Material personality:

```yaml
roughness: 0.6-0.9
metallic: 0.0-0.35
emissive: low_to_medium, signage and consent markers
alpha: glass boards, transparent partitions
```

Avoid: sterile hospital, corporate mall, authoritarian monumentality.

### 4.8 Robotics Field Ops

**Use:** care machines, scouts, hydrology sentinels, repair drones, field robotics.

| Token | HSL range | Notes |
|---|---|---|
| ceramic_shell | H 35-60, S 5-25, L 60-85 | friendly shells, non-military profile |
| worn_metal | H 200-230, S 5-20, L 35-65 | steel, field repairs |
| sensor_cyan | H 185-205, S 55-90, L 55-80 | active sensing, diagnostics |
| repair_orange | H 25-45, S 55-90, L 45-65 | tools, hazard edges, maintenance |
| rubber_black | H 220-260, S 5-20, L 8-25 | tires, seals, grippers |

Material personality:

```yaml
roughness: 0.35-0.8
metallic: 0.25-0.85
emissive: medium, diagnostic lines not cyberpunk excess
alpha: sensor glass, LiDAR cones, UI overlays
```

Avoid: weaponized silhouette, black tactical sci-fi, pure corporate product render.

### 4.9 Red Bloom Hazard

**Use:** dangerous absorption ecologies, contamination zones, consent failures, quarantine disputes.

| Token | HSL range | Notes |
|---|---|---|
| bloom_red | H 350-10, S 65-95, L 35-65 | hazard tissue and absorption fronts |
| fever_pink | H 320-345, S 60-95, L 45-75 | seductive danger, toxic beauty |
| warning_amber | H 35-50, S 70-95, L 50-70 | official boundary markers |
| dead_green | H 90-130, S 15-45, L 20-45 | stressed ecology |
| ash_gray | H 220-260, S 5-15, L 20-55 | dead surfaces, residue |

Material personality:

```yaml
roughness: 0.45-0.9
metallic: 0.0-0.25
emissive: high but localized; pulsing hazard language
alpha: spores, haze, membrane sheets
```

Avoid: gore aesthetic, sexy-horror default, impossible-to-read red wash.

### 4.10 Ice-Shell Ocean

**Use:** cryogenic habitats, Europa-like worlds, under-ice seas, cold research settlements.

| Token | HSL range | Notes |
|---|---|---|
| ice_cyan | H 180-205, S 25-65, L 65-90 | ice, shafts, cold light |
| deep_undersea | H 205-235, S 45-85, L 12-40 | water depth |
| mineral_blue | H 220-250, S 20-50, L 35-60 | rock, stress fractures |
| life_warmth | H 30-50, S 40-75, L 50-75 | habitats, humans, care spaces |
| signal_teal | H 160-185, S 55-90, L 45-75 | life detection, biospheric signals |

Material personality:

```yaml
roughness: 0.2-0.75
metallic: 0.0-0.35
emissive: low_to_medium, high contrast against cold base
alpha: ice, water, vapor, frost
```

Avoid: sterile ice cave, all-blue monotony, generic sci-fi lab.

---

## 5. Material Family Defaults

| Material family | Roughness | Metallic | Alpha | Emissive | Notes |
|---|---:|---:|---:|---:|---|
| living_infrastructure | 0.65-0.95 | 0.0-0.25 | low | low | patched civic materials with moss, algae, biopolymer, wood, concrete |
| biospheric_tissue | 0.55-0.95 | 0.0-0.05 | medium | low-medium | bark-skin, fungal mats, wet membranes, coral logic |
| robotics_care_machine | 0.35-0.8 | 0.25-0.85 | low-medium | medium | ceramic shells, worn metal, sensor glass, rubber |
| governance_ritual_surface | 0.45-0.85 | 0.0-0.35 | medium | low-medium | ledgers, oath-stones, consent seals, translucent walls |
| hazard_boundary | 0.45-0.9 | 0.0-0.25 | medium | medium-high | quarantine barriers, Red Bloom, warning pigments |
| habitat_shell | 0.4-0.85 | 0.15-0.65 | medium | low-medium | orbital shells, pressure walls, transparent panels |
| ui_holographic | 0.2-0.6 | 0.0-0.15 | high | medium | crisp holographic surfaces, low bloom, high readability |
| archive_matter | 0.85-1.0 | 0.0-0.25 | low-medium | low | clay, chalk, mycelial records, old stone |

---

## 6. Surface Archetypes

| Archetype | Pattern language | Good source assets | Required style transform |
|---|---|---|---|
| bark_skin | longitudinal fibers, healed seams, subtle pores | bark, wood, rough plaster | recolor to biome; add low emissive root/thread accents |
| fungal_mat | soft branching, radial blooms, porous surface | moss, lichen, ground cover | add controlled mycelial patterns; avoid horror slime |
| coral_logic | branching ridges, cellular cavities, repeated polyps | coral, rock, reef, clay | add signal colors; preserve readable silhouette |
| civic_patchwork | repair seams, labels, public utility markings | concrete, metal, tile, panels | add audit labels, consent markings, wear gradients |
| sensor_membrane | translucent film, gridded microtexture | glass, plastic, fabric, water | add controlled alpha and sensor emissive edge |
| ritual_stone | worn touch surfaces, inscriptions, soft glow | stone, ceramic, clay | add memory colors and worn human-scale details |
| field_repaired_machine | scratches, replaced panels, tool marks | metal, rubber, ceramic | add repair orange, serial tags, non-military silhouette |
| quarantine_skin | boundary tape, warning pigments, abnormal bloom | plastic, cloth, organic tissue | use hazard palette and limit red wash |

---

## 7. Style Pass Automation Rules

The style pass should be a mixture of scripts and review. Automation should propose; humans approve.

### Safe automation

- Normalize scale, origin, and orientation.
- Generate thumbnails and turntables.
- Clamp texture budgets by export profile.
- Rename files and stable asset IDs.
- Apply subtle palette harmonization within selected HSL ranges.
- Generate metadata tags from source, branch, biome, and material family.
- Generate roughness/metallic defaults for missing material channels.

### Review-required automation

- Strong hue shifts.
- Adding emissive veins, ritual marks, consent seals, civic decals, or hazard markings.
- Adding biological overlays to human-made structures.
- Converting generic props into lore-specific artifacts.
- Any style transform that changes semantic meaning.

### Blocked automation

- Removing provenance or license data.
- Auto-approving style status from source tags alone.
- Adding trademark-like symbols or recognizable third-party references.
- Using AI-generated style transforms without AI provenance logging.
- Making all biomes share one global color grade.

---

## 8. Asset Style Review Checklist

| Question | Pass condition |
|---|---|
| Does the asset have legal/provenance approval? | `legal_status` is not pending or quarantined. |
| Does the asset have a biome palette? | `palette_id` is set and valid. |
| Does the material family match its world role? | Family is one of the canonical material families. |
| Is emissive color meaningful? | Emissive role is consent, trust, warning, memory, ritual, care, machine state, or biospheric signal. |
| Is the asset readable in Low/Medium settings? | Thumbnail and validation scene pass. |
| Does it avoid generic asset-pack appearance? | It has at least one Symtropy-specific style transform or approved reason to remain natural. |
| Is it culturally/worldbuilding safe? | No accidental logos, real people, sacred/cultural misuse, or fan-IP residue. |
| Does it preserve gameplay clarity? | Does not hide interactable shapes, routes, hazards, or UI. |

---

## 9. Metadata Additions for Foundry Manifests

```yaml
style:
  visual_spec_version: symtropy_visual_vocab_v0.1
  palette_id: wetland_mycelial
  material_family: biospheric_tissue
  surface_archetype: fungal_mat
  emissive_role: biospheric_signal
  style_transform_recipe:
    - normalize_scale_origin
    - clamp_texture_budget_medium
    - palette_harmonize_wetland_mycelial
    - add_low_mycelial_emissive_veins
  review:
    status: pending_review
    reviewer: null
    notes: null
```

---

## 10. First Ten Style Recipes

| Recipe ID | Use | Operations |
|---|---|---|
| `wetland_mycelial_materialize` | bark, moss, mud, wet concrete | green-blue palette clamp, high roughness, mycelial vein option, humidity stains |
| `desert_spore_weather` | sand, plaster, rust, solar hardware | ochre/rust palette, dust overlay, spore-edge accents, heat-worn roughness |
| `civic_commons_warmth` | clinics, halls, schools, markets | warm whites, trust blue signage, care green repair marks, readable contrast |
| `robotics_field_care` | drones, sentinels, field machines | ceramic/worn metal palette, sensor cyan, repair orange, no military black profile |
| `ocean_reef_signal` | coral, shells, aquatic surfaces | reef blues, coral memory accents, plankton glow, controlled caustic overlay |
| `orbital_commons_softtech` | stations, habitats, docking | warm white panels, garden green, civic blue lines, limited chrome |
| `subterranean_archive_memory` | archives, seed banks, vaults | earth/chalk/violet palette, fungus lamplight, dust motes, inscription marks |
| `red_bloom_boundary` | hazards, quarantine, absorption | red/pink hazard accents, amber boundaries, pulse emissions, avoid full red wash |
| `ice_shell_contrast` | under-ice, cryo, research | cyan/blue base, warm habitat light, teal life signals, frost overlays |
| `mist_forest_recovery` | sensory recovery, cloud roots | pale fog, muted greens, recovery gold markers, soft silhouettes |

---

## 11. Relationship to Other Docs

This document is now a dependency of:

- **Symtropy Public Asset Automation Foundry v0.2**: style gate, manifest fields, material family, palette IDs.
- **Symtropy Bevy Rendering Strategy v0.2**: atmosphere presets, material families, emissive conventions, LOD/render validation scenes.

The Foundry decides whether an asset is legal and technically processable. This Visual Vocabulary decides whether it belongs in Symtropy.

---

## 12. Acceptance Tests for v0.1

The visual vocabulary v0.1 is successful when:

1. Every approved Foundry asset can point to a valid `palette_id`.
2. Every approved Foundry asset can point to a valid `material_family`.
3. Every emissive material has an `emissive_role`.
4. The renderer validation scene can preview each palette in a neutral studio scene.
5. At least ten public assets can be transformed with style recipes without losing provenance.
6. Reviewers can reject an asset for style reasons without changing its legal/provenance status.
7. No asset can enter `bevy_export/` with `style_status: pending_review`.

---

## 13. Decision Record

```text
Decision: Symtropy uses a palette-and-material vocabulary as the authority for asset style approval.

Color: Use biome-specific HSL ranges plus semantic emissive roles.

Materials: Use canonical families: living infrastructure, biospheric tissue, robotics care machine, governance ritual surface, hazard boundary, habitat shell, UI holographic, archive matter.

Style pass: Automation proposes; humans approve expressive semantic transforms.

Export gate: Legal approval + technical approval + style approval required before Bevy export.
```
