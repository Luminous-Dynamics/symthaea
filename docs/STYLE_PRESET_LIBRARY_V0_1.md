# Luminous Atelier — style and function preset library v0.1

## Why presets are layered

A style preset and a functional preset answer different questions.

- A **style preset** suggests visual vocabulary, silhouette and editable component directions.
- A **functional preset** suggests activity, comfort, accessibility, climate, repair or lifecycle priorities.
- A **user override** expresses the person's actual requirements and always takes precedence.

A preset is a starting point, never a style ranking, a fit recommendation or an evidence claim.

## Initial library

### Functional and hybrid starting points

| Preset | Main intent |
|---|---|
| Everyday Core | Versatile daily clothing and balanced priorities |
| Heat-Adaptive | User-selected ventilation, low bulk and warm-climate comfort |
| Adaptive Ease | Wearer-led closures, openings, movement and posture |
| Repair-First | Replaceable components, repairability and lifecycle considerations |
| Creative Statement | Expressive visual exploration and novelty |

### Style directions

| Preset | Visual vocabulary | User controls |
|---|---|---|
| Streetwear Volume | Relaxed proportions, layering, graphics | Volume, pocket layout, rights-cleared graphics |
| Classic Tailoring | Structured, refined, controlled ease | Shoulder, lapel, drape, fit and finishing |
| Utility Workwear | Practical storage and serviceable components | Task-specific pockets, panels and closures |
| Outdoor Layering | Layer-compatible, activity-aware silhouettes | Ventilation, coverage, hood and storage |
| Movement Active | Articulation and movement | Compression/ease, seam placement and mobility |
| Modest Coverage | User-defined coverage and layering | Opacity, length, openings and coverage zones |
| Heritage Craft | User-directed craft and cultural references | Motif rights, technique, attribution and provenance |
| Formal Occasion | Refined, structured or fluid occasionwear | Drape, structure, embellishment and event intent |
| Avant-Garde Sculptural | Asymmetry and experimental forms | Panel topology, volume, modularity and proportion |
| Vintage Rework | Reuse, adaptation and reclaimed components | Source garment condition, panel reuse and alteration |
| Sensory Calm | Wearer-led tactile preferences | Seam placement, labels, closures and surface feel |
| Hot Climate Lightweight | Warm-climate low-bulk directions | Ventilation, coverage, layering and fabric choices |

These 12 styles are stored as separate JSON presets under `presets/style/`.

## Composition rules

A future `PresetStack` implementation should use explicit layer roles and preserve the source of every value.

1. Combine style tags as a de-duplicated set, preserving origin references.
2. Use the style layer's silhouette unless the user overrides it.
3. Merge component defaults by semantic path. When two layers specify different values for the same path, surface a conflict; never use silent last-write-wins.
4. Treat objective weights as preference suggestions. Do not sum, normalize or change them silently. Ask the user to resolve duplicates or explicitly choose a merge policy.
5. Combine accessibility and circularity notes as suggestions, not as proof or hard requirements.
6. Presets do not create hard constraints. Hard constraints are entered or confirmed by the user.
7. Preserve each preset ID, revision and content digest in the composed brief.
8. When a preset revision changes, show which derived fields would change and require user approval before applying it to an existing design.

## Inclusion and rights rules

- Do not infer gender, religion, disability, ethnicity or cultural identity from a style selection.
- Heritage-inspired presets must require user-supplied or rights-cleared cultural references; do not treat motifs as free generic content.
- Modest-coverage settings must be user-defined; no assumptions about why a person wants coverage.
- Adaptive and sensory presets are wearer-led design aids, not clinical or medical recommendations.
- Outdoor, workwear and movement presets are not PPE or safety certifications.
- Environmental, washability, durability, comfort, weatherproofness and protective-performance claims require separately qualified evidence.

## Benchmark for presets

Evaluate presets on:

- editability and preservation of user intent;
- style-family diversity without generic homogenization;
- conflict detection in composed preset stacks;
- user acceptance, override and reversal rates;
- coverage across different garment families and body/fit profiles;
- time to an accepted first brief;
- rights/provenance completeness for references and motifs;
- no increase in hard-constraint violations or unsupported physical claims.

Do not evaluate a cultural or aesthetic style with one universal preference score. Use user-defined intent and appropriate reviewers.
