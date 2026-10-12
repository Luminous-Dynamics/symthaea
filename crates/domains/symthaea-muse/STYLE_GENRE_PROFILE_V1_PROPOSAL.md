# Muse Style / Genre Profiles V1 — Architecture Proposal

**Status:** design proposal; no runtime behavior changes yet  
**Related:** [Issue #7417](https://github.com/Luminous-Dynamics/symthaea/issues/7417), [152-style catalog](./MUSE_152_STYLE_CATALOG_2026-07-18.md), [instrument support #7416](https://github.com/Luminous-Dynamics/symthaea/issues/7416)

## Executive proposal

Do not add more names to a flat genre enum. Build an editable, versioned **music recipe** whose independently defined layers describe how a piece is composed, performed, and rendered. A genre/style name selects a meaningful starting recipe and adds provenance; it is not a substitute for the recipe itself.

This builds on work already present in Muse:
- 152 catalog entries arranged in 19 constellations;
- catalog status and optional routing to the 29-variant executable `symthaea_music_theory::Style` enum;
- eight-field `StyleAnatomy`;
- `HybridStyleSpec` with one structural owner and optional rhythmic, harmonic, ensemble, performance, and production layers;
- current readiness/evidence types and tests.

Those foundations are valuable and should not be replaced. The gap is that the current anatomy frequently inherits constellation-level descriptions, executable styles and research catalog entries are different kinds of things, and the current hybrid validator has only a small number of hard-coded compatibility cases.

## Design principle: distinguish identity from mechanism

**Genre/tradition is a contextual identity claim. Style profile is an executable compositional recipe. Palette is the sound. Mood is intent.**

Keep these concepts independent:

| Layer | Owns | Does not own |
| --- | --- | --- |
| Tradition / genre identity | canonical name, aliases, lineage/context, attribution and claim provenance | the actual note generator |
| Compositional grammar | form, phrase/sentence grammar, theme relations, development, cadence, section transitions | soundfont selection |
| Pitch language | tonal/modal system, pitch hierarchy, intonation/tuning, permitted note and ornament behavior | drum pattern |
| Harmony | harmonic system, harmonic rhythm, voice leading and cadence constraints | production or mix |
| Groove / rhythm | meter/tala/cycle, accent positions, subdivision, swing/syncopation, interlocking lines, cycle continuity | historical authenticity claim |
| Melodic language | contour tendencies, phrase length, repetition/transformation, ornament grammar and range | ensemble selection |
| Orchestration | role-to-part assignment, instrument identity, ranges, playability, ensemble constraints | musical role inferred from register |
| Performance dialect | articulation, timing feel, rubato, humanization and phrase behavior | structural form |
| Production palette | sample/procedural instrument realization, room, stereo, effects, dynamics and mix | compositional grammar |
| Intent axes | mood, energy, density, tension, intimacy, dance function or other user goals | exclusive genre label |

Profiles are sparse. A layer that is unknown, unsupported, or not relevant is explicit; it must not be silently guessed from another axis. Values should record whether they were directly specified, inherited, supplied by the user, or defaulted by the engine.

## Proposed versioned contract (illustrative, not implemented Rust)

```json
{
  "schema": "music-recipe-v1",
  "identity": {
    "kind": "hybrid",
    "display_name": "Sonata development + son-clave groove",
    "genre_claim": null,
    "source_profile_ids": ["form.sonata.v1", "groove.son-clave.v1"]
  },
  "layers": {
    "grammar": {"profile_id": "form.sonata.v1"},
    "pitch_language": {"profile_id": "pitch.functional-tonality.v1"},
    "harmony": {"profile_id": "harmony.functional.v1"},
    "groove": {"profile_id": "groove.son-clave.v1"},
    "orchestration": {"profile_id": "ensemble.chamber.v1"},
    "performance": {"profile_id": "performance.dance-locked.v1"},
    "production": {"profile_id": "production.dry-chamber.v1"},
    "intent": {"mood": ["tense", "energetic"]}
  },
  "overrides": [],
  "provenance": {"recipe_author": "user", "sources": []}
}
```

The example is a newly constructed hybrid recipe, **not** a claim that the historical traditions represented by its layers are one historical genre. IDs are illustrative and must not be treated as IDs that exist in the runtime today.

A resolved recipe receipt should bind:
- schema version;
- exact profile IDs and versions/digests;
- explicit overrides and their provenance;
- engine revision and deterministic seed;
- validation warnings, unsupported layers, and rendering fallbacks;
- final resolved plan digest.

Same resolved inputs + same engine version + same seed should produce the same plan. This is a reproducibility contract, not a promise that every backend creates bit-identical audio unless that backend separately guarantees it.

## Catalog graph, not a single taxonomy tree

Keep the 19 constellations for browsing. Add typed relations to represent the actual web of musical knowledge:
- `substyle_of` / `broader_than`;
- `historically_related_to`;
- `shares_form_with`;
- `shares_groove_grammar_with`;
- `influenced_by` (requires a sourced claim);
- `hybrid_component_of`;
- `contrasts_with`.

Similarity does not prove lineage. Alias and name claims need source/provenance. Multiple attributed genre tags can coexist because genre is subjective and community-dependent. A generated recipe must not be used as evidence that a work or tradition belongs to a genre.

## Hybrid validation as declared constraints and transitions

Replace additional scattered `if constellation == ...` checks with a versioned set of declared compatibility rules. Keep current special-case behavior as regression fixtures during migration.

Use structured outcomes:
- `compatible`;
- `compatible_with_transition` (e.g. free-time opening, then an explicit metered cycle);
- `conflict` (a hard musical invariant is violated);
- `needs_provenance_review` (only when making the relevant historical/authenticity claim);
- `unsupported` (the current engine cannot realize the layer faithfully).

Every result should include stable reason codes, rule/profile IDs, affected section or voice, severity, and whether an explicit override is allowed. A conflict between layers in one section must not automatically forbid a valid section-by-section transition. Conversely, a transition that has no representation in the formal plan must not be assumed.

## Capability and authenticity are separate status axes

Do not compress all readiness into one status. At minimum report independently:
1. **Catalog status:** research / developing / supported / retired.
2. **Execution status:** catalog-only / structure executable / renderable / unsupported layer present.
3. **Evidence status:** deterministic probes present / capture reviewed / behavior demonstrated / known regressions.
4. **Distinctness status:** grammar differs / only sonic palette differs / insufficient evidence.
5. **Cultural-claim status:** sourced description / informed-by framing / qualified review / authenticity claim approved.

Expert/source review is required before making authenticity claims. It is not proof of engine correctness. Unit tests prove code contracts, not cultural authenticity or listener response. Conversely, a culturally informed experimental recipe can still be useful if labelled honestly. The release process should prioritize automated conformance tests, deterministic capture comparison, and owner/user playtesting; none of a genre classifier's label agreement or an unrepeatable listening anecdote should be the sole evidence gate.

## Style distinctness: test grammar, not only audio color

For every executable profile, generate a reproducible report across:
- form/section transitions;
- phrase lengths and rhetorical behavior;
- motif identity and allowed transformations;
- pitch/harmonic constraints and cadence behavior;
- meter/cycle/accent and interlocking constraints;
- orchestration roles, register/playability, and actual instrument capability;
- performance timing/articulation settings;
- production/sample/soundfont path and substitutions.

Compare styles with a **dimension vector**, not a global musical-quality number. The report should show which layers genuinely differ and which are still inherited/unimplemented. Test at least these failure patterns:
- two names resolve to the same executable recipe;
- two recipes differ only in instrument or tempo but are advertised as fully distinct compositional grammars;
- same grammar, different palette (valid and expected);
- same named tradition with a user override (keep the identity/override distinction honest);
- a culturally-specific profile with incomplete provenance;
- a hybrid requiring an explicit staged transition;
- absent instruments, sample banks, production effects, or unsupported articulations.

Do not use single-label genre-classifier accuracy as the universal genre-quality measure. Genre labels are not an objective partition and the validity of common genre-recognition evaluation practices has been challenged.

## Implementation sequence

### M0 — Coverage inventory (first coding increment)
Emit a deterministic machine-readable report linking each of the 152 catalog entries to:
- routed executable style (or none);
- implemented grammar family and form path;
- actual per-style deviations from inherited anatomy;
- rhythm/pitch/harmony/orchestration/performance/production profiles actually used;
- tests and deterministic captures;
- current unsupported/fallback behavior.

Inventory only verified source evidence. A catalog row, enum variant, or non-empty anatomy field is not proof of an independent implementation.

### M1 — Profile and receipt contract
Add schema/version types, sparse values with origin, canonical resolver output, stable validation codes, profile digest and round-trip tests. Keep the existing `Style` enum and stored formats as adapters; pin Classical's current behavior.

### M2 — Migrate a small set of established executable styles
Select varied contrasting examples (e.g. son-clave/cycle-driven, Passacaglia/ground transformation, Fugue/contrapuntal grammar, Ambient/process). Only map a layer when the underlying behavior exists. Preserve existing outputs unless an explicit migration proposal authorizes a change.

### M3 — Declarative hybrid rules and stage transitions
Move existing hybrid validation rules into data-driven constraints. Add counterexample fixtures and preserve diagnostic specificity. Do not claim a general-purpose constraint solver if the implemented rules remain finite and explicit.

### M4 — User-directed recipe construction
Offer two entry paths: choose a canonical named style and modify layers, or start from explicit musical intent and compose a recipe. Always show the result as a hybrid/derived recipe when an override changes its historical identity. Explain unsupported settings and fallbacks.

### M5 — Distinctness and regression evidence
Run deterministic structural probes and audio-capture comparisons, then owner/user playtest the changes. Publish coverage, collisions, unsupported axes, and regressions alongside showcases rather than claiming every one of 152 entries is equally mature.

## Research basis

- MusicBrainz notes genres are subjective and represented through community tags: https://musicbrainz.org/doc/Genre and https://musicbrainz.org/doc/Folksonomy_Tagging.
- AudioSet separates music genre, instrument, musical concept, role, and mood: https://research.google.com/audioset/ontology/music_1.html.
- Genre recognition research questions the validity of commonly used benchmark practice, including GTZAN-related evaluation: https://www.tandfonline.com/doi/full/10.1080/09298215.2014.894533 and https://link.springer.com/article/10.1007/s44206-025-00191-w.

## Acceptance criteria

- No larger flat list is added in place of the profile model.
- Existing catalog and enum APIs remain compatible during staged migration.
- Unsupported or inherited capabilities are explicit and machine-readable.
- Recipes and resolution receipts are versioned, deterministic and provenance-bound.
- Hybrid compatibility returns reasons and can model section transitions.
- A style's status cannot imply cultural authenticity, rendering fidelity, or distinctness unless the corresponding evidence axis supports that claim.
- The first coverage report counts actual implemented grammar and layers, not names or labels.
