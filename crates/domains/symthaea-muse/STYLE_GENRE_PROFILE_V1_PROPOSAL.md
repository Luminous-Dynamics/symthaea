# Muse Style / Genre Profiles V1 — Architecture Proposal

**Status:** design proposal; no runtime behavior changes yet  
**Related:** [Issue #7417](https://github.com/Luminous-Dynamics/symthaea/issues/7417), [original 152-style catalog (historical)](./MUSE_152_STYLE_CATALOG_2026-07-18.md), [instrument support #7416](https://github.com/Luminous-Dynamics/symthaea/issues/7416)

## Executive proposal

Do not add more names to a flat genre enum. Build an editable, versioned **music recipe** whose independently defined layers describe how a piece is composed, performed, and rendered. A genre/style name selects a meaningful starting recipe and adds provenance; it is not a substitute for the recipe itself.

This builds on work already present in Muse:
- 160 catalog entries arranged in 20 constellations;
- catalog status and optional routing to the 29-variant executable `symthaea_music_theory::Style` enum;
- eight-field `StyleAnatomy`;
- `HybridStyleSpec` with one structural owner and optional rhythmic, harmonic, ensemble, performance, and production layers;
- current readiness/evidence types and tests.

Those foundations are valuable and should not be replaced. The gap is that the current anatomy frequently inherits constellation-level descriptions, executable styles and research catalog entries are different kinds of things, and the current hybrid validator has only a small number of hard-coded compatibility cases.

### Measured source baseline (2026-10-12)

The frozen inventory in [style-genre-coverage-baseline-v1.json](./style-genre-coverage-baseline-v1.json) is pinned to the current catalog source and declared `Style::ALL` source. It contains 160 stable catalog IDs across 20 constellations: 29 entries have composer routes and 131 do not. Of the unrouted entries, 123 are Research and eight Southern African additions are explicitly `ResearchExpertReviewRequired`. Status counts are 24 Developing, 123 Research, eight ResearchExpertReviewRequired, and five ExpertReviewRequired; none are Foundation. The executable `Style::ALL` declaration still contains 29 variants. This is a route/status inventory, **not** evidence that the 29 routes are musically distinct, renderer-backed, or culturally authenticated. The snapshot remains frozen for comparison; future inventory generation reads the Rust sources rather than treating JSON as a second source of truth.

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

Keep the constellation model for browsing and add typed relations to represent the actual web of musical knowledge. Add typed relations to represent the actual web of musical knowledge:
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

### Controlled evaluation protocol

Treat the following as separate evidence gates; do not collapse them into a single score:

1. **Inventory integrity:** catalog-to-runtime and runtime-to-catalog routing, stable IDs, statuses, and source digests. A route existing is inventory evidence only.
2. **Control sensitivity:** change one requested recipe dimension while holding seed, form scaffold, and all other inputs fixed. Measure whether the corresponding musical descriptors move in the intended direction.
3. **Non-target invariance:** for a change to groove, for example, check that unrelated constraints such as requested pitch language or section order remain within explicitly declared tolerances. Record unintended cross-layer effects instead of silently averaging them away.
4. **Seed robustness:** repeat the matched contrast over a fixed, published seed set. Report per-seed values and robust summaries; do not select only the most flattering generation.
5. **Execution and rendering:** separately record whether the requested layer is implemented, whether the needed instrument/sample/effect exists, which substitutions occurred, and whether an audio artifact was successfully produced.
6. **Musical distinctness and review:** compare symbolic grammar descriptors, captured audio evidence, and owner/user listening notes as separate modalities. Establish tolerances from calibration data before making a pass/fail claim; do not invent universal thresholds. Provenance and cultural review remain a separate gate.

Every contrast should carry a resolution receipt binding the recipe/profile digests, engine revision, fixed seed, input scaffold, explicit overrides, capability fallbacks, and artifact digests. This makes a failed or successful comparison reproducible and prevents a good sample from concealing unsupported requested behavior.

This protocol is motivated by recent evidence that nominally disentangled music representations do not always encode their intended semantics (Ibáñez-Martínez et al., ICASSP 2026, https://arxiv.org/abs/2602.10058), alongside the broader 2026 survey conclusion that no single objective metric captures music-generation quality and that controllability and evaluation remain open problems (Duan, Yu & Briot, https://link.springer.com/article/10.1007/s10462-026-11582-x). The lesson for Muse is to measure intended control, non-target invariance, realization, and perceptual evidence separately—not to treat an embedding distance or a genre classifier as proof of success.

A study published on 1 October 2026 examined 75 Suno outputs using the same Turkish lyric pattern, five style-influence settings (0/25/50/75/100), and five styles. Its qualitative analysis found that the effect of style influence varied by genre, and was not consistently monotonic for every style (Yücebağ, 2026, https://doi.org/10.7456/tojdac.1993191). This is a single-platform qualitative case study, not a universal result, but it motivates one more evaluation requirement for Muse: for tunable controls, measure a predeclared response curve over multiple levels and fixed seeds, not only one baseline/perturbation pair. Record whether the intended feature changes, whether non-target features remain stable, and whether responses are monotonic, thresholded, or context-dependent. Do not assume a higher setting inherently means a more correct or authentic style.

## Implementation sequence

### M0a — Bidirectional source-routing inventory (implemented foundation)
Emit a deterministic source-derived report for the 160 catalog entries and 29 executable `Style` variants. The current audit checks stable catalog IDs, source digests, status/route consistency, routes that reference enum variants, and executable variants that lack a catalog route. It lists routed/unrouted catalog IDs per constellation and pins the aggregate inventory in a frozen baseline.

This phase reports **routing and source integrity only**. A route, enum variant, catalog row, or non-empty anatomy field is not proof of an independent musical implementation. The audit explicitly does not claim musical distinctness, renderer availability, or cultural authenticity.

### M0b — Source-declared runtime capability map (first pass implemented)
`scripts/muse_style_capability_inventory.py` now reads the actual `Style` declarations, grammar dispatcher, and Muse's style-to-instrument-pool mapping. For all 29 executable styles it inventories:
- grammar family plus whether dispatch reaches a dedicated engine, the named period pipeline, or the compatibility fallback;
- whether a `CompositionSpec` preset is declared;
- meter, tempo range, and the intent axes declared by the existing census-informed source contract;
- source-declared instrument names and the count of three-instrument pool choices;
- source blob digests for the files that define these mappings.

The independent source pass currently resolves 12 styles to the named period pipeline, 6 to dedicated grammar engines, and 11 to the compatibility fallback. These are dispatch categories, not quality grades: a dedicated call site does not prove an engine has the intended causal effect, a sound bank exists, or the rendered audio is distinctive. The report explicitly retains that distinction.

The report now also joins all 160 catalog rows to their runtime profile by composer route: 29 rows expose the parsed source profile, while 131 catalog-only rows carry an explicit null runtime profile. Eight catalog-only entries explicitly require expert review and are not composable. The intent-axis source metadata points to the existing `diversity_census.rs` one-factor-at-a-time census using a blob digest and function-marker check. It does **not** execute that census or claim to independently reproduce its measurements. A further source-derived count makes control gaps visible: 13 styles declare all four intent axes, 10 declare three, 4 declare two, and 2 declare none (`Passacaglia` and `RenaissancePolyphony`). The latter reflects the existing `supported_intent_axes()` declaration and its cited census contract; it is a priority for measured M0c follow-up, not proof that every possible input can never affect those styles.
### M0c — Routed symbolic control-response curves (probe implemented; execution pending)
`crates/domains/symthaea-music-theory/examples/style_control_response_curves.rs` measures the real public styled-composition dispatch path via `compose_with_grammar_plan(style.grammar_profile(), ...)`. This matters because the existing `diversity_census.rs` sensitivity matrix calls `compose_with_spec_and_form`, which does not exercise the same grammar-dispatch path for styles routed to dedicated engines.

The new probe varies one input at a time across 29 styles:
- tonic: C, Eb, E, G, Bb;
- valence, arousal, energy: five ordered levels each;
- requested bars: 8, 12, 16, 20, 24;
- four fixed baseline seeds (`90210`, `130363`, `222731`, `918273`) for each control curve; the seed control itself tests four adjacent offsets from each baseline.

It emits 696 style/control curves and 3,364 tested points (29 styles × four baseline seeds × six controls), each carrying the full input, exact fingerprint, distances for every named structural layer, tempo, meter, duration, note count, pitch diversity/range, absolute tonic, velocity mean/spread, section-intensity mean/spread, emphasis counts, and explicit optional form metadata. Dynamics are reported separately because the exact fingerprint deliberately ignores velocity/emphasis and the structural pitch-class layer is transposition-invariant. Continuous response shapes are reported as descriptive observations, not pass/fail quality scores; tonic and seed are not treated as ordinal controls. The per-curve record also flags `no_observed_change_at_tested_points` after checking note-event fingerprint, overall structural distance and layers, tempo/meter, durations, note/pitch ranges, absolute tonic, velocity and section-intensity statistics, emphasis counts, and form availability. This is a triage signal, not an automatic failure or proof a control is dead; target-specific expectations still need to be defined. Per-style JSON now aggregates these outcomes across the four baseline seeds by control and by output channel, and the workflow summary publishes the seed counts and specific baseline seeds with no observed change. A dedicated `cargo test -p symthaea-music-theory --example style_control_response_curves` gate checks the control grids and monotonicity labeling. Per-style summaries aggregate the fraction of the four baseline seeds that show any observed change and count seed-level changes for each output channel; the workflow summary includes the no-change seed IDs and a style/control robustness table.

A dedicated [Muse Style Response Curves workflow](../../../.github/workflows/muse-style-response-curves.yml) now runs the example tests, executes the probe, binds the JSON report to the exact commit plus Git blob and SHA-256 digests, and uploads both report and receipt as a CI artifact. The latest revision has not yet produced an execution result, so no generated measurements are claimed in this proposal. Once produced, inspect zero/partial-axis styles first and compare intended target changes against non-target invariance. Audio-render and listener evidence remain separate gates.

### M0d — Measured implementation and evidence inventory
Extend the source map with per-style evidence for:
- actual form and section-transition paths and deviations from inherited anatomy;
- measured rhythm, pitch, harmony, orchestration, performance, and production behavior;
- deterministic symbolic/audio captures and their source/engine/input digests;
- unsupported layers, sample/effect substitutions, and fallback behavior.

Each capability claim must cite the executable source and relevant test/capture receipt. Unknown capability remains unknown; do not infer it from a name or constellation. Source-declared controls and observed effects must remain separate.

### M1 — Profile and receipt contract
Add schema/version types, sparse values with origin, canonical resolver output, stable validation codes, profile digest and round-trip tests. Keep the existing `Style` enum and stored formats as adapters; pin Classical's current behavior.

### M2 — Migrate a small set of established executable styles
Select varied contrasting examples (e.g. son-clave/cycle-driven, Passacaglia/ground transformation, Fugue/contrapuntal grammar, Ambient/process). Only map a layer when the underlying behavior exists. Preserve existing outputs unless an explicit migration proposal authorizes a change.

### M3 — Declarative hybrid rules and stage transitions
Move existing hybrid validation rules into data-driven constraints. Add counterexample fixtures and preserve diagnostic specificity. Do not claim a general-purpose constraint solver if the implemented rules remain finite and explicit.

### M4 — User-directed recipe construction
Offer two entry paths: choose a canonical named style and modify layers, or start from explicit musical intent and compose a recipe. Always show the result as a hybrid/derived recipe when an override changes its historical identity. Explain unsupported settings and fallbacks.

### M5 — Distinctness and regression evidence
Run deterministic structural probes and audio-capture comparisons, then owner/user playtest the changes. Publish coverage, collisions, unsupported axes, and regressions alongside showcases rather than claiming every one of 160 entries is equally mature.

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
- The routing inventory reports source-declared coverage without presenting route counts as musical capability evidence.
- The subsequent runtime-capability inventory cites source/tests/captures for actual grammar and layer behavior, rather than inferring implementation from names or labels.

## Follow-up: source-grounded control targets

The routed response-curve probe now has a separate, versioned contract for target fidelity, observed response, non-target invariance, and evidence boundaries: [Muse Control-Target Contracts v0.1](MUSE_CONTROL_TARGET_CONTRACTS_V0_1.md). It treats zero valence as neutral/unscored for the directional major/minor proxy and scores requested-bar fidelity only on the Blues and Jazz chorus routes whose plans expose an explicit quantization contract and can be checked against the emitted note span. These are evaluation changes, not composer behavior changes; the exact-head Actions artifact remains the authority for runtime outcomes. The probe also reports candidate non-target invariance for key tonic, key tonality, meter, and tempo, excluding the active target channel; for preset/fixed-mode valence routes it checks mode-lock instead. Deviations are diagnostics rather than a quality score, and the workflow summary aggregates them by style, control, channel, and seed-curve coverage. Numeric response curves also carry tie-aware Spearman rank correlations for key descriptors, with per-seed ranges in the Actions summary; these are descriptive associations, not control-success scores.


## M0e — Whole-catalog completion roadmap and Southern African expansion

The public catalog now contains 160 stable IDs across 20 constellations. Eight additional Southern African entries—Kwaito, Maskandi, Isicathamiya, Mbaqanga, Kwela, Marabi, Township Jazz, and South African Bubblegum Pop—use the explicit ResearchExpertReviewRequired status. They remain non-composable and have no composer route. This constellation is an index for separate research tracks, not a shared engine or authenticity claim. See [the research brief](./SOUTHERN_AFRICAN_GENRE_RESEARCH_BRIEF_V1.md).

The generated [genre-completion-roadmap-v1.json](./genre-completion-roadmap-v1.json) binds a proposed next-step track to every catalog row and maintains a separate proposal-only queue for further genres not yet cataloged. Candidate track names and P0/P1/P2 labels are design hypotheses and work sequencing only; they do not improve runtime coverage until a route and evidence exist.

Every track must earn its route through source-grounded target contracts, measured response and invariance, nearest-neighbor distinctness evidence, renderer receipts, and applicable specialist review.
