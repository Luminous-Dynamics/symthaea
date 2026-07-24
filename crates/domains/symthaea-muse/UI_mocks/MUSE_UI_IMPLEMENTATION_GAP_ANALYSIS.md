# Muse UI Implementation Gap Analysis

**Status:** Active implementation ledger
**Scope:** Current `symthaea-muse` executable bridge versus the Listen, Research, and Studio design specifications
**Last reviewed:** 2026-07-18

## 1. Current architecture truth

The current executable product path is still an Axum server with an embedded HTML/CSS/JavaScript client:

- `src/bin/muse_studio.rs` serves the shell, audio artifacts, and inspectable piece bundles.
- `studio/index.html` defines the product shell.
- `studio/muse-studio.css` and `studio/muse-studio.js` implement the current interaction layer.
- `symthaea-muse-protocol` owns versioned composition, performance, and provenance wire contracts.
- `/api/compose` returns candidate metadata.
- `/api/audio/{id}` and `/api/midi/{id}` expose rendered artifacts.
- `/api/piece/{id}/listen-bundle` exposes score, motif, sonority, cadence, orchestration, and bounded structural-activity evidence.
- `/api/piece/{id}/performance-bundle` exposes realized notes and score-event mappings where the renderer preserves them.
- `/api/piece/{id}/provenance` exposes artifact hashes, version facts, and bounded reproducibility claims.

This remains an executable migration bridge rather than the canonical Leptos architecture in the specifications. The bridge now has enough typed evidence to serve as a behavior reference, but it should not become a reason to postpone shared Leptos state, route ownership, or semantic Studio work.

## 2. What the bridge now supports

### Listen

- Product-level Listen / Create / Research navigation.
- Persistent current-piece identity in the global header.
- A three-region Listen layout: current piece, central visualizer, and context rail.
- Hybrid, Form, Score, Compare, Resonance, Motifs, Harmony, and Orchestration purposes.
- A real form ring derived from emitted section regions.
- Motif occurrence and cadence markers layered on the whole-piece map.
- Seekable section labels and current-section highlighting.
- A symbolic score view independent of performed notes.
- A composed-versus-performed comparison view with source-note mapping where available.
- A motif timeline with transformation label, similarity, evidence status, and method.
- A score-side sonority path with exact declared-home-key triad labels only where justified.
- Marked cadential arrivals without invented cadence types.
- Composition-side voice assignment by structural region, explicitly separate from rendered prominence.
- A score-derived structural-activity curve with explicit proxy limitations and a performed-density fallback.
- A current-moment card linking motif, sonority, cadence, and assigned voices to playback time.
- Grounded `why` explanations emitted by the composer.
- Discovery, Resonance, and Contrast journey policies spanning the documented valence, arousal, and energy ranges.
- Unavailable visualization modes disabled rather than populated with fabricated data.
- Reduced-motion throttling and suspension of hidden Listen visualization work.

### Research

- An evidence-oriented Overview rather than a decorative all-panels dashboard.
- Dedicated Performance, Motifs, Harmony, Orchestration, Identity, and Provenance views.
- A synchronized composition/performance timeline.
- Motif and cadence overlays on the shared timeline.
- Observed form, sections, phrases, symbolic-note count, meter, and duration.
- A motif workspace with recipe definition, bounded occurrences, classifications, confidence, method, and declared limitations.
- A harmony workspace that shows observed sounding pitch classes, exact home-key triads, and marked arrivals without pretending to perform modulation or cadence-type analysis.
- An orchestration workspace that shows symbolic voice assignments, note shares, and register span without calling them perceptual prominence.
- A selected-moment inspector that cites score, performance, and structural evidence identifiers.
- Derived score metrics with explicit caveats.
- Identity traits and novelty channels only when the candidate emits them.
- Piece-level interpretations kept separate from observed, reconstructed, and inferred data.
- Explicit availability reporting for each evidence family.
- Artifact hashes, engine versions, renderer facts, exactness flags, and declared reproducibility limitations.

### Create

- Prompt-first candidate generation.
- Guided, Composer, and Advanced density profiles that no longer masquerade as product modes.
- Candidate-level navigation into Research.
- Integration metrics labeled as score analysis rather than an unexplained `Φ` claim.

Create is still not the semantic Studio described by the design specification.

## 3. Versioned data now available

### Composition bundle v2

`ListenCompositionBundle` provides one shared tick, beat, and rendered-seconds coordinate system plus:

- tempo and meter maps;
- form kind;
- score-grounded section regions;
- phrase regions reconstructed from explicit phrase and cadential emphasis annotations;
- exact symbolic note events with stable IDs, pitch, duration, role, emphasis, and section intensity;
- the exact motif definition selected by the resolved recipe;
- score-window motif occurrences with transformation classification, similarity, source note IDs, method, confidence, and limitations;
- exact cadential-emphasis arrival markers with source note IDs but no invented cadence type;
- active score pitch-class regions per beat;
- optional home-key scale degree and function only for an exact diatonic triad in the declared home key;
- symbolic voice assignment, note count, register span, and mean velocity by structural region;
- score-derived energy, density, and melodic-motion samples with method and limitations.

The new evidence fields use `serde(default)` so v1 payloads remain readable. Evidence carries an explicit epistemic status:

- `observed` — copied directly from recipe or score fields;
- `reconstructed` — deterministically rebuilt from emitted fields;
- `inferred` — a bounded analysis result with method, confidence where meaningful, and limitations.

### Performance bundle

`ListenPerformanceBundle` provides:

- rendered voices and instruments;
- performed note events;
- stable performed-event IDs;
- source symbolic-note IDs for primary voices when event counts preserve a one-to-one ordered mapping;
- actual onset and duration;
- onset and duration deviation from the symbolic score;
- warnings when source mapping cannot be claimed.

Renderer-added color or doubling voices intentionally remain unmapped rather than receiving invented source IDs.

### Provenance bundle

`PieceProvenanceBundle` provides:

- recipe schema version and SHA-256;
- score SHA-256;
- rendered-audio SHA-256;
- seed and style;
- Muse and theory engine versions;
- renderer name and optional version;
- source revisions and renderer/input digests when recorded;
- artifact references;
- separate exactness claims for symbolic score, MIDI, and rendered audio;
- explicit limitations and warnings when independent exact reproduction is not supportable.

## 4. Honest boundaries of the new evidence

The new panels are real, but they are intentionally narrower than their final design-spec meanings.

### Motif

Current truth:

- the recipe motif definition is exact;
- occurrence candidates are discovered by a bounded symbolic score-window scan;
- source note IDs, similarity, transformation label, method, and limitations are inspectable.

Still missing:

- composer-owned motif occurrence IDs emitted during construction;
- phrase-aware segmentation and voice-crossing relationships;
- competing classifications and threshold controls;
- confidence calibration against a labeled corpus;
- explicit fragmentation, recombination, erosion, restoration, and lineage events where the composer already knows them.

### Harmony and cadence

Current truth:

- sounding pitch classes are observed from active symbolic notes;
- declared-home-key degree and function appear only for exact diatonic triads;
- cadential-emphasis arrivals are observed exactly from score markers.

Still missing:

- key-region and modulation analysis;
- chord spelling and inversion;
- non-chord-tone treatment;
- harmonic function under ambiguity;
- cadence-type classification and supporting evidence;
- alternate plausible analyses and confidence.

### Orchestration

Current truth:

- score voice roles, note counts, register span, and mean velocity are grouped by structural region.

Still missing:

- explicit orchestration-role events independent of current voice-role labels;
- doubling and role-transfer relationships;
- entrance and exit event semantics;
- performed prominence and masking;
- renderer-added color-voice mappings;
- spectral evidence from the produced audio.

### Resonance

Current truth:

- energy, density, and motion are a versioned score-activity proxy;
- the method and limitations are exposed;
- the UI does not call this objective emotion.

Still missing:

- tension, brightness, warmth, and orchestration intensity;
- authored expressive intent as a separate channel;
- performed and audio-derived measurements;
- comparison baselines and uncertainty calibration;
- validated relationships between displayed curves and listener judgments.

## 5. Do not fake these panels

The UI must continue to avoid:

- sonata labels inferred from equal time quarters;
- authored motif returns inferred from similarity alone;
- harmonic paths inferred from brightness or energy;
- modulation inferred from one local pitch-class set;
- cadence types inferred from an arrival marker alone;
- emotional truth inferred from amplitude, density, or the structural-activity proxy;
- perceptual prominence inferred from score note counts alone;
- lineage edges without stored parent identifiers;
- exact reproducibility without the required hashes, versions, revisions, and environment inputs.

A visible **Not emitted** state remains a product feature, not a failure.

## 6. Protocol and endpoint status

Implemented versioned endpoints:

- `GET /api/piece/{id}/listen-bundle`
- `GET /api/piece/{id}/performance-bundle`
- `GET /api/piece/{id}/provenance`

Each uses a shared envelope containing:

- piece ID;
- optional render ID;
- bundle version;
- candidate creation time;
- non-fatal warnings;
- typed payload.

Protocol serialization tests cover v2 evidence and v1-compatible defaults. Server helper tests cover stable symbolic IDs, section boundaries, cadence note binding, exact home-key triad recognition, and orchestration grouping.

The next endpoints should be added only when their underlying state becomes independently useful or independently versioned:

- `GET /api/piece/{id}/identity-bundle`
- `GET /api/piece/{id}/lineage`
- optional dedicated motif, harmony, orchestration, or resonance endpoints if those analyses outgrow the composition bundle or require asynchronous computation.

Do not split the current evidence into extra endpoints merely to imitate the document component map.

## 7. Leptos migration sequence

The bridge should be migrated without pausing product progress.

### Phase A — shared state and protocol

1. Keep the protocol crate authoritative for wire types and compatibility tests.
2. Introduce a persistent `PlaybackStore` independent of route-local state.
3. Introduce `CurrentPieceStore`, `SelectedMomentStore`, `ResearchSelectionStore`, and `AnalysisBundleStore`.
4. Preserve evidence IDs and epistemic status in every store projection.
5. Add explicit loading, stale-version, warning, and unavailable states.

### Phase B — shell and Listen

1. Recreate the current global shell as Leptos components.
2. Move audio ownership into `PlaybackStore`.
3. Implement `ListenPage`, `JourneyRail`, `ListenContextRail`, and transport components.
4. Move dense Canvas rendering behind one renderer interface that consumes protocol bundles.
5. Preserve visualization purposes as musical concepts rather than geometry names.
6. Expose a textual current-moment summary and reduced-motion path from the start.
7. Ensure hidden routes do not redraw evidence canvases.

### Phase C — Research

1. Implement stable Research routes rather than one monolithic dashboard.
2. Reuse one selected-moment and selected-object model across Score, Performance, Motifs, Harmony, Orchestration, and the inspector.
3. Convert evidence IDs into typed `EvidenceReference` values rather than composing endpoint strings in UI text.
4. Add keyboard linked selection, range selection, pinning, and comparison.
5. Add metric definitions, versions, baselines, warnings, and evidence references.
6. Add provenance and export views.
7. Add Compare only after stable cross-piece and cross-version identifiers exist.

### Phase D — Studio

Studio must not be represented as a restyled Create form. It requires a semantic editing backend.

The existing `studio_contract` module is a strong starting point, but the UI still needs:

- stable selections for piece, section, phrase, motif, harmony region, voice, and note range;
- a general endpoint that resolves natural-language or structured requests into a visible Preserve / Change contract;
- exact versus semantic lock semantics;
- constraint conflict reporting;
- generation of a small alternative set;
- structural, score, and performance diffs;
- audition without changing the canonical version;
- version nodes and commit semantics;
- recipe and artifact lineage.

Recommended endpoints:

- `POST /api/studio/contracts/resolve`
- `POST /api/studio/alternatives`
- `GET /api/studio/alternatives/{id}`
- `POST /api/studio/alternatives/{id}/commit`
- `GET /api/studio/versions/{piece_id}`
- `POST /api/studio/versions/{piece_id}/canonical`

The Sonata intervention endpoint may demonstrate parts of this loop, but it must not become the implicit general Studio contract.

## 8. Highest-value next patches

1. Retain explicit composer-owned section, phrase, motif-occurrence, cadence-role, and development-operation identifiers in the score or recipe.
2. Add a versioned harmonic-analysis result with ambiguity and evidence, rather than expanding the current exact-triad recognizer into unsupported claims.
3. Add performed phrase shape, articulation, sustain, and voice-prominence bundles.
4. Introduce typed evidence references and one shared Research selection model.
5. Move playback and current-piece ownership into Leptos stores.
6. Implement the first complete semantic Studio loop: select → preserve/change → alternatives → compare → commit.
7. Add lineage only when version nodes and parent relationships are stored, not inferred.

## 9. Visual refinement priorities

1. Keep gold for committed identity and meaningful focus.
2. Use violet for composition and proposals, cyan for performed realization, and explicit status text for epistemic differences.
3. Reduce simultaneous high-contrast borders; hierarchy should come primarily from spacing and luminance.
4. Make the central visualization the brightest region in Listen.
5. Keep Research cards quieter than selected evidence.
6. Use deterministic cover sigils distinct from the structural visualizer.
7. Keep text at practical reading sizes rather than reproducing the mockup's smallest labels literally.
8. Preserve restrained motion and never animate structure independently from music.
9. Use warning and unavailable states as part of the visual system, not generic error banners.
10. Keep confidence visually subordinate to the evidence and method that produced it.

## 10. Verification gates

Before calling the design-spec migration complete:

- Rust workspace tests and clippy pass under the canonical Nix environment.
- Protocol/server wire-compatibility tests pass.
- JavaScript syntax checks pass for the bridge.
- Browser smoke tests exercise every Listen visualization purpose and each Research evidence route.
- Playback continues across mode and route changes.
- Hidden views do not run continuous animation or analysis work.
- Reduced-motion mode remains fully usable.
- A screen reader receives current piece, current time, current section, and selected-evidence summaries.
- Keyboard users can seek, play, change view, and inspect evidence.
- Missing analysis is disclosed rather than rendered as zero-valued charts.
- Every motif, harmony, cadence, orchestration, and resonance claim exposes status, method, evidence identifiers where applicable, and limitations.
