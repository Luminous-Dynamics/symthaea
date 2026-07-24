# Muse Research Mode Design Specification

**Status:** Proposed  
**Audience:** Muse product, UX, frontend, music-theory, rendering, and research contributors  
**Primary implementation:** Leptos web application  
**Scope:** Research Mode information architecture, interaction model, visualization behavior, data contracts, and alpha acceptance criteria  
**Related documents:**

- `MUSE_STUDIO_UI_INTERACTION_DESIGN_SPEC.md`
- `MUSE_LISTEN_MODE_VISUALIZATION_DESIGN_SPEC.md`

---

## 1. Purpose

Research Mode is Muse's evidence-oriented environment for understanding, comparing, and verifying music.

It should let a user answer questions such as:

- What is this piece made from?
- How does its form develop over time?
- Where do motifs return, transform, fragment, or disappear?
- How does the rendered performance differ from the symbolic score?
- What harmonic, rhythmic, and orchestral choices create its character?
- Why did Muse rank or select this piece?
- How similar is it to other candidates, styles, or lineages?
- Which claims are measured, which are inferred, and what evidence supports them?
- Can the exact piece, rendering, metrics, and provenance be reproduced?

Research Mode is not a decorative analytics dashboard and not a substitute for a DAW. It is an inspectable musical laboratory connected directly to Muse's score, performance, identity, and provenance data.

---

## 2. Product principles

### 2.1 Evidence before interpretation

Every interpretation should link to inspectable evidence.

A claim such as “the opening motif returns inverted” should provide a **Show Evidence** action that highlights the relevant score regions and transformation relationship.

### 2.2 Observation, interpretation, and uncertainty remain distinct

Research Mode should clearly separate:

1. **Observed data** — notes, timing, velocity, sections, harmonic labels, rendering metadata.
2. **Derived measurements** — novelty, density, motif similarity, tension, orchestration balance.
3. **Interpretations** — why a moment matters, how a passage functions, why a candidate was selected.

Interpretations must not be presented as raw measurements.

### 2.3 No false precision

Metrics should display:

- units or scale;
- method or source;
- confidence where applicable;
- comparison baseline;
- known limitations.

A value such as `Φ 0.862` should never appear without an accessible explanation of what it measures, how it was calculated, and what it does not establish.

### 2.4 Linked exploration

Selecting a section, motif, note region, harmonic event, or timeline moment should update every compatible view.

This linked-selection model is the central interaction pattern of Research Mode.

### 2.5 Static composition and dynamic performance remain separate

Research Mode must preserve the distinction between:

- the symbolic composition;
- the current rendered performance;
- perceptual or resonance analysis.

These layers may be overlaid, but should never become ambiguous.

### 2.6 Reproducibility is a product feature

Every analysis should retain:

- piece identifier;
- recipe hash;
- engine version;
- analysis profile;
- renderer and sound-pack version;
- seed;
- metric version;
- creation timestamp;
- source artifacts.

### 2.7 Density is earned through navigation

The concept mockup is a useful design board, but the final product should not display every panel at full prominence simultaneously.

Research Mode should use:

- a concise Overview;
- dedicated deep-dive pages;
- linked drawers and inspectors;
- pinning and comparison workspaces.

---

## 3. Intended users

### 3.1 Curious musician

Wants accessible explanations without needing to read research terminology.

### 3.2 Composer or arranger

Wants to inspect motifs, harmony, form, voice leading, orchestration, and performance behavior.

### 3.3 Researcher

Wants exact definitions, methods, uncertainty, comparisons, exports, and reproducibility.

### 3.4 Muse developer

Wants diagnostic evidence, metric provenance, experiment outputs, and implementation-level traces.

Research Mode should support all four through analysis profiles rather than separate products.

---

## 4. Analysis profiles

The global **Analysis Profile** selector changes density and terminology without changing the underlying evidence.

### 4.1 Guided

For curious listeners and musicians.

- plain-language labels;
- reduced metrics;
- explanations before formulas;
- automatic highlighting;
- limited configuration.

### 4.2 Composer

For practical musical inspection.

- notation and piano-roll detail;
- harmony and form labels;
- motif transformation tools;
- orchestration and performance views;
- comparison controls.

### 4.3 Deep Dive

For researchers and developers.

- metric definitions;
- confidence and uncertainty;
- raw values;
- analysis versions;
- diagnostic overlays;
- export options.

The profile is a presentation preference, not a change to the piece or analysis result.

---

## 5. Information architecture

Research Mode uses the following primary routes or tabs.

1. **Overview**
2. **Score**
3. **Harmony**
4. **Motifs**
5. **Orchestration**
6. **Performance**
7. **Identity**
8. **Lineage**
9. **Metrics**
10. **Provenance**
11. **Compare**
12. **Experiments** — later phase

The top-level route should remain stable while the selected piece and playback position persist.

---

## 6. Global Research shell

### 6.1 Header

Contains:

- Muse brand;
- Listen, Create, and Research mode switcher;
- current piece;
- current playback position;
- audio controls;
- analysis profile selector;
- export menu;
- settings and user identity.

Behavior:

- entering Research Mode does not stop playback;
- opening a different analysis view preserves the selected moment;
- changing profile does not recompute analysis unless the profile requires unavailable data;
- Export applies to the active piece and active analysis context.

### 6.2 Piece context bar

Persistent beneath the header.

Contains:

- piece artwork or deterministic visual identity;
- title;
- style, form, key, meter, tempo, duration;
- current render identifier;
- current section and bar;
- Open in Listen;
- Open in Create;
- Compare;
- pin status.

### 6.3 Research navigation

Desktop uses horizontal tabs or a compact secondary sidebar.

Mobile and narrow tablet layouts use an overflow menu or bottom sheet.

### 6.4 Shared timeline

A compact synchronized timeline remains available across all research routes.

It may show:

- form regions;
- current playhead;
- selected moment;
- motif markers;
- harmonic events;
- performance dynamics;
- annotations.

The route-specific page may expand this into a full analysis timeline.

---

## 7. Research Overview

Overview is a summary and navigation surface, not the entire laboratory on one screen.

### 7.1 Primary overview regions

Recommended desktop composition:

- **Structural portrait** — radial or linear form summary;
- **Emotional and resonance contour**;
- **Motif activity summary**;
- **Structural density summary**;
- **Why this piece?**;
- **Identity summary**;
- **Provenance summary**;
- **Similarity neighborhood**;
- **Analysis timeline**.

No more than four major analytical cards should compete at the same visual level above the fold.

### 7.2 Card behavior

Every card supports:

- Open full view;
- Show definition;
- Show evidence;
- pin to comparison;
- export current data where appropriate.

### 7.3 Overview narrative

The Overview should generate a concise, evidence-grounded summary:

- formal identity;
- primary motif behavior;
- harmonic journey;
- performance character;
- most distinctive measured trait;
- known caveat or uncertainty.

---

## 8. Linked-selection model

Research Mode maintains one shared `ResearchSelection`.

A selection may represent:

- time or bar;
- time range;
- section;
- phrase;
- motif occurrence;
- harmonic region;
- voice or instrument;
- metric interval;
- lineage node;
- comparison candidate.

### 8.1 Selection behavior

When the user selects a motif occurrence:

- Score highlights the notes;
- Harmony highlights the underlying region;
- Performance shows expressive treatment;
- Orchestration highlights active voices;
- Identity shows relevant identity events;
- Lineage shows related transformations;
- the explanation inspector updates.

### 8.2 Hover versus selection

- Hover previews relationships without changing persistent state.
- Click selects and synchronizes all views.
- Shift-click adds to a comparison selection.
- Escape clears the active selection.

### 8.3 Pinning

Pinned selections survive route changes and can be used in Compare.

---

## 9. Score view

Score view answers: **What was composed?**

### 9.1 Representations

Supported representations:

- standard notation;
- piano roll;
- voice-leading reduction;
- event table — Deep Dive profile.

### 9.2 Core controls

- representation selector;
- voice and instrument visibility;
- concert versus transposed pitch;
- bar and beat grid;
- motif labels;
- harmonic labels;
- section labels;
- zoom;
- follow playhead;
- loop selection.

### 9.3 Evidence actions

The user can select notes and ask:

- Which motif is this?
- Where does it recur?
- How was it transformed?
- What harmony supports it?
- How was it performed?
- Which identity rule produced it?

### 9.4 Editing boundary

Research Mode may allow annotations and selections, but symbolic editing belongs in Create.

An **Edit in Create** action should preserve the exact selected region.

---

## 10. Harmony view

Harmony view answers: **How does tonal or modal organization move through the piece?**

### 10.1 Primary visualizations

- key or mode journey;
- chord timeline;
- harmonic-function map;
- cadence map;
- tonal-center wheel or graph;
- harmonic rhythm;
- stability or tension curve.

### 10.2 Label levels

Users can choose:

- plain language;
- chord symbols;
- scale degrees;
- Roman numerals;
- pitch-class detail.

### 10.3 Uncertainty

If harmonic analysis is inferred rather than generated directly from the recipe, Research Mode must display confidence or ambiguity.

Alternative analyses may be shown when plausible.

### 10.4 Evidence links

Selecting a chord or region highlights:

- sounding notes;
- non-chord tones;
- bass motion;
- cadence evidence;
- motif relationship;
- current key context.

---

## 11. Motifs view

Motifs view answers: **What musical identities persist and how do they change?**

### 11.1 Primary representations

- motif network;
- motif timeline;
- occurrence table;
- transformation tree;
- notation or piano-roll snippets.

### 11.2 Transformation vocabulary

Muse may identify:

- literal return;
- transposition;
- inversion;
- retrograde where supported;
- augmentation;
- diminution;
- fragmentation;
- ornamentation;
- rhythmic displacement;
- recombination;
- erosion;
- restoration;
- sequence;
- register transfer.

### 11.3 Confidence and thresholds

Similarity thresholds must be inspectable.

A motif relation should expose:

- similarity value;
- feature channels used;
- threshold;
- transformation classification;
- uncertain or competing classifications.

### 11.4 Interaction

Selecting a motif node highlights every occurrence across the product.

The user can compare two occurrences directly.

---

## 12. Orchestration view

Orchestration view answers: **Who plays what, where, and with what structural role?**

### 12.1 Primary visualizations

- instrument activity timeline;
- register distribution;
- density stack;
- role map: lead, counterline, harmony, bass, pulse, texture;
- doubling graph;
- entrance and exit events;
- prominence curves.

### 12.2 Interaction

Selecting an instrument filters Score and Performance to that voice.

Selecting a time region explains:

- active ensemble;
- role changes;
- register movement;
- texture density;
- doubling;
- orchestration contrast from the previous section.

### 12.3 Static versus performed orchestration

The composition layer shows assigned parts.

The performance layer shows actual prominence and dynamics.

---

## 13. Performance view

Performance view answers: **How was the symbolic piece realized?**

### 13.1 Primary channels

- actual timing;
- timing deviation from grid;
- velocity;
- dynamics;
- articulation;
- sustain or pedal;
- phrase shape;
- voice prominence;
- optional renderer-specific expression.

### 13.2 Compare composed versus performed

Performance view should support three states:

- Composed;
- Performed;
- Overlay.

Overlay should show:

- score onset versus rendered onset;
- score duration versus rendered duration;
- dynamic shaping;
- phrase-level deviation;
- articulation changes.

### 13.3 Renderer comparison

A later version may compare multiple renderings of the same composition.

The symbolic score remains fixed while performance layers change.

---

## 14. Identity view

Identity view answers: **What makes this piece itself?**

### 14.1 Identity channels

Current supported channels may include:

- melodic;
- rhythmic;
- harmonic;
- orchestration;
- premise distance;
- form;
- development behavior;
- identity grammar.

Do not force every identity concept into one radar chart.

### 14.2 Recommended layout

- compact identity profile;
- channel-by-channel evidence;
- strongest and weakest identity carriers;
- identity events over time;
- nearby and distant pieces;
- preserved versus changed features.

### 14.3 Explainability

Each channel should answer:

- what is measured;
- how it is normalized;
- comparison baseline;
- where in the piece the evidence occurs;
- limitations.

### 14.4 Identity grammar

Show grammar behavior such as:

- Persistence;
- Lineage;
- Erosion;
- Memory;
- Auto or style-default identity.

The view should link abstract identity claims to concrete motif, form, harmony, and orchestration events.

---

## 15. Lineage view

Lineage view answers: **How is this piece related to its sources, variants, and descendants?**

### 15.1 Node types

- original recipe;
- generated candidate;
- kept piece;
- variation;
- transformed piece;
- performance render;
- user edit;
- published artifact.

### 15.2 Edge types

- generated from;
- varied from;
- preserves motif;
- preserves form;
- changes harmony;
- re-orchestrates;
- re-renders;
- user-edited from;
- published as.

### 15.3 Interaction

Selecting an edge reveals:

- preserved features;
- changed features;
- recipe diff;
- score diff;
- performance diff;
- provenance evidence.

### 15.4 Trust

Lineage claims must be backed by stored identifiers or computed evidence.

Inferred relationships should be labeled as inferred.

---

## 16. Metrics view

Metrics view answers: **What has Muse measured, and how should the result be interpreted?**

### 16.1 Metric card contract

Every metric card must include:

- metric name;
- current value;
- unit or range;
- definition;
- algorithm version;
- input artifacts;
- baseline or comparison set;
- uncertainty or caveat;
- time-local evidence where relevant.

### 16.2 Metric categories

- structure;
- motif identity;
- novelty and distance;
- harmonic behavior;
- rhythmic behavior;
- orchestration;
- performance;
- resonance;
- coherence or integration;
- prompt fit;
- listener-derived signals, if enabled.

### 16.3 Metric comparisons

Users should be able to compare:

- this piece against candidates from the same generation;
- this piece against a style distribution;
- multiple seeds;
- multiple renderers;
- kept versus rejected pieces;
- original versus variation.

### 16.4 Metric misuse prevention

Research Mode should warn when:

- a metric is being compared across incompatible analysis versions;
- the baseline is too small;
- the metric does not support the requested conclusion;
- a value is descriptive rather than evaluative.

---

## 17. Provenance view

Provenance view answers: **Can this result be traced and reproduced?**

### 17.1 Required fields

- piece ID;
- recipe hash;
- seed;
- engine version;
- source commit where available;
- style and grammar versions;
- analysis profile and version;
- renderer version;
- sound-pack or sample source;
- license metadata;
- creation time;
- render time;
- platform details where relevant;
- exported artifact hashes.

### 17.2 Artifact graph

Show relationships among:

- recipe;
- score;
- MIDI;
- rendered audio;
- analysis bundles;
- cover or sigil;
- export package.

### 17.3 Reproduce

A **Reproduce** action should verify whether the current environment can recreate:

- symbolic score exactly;
- MIDI exactly;
- audio exactly or within declared renderer limits;
- metrics under the same algorithm version.

Any non-deterministic boundary must be disclosed.

---

## 18. Compare workspace

Compare is a dedicated workspace, not a small modal.

### 18.1 Comparison targets

- two or more pieces;
- two motifs;
- two sections;
- two performances;
- original and variation;
- candidate and kept piece;
- style archetypes;
- analysis versions.

### 18.2 Layout

Recommended desktop layout:

- pinned comparison items across the top;
- synchronized timeline;
- selectable comparison dimension;
- difference summary;
- evidence panels;
- metric table;
- export.

### 18.3 Difference categories

- form;
- melody;
- rhythm;
- harmony;
- orchestration;
- performance;
- identity;
- provenance;
- metric deltas.

### 18.4 Honest comparison

The interface must distinguish:

- exact equality;
- measured similarity;
- inferred relationship;
- subjective interpretation.

---

## 19. Experiments workspace

**Phase P2 or later.**

Experiments connects Muse's internal research discipline to a usable interface.

### 19.1 Experiment record

- question;
- hypothesis;
- pre-registered success and failure criteria;
- frozen inputs;
- conditions and controls;
- seeds;
- metric versions;
- results;
- artifacts;
- limitations;
- conclusion;
- status: planned, running, complete, invalidated, superseded.

### 19.2 Experiment views

- run matrix;
- result distributions;
- listening-test results;
- failure analysis;
- artifact browser;
- reproducibility status.

### 19.3 Negative results

Negative and null results should be first-class outcomes, not hidden failures.

---

## 20. Explanation inspector

The right-side inspector is shared across Research Mode.

Recommended sections:

- selected object or moment;
- observation;
- interpretation;
- significance;
- confidence;
- evidence links;
- related structures;
- provenance.

The inspector should never obscure the underlying evidence permanently.

On narrow screens it becomes a bottom sheet.

---

## 21. Visual language

Research Mode is denser than Listen or Create, but should remain recognizably Muse.

### 21.1 Semantic color roles

Recommended mapping:

- **Purple:** composition and static structure;
- **Cyan:** performance and realized behavior;
- **Gold/amber:** resonance, focus, and meaningful selection;
- **Rose/red:** selected event, anomaly, or conflict;
- **Warm gray:** relationship, baseline, and inactive data.

Color must not be the only differentiator.

### 21.2 Typography

- serif for piece titles and major conceptual headings;
- sans-serif for controls, labels, metrics, and tables;
- monospace for hashes, versions, seeds, and exact values.

### 21.3 Density

Use strong surface hierarchy and spacing rather than excessive borders.

Avoid presenting six equally bright charts in one viewport.

---

## 22. Interaction conventions

### 22.1 Playback

- Space: play or pause;
- Left/Right: seek by configured interval;
- Shift + Left/Right: move by bar or phrase;
- `L`: toggle loop selection;
- `F`: follow playhead;
- Escape: clear selection or close inspector.

### 22.2 Selection

- click: select;
- shift-click: add to comparison;
- drag: select time range;
- double-click: zoom to object or region;
- hover: preview linked evidence.

### 22.3 Navigation

Opening evidence from an explanation changes the active route and preserves a breadcrumb back to the originating claim.

---

## 23. Responsive behavior

### 23.1 Desktop

- persistent header and piece context bar;
- route navigation;
- primary canvas;
- right explanation inspector;
- compact shared timeline.

### 23.2 Tablet

- inspector collapses into tabs or drawer;
- route tabs become horizontally scrollable;
- comparison uses stacked panes;
- timeline remains available below primary visualization.

### 23.3 Mobile

Research Mode should prioritize inspection rather than replicate the desktop laboratory.

Mobile surfaces:

- overview summary;
- selected-moment explanation;
- form;
- motifs;
- harmony;
- provenance;
- compact compare.

Full notation and dense multi-track views may direct users to landscape or desktop mode.

---

## 24. Accessibility

Research Mode must remain understandable without color, animation, or chart interpretation.

Requirements:

- keyboard navigation;
- visible focus states;
- semantic headings and regions;
- text summaries for charts;
- data-table alternatives;
- reduced motion;
- high contrast;
- no hover-only actions;
- screen-reader announcement of playback and selection changes without excessive interruption;
- confidence and uncertainty expressed in text.

---

## 25. Performance requirements

Audio playback has priority over visualization.

Recommended constraints:

- precompute expensive analysis;
- render dense plots with Canvas or WebGL;
- use DOM for controls, labels, accessibility, and low-density graphics;
- virtualize long tables and event lists;
- suspend hidden visualizations;
- decimate curves at low zoom;
- retain full-resolution source data for inspection and export;
- do not recompute analyses on every hover or playback frame.

Research Mode should become useful incrementally as analysis bundles arrive.

---

## 26. Data contracts

All research bundles share stable identifiers and a common musical-time mapping.

### 26.1 Shared analysis envelope

```rust
pub struct AnalysisEnvelope<T> {
    pub piece_id: PieceId,
    pub render_id: Option<RenderId>,
    pub analysis_id: AnalysisId,
    pub analysis_version: String,
    pub created_at: Timestamp,
    pub confidence: Option<Confidence>,
    pub warnings: Vec<AnalysisWarning>,
    pub payload: T,
}
```

### 26.2 Research selection

```rust
pub struct ResearchSelection {
    pub time_range: Option<TimeRange>,
    pub section_ids: Vec<SectionId>,
    pub phrase_ids: Vec<PhraseId>,
    pub motif_occurrence_ids: Vec<MotifOccurrenceId>,
    pub harmony_region_ids: Vec<HarmonyRegionId>,
    pub voice_ids: Vec<VoiceId>,
    pub lineage_node_ids: Vec<LineageNodeId>,
    pub comparison_piece_ids: Vec<PieceId>,
}
```

### 26.3 Evidence reference

```rust
pub struct EvidenceReference {
    pub claim_id: ClaimId,
    pub evidence_type: EvidenceType,
    pub artifact_id: ArtifactId,
    pub time_range: Option<TimeRange>,
    pub entity_ids: Vec<EntityId>,
    pub explanation: String,
}
```

### 26.4 Metric result

```rust
pub struct MetricResult {
    pub metric_id: MetricId,
    pub metric_version: String,
    pub value: MetricValue,
    pub scale: MetricScale,
    pub baseline: Option<MetricBaseline>,
    pub uncertainty: Option<MetricUncertainty>,
    pub evidence: Vec<EvidenceReference>,
    pub limitations: Vec<String>,
}
```

---

## 27. Leptos component map

```text
ResearchPage
├── GlobalHeader
├── PieceContextBar
├── ResearchNav
├── ResearchWorkspace
│   ├── ResearchOverview
│   ├── ScoreWorkspace
│   ├── HarmonyWorkspace
│   ├── MotifWorkspace
│   ├── OrchestrationWorkspace
│   ├── PerformanceWorkspace
│   ├── IdentityWorkspace
│   ├── LineageWorkspace
│   ├── MetricsWorkspace
│   ├── ProvenanceWorkspace
│   ├── CompareWorkspace
│   └── ExperimentsWorkspace
├── SharedAnalysisTimeline
├── ExplanationInspector
└── ResearchTransport
```

Recommended shared stores:

- `PlaybackStore`
- `CurrentPieceStore`
- `ResearchRouteStore`
- `ResearchSelectionStore`
- `PinnedComparisonStore`
- `AnalysisBundleStore`
- `ExplanationStore`
- `ResearchPreferencesStore`

---

## 28. State model

```rust
pub enum ResearchRoute {
    Overview,
    Score,
    Harmony,
    Motifs,
    Orchestration,
    Performance,
    Identity,
    Lineage,
    Metrics,
    Provenance,
    Compare,
    Experiments,
}

pub enum AnalysisProfile {
    Guided,
    Composer,
    DeepDive,
}

pub struct ResearchUiState {
    pub route: ResearchRoute,
    pub profile: AnalysisProfile,
    pub selection: ResearchSelection,
    pub pinned_claims: Vec<ClaimId>,
    pub pinned_items: Vec<ComparisonItem>,
    pub follow_playhead: bool,
    pub inspector_open: bool,
    pub visible_layers: ResearchLayerVisibility,
}
```

Playback state remains independent from Research Mode state.

---

## 29. Loading and failure states

### 29.1 Score ready, analysis pending

Show the score immediately and mark unavailable views as computing.

### 29.2 Partial analysis

Render available cards and identify missing channels.

Do not fabricate empty charts.

### 29.3 Analysis failure

Show:

- failed analysis stage;
- error summary;
- retry;
- diagnostics in Deep Dive profile;
- unaffected available views.

### 29.4 Version mismatch

Warn before comparing incompatible analysis versions.

Offer recomputation where possible.

### 29.5 Missing provenance

Mark the piece as partially reproducible and list missing artifacts.

---

## 30. Export behavior

Export should be context-sensitive.

Supported outputs may include:

- JSON analysis bundle;
- CSV metrics or event table;
- MIDI;
- MusicXML;
- WAV or rendered audio;
- PNG or SVG visualization;
- provenance manifest;
- comparison report;
- reproducibility package.

Every export should include relevant version metadata.

---

## 31. Implementation priorities

### P0 — Research alpha

- Research shell and routing;
- Overview;
- Score view with notation or piano roll;
- shared timeline and linked playhead;
- Harmony view;
- Motifs view;
- Performance composed/performed overlay;
- explanation inspector;
- Why This Piece? with evidence links;
- Provenance summary;
- Guided and Composer profiles;
- desktop layout and basic responsive support.

### P1 — Full research workflow

- Orchestration view;
- Identity view;
- Lineage view;
- Metrics view with definitions and caveats;
- Compare workspace;
- pinning and linked brushing;
- exports;
- Deep Dive profile;
- tablet and mobile refinement;
- accessibility data tables.

### P2 — Research platform

- Experiments workspace;
- renderer comparison;
- analysis-version comparison;
- listening-test integration;
- reproducibility verification;
- user annotations;
- shareable research reports;
- collaborative review.

---

## 32. Alpha acceptance criteria

Research Mode is ready for alpha when:

1. Playback persists across Research routes and mode switches.
2. Selecting a time region synchronizes Score, Harmony, Motifs, Performance, and the inspector.
3. Score and performance are independently viewable and can be overlaid.
4. Every generated interpretation offers inspectable evidence or explicitly states that evidence is unavailable.
5. Metric cards include definitions, versions, scales, and limitations.
6. Research Overview remains legible without showing every deep-dive panel simultaneously.
7. Why This Piece? explains ranking or selection rather than making unsupported claims about the user.
8. Provenance identifies the recipe, engine, renderer, analysis version, and available artifacts.
9. Missing or failed analyses are disclosed rather than represented by invented values.
10. Visual performance does not interrupt audio playback.
11. Keyboard navigation and reduced-motion behavior work for all P0 routes.
12. A composer can move from an identified issue or insight directly into the corresponding region in Create.

---

## 33. Open questions

- Which harmonic and motif analyses are generated directly versus inferred after composition?
- What confidence model should be used across heterogeneous analysis channels?
- Which metrics are mature enough for alpha exposure?
- How should Musical Φ be defined and bounded in the UI?
- Which comparisons require frozen reference distributions?
- Should user annotations become provenance-bearing artifacts?
- How much raw diagnostic data belongs in the public Deep Dive profile?
- Which research views should be available for imported MIDI or audio?
- Can the same evidence model support MuseBench and external validation?

---

## 34. Final design intent

Research Mode should feel like:

- a score reader when the user wants musical detail;
- a laboratory when the user wants evidence;
- a map when the user wants relationships;
- a provenance ledger when the user wants trust.

Its core promise is:

> Muse does not merely describe a piece. It lets the user inspect the score, hear the performance, trace the structure, test the interpretation, and reproduce the result.
