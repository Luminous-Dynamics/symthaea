# Muse Listen Mode Visualization Design Specification

**Status:** Proposed  
**Audience:** Muse product, UX, frontend, audio, and visualization contributors  
**Primary implementation:** Leptos web application  
**Scope:** Listen Mode visualization, interaction, state, and data contracts  
**Related:** `MUSE_STUDIO_UI_INTERACTION_DESIGN_SPEC.md`

---

## 1. Purpose

Listen Mode is Muse's immersive playback and guided-understanding experience.

It should let a listener:

- enjoy a composition without needing to understand the machinery;
- see where they are in the piece;
- distinguish the composition from its performed realization;
- follow form, motifs, harmony, orchestration, and emotional development;
- ask why a moment matters;
- move a piece into Create or Research without interrupting playback;
- continue through a meaningful listening journey rather than a random queue.

Listen Mode is not a decorative visualizer and not a reduced DAW. It is a listening instrument that presents three synchronized truths:

1. **Composition** — how the piece is written.
2. **Performance** — how the current rendering plays it.
3. **Resonance** — how the piece develops perceptually and emotionally.

---

## 2. Product principles

### 2.1 Immersion first

The default experience supports listening rather than demanding analysis. A user should be able to open Listen Mode, press play, and understand the primary interface without learning Muse-specific terminology.

### 2.2 Meaningful motion

Every animation should correspond to a musical event, expressive parameter, or timeline transition. Decorative movement must never imply musical meaning where none exists.

### 2.3 Progressive disclosure

The first view shows only the most useful structure. Detailed analysis appears through explicit user actions.

### 2.4 Composition and performance remain distinct

The symbolic score and rendered expression must not be collapsed into one ambiguous representation. A listener should be able to see what was composed, what the renderer changed, and how those two layers relate.

### 2.5 Identity and provenance remain attached

Every visualization remains traceable to the piece recipe, score, rendering, motifs, and identity metadata.

### 2.6 Listening continues across the product

Switching between Listen, Create, Research, and Library does not stop playback unless the user explicitly stops it or the selected piece becomes unavailable.

---

## 3. Primary user modes

### Passive listening

The user wants a beautiful now-playing experience with minimal controls.

### Curious listening

The user wants concise explanations of form, emotion, motifs, and current musical events.

### Analytical listening

The user wants to inspect score, performance, harmony, orchestration, and structural data in detail.

The same experience should support all three through progressive disclosure rather than separate applications.

---

## 4. Core visualization model

Listen Mode uses one synchronized visualization system with three layers.

### 4.1 Composition layer

Derived from the symbolic score and readable even when playback is stopped:

- sections and phrase boundaries;
- bars and beats;
- symbolic note events;
- voice and instrument assignments;
- motif identities and transformations;
- form grammar;
- key journey and harmonic events;
- cadence markers;
- identity grammar events;
- structural climax and return markers.

### 4.2 Performance layer

Represents the current rendered realization:

- actual onset timing;
- timing displacement from the symbolic grid;
- velocity and dynamic shaping;
- articulation and note-length changes;
- sustain and pedal behavior;
- rubato;
- phrase-level shaping;
- active voice prominence;
- rendering-specific expression metadata.

### 4.3 Resonance layer

Represents continuous perceptual and musical development:

- energy;
- tension;
- brightness;
- warmth;
- density;
- motion;
- orchestration intensity;
- climax and release;
- emotional contour;
- optional coherence or integration metrics where clearly explained.

These values are derived interpretations, not objective emotional truth.

---

## 5. Information architecture

Listen Mode is divided into five primary regions.

### 5.1 Global header

Contains:

- Muse identity;
- Listen, Create, and Research switcher;
- compact now-playing information;
- global volume and audio output;
- settings;
- user or local identity menu.

Behavior:

- mode switching preserves playback;
- current piece and progress remain visible;
- Space controls playback unless focus is inside a text field;
- volume and output settings affect all modes.

### 5.2 Now-playing hero

Contains:

- composition title;
- concise musical premise;
- style, form, key, meter, tempo, and duration;
- primary visualization;
- progress;
- transport controls;
- Keep, Next, and More actions.

This is the visual and emotional center of Listen Mode.

### 5.3 Context rail

Contains:

- Why This Piece?;
- current section;
- current motif;
- current harmony;
- current resonance state;
- identity summary;
- quick navigation to Create or Research.

The rail updates with playback but should not flicker or replace text too aggressively.

### 5.4 Detail timeline

Contains:

- score and performance overlays;
- section boundaries;
- motifs;
- harmony;
- orchestration lanes;
- selected-moment marker;
- detailed scrubber.

This region may collapse in passive mode and expand in curious or analytical mode.

### 5.5 Journey rail

Contains:

- previous piece;
- current piece;
- next recommendations;
- journey title and intent;
- relationship labels such as Nearby, Contrast, Return, or Distant Cousin.

The journey rail should communicate why the sequence exists.

---

## 6. Visualization modes

The user-facing modes describe musical purpose rather than geometry.

### 6.1 Hybrid

**Default mode.**

Combines:

- radial whole-piece map;
- compact local timeline;
- performance and resonance overlays;
- current-section and motif context.

Use Hybrid for the best balance of immersion and understanding.

### 6.2 Form

Shows:

- formal sections;
- phrase groups;
- returns and transitions;
- key journey;
- cadence locations;
- form-specific structures.

Examples include sonata exposition/development/recapitulation, rondo returns, passacaglia cycles, fugue exposition/episode/stretto, and variation trajectories.

### 6.3 Score

Shows the static symbolic composition:

- note events;
- voices;
- instrumentation;
- motifs;
- phrase and bar boundaries;
- formal labels.

Active notes highlight during playback, but the base representation does not move.

### 6.4 Performance

Shows the current rendering:

- actual onset times;
- dynamics;
- timing deviations;
- articulation;
- sustain;
- rubato;
- phrase curves;
- voice prominence.

Performance mode must retain a visible reference to the symbolic score.

### 6.5 Compare

Overlays score and performance.

Recommended visual mapping:

- symbolic note event: stable base rectangle or line;
- performed onset: offset edge or marker;
- performed duration: extended or shortened outline;
- velocity: luminance or thickness;
- phrasing: continuous curve;
- pedal or sustain: translucent span.

Compare mode is a signature Muse capability.

### 6.6 Resonance

Shows continuous perceptual development:

- energy;
- tension;
- brightness;
- warmth;
- density;
- climax and release.

This mode should be visually calm and suitable for passive listening.

### 6.7 Motifs

Shows:

- motif identity;
- first appearance;
- literal returns;
- fragmentation;
- inversion;
- augmentation;
- ornamentation;
- erosion;
- completion or restoration.

Hovering a motif occurrence highlights related occurrences.

### 6.8 Harmony

Shows:

- key center;
- chord path;
- modulation;
- cadence events;
- tonal stability;
- pedal tones;
- harmonic rhythm.

Harmony labels should remain optional for listeners who do not read Roman numerals.

### 6.9 Orchestration

Shows:

- active instruments and voices;
- register distribution;
- lead, counterline, harmony, and bass roles;
- density;
- doubling;
- entrances and exits;
- prominence over time.

---

## 7. Visual primitives

Radial, bars, and waves are reusable primitives rather than isolated modes.

### 7.1 Radial primitive

Best for whole-piece relationships.

Recommended ring mapping:

- outer ring: form sections;
- second ring: phrase groups and cadence points;
- middle ring: motif occurrences and transformations;
- inner ring: orchestration density and resonance;
- center: current state or piece identity;
- radial playhead: current playback position.

Interactions:

- click a section to seek;
- hover to preview section metadata;
- select a motif occurrence to highlight related returns;
- zoom into a section without changing playback.

### 7.2 Bars primitive

Best for discrete symbolic events:

- notes;
- voices;
- motifs;
- harmony blocks;
- orchestration lanes;
- bars and beats.

Bars must remain readable at multiple zoom levels.

### 7.3 Waves primitive

Best for continuous performed and perceptual behavior:

- amplitude;
- phrase dynamics;
- timing flow;
- brightness;
- tension;
- density;
- spectral or orchestration activity.

A raw waveform should never substitute for structural analysis.

---

## 8. Default Hybrid visualization

The Hybrid view combines all three primitives without making them equally prominent.

### 8.1 Main radial map

Default visible layers:

- form;
- current playback position;
- motif returns;
- compact resonance glow.

Optional layers:

- harmony;
- orchestration;
- cadence markers;
- identity events.

### 8.2 Local timeline

Default visible layers:

- symbolic note activity;
- current section;
- active motif;
- dynamics curve.

Optional overlays:

- performance timing;
- harmony;
- orchestration;
- resonance curves.

### 8.3 Context rail

The context rail explains the selected or current moment. It should never merely repeat labels already visible in the chart.

---

## 9. Playback behavior

### 9.1 Stopped

Show the whole-piece structure, selected section, readable labels, and a calm low-motion state.

### 9.2 Playing

Animate:

- radial playhead;
- active section;
- active motif;
- note activity;
- dynamics;
- orchestration emphasis;
- resonance state.

Animation must remain synchronized with audio and degrade gracefully under load.

### 9.3 Paused

Freeze the current visual state and preserve the selected moment.

### 9.4 Scrubbing

Scrubbing updates:

- current section;
- active motif;
- harmony;
- orchestration;
- resonance;
- context rail;
- all visible playheads.

Audio preview during scrubbing is optional and disabled by default.

### 9.5 Track transition

At the end of a piece:

- preserve the final visual state briefly;
- show the relationship to the next journey item;
- begin the next composition with a short, non-destructive visual transition;
- do not crossfade audio unless the rendering pipeline and composition boundaries support it cleanly.

---

## 10. Explain This Moment

Explain This Moment is a first-class interaction.

The user can trigger it by:

- clicking the radial visualization;
- clicking the timeline;
- pressing a dedicated action;
- using a keyboard shortcut;
- opening the current-moment card.

The explanation should contain only grounded information available from the piece data:

- **Location:** section, phrase, bar, and time;
- **Motif:** active motif and transformation;
- **Harmony:** key or harmonic function;
- **Performance:** expressive emphasis;
- **Change:** what changed from the preceding phrase;
- **Significance:** why the moment matters in the form.

Example:

> Development, bars 49–56. The opening subject has been fragmented and moved into a less stable harmonic region. The renderer broadens the phrase and increases upper-string intensity. This creates the peak of instability before the recapitulation restores the subject in the home key.

A **Show Evidence** action highlights the referenced notes, motifs, curves, or section boundaries.

---

## 11. Why This Piece?

Why This Piece? explains selection rather than composition mechanics.

It should answer:

- why the piece fits the current prompt or journey;
- what makes it different from nearby candidates;
- which musical traits were most relevant;
- whether the selection was driven by resonance, contrast, novelty, or user history.

Suggested structure:

- **Fit:** strongest relationship to the current intent;
- **Distinctive trait:** most differentiating musical feature;
- **Journey role:** why it appears here;
- **Evidence:** links to relevant form, motif, harmony, or identity data.

Avoid ungrounded claims such as “Muse knew you needed this.”

---

## 12. Journey behavior

A journey is an ordered listening context with explicit transition logic.

Supported journey kinds may include:

- Discovery;
- Resonance;
- Style;
- Mood;
- Identity;
- Nearby;
- Contrast;
- User-curated;
- Kept-piece journey.

Each transition has a relation label, for example:

- Similar motif, different harmony;
- Same mood, higher energy;
- Distant style, shared form;
- Return to an earlier identity;
- Lower density after climax.

Journey state includes:

- journey identifier;
- ordered piece references;
- current index;
- transition explanations;
- generation or retrieval state;
- user edits;
- saved status.

---

## 13. Actions

### 13.1 Keep

Stores:

- exact rendered audio;
- MIDI;
- symbolic recipe;
- visualization identity;
- journey context;
- explanation metadata;
- rendering metadata.

The interface must confirm what is preserved.

### 13.2 Next Piece

Advances within the journey. If the next item is still rendering, show progress and allow the current piece to continue or replay.

### 13.3 Nearby

Finds or generates a piece close in identity space. The interface summarizes what remains similar.

### 13.4 Distant Cousin

Finds or generates a piece that differs in style or surface while preserving a meaningful structural relationship. The interface explains the shared relationship.

### 13.5 Open in Create

Opens the exact piece and recipe in Create mode without stopping playback.

### 13.6 Open in Research

Opens the current piece in Research mode at the active moment.

### 13.7 Generate Variation

Creates a related candidate while preserving explicitly selected invariants:

- motif;
- form;
- harmony;
- orchestration;
- identity grammar;
- performance.

---

## 14. Responsive behavior

### 14.1 Desktop

Show:

- full radial hero;
- context rail;
- journey rail;
- expandable timeline;
- persistent transport.

### 14.2 Tablet

Show:

- radial hero;
- context rail as tabs or drawer;
- timeline below the hero;
- fixed compact transport.

### 14.3 Mobile

Use a single-column experience with primary cards for:

- Now Playing;
- Current Moment;
- Visualization;
- Journey.

Recommended gestures:

- horizontal swipe between visualization and journey;
- upward sheet for details;
- tap radial area to seek;
- long press for Explain This Moment.

Do not attempt to preserve desktop density on mobile.

---

## 15. Accessibility

Listen Mode must remain useful without animation, color, or visual interpretation.

Requirements:

- semantic control labels;
- full keyboard operation;
- screen-reader summaries of current section and moment;
- high-contrast theme;
- reduced-motion mode;
- no critical meaning conveyed only by color;
- visible focus;
- accessible chart summaries;
- textual alternatives for motifs, form, harmony, and resonance.

Reduced-motion mode preserves active section, playhead position, selected motif, and context text while removing particles, continuous glow pulses, decorative rotation, and nonessential interpolation.

---

## 16. Performance requirements

Visualization must never compromise audio playback.

Recommended constraints:

- audio scheduling runs independently of visualization rendering;
- visualization uses `requestAnimationFrame` or equivalent;
- expensive derived data is precomputed;
- Canvas or WebGL is preferred for dense note and radial rendering;
- DOM is reserved for controls, labels, accessibility, and low-density graphics;
- visualization frame drops do not affect audio;
- inactive panels suspend animation;
- mobile defaults to reduced visual density.

Target behavior:

- smooth interaction on a typical integrated-GPU laptop;
- acceptable reduced-density behavior on mid-range mobile hardware;
- fast first paint before all analytic bundles are available.

---

## 17. Data contracts

All visualization data uses one shared time coordinate system:

- integer musical ticks for symbolic alignment;
- seconds for rendered playback;
- an explicit mapping between ticks and rendered time.

### 17.1 Composition bundle

```rust
pub struct ListenCompositionBundle {
    pub piece_id: PieceId,
    pub duration_ticks: u64,
    pub duration_seconds: f64,
    pub tempo_map: Vec<TempoEvent>,
    pub meter_map: Vec<MeterEvent>,
    pub sections: Vec<SectionRegion>,
    pub phrases: Vec<PhraseRegion>,
    pub notes: Vec<SymbolicNoteEvent>,
    pub motifs: Vec<MotifOccurrence>,
    pub harmony: Vec<HarmonyRegion>,
    pub orchestration: Vec<OrchestrationRegion>,
    pub cadences: Vec<CadenceEvent>,
    pub identity_events: Vec<IdentityEvent>,
}
```

### 17.2 Performance bundle

```rust
pub struct ListenPerformanceBundle {
    pub piece_id: PieceId,
    pub render_id: RenderId,
    pub rendered_notes: Vec<RenderedNoteEvent>,
    pub dynamics: Vec<TimedScalar>,
    pub articulation: Vec<ArticulationRegion>,
    pub sustain: Vec<SustainRegion>,
    pub phrase_shapes: Vec<PhraseShape>,
    pub voice_prominence: Vec<VoiceProminenceRegion>,
}
```

### 17.3 Resonance bundle

```rust
pub struct ListenResonanceBundle {
    pub piece_id: PieceId,
    pub energy: Vec<TimedScalar>,
    pub tension: Vec<TimedScalar>,
    pub brightness: Vec<TimedScalar>,
    pub warmth: Vec<TimedScalar>,
    pub density: Vec<TimedScalar>,
    pub motion: Vec<TimedScalar>,
    pub climax_markers: Vec<TimedMarker>,
}
```

### 17.4 Selected moment

```rust
pub struct SelectedMoment {
    pub seconds: f64,
    pub tick: u64,
    pub section_id: Option<SectionId>,
    pub phrase_id: Option<PhraseId>,
    pub motif_ids: Vec<MotifId>,
    pub harmony_id: Option<HarmonyId>,
}
```

---

## 18. Leptos component map

```text
ListenPage
├── GlobalHeader
├── ListenHero
│   ├── PieceHeader
│   ├── HybridVisualizer
│   │   ├── RadialStructureCanvas
│   │   ├── LocalTimelineCanvas
│   │   └── VisualizationOverlayControls
│   └── PrimaryTransport
├── ListenContextRail
│   ├── WhyThisPieceCard
│   ├── CurrentMomentCard
│   ├── IdentitySummaryCard
│   └── ListenQuickActions
├── ListenDetailPanel
│   ├── VisualizationModeTabs
│   ├── ScoreView
│   ├── PerformanceView
│   ├── CompareView
│   ├── MotifView
│   ├── HarmonyView
│   └── OrchestrationView
├── JourneyRail
└── ExplainMomentSheet
```

Recommended shared stores:

- `PlaybackStore`
- `CurrentPieceStore`
- `ListenVisualizationStore`
- `JourneyStore`
- `SelectedMomentStore`
- `AccessibilityPreferencesStore`

---

## 19. State model

```rust
pub enum ListenVisualizationMode {
    Hybrid,
    Form,
    Score,
    Performance,
    Compare,
    Resonance,
    Motifs,
    Harmony,
    Orchestration,
}

pub enum ListenLayerMode {
    Composed,
    Performed,
    Overlay,
}

pub struct ListenUiState {
    pub visualization_mode: ListenVisualizationMode,
    pub layer_mode: ListenLayerMode,
    pub selected_moment: Option<SelectedMoment>,
    pub selected_motif: Option<MotifId>,
    pub selected_section: Option<SectionId>,
    pub detail_panel_open: bool,
    pub context_rail_tab: ListenContextTab,
    pub visual_density: VisualDensity,
}
```

Playback state remains independent from page-local visualization state.

---

## 20. Loading and failure states

### 20.1 Composition ready, audio rendering

Show the static composition visualization immediately. Display rendering progress without blocking inspection.

### 20.2 Audio ready, analysis pending

Allow playback. Use available composition and performance data, then add resonance or analysis layers when ready.

### 20.3 Missing analysis

Hide unavailable modes and explain why. Do not display fabricated placeholders.

### 20.4 Render failure

Preserve the composition and allow:

- retry;
- MIDI export;
- inspection in Create;
- alternate renderer selection.

### 20.5 Offline

Local pieces and local rendering remain available where supported. Network-dependent journey generation clearly indicates its unavailable state.

---

## 21. Implementation phases

### P0 — Core Listen Mode

- global persistent playback;
- now-playing hero;
- Hybrid radial visualization;
- compact timeline;
- form and current-section display;
- composed/performed/overlay toggle;
- Why This Piece?;
- Keep;
- Next Piece;
- Open in Create;
- Open in Research;
- desktop and basic mobile layout.

### P1 — Structural understanding

- Explain This Moment;
- Motifs mode;
- Harmony mode;
- Orchestration mode;
- evidence highlighting;
- improved journey relations;
- tablet and mobile refinement;
- reduced-motion and chart summaries.

### P2 — Advanced listening

- compare multiple performances;
- user-saved moments;
- shareable visualization exports;
- adaptive visual density;
- animated lineage;
- user-authored journeys;
- real-time renderer comparison.

---

## 22. Acceptance criteria

Listen Mode is ready for alpha when:

1. Playback persists across Listen, Create, and Research.
2. A user can identify the current section without opening Research.
3. Hybrid mode remains understandable while stopped, playing, paused, and scrubbing.
4. Composition and performance can be viewed separately and overlaid.
5. Explain This Moment uses only grounded piece data.
6. Why This Piece? explains selection rather than making emotional claims about the user.
7. Keeping a piece preserves exact audio, MIDI, recipe, and journey context.
8. Visualization frame drops do not interrupt audio.
9. The primary experience works with reduced motion.
10. Mobile presents a usable single-column listening experience.
11. Unavailable analysis is disclosed rather than invented.
12. A new user can play, keep, and advance without learning research terminology.

---

## 23. Open questions

- Which resonance channels are reliable enough to expose in alpha?
- Should raw waveform display be available at all, or only derived performance curves?
- Which renderer metadata can be emitted directly rather than reconstructed?
- How should motif confidence or uncertain analysis be communicated?
- Should users be able to edit journey transition logic in Listen Mode?
- Which visualization settings persist globally versus per piece?
- How much of Compare mode can be supported before a full score editor exists?

---

## 24. Final design intent

Listen Mode should feel like:

- a music player when the user wants simplicity;
- a score reader when the user becomes curious;
- an observatory when the user wants understanding.

Its core promise is:

> Muse lets you see how a piece was composed, how it is being performed, and how its structure develops through time—without making analysis a prerequisite for listening.
