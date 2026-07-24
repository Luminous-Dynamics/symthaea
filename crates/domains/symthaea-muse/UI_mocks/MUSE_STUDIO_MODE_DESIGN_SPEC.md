# Muse Studio Mode Design Specification

**Status:** Proposed  
**Audience:** Muse product, UX, frontend, composition-engine, audio, and research contributors  
**Primary implementation:** Leptos web application  
**Scope:** Studio Mode interaction model, information architecture, editing workflows, state, data contracts, and implementation priorities  
**Related documents:**
- `MUSE_STUDIO_UI_INTERACTION_DESIGN_SPEC.md`
- `MUSE_LISTEN_MODE_VISUALIZATION_DESIGN_SPEC.md`
- `MUSE_RESEARCH_MODE_DESIGN_SPEC.md`

---

## 1. Purpose

Studio Mode is Muse's precision composition workspace.

Create mode answers:

> What kind of piece do I want?

Listen mode answers:

> What does this piece feel like, and how does it unfold?

Research mode answers:

> How is this piece constructed, performed, and related to other works?

Studio mode answers:

> I chose this piece. What should remain, what should change, and which version should become canonical?

Studio Mode should let users shape musical structure without forcing them to work only at the level of raw MIDI events. It should support edits at the level of:

- whole pieces;
- sections;
- phrases;
- motifs and themes;
- harmony regions;
- orchestration roles;
- performance gestures;
- individual notes when exact repair is needed.

Its defining promise is:

> Edit musical thought, not merely MIDI events.

---

## 2. Product principles

### 2.1 Semantic editing first

The default editing units are musical objects, not arbitrary selections of bytes or rendered audio.

Muse should understand the difference between:

- a motif and one occurrence of that motif;
- a section and the bars that currently realize it;
- the score and its performance;
- an orchestration role and a specific instrument;
- a cadence and the notes that implement it.

### 2.2 Preserve before transform

Every generative edit must explicitly state:

- the selected scope;
- the intended change;
- the invariants that must survive;
- the expected output.

Muse must never silently rewrite unrelated material.

### 2.3 Non-destructive by default

Substantial edits produce auditionable alternatives or branches.

The current canonical piece changes only after the user commits an alternative.

### 2.4 Structure remains visible

The user should always know:

- where they are in the piece;
- what is selected;
- what relationships depend on the selection;
- what changed between versions.

### 2.5 Score and performance remain separable

A user must be able to:

- edit the composition without regenerating the performance;
- edit the performance without recomposing the score;
- compare composed and performed realizations.

### 2.6 Provenance is part of authorship

Every committed version should preserve:

- the parent version;
- transformation request;
- locked invariants;
- score;
- recipe;
- render;
- manual edits;
- timestamp;
- optional annotation.

### 2.7 The interface should remain artistic

Studio is powerful, but it should not become a parameter cockpit.

Engine internals should appear only when they help the user make a musical decision.

---

## 3. Primary user workflow

The core Studio loop is:

1. Open a selected piece or candidate.
2. Select a musical scope.
3. Define what must remain invariant.
4. Choose or describe a transformation.
5. Generate a small set of alternatives.
6. Audition and compare the alternatives.
7. Inspect structural and performance differences.
8. Commit one result as a new version.
9. Continue editing, branch, or export.

Example:

> Select the Development section. Preserve the primary motif and section length. Increase harmonic instability and orchestral density. Generate three alternatives.

Muse should resolve this into a visible edit contract before execution.

---

## 4. Information architecture

Studio Mode is divided into six persistent regions.

### 4.1 Global header

Contains:

- Muse identity;
- Listen / Create / Research mode switcher;
- compact now-playing information;
- render status;
- global volume and output;
- settings;
- profile or local identity.

Behavior:

- playback persists across modes;
- current piece persists across modes;
- render progress remains visible;
- switching to Listen or Research preserves the selected bar or moment.

### 4.2 Left workspace rail

Recommended workspaces:

- Overview
- Structure
- Themes
- Harmony
- Orchestration
- Performance
- Score
- Versions
- Render & Export

The workspace rail changes the interpretation of the central canvas rather than opening unrelated applications.

### 4.3 Central semantic timeline

The timeline is the primary editing surface.

It supports semantic zoom.

#### Far zoom

Shows:

- full form;
- section proportions;
- key journey;
- major motif returns;
- climax;
- orchestration density;
- version markers.

#### Medium zoom

Shows:

- phrases;
- motif regions;
- harmony regions;
- instrumental roles;
- development operations;
- transitions;
- cadence events.

#### Close zoom

Shows:

- notation or piano roll;
- individual notes;
- velocity;
- articulation;
- timing;
- voice leading;
- exact manual edits.

The user should remain on one continuous musical timeline while moving from form to note detail.

### 4.4 Context inspector

The right-side inspector answers:

- What is selected?
- What is it doing?
- What relationships depend on it?
- What may be changed?
- What is currently locked?
- What consequences will the proposed edit have?

The inspector should adapt to the current selection type.

### 4.5 Alternative tray

The alternative tray contains uncommitted proposals.

Each proposal includes:

- concise description;
- affected scope;
- preserved invariants;
- structural differences;
- performance differences;
- duration impact;
- play;
- compare;
- commit;
- discard.

### 4.6 Bottom transport and version dock

Contains:

- play / pause;
- previous / next;
- scrubber;
- loop selection;
- metronome;
- playback speed;
- composed / performed / overlay switch;
- active version;
- compare mode;
- undo / redo for local manual edits;
- render status.

---

## 5. Selection model

Selection is foundational.

Supported scopes:

- whole piece;
- section;
- phrase;
- motif definition;
- motif occurrence;
- harmony region;
- orchestration role;
- instrument or voice;
- bar range;
- note range;
- individual note;
- performance gesture;
- cadence;
- climax marker;
- version branch.

The current scope must always be visible as a breadcrumb.

Example:

> Copper Meridian → Development → Bars 49–64 → Motif A fragments

### 5.1 Selection behavior

Selecting an object should:

- highlight it in every visible view;
- update the inspector;
- show dependent objects;
- reveal relevant transformations;
- preserve playback position;
- optionally loop the selected scope.

### 5.2 Dependency visibility

Muse should show when a selected object is linked to:

- other motif occurrences;
- a cadence;
- a transition;
- a key journey;
- an orchestration role;
- a performance phrase;
- a climax;
- an identity invariant.

Dependencies should be visible before destructive or wide-reaching edits.

---

## 6. Preserve / Change contract

The Preserve / Change contract is the defining Studio interaction.

Every generative edit should expose:

- **Scope**
- **Preserve**
- **Change**
- **Output**
- **Expected consequences**

### 6.1 Preserve controls

Users may lock:

- melody;
- motif identity;
- rhythm;
- harmony;
- form;
- section length;
- meter;
- key;
- orchestration;
- bass line;
- climax;
- ending;
- identity grammar;
- cadence type;
- phrase count;
- seed;
- deterministic recipe;
- manual note edits;
- performance.

Locks may be exact or semantic.

#### Exact lock

The underlying events remain unchanged.

#### Semantic lock

The musical identity remains within defined transformation limits.

Example:

> Preserve thematic identity

may allow ornamentation, sequence, or fragmentation while preventing replacement by unrelated material.

### 6.2 Change controls

Available transformations depend on the selected scope.

#### Structure transformations

- extend;
- compress;
- reorder;
- insert transition;
- remove transition;
- strengthen return;
- replace ending;
- create false recovery;
- redistribute climax;
- change section role;
- add variation;
- alter recurrence spacing.

#### Theme transformations

- fragment;
- invert;
- augment;
- diminish;
- sequence;
- ornament;
- simplify;
- combine;
- split;
- erode;
- restore;
- transfer to another voice;
- alter register;
- preserve contour while changing rhythm;
- preserve rhythm while changing contour.

#### Harmony transformations

- reharmonize;
- delay cadence;
- change cadence type;
- increase instability;
- reduce instability;
- change tonal destination;
- modalize;
- tonicize;
- simplify harmonic rhythm;
- intensify dominant preparation;
- add or remove pedal tone;
- preserve melody while changing harmony.

#### Orchestration transformations

- thin texture;
- broaden texture;
- redistribute roles;
- change lead instrument;
- add counterline;
- remove doubling;
- alter register;
- increase contrast;
- stage entrances;
- soften climax;
- preserve score while changing color.

#### Performance transformations

- increase or reduce rubato;
- reshape phrase;
- soften attack;
- broaden climax;
- change articulation;
- alter dynamic range;
- humanize timing;
- tighten timing;
- adjust sustain;
- alter voice prominence;
- create multiple performances of the same score.

### 6.3 Natural-language request

The user may describe an edit naturally.

Example:

> Make the recapitulation feel earned rather than merely repeated.

Muse should translate this into a proposed plan before running it.

Example resolution:

- **Scope:** Recapitulation, bars 65–88
- **Preserve:** themes, home key, section length
- **Change:** orchestration trajectory, cadence preparation, counterline activity
- **Output:** three alternatives

The user can edit or confirm the plan.

---

## 7. Alternative generation

Generative operations should normally return three alternatives.

Avoid overwhelming users with large candidate sets.

Each alternative card should show:

- title;
- one-sentence musical intent;
- what remains unchanged;
- what changed;
- affected sections;
- motif changes;
- harmony changes;
- orchestration changes;
- performance changes;
- metric deltas where useful;
- confidence or constraint warnings;
- play;
- solo;
- compare;
- commit;
- discard.

### 7.1 Alternative visual state

- uncommitted alternative: violet;
- selected alternative: brighter violet with focus ring;
- committed version: gold/copper;
- invalid alternative: muted with explicit reason;
- conflict: red only when a real constraint conflict exists.

### 7.2 Example alternative set

#### A — Harmonic intensification

- motif unchanged;
- form unchanged;
- development moves through a darker tonal region;
- cadence delayed by four bars;
- string density increases near climax.

#### B — Thematic fragmentation

- opening motif split into smaller cells;
- harmony largely preserved;
- more rhythmic interruption;
- earlier local climax.

#### C — Orchestral escalation

- score materially unchanged;
- lower strings and brass enter progressively;
- dynamic range widened;
- recapitulation softened for contrast.

---

## 8. Structural diff

Every proposal and version comparison should support a structural diff.

Possible differences:

- notes added, removed, or moved;
- rhythm changed;
- motif transformed;
- motif occurrence inserted or deleted;
- harmony substituted;
- cadence changed;
- key area changed;
- section resized;
- instrumentation changed;
- role reassigned;
- performance timing changed;
- dynamics changed;
- articulation changed;
- recipe changed.

Recommended visual language:

- gold: current committed material;
- translucent gray: previous material;
- violet: proposed material;
- cyan: performance-only difference;
- connectors: moved or transformed relationships;
- red: unresolved conflict only.

Every visual diff must have a textual summary.

---

## 9. Workspaces

## 9.1 Overview

Purpose:

- provide a concise picture of the current piece;
- show the most important editable structures;
- surface recent changes;
- suggest useful next actions.

Contains:

- form map;
- key journey;
- motif overview;
- emotional contour;
- orchestration density;
- current version;
- recent edits;
- unresolved warnings;
- suggested transformations.

## 9.2 Structure

Purpose:

- edit form and large-scale temporal architecture.

Contains:

- section blocks;
- phrase groups;
- transition dependencies;
- climax location;
- cadence map;
- recurrence pattern;
- timing and proportion controls.

Actions:

- resize sections;
- insert or remove sections;
- alter section roles;
- move climax;
- change ending strategy;
- edit transitions;
- change recurrence spacing.

Structural edits must expose dependent motifs, harmony, and orchestration before recomputation.

## 9.3 Themes

Purpose:

- edit motifs, themes, and thematic relationships.

Contains:

- motif definitions;
- motif occurrences;
- transformation family;
- recurrence timeline;
- theme relationship graph;
- identity-preservation controls;
- selected motif contour and rhythm.

Actions:

- fragment;
- invert;
- augment;
- diminish;
- sequence;
- ornament;
- simplify;
- merge;
- split;
- transfer between voices;
- preserve or relax thematic identity.

## 9.4 Harmony

Purpose:

- edit tonal and harmonic structure.

Contains:

- key regions;
- chord timeline;
- harmonic function;
- cadence map;
- tonal stability;
- modulation path;
- pedal tones;
- harmonic rhythm.

Actions:

- reharmonize;
- change cadence;
- delay cadence;
- alter tonal destination;
- simplify or intensify harmony;
- preserve melody;
- preserve bass;
- preserve formal arrival.

Provide musician-friendly labels by default and optional Roman-numeral detail.

## 9.5 Orchestration

Purpose:

- edit musical roles and instrumental realization.

Primary role lanes:

- lead;
- counterline;
- harmonic support;
- bass;
- pulse;
- texture;
- percussion.

The user may then inspect the instruments assigned to each role.

Actions:

- reassign role;
- change register;
- add or remove doubling;
- stage entrances;
- thin or broaden texture;
- create contrast;
- change instrumental color;
- preserve notes while changing orchestration.

## 9.6 Performance

Purpose:

- edit expressive realization without changing the symbolic score.

Contains:

- symbolic grid;
- performed timing;
- dynamics;
- articulation;
- sustain;
- phrase curves;
- voice prominence;
- timing deviation;
- performance versions.

Actions:

- regenerate performance;
- adjust phrase shape;
- alter rubato;
- change articulation;
- modify dynamic range;
- tighten or loosen timing;
- preserve score exactly.

## 9.7 Score

Purpose:

- support exact repair and detailed authorship.

Views:

- notation;
- piano roll;
- voice-leading view;
- event list where needed.

Actions:

- edit pitch;
- edit rhythm;
- edit duration;
- edit velocity;
- edit articulation;
- move voice;
- repair collision;
- adjust register;
- lock manual edits.

Manual edits become part of the deterministic recipe and remain protected unless explicitly unlocked.

## 9.8 Versions

Purpose:

- show the creative lineage of the piece.

Use a branch graph rather than a flat undo stack.

Each node stores:

- parent;
- version name;
- transformation request;
- locked invariants;
- recipe;
- score hash;
- render identifier;
- timestamp;
- author;
- optional note;
- committed status.

Actions:

- branch;
- compare;
- rename;
- revert;
- mark canonical;
- merge compatible changes;
- export version;
- open in Listen or Research.

## 9.9 Render & Export

Purpose:

- manage final realization and external workflow.

Contains:

- renderer choice;
- instrument library;
- expression profile;
- sample rate;
- output format;
- MIDI;
- WAV;
- MusicXML;
- stems where supported;
- render queue;
- provenance summary.

Studio should communicate clearly when an export represents:

- score;
- performance;
- audio render;
- recipe;
- version lineage.

---

## 10. Command bar

Studio includes a natural-language command bar, but it is not the only editing interface.

Example commands:

- Keep the theme, but make the middle less repetitive.
- Give the cello more agency during the second subject.
- Try a darker harmony without changing the melody.
- Make the coda half as long.
- Preserve the score and generate three restrained performances.
- Turn this return into a false recovery.
- Repair the climax without increasing volume.

Before execution, Muse shows the resolved contract.

The user must be able to modify:

- scope;
- invariants;
- transformations;
- number of alternatives;
- render behavior.

---

## 11. Visual system

Studio retains Muse's dark, refined visual identity.

Semantic color use:

- **Gold / copper:** committed material and canonical version;
- **Violet:** generated proposals and alternate branches;
- **Cyan / teal:** performance and expression;
- **Warm white:** selected text and exact score information;
- **Muted gray:** inactive, historical, or hidden material;
- **Red:** conflicts, invalid states, clipping, or broken constraints only.

Avoid assigning a bright color to every track.

Use:

- labels;
- role icons;
- line patterns;
- restrained hue variations;
- focus and luminance.

The workspace should feel like an illuminated score table, not a neon cockpit.

---

## 12. Interaction details

### 12.1 Hover

Hover may reveal:

- object name;
- role;
- bar range;
- dependencies;
- available transformations;
- version origin.

Hover must not be required for essential information.

### 12.2 Double click

Recommended behavior:

- section: zoom to section;
- phrase: zoom to phrase;
- motif occurrence: select motif family;
- note: open exact edit state;
- version: open compare.

### 12.3 Dragging

Dragging may support:

- moving section boundaries;
- moving selected motifs within valid constraints;
- reassigning orchestration roles;
- adjusting phrase curves;
- changing note timing or pitch at close zoom.

Every drag should provide:

- preview;
- validity feedback;
- snapping;
- cancel;
- undo.

### 12.4 Keyboard

Recommended shortcuts:

- Space: play / pause
- L: loop selection
- F: focus selection
- C: compare current proposal
- Enter: commit selected proposal
- Escape: close inspector state or cancel operation
- Command/Ctrl+K: command bar
- Command/Ctrl+Z: undo local edit
- Shift+Command/Ctrl+Z: redo
- `[` / `]`: previous / next alternative
- `1`–`9`: switch workspace when focus is not in an editor

---

## 13. Accessibility

Requirements:

- complete keyboard navigation;
- visible focus;
- semantic labels;
- screen-reader descriptions for timeline objects;
- textual alternatives for visual diffs;
- reduced-motion mode;
- high-contrast mode;
- no information conveyed by color alone;
- minimum target sizes;
- accessible confirmation for commit and discard.

Dense timelines should support:

- list view;
- current-selection summary;
- keyboard stepping by bar, phrase, or section.

---

## 14. Responsive behavior

## 14.1 Desktop

Show:

- workspace rail;
- central timeline;
- context inspector;
- alternative tray;
- persistent transport.

## 14.2 Tablet

Use:

- collapsible workspace rail;
- bottom-sheet inspector;
- timeline as primary surface;
- swipeable alternative tray;
- persistent compact transport.

## 14.3 Mobile

Studio mobile is a focused companion, not the full desktop editor.

Support:

- playback;
- section and phrase selection;
- preserve/change contract;
- audition alternatives;
- commit;
- simple score repairs;
- version browsing;
- export.

Avoid attempting full orchestration and notation density on a narrow screen.

---

## 15. State model

Recommended state domains:

```rust
pub struct StudioUiState {
    pub workspace: StudioWorkspace,
    pub zoom: SemanticZoomLevel,
    pub selection: StudioSelection,
    pub inspector_open: bool,
    pub alternative_tray_open: bool,
    pub compare_mode: Option<StudioCompareMode>,
    pub layer_mode: StudioLayerMode,
    pub active_version: VersionId,
    pub draft_contract: Option<EditContract>,
}
```

```rust
pub enum StudioWorkspace {
    Overview,
    Structure,
    Themes,
    Harmony,
    Orchestration,
    Performance,
    Score,
    Versions,
    RenderExport,
}
```

```rust
pub enum StudioSelection {
    Piece(PieceId),
    Section(SectionId),
    Phrase(PhraseId),
    Motif(MotifId),
    MotifOccurrence(MotifOccurrenceId),
    HarmonyRegion(HarmonyRegionId),
    OrchestrationRole(OrchestrationRoleId),
    Voice(VoiceId),
    BarRange(BarRange),
    NoteRange(Vec<NoteId>),
    Note(NoteId),
    PerformanceGesture(PerformanceGestureId),
    Version(VersionId),
}
```

```rust
pub struct EditContract {
    pub scope: StudioSelection,
    pub preserve: Vec<Invariant>,
    pub change: Vec<TransformationIntent>,
    pub alternative_count: u8,
    pub render_policy: RenderPolicy,
}
```

Playback state must remain separate from Studio page state.

---

## 16. Data contracts

### 16.1 Version node

```rust
pub struct StudioVersion {
    pub id: VersionId,
    pub parent_ids: Vec<VersionId>,
    pub name: String,
    pub piece_id: PieceId,
    pub recipe_hash: String,
    pub score_hash: String,
    pub render_ids: Vec<RenderId>,
    pub transformation_request: Option<String>,
    pub invariants: Vec<Invariant>,
    pub manual_edits: Vec<ManualEdit>,
    pub created_at: Timestamp,
    pub author: AuthorId,
    pub note: Option<String>,
    pub is_canonical: bool,
}
```

### 16.2 Alternative proposal

```rust
pub struct StudioAlternative {
    pub id: AlternativeId,
    pub parent_version: VersionId,
    pub contract: EditContract,
    pub score_diff: ScoreDiff,
    pub structure_diff: StructureDiff,
    pub performance_diff: Option<PerformanceDiff>,
    pub warnings: Vec<ConstraintWarning>,
    pub preview_render: Option<RenderId>,
    pub status: AlternativeStatus,
}
```

### 16.3 Structural diff

```rust
pub struct StructureDiff {
    pub section_changes: Vec<SectionChange>,
    pub phrase_changes: Vec<PhraseChange>,
    pub motif_changes: Vec<MotifChange>,
    pub harmony_changes: Vec<HarmonyChange>,
    pub orchestration_changes: Vec<OrchestrationChange>,
    pub cadence_changes: Vec<CadenceChange>,
    pub climax_changes: Vec<ClimaxChange>,
}
```

---

## 17. Leptos component map

Recommended high-level structure:

```text
StudioPage
├── GlobalHeader
├── StudioWorkspaceRail
├── StudioTopBar
│   ├── PieceBreadcrumb
│   ├── CommandBar
│   └── VersionSelector
├── StudioMain
│   ├── SemanticTimeline
│   │   ├── StructureLayer
│   │   ├── MotifLayer
│   │   ├── HarmonyLayer
│   │   ├── OrchestrationLayer
│   │   ├── ScoreLayer
│   │   └── PerformanceLayer
│   └── ContextInspector
├── PreserveChangePanel
├── AlternativeTray
│   └── AlternativeCard
├── VersionGraphDrawer
└── StudioTransportDock
```

Recommended stores:

- `PlaybackStore`
- `CurrentPieceStore`
- `StudioSelectionStore`
- `StudioVersionStore`
- `StudioEditContractStore`
- `StudioAlternativeStore`
- `RenderQueueStore`
- `AccessibilityPreferencesStore`

---

## 18. Loading and failure states

### 18.1 Piece loading

Show:

- form skeleton;
- known metadata;
- playback readiness;
- unavailable workspaces until data arrives.

### 18.2 Alternative generation

Show:

- scope;
- preserved invariants;
- generation progress;
- partial availability;
- cancel.

### 18.3 Constraint conflict

Explain:

- which invariants conflict;
- why the requested transformation cannot satisfy all locks;
- suggested relaxations.

Do not silently drop constraints.

### 18.4 Render failure

Preserve the score proposal.

Allow:

- retry render;
- choose different renderer;
- inspect score;
- commit score without audio only if explicitly allowed.

### 18.5 Merge conflict

Show incompatible changes by musical object, not only by file or event identifier.

---

## 19. Implementation priorities

## P0 — Complete editing loop

- open selected candidate;
- semantic timeline with sections and motifs;
- select piece, section, phrase, or motif;
- preserve/change contract;
- generate three alternatives;
- audition;
- compare;
- commit;
- version creation;
- persistent playback;
- MIDI and WAV export;
- basic score/performance overlay.

## P1 — Structural editing

- Structure workspace;
- Themes workspace;
- Harmony workspace;
- Orchestration workspace;
- motif transformations;
- version graph;
- command-bar planning;
- evidence-linked diffs;
- section looping;
- constraint conflict handling.

## P2 — Precision and integration

- exact notation editing;
- compatible branch merging;
- collaborative annotation;
- advanced performance editing;
- DAW bridge;
- MusicXML round-trip;
- reusable theme library;
- reusable orchestration library;
- stem rendering;
- external plugin integration.

---

## 20. Acceptance criteria

Studio Mode is ready for alpha when:

1. A user can open a candidate and preserve playback.
2. A user can select a whole piece, section, phrase, or motif.
3. The selected scope is visible as a breadcrumb.
4. A generative edit requires a visible Preserve / Change contract.
5. Muse returns no more than a small, usable set of alternatives by default.
6. Each alternative explains what was preserved and changed.
7. The user can audition alternatives without changing the canonical version.
8. The user can compare a proposal against the current version.
9. Committing creates a new version node with provenance.
10. Score and performance can be edited separately.
11. Manual note edits can be locked against regeneration.
12. Constraint conflicts are disclosed rather than silently resolved.
13. Playback remains independent from visualization frame rate.
14. MIDI and WAV export clearly identify the active version.
15. The interface remains usable in reduced-motion mode.
16. A new user can complete one edit without understanding internal metrics or engine terminology.

---

## 21. Open questions

- Which semantic invariants are reliable enough for alpha?
- How should users distinguish exact and semantic locks?
- Which transformations can safely operate locally without full-piece recomposition?
- How should compatible branch merging be defined musically?
- How much notation editing should ship before MusicXML round-trip?
- Should alternatives be generated sequentially or concurrently?
- Which metric deltas are useful to artists without becoming prescriptive?
- How should external MIDI edits be reconciled with version lineage?
- Which renderers support fully separable score and performance workflows?
- How should collaborative edits attribute authorship and intent?

---

## 22. Final design intent

Studio Mode should feel like a composer's structural workbench.

It should not feel like:

- a black-box song generator;
- a conventional multitrack recorder;
- a research dashboard;
- a wall of engine parameters.

Its signature workflow is:

> Select a meaningful musical object, declare what must survive, describe what should change, audition several honest alternatives, and commit one into the composition's lineage.

That interaction unifies Muse's strongest capabilities:

- symbolic composition;
- development DNA;
- identity grammar;
- structural understanding;
- deterministic provenance;
- explainable transformation;
- editable ownership.
