# Muse Listen Mode Development Plan

**Status:** implementation-ready architecture after repository census and review  
**Date:** 2026-07-19  
**Scope:** migrate the complete Muse product surface to Leptos, retire the
embedded HTML/JavaScript client, and then evolve the canonical Leptos Listen Mode
into Muse's personal artistic-refinement loop.

## 1. Outcome

Listen Mode already works as an immersive player. The next milestone is to add a
bounded **Review Session** inside it so one serious listener can make blinded,
repeatable, longitudinal judgments and reconcile them with Analyst only after
submission.

The product should support this loop:

```text
Listen naturally
      ↓
Enter a short blinded Review Session
      ↓
Submit an absolute judgment or A/B decision
      ↓
Reveal identity and Analyst evidence
      ↓
Record agreement, disagreement, or revision target
      ↓
Revisit selected artifacts after a delay
```

This is an N-of-1 artistic-development instrument. It provides engineering and
personal-preference evidence; it does not establish population-level listener
claims.

### Architectural decision

Leptos is the only forward product UI. The embedded `studio/index.html`,
`muse-studio.js`, and `muse-studio.css` implementation is frozen as a behavioral
and visual reference during migration. It receives no new product features.

The migration is complete only when `muse_studio` serves the built Leptos
application at `/` and the embedded client is removed from the runtime and build.
There must be one player, one state model, one route tree, and one implementation
of every Listen visualization.

“Everything to Leptos” applies to the product UI and browser-side interaction
code. The Axum service remains the authoritative API, artifact, composition, and
evidence backend; musical and study rules stay in Rust domain crates rather than
moving into view components.

## 2. Repository census

### 2.1 What exists and should be reused

The embedded Studio client currently served by `muse_studio` already has:

- continuous composition with a one-item prefetch queue;
- Resonance, Discovery, and Contrast journey policies;
- current, next, and previous navigation;
- persistent playback while moving between the embedded product tabs;
- exact WAV/MIDI playback and keeper artifact persistence;
- Hybrid, Form, Score, composed/performed Compare, Resonance, Motif, Harmony,
  and Orchestration visualizations;
- score, performance, provenance, Listen, and Analyst bundles;
- grounded current-moment and “Why this piece?” explanations;
- reduced-motion and hidden-view rendering controls;
- a Library backed by the keeper log.

The Leptos client already has:

- route-independent audio ownership through `MuseState`;
- Listen, Create, Research, Library, and Atlas routes;
- a lightweight Listen player and prefetch queue;
- keeper persistence and exact artifact playback;
- shared protocol types.

The analysis backend already has independently verified structural traces,
audio-integrity evidence, review escalation, motif occurrences, and artifact
hashes. The cognition-study subsystem also contains rigorous blinded-schedule
and append-only evidence patterns that can be simplified for personal use.

### 2.2 Important gaps found

1. There are two Listen clients. The embedded HTML/JavaScript client is the
   executable feature reference; the Leptos Listen route is a partial migration.
   This is temporary migration debt: all product behavior moves to Leptos and the
   embedded client is retired.
2. Normal Listen is intentionally unblinded: it exposes title, style, grammar,
   visual identity, and explanations before playback.
3. `Compare` currently means symbolic score versus performed realization of one
   piece. It is not a two-piece preference trial.
4. Keep is the only explicit taste signal. Skip, dwell, replay, completion,
   rating, recall, and pairwise preference are not persisted.
5. Listen session history and journey state are client-memory only and disappear
   on reload.
6. Listen composes one candidate at a time. That bypasses the server's current
   multi-candidate Identity Explorer and premise layer. The embedded reference
   varies valence/arousal/energy, but the Leptos client still uses fixed intent
   values; neither path selects from finished scores against recent history.
7. The protocol client type does not expose a `vary_premise` control, despite an
   older plan describing one. The running server also applies premise variation
   only when `n_candidates > 1` and exploration is active.
8. The keeper log is append-only but not a versioned listening ledger. Historic
   records vary by schema; repeat keeps are not idempotent by score/audio hash.
9. A rated but unkept in-memory candidate cannot support delayed listening after
   server restart unless its exact artifact is pinned.
10. Leptos transport synchronization methods exist, but its persistent `<audio>`
    currently wires only `ended`; time/play/pause/metadata signals are therefore
    not kept live by the root component.

## 3. Product boundary

Do not turn ordinary Listen into a laboratory. Add two explicit experiences:

### Immerse

The current product experience. Identity, structure, visualizations, journey
relations, and explanations remain visible.

### Review Session

A short, opt-in, blinded session of four to eight exposures. Before judgment it
shows only:

- an anonymous presentation code;
- playback and progress;
- optional neutral cover art with no style encoding;
- the question being answered;
- fatigue/stop controls.

It must hide titles, styles, grammar labels, motifs, seeds, filenames, journey
explanations, Analyst warnings, structural charts, and download names until the
judgment has been durably recorded.

The existing one-piece `Compare` visualization keeps its name and meaning.
Cross-piece comparison is called **A/B Review**.

## 4. Authoritative data model

Add versioned wire types to `symthaea-muse-protocol`. Store records server-side;
clients only submit typed commands.

### 4.1 Artifact identity and references

Muse needs three typed identities:

```text
CompositionArtifactId — the symbolic musical work
RenditionArtifactId   — one performed/rendered realization of that work
EvidenceBundleId      — one versioned analysis of an exact artifact
```

A composition identity binds:

- score and recipe hashes;
- composition and theory engine versions;
- seed and intent provenance;
- grammar plan and verified structural trace;
- later, the requested/asserted/verified `DiversityPlan`.

A rendition identity binds:

- its `CompositionArtifactId`;
- performance dialect and performance-model identity;
- renderer and soundfont/synthesis identity;
- mastering/render parameters;
- audio hash;
- neutral, natural, or another declared performance arm.

Natural and neutral renditions therefore share a composition without becoming
the same review artifact. Candidate IDs, keeper keys, and URIs are locators and
provenance—not identity.

Every durable judgment references immutable evidence, not a transient candidate
ID alone:

```rust
struct ListenArtifactRef {
    composition_id: CompositionArtifactId,
    rendition_id: RenditionArtifactId,
    evidence_bundle_id: Option<EvidenceBundleId>,
    audio_sha256: String,
    score_sha256: String,
    recipe_sha256: String,
    candidate_id: Option<u64>,
    keeper_audio_key: Option<String>,
    artifact_uri: Option<String>,
}
```

The server resolves and verifies hashes. A client may not assert them.

### 4.2 Pure playback kernel

`PlaybackStore` is a Leptos projection of a pure Rust reducer, not a bag of
signals coordinated by browser callbacks.

```text
Playback reducer
    ↕ typed commands/events
Browser audio adapter
    ↕
One persistent <audio> element
    ↕
Leptos projections
```

The reducer owns explicit states:

```text
Empty → Loading → Ready → Playing / Paused / Seeking → Ended
                  ↘ Failed ↗
```

It handles `LoadRequested`, `MetadataLoaded`, `PlayRequested`,
`PlaybackStarted`, `PauseRequested`, `PlaybackPaused`, `SeekRequested`,
`TimeAdvanced`, `Ended`, `PlaybackFailed`, and `SourceSuperseded`.

Every load increments `load_epoch`. Browser events are accepted only when their
epoch matches the current source. This prevents late `ended`, `error`, or
`loadedmetadata` events from changing a replacement source.

### 4.3 Pure journey kernel

`JourneyStore` projects an event-sourced deterministic Rust reducer. Durable
state includes:

```text
journey ID and policy
policy parameters
traversal seed and RNG position
previous/current/next artifact references
pending composition requests
recent-history window
step sequence
relation explanations
```

Commands include `BeginJourney`, `Advance`, `ReturnToPrevious`,
`SelectAlternative`, `PrefetchCompleted`, `CompositionFailed`, `ChangePolicy`,
and `RestoreJourney`. The same starting state and event sequence must reproduce
the same journey.

Each asynchronous request carries `journey_step_id`, `composition_request_id`,
and `prefetch_epoch`. Superseded results never enter the queue.

Persist meaningful journey checkpoints or domain events, never arbitrary Leptos
component state.

### 4.4 Coordinated application state

Stores remain separated by responsibility but components do not orchestrate
cross-store transitions. Components dispatch commands to a coordinator. For
example:

```text
MuseAppEvent::JourneyAdvanced
    ├── current artifact changes
    ├── stale analysis bundle clears
    ├── selected moment resets
    └── playback loads the verified rendition
```

This prevents a selected moment, performance bundle, or visualization from
remaining attached to the prior composition.

### 4.5 Session and presentation

```rust
struct PersonalListenSession {
    session_id: String,
    protocol_version: String,
    created_at_unix_ms: u64,
    mode: PersonalListenMode,
    presentations: Vec<PersonalPresentation>,
    reveal_policy: RevealPolicy,
    status: SessionStatus,
}

enum PersonalListenMode {
    Absolute,
    Pairwise,
    DelayedRecall,
    RandomAudit,
    ChampionChallenger,
    RevisionComparison,
}
```

Presentation order and anonymous codes are generated on the server. Private
identity bindings remain unavailable to the client until the relevant response
is committed.

### 4.6 Append-only events

Use separate event variants rather than one nullable monolith:

```text
SessionStarted
ExposureSummary
AbsoluteJudgmentSubmitted
PairwiseDecisionSubmitted
RecallJudgmentSubmitted
IdentityRevealed
AnalystReconciliationSubmitted
SessionStoppedForFatigue
SessionCompleted
```

Each record carries:

- schema and event versions;
- unique event and session IDs;
- server timestamp and monotonic session sequence;
- previous-event hash and event hash;
- anonymous presentation ID;
- resolved artifact hashes where permitted;
- first exposure versus revisit;
- confidence and optional concise note;
- evidence origin (`HumanObservation`);
- supersedes-event ID for corrections.

No record is edited in place. A correction appends a linked event.

### 4.7 Judgments

Absolute judgment fields use an integer 1–5 scale:

- love;
- replay desire;
- identity/memorable idea;
- development;
- ending;
- mechanicalness;
- distinctness/interchangeability;
- confidence.

`motif_memory` is not asked on a first exposure. A revisit instead records:

- remembered anything before playback;
- remembered or anticipated a motif/return/ending;
- stronger, unchanged, or weaker with familiarity;
- replay desire now.

Pairwise records store left/right/tie, confidence, and a constrained reason plus
optional note. They do not duplicate the absolute-rating fields.

### 4.8 Artifact retention

Exposure telemetry alone does not archive every generated WAV. Any artifact
selected for delayed recall, champion status, revision comparison, or an explicit
judgment must be pin-able into the existing keeper-style immutable artifact
layout. Use a distinct `review-artifact-v1` role so “rated” does not silently mean
“loved.”

## 5. Development sequence

### Phase 0A — State and identity foundation

1. Freeze the embedded client and capture deterministic parity fixtures before
   changing its behavior.
2. Implement typed composition, rendition, and evidence identities plus one
   authoritative artifact resolver.
3. Project legacy keepers into the new identity index without rewriting historic
   JSONL records. Make new keeps idempotent by typed identity.
4. Implement the pure playback reducer, `load_epoch`, invariants, and a non-browser
   test adapter.
5. Implement the pure journey reducer, event log, restoration, traversal seed,
   relation explanations, request epochs, and stale-prefetch rejection.
6. Add a coordinator so components dispatch commands rather than mutating several
   stores in arbitrary order.
7. Repair the composition request contract: expose `vary_premise`, generate
   several symbolic candidates, assign canonical composition identities, reject
   exact and recent-history collisions, return a small diverse frontier, and let
   authoritative Rust journey policy select the next candidate.
8. Render only selected candidates and enqueue verified rendition identities.
9. Register the neutral and natural paired packs by composition and rendition
   identity, preserving their shared-composition relationships where present.

**Gate:** reducers replay deterministically; stale browser and composition events
cannot mutate current state; candidate selection is multi-candidate and
history-aware; Candidate, Keeper, Library, Atlas, Analyst, Journey, and future
Review resolve the same typed identities.

### Phase 0B — Leptos behavioral parity

1. Bind one root `<audio>` element to the playback adapter and wire play, pause,
   time, duration, seeking, volume, ended, error, metadata, and autoplay rejection.
2. Project playback, journey, current artifact, analysis, selected moment, and UI
   state into focused Leptos stores coordinated by typed application events.
3. Port current/previous/next, queue, prefetch, policies, relation explanations,
   keeper behavior, section seeking, and current-moment synchronization.
4. Define a typed visualization contract. Every visualization declares required
   bundles, optional evidence, interactions, empty-state reason, synchronization,
   reduced-motion behavior, and supported rendition types.
5. Port Hybrid, Form, Score, composed/performed Compare, Resonance, Motifs,
   Harmony, and Orchestration behind that contract.
6. Port grounded explanations, missing-evidence states, responsive behavior,
   reduced motion, hidden-route suspension, keyboard use, and accessibility text.
7. Port all remaining embedded Create and Research behavior so cutover strands no
   product capability.
8. Build a parity harness from the frozen fixtures.

Behavioral parity checks queue transitions, journey relations, selected artifact,
seek behavior, keeper identity, playback continuity, and explanation evidence.
Visual parity checks product intent through screenshots and interaction tests; it
does not require identical DOM or pixels.

**Gate:** the full vertical workflow passes: open Leptos Listen, receive an
authoritatively selected multi-candidate result, play it, navigate across routes
without interruption, advance exactly once on `ended`, preserve coherent
previous/current/next state, replenish prefetch, and restore or explicitly restart
after refresh. The underlying event log replays to the same state.

### Phase 0C — Runtime cutover

1. Serve content-hashed Trunk/Leptos assets from Axum with correct cache headers.
2. Add client-route history fallback that can never intercept `/api/*`.
3. Run behavioral, browser, screenshot, accessibility, reduced-motion,
   back/forward, refresh, failure-recovery, and performance checks.
4. Switch `/` to Leptos and retain a feature-flagged read-only rollback for one
   stabilization release.
5. Verify production loads no embedded JavaScript asset and contains one audio
   element across the complete route tree.
6. Remove the embedded runtime, asset handlers, and rollback after stabilization;
   retain only fixtures and screenshots as historical references.

**Gate:** Leptos is the only production client; route changes never restart
playback; all stale-event race tests pass; journey replay is deterministic;
candidate selection is diverse and authoritative; keeper operations are
idempotent; legacy keepers remain readable; back/forward and refresh are coherent;
static caching is correct; `/api/*` is never handled by history fallback.

### Phase 1 — Personal listening ledger

1. Add protocol types and validation.
2. Add an append-only, hash-linked local ledger separate from `keepers.jsonl`.
3. Add endpoints to create a session, fetch the next blinded presentation,
   submit an event, reveal a committed result, and fetch personal history.
4. Generate server-side anonymous codes and schedules.
5. Record compact exposure summaries on transition rather than high-frequency
   `timeupdate` events: listened seconds, completion fraction, play starts, seeks,
   replays, and abandonment reason.
6. Provide an export containing ledger, manifests, and referenced hashes without
   requiring Analyst internals.

**Gate:** restart-safe append-only behavior; duplicate submissions are
idempotent; sequence/hash tampering is detected; no identity field is returned by
a blinded endpoint before response commitment.

### Phase 2 — Review Session inside canonical Leptos Listen

1. Add an explicit `Immerse | Review` switch to Leptos Listen.
2. Generate sessions of at most eight presentations, defaulting to four.
3. Add Absolute, A/B, Random Audit, and Delayed Recall session types.
4. Add hidden repeat support and a visible stop-for-fatigue action.
5. Keep the normal player and audio element; swap only the information and
   response surfaces.
6. After durable submission, reveal identity and offer “Open evidence.”
7. Keep scheduling, blinding, validation, and reveal rules server-side. Leptos
   renders typed state and submits commands; it does not own study logic.
8. Stream review audio through an opaque endpoint such as
   `/api/review/presentation/{opaque_id}/audio`; do not disclose the rendition URI
   or predictable artifact identifier before commitment.
9. Strip or neutralize identifying WAV metadata, HTTP headers, Media Session and
   browser-title metadata, cover images, analytics labels, source-map strings,
   filenames in errors, and cache keys.

**Gate:** a browser test proves that title, style, grammar, seed, filenames,
palette identity, Analyst warnings, artifact routes, media metadata, headers,
cache entries, analytics, source maps, and errors do not reveal identity before a
committed judgment. Anonymous presentation IDs are opaque and unpredictable.

### Phase 3 — Analyst reconciliation

After reveal, display five separate columns where present:

```text
Personal judgment
Composer assertion
Symbolic verification
Audio measurement / external witness
Analyst prediction
```

Allow one reconciliation label:

- Analyst true positive;
- Analyst false positive;
- Analyst false negative;
- genuine structural defect;
- personal preference;
- technically valid but artistically weak;
- musically strong despite warning;
- culturally unresolved.

Deduplicate the current review queues by artifact hash while retaining every
reason. Mix targeted alerts with deterministic random audits so false negatives
remain observable.

**Gate:** reports can calculate warning precision, random-audit defect rate,
minutes per reviewed artifact, and metric/judgment disagreement without treating
personal taste as population truth.

### Phase 4 — Longitudinal listening and champion shelves

1. Schedule opt-in revisit prompts at one day, one week, and one month.
2. Maintain Champion, Runner-up, Failure Reference, and Template Collision
   Reference shelves per grammar.
3. Add motif, ending, development, and performance shelves only after their IDs
   are stable.
4. Use pairwise decisions to update a transparent ranking; retain ties and
   uncertainty.
5. Measure hidden-repeat consistency and surface drift rather than silently
   averaging it away.

**Gate:** a promoted champion is backed by an exact retained artifact, at least
one delayed judgment, and its complete decision lineage.

### Phase 5 — DiversityPlan and targeted revision

Add an explicit `DiversityPlan` before expanding the catalog. It should specify:

- form topology;
- phrase/cycle proportions;
- harmonic route;
- motif-development strategy;
- texture trajectory;
- silence strategy;
- climax strategy;
- ending strategy.

Store requested plan, composer assertion, and independent verification
separately. Compare plans and realized structures against recent Listen history
and champion shelves.

Add targeted revision contracts that lock successful dimensions and change one
diagnosed dimension. Every revision is a new version node and a blinded A/B
Review against its parent; it never overwrites the parent.

**Gate:** Muse can demonstrate that a requested revision changed the target
dimension, preserved its locks, and improved or lost the personal A/B decision.

### Phase 6 — Motif Foundry pilot

The schema-validation pilot may run before Review so its artifact contract can
be exercised without making promotion claims. Generate twenty candidates:

- four lyrical;
- four contrapuntal;
- four groove-cycle;
- four process identities;
- four modal identities.

Each receives canonical, moderate, severe, and difficult-lure evidence first.
Add two-grammar-transfer and two-formal-role renditions when Review can collect
immediate and delayed decisions without exposing labels.

**Gate:** promotion depends on symbolic validity, provenance, lure separation,
recognizable transformation, two compatible uses, delayed replay interest, and
expert review where culturally required.

### Phase 7 — Shadow preference models

Do not begin before 50–100 reasonably independent personal decisions including
hidden repeats and delayed judgments. Train separate uncertain predictors for:

- replay;
- mechanicalness;
- weak endings;
- motif memory;
- template collision;
- pairwise preference.

Predictions remain `ModelPrediction`, never `HumanObservation`. Initially they
may rank review candidates but may not delete or silently reject music. Preserve
an exploration allocation for structurally novel, uncertain, and deliberately
challenging work.

**Gate:** leave-one-artifact-family-out evaluation improves on simple symbolic and
audio baselines; calibration and uncertainty are reported; disabling the model
does not change authoritative composition or evidence.

## 6. First development corpus

Do not generate a larger factorial pack yet. Begin with the existing 32
neutral/natural artifacts:

1. deduplicate review reasons by artifact hash;
2. create four-item blinded sessions;
3. compare natural versus neutral versions where correctly paired;
4. resolve flagged cases;
5. include deterministic random accepted cases;
6. revisit selected keepers after one day and one week;
7. use the disagreements to choose the first compositional revision.

The corpus is exhausted only when it has produced actionable findings about
grammar ownership, motif preservation, performance, ending, or template
collapse—not merely when every clip has a rating.

## 7. Verification matrix

### Protocol and storage

- round-trip and older-version defaults;
- composition/rendition/evidence identity relationships;
- append-only sequence and hash-chain validation;
- idempotent submission and keep operations;
- exact artifact-hash resolution;
- correction-by-supersession;
- crash-safe publish ordering;
- legacy keeper compatibility.

### Playback and journey kernels

- rapid next/previous commands;
- route changes while loading;
- source replacement during playback;
- stale metadata, error, time, and ended events;
- seeking before metadata;
- autoplay rejection and explicit recovery;
- duplicate `ended` events;
- stale prefetch and composition completions;
- policy changes during pending composition;
- deterministic reducer replay and checkpoint restoration;
- refresh and browser back/forward behavior;
- multi-candidate selection and recent-history collision gates.

### Blinding

- identity absent from pre-reveal response payloads;
- identity absent from DOM, accessibility labels, filenames, colors, and URLs;
- identity absent from audio metadata, HTTP headers, Media Session state, browser
  titles, cover/cache keys, analytics labels, source maps, and error messages;
- reveal impossible before a committed judgment;
- refresh/restart preserves blinding state;
- A/B left/right assignment randomized and stored.

### Listening behavior

- four-to-eight item limit;
- hidden repeat scheduling;
- first-listen versus recall question separation;
- fatigue stop does not coerce incomplete ratings;
- audio continues across ordinary routes;
- Review prevents automatic advance before response when required.

### Evidence integrity

- human observations cannot be written as symbolic facts or predictions;
- reconciliation never changes the original judgment;
- random-audit sampling is deterministic and inspectable;
- delayed judgments bind the same audio hash;
- culturally required work remains escalated regardless of personal preference.

### Accessibility and performance

- keyboard playback, seek, response, reveal, and stop;
- screen-reader-safe anonymous presentation summaries;
- reduced-motion Review remains complete;
- hidden views stop animation work;
- event persistence never interrupts audio.

### Migration parity and cutover

- one audio element across the route tree;
- fixture-equivalent journey transitions, relations, seeks, keeps, and evidence;
- visual intent retained across desktop, tablet, mobile, and reduced motion;
- no embedded JavaScript loaded after cutover;
- content-hashed static assets and cache headers;
- history fallback excludes every `/api/*` route;
- feature-flag rollback is read-only and removed after stabilization.

## 8. Concrete first patch series

### Foundation

1. `listen-artifact-identity-v1`: typed composition, rendition, and evidence
   identities; resolver; legacy keeper projection; idempotent keep.
2. `playback-kernel`: pure Rust reducer, load epochs, invariants, and test adapter.
3. `leptos-audio-adapter`: the single root audio element and complete event
   synchronization.
4. `journey-kernel`: deterministic reducer, history, policies, explanations,
   restoration, request epochs, and stale-result rejection.
5. `listen-composition-queue`: `vary_premise`, multi-candidate symbolic
   generation, typed identities, exact/history collision gates, diverse frontier,
   authoritative selection, and verified-rendition prefetch.

### Parity

6. `leptos-listen-shell`: current/next/previous, context rail, transport, keeper,
   explanations, and coordinator projections.
7. `leptos-visualization-contract`: typed requirements, evidence, interactions,
   unavailable states, synchronization, and reduced-motion behavior.
8. `leptos-listen-visualizations`: Hybrid, Form, Score, composed/performed
   Compare, Resonance, Motifs, Harmony, and Orchestration.
9. `leptos-create-research-parity`: remaining embedded capabilities and edge
   states.
10. `leptos-parity-harness`: fixture replay, browser workflows, screenshots,
    accessibility, stale-event races, and performance checks.
11. `leptos-runtime-cutover`: Axum static serving, safe history fallback, content
    hashes/cache policy, one-release rollback, and final legacy removal.

### New development after cutover

12. `listen-ledger-protocol-storage`.
13. `listen-review-api`.
14. `listen-review-leptos`.
15. `listen-analyst-reconciliation`.
16. `listen-longitudinal`.
17. `diversity-plan-v1`.
18. `targeted-ending-revision-v1`.

## 9. Definition of the first vertical slice

The first deliverable is not store scaffolding. It must demonstrate:

```text
Open Leptos Listen
→ server generates several symbolic candidates
→ candidates receive typed composition identities
→ collision and recent-history gates produce a diverse frontier
→ journey policy selects deterministically
→ a verified rendition enters prefetch
→ the current rendition plays through the one root audio element
→ route changes do not interrupt playback
→ ended advances exactly once
→ previous/current/next remain coherent
→ prefetch replenishes without admitting stale results
→ refresh restores the journey or explicitly starts a new one
→ reducer events replay to the same final state in a deterministic test
```

This vertical slice is the Phase 0A gate and the foundation for every subsequent
Leptos component.

## 10. Definition of the migration milestone

Migration is complete when:

- ordinary Listen is at least as usable as the embedded reference;
- Leptos is the only served product client;
- the embedded runtime and temporary rollback are retired;
- all critical behavior and visualization parity gates pass;
- typed artifact identity is used across the product;
- multi-candidate journey selection is authoritative and history-aware;
- one audio element survives every route transition;
- journey restoration and stale-event race tests pass;
- Create and Research lose no embedded capability.

Only then begin the listening ledger and Review Session. At that point every new
product feature is written once, in the architecture that survives.

## 11. Implementation status — 2026-07-19

The Phase 0A vertical slice is implemented:

- composition/rendition/evidence identities are typed in the shared protocol;
- candidates and provenance carry canonical identities;
- keep operations are idempotent by rendition identity while legacy entries
  remain readable;
- playback and journey behavior live in pure deterministic reducers;
- load/prefetch epochs reject stale events and duplicate `ended` advances;
- Listen requests a four-candidate frontier from the authoritative Rust server;
- recent exact composition identities are excluded before selection;
- the single root audio element survives Leptos route transitions;
- the Listen shell exposes previous/play/pause/seek/next, journey policies, and
  an honest next-item rail;
- a typed visualization capability contract now defines evidence requirements;
- Axum serves the content-hashed Leptos distribution with safe history fallback,
  an API 404 boundary, cache policy, and a `/legacy` rollback route;
- deterministic reducer, identity, protocol, and runtime-serving checks pass.

Phase 0B remains active: the richer visualization renderers and final
Create/Research behavioral fixtures must be completed before the one-release
legacy rollback route is removed. Review and longitudinal listening remain
deliberately outside that cutover gate.

## 12. Motif Foundry integration status — 2026-07-19

Motif work is now part of the first vertical slice rather than a parallel
research island:

- `MotifFamilyId` and `MotifAssignment` are shared protocol types;
- mechanical, listening, originality, cultural, and promotion states are
  independent typed enums and cannot imply one another;
- Create can request Foundry material explicitly and Listen enables it for
  compatible styles;
- Foundry selection happens before composition and installs the exact canonical
  motif despite seed-driven inversion and retrograde behavior;
- incompatible styles receive no fabricated assignment or authenticity claim;
- motif lineage participates in composition and rendition identity;
- candidates, keepers, provenance, Listen evidence, and Analyst evidence retain
  the same assignment;
- grammar owners emit occurrence assertions with stable score-event references;
- Analyst independently checks the asserted family, event order, and claimed
  structural distance against the emitted motif definition and score;
- the paired grammar study uses the same exact-installation path as the product,
  eliminating its former duplicated-motif-bank workaround;
- Leptos shows the assignment and its evidence status while explicitly stating
  that mechanical validity is not listener recognition.

The first five contrasting grammar systems are now integrated end to end:

- lyrical/period material is typed as melodic identity;
- contrapuntal material privileges ordered intervals and contour;
- groove-cycle material privileges rhythm, accent, and metric phase;
- process/additive material is typed as a transformation-rule identity;
- modal-arc-informed material privileges anchors and directed contour and is
  always marked expert-review-required.

Listen's multi-candidate frontier cools recently heard motif hashes as well as
recent composition identities. A reproducible twenty-candidate pilot emits four
families per grammar with canonical forms, transformations, difficult lures,
lineage, provenance, hashes, and honest empty psychometric curves. These remain
mechanically valid candidates, not listener-validated or foundational motifs.
Experimental cross-grammar transfer remains separately marked until the
minimal-pair, lure, and context-transfer evidence gates pass.

## 13. Foundry Twelve maturation status — 2026-07-19

Foundry v2 now targets twelve reference champions rather than treating every
mechanically valid candidate as a promotion:

- ten specialist portfolio cells cover two contrasting identities in each of
  the five material systems;
- two bridge cells are earned from transfer evidence rather than populated by
  a separate bridge generator;
- the reproducible maturation corpus contains forty candidates, eight per
  system, generated through persistent per-system duplicate registries;
- material-specific symbolic distance uses declared carriers rather than one
  universal fingerprint;
- each candidate emits canonical, light/moderate/severe valid variants, a
  boundary relative, an invalid identity, difficult lures, two role contexts,
  two explicitly experimental transfers, a complete symbolic score, and
  default-generator/identity-disabled ablations;
- a Pareto screen creates a ten-item mechanical shortlist without collapsing
  independent qualities into one opaque score;
- reserve research, failure, and boundary references remain preserved;
- reference-champion and foundational fields remain false until the required
  blinded, delayed, artistic, originality, audio, and cultural evidence exists.

Analyst bundle v3 separately reports exact ingress installation, final literal
survival, final contract-valid survival, composer assertions, independent
recovery, and their reconciliation. Cultural status is independent at motif,
style-authenticity, and finished-piece scope. Artifact identity now preserves
score-content, composition-lineage, and rendition identities separately.

That archive is frozen as the immutable `pilot_v2` diagnostic baseline. Its
renderer-blocked artifacts and compatibility-backed plans remain truthful
historic evidence rather than being rewritten in place.

## 14. Foundry qualification v2.1 — 2026-07-19

The qualification corpus is a new linked artifact set, not a mutation of v2.
It closes the mechanical ambiguities discovered during the v2 audit:

- canonical, valid transformations, boundary relatives, invalid identities,
  and lures must be content-hash-disjoint or generation fails;
- identity preservation, transformation magnitude, relation distance,
  boundary confidence, and contract verdict are separate measurements;
- groove, process, contrapuntal, modal, and melodic material carry typed,
  material-specific signatures rather than relying on contour alone;
- each system generates four candidates in each of two declared portfolio
  cells, preventing role-collapsed populations;
- transfer evidence distinguishes generation, ingress installation,
  final-score verification, surviving occurrences, and reconciliation;
- an unmeasured psychometric curve is explicitly `not_measured`, never an
  apparently complete empty result;
- Period/Sentence and Contrapuntal now emit dedicated grammar-plan evidence;
  the latter is realized through the native fugue engine;
- every mechanical finalist receives hash-addressed motif-card,
  transformation, 30-second neutral/natural, complete-form, and targeted
  revision auditions plus a Foundry/default/identity-disabled causal ablation
  from a deterministic pinned diagnostic renderer;
- every audition passes the same audio-integrity analyzer before it enters a
  review pack.

The twelve outputs remain **mechanical finalists**, not automatically selected
champions. A blinded Leptos-only `/review/foundry` workflow serves opaque
presentation IDs through Axum, uses the persistent root audio element, presents
six-item sessions, and appends immutable artifact-hash-bound judgments. The
sealed candidate mapping is returned only after a successful commitment.
First, next-day, one-week, and one-month exposure states remain explicit.
Delayed replay evidence, complete-piece
preference, originality clearance, and cultural review remain separate gates;
software generation cannot fabricate them.
