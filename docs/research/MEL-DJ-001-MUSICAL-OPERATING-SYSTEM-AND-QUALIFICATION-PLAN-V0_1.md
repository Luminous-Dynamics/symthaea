# MEL-DJ-001 — Musical Operating System and DJ Qualification Plan v0.1

**Status:** Research/design proposal; no DJ capability is qualified by this document.  
**Date:** 2026-11-28  
**Scope:** Melothaea UI, DJ preparation, transition intelligence, performance runtime, and evaluation.

## 1. Thesis

Melothaea should not be a playlist recommender with a crossfader, nor a clone of a hardware-vendor ecosystem. It should be a musical operating system spanning composition, preparation, performance, collaboration, and post-set analysis.

The DJ is the author/director. The system provides inspectable options, evidence, and bounded assistance. Automatic action is opt-in, explicitly authorized, reversible where possible, and never silently inferred from a recommendation.

The core object is not a track pair but a **time-varying musical transition** embedded in a set-level trajectory. Evaluate three nested decision scales:

1. Transition: what to do with currently loaded material, and when.
2. Track choice: what material should follow, given current musical and contextual state.
3. Set trajectory: how the whole performance should evolve over time.

This framing follows the temporal hierarchy identified in *Temporal Considerations in DJ Mix Information Retrieval and Generation* (TIME 2025), while extending it with explicit performer agency, evidence, and live safety contracts.

## 2. Paradigm shifts

### P1. From track matching to transition contracts

A transition is a typed, executable proposal with:
- source and target track identities and analysis versions;
- selected in/out windows and phrase/beat alignment;
- tempo-map and time-stretch constraints;
- stem routing and gain trajectories;
- protected musical properties and prohibited overlaps;
- expected energy/timbre/harmonic trajectory;
- preconditions, postconditions, uncertainty, and fallback;
- a deterministic replay recipe and evidence receipt.

A proposal is not "good" because its tracks share a key or BPM. It must satisfy hard technical constraints and be evaluated perceptually in context. The system must be able to abstain.

### P2. From fixed BPM to uncertainty-aware musical time

Represent beat positions as hypotheses with confidence and provenance, not a single infallible grid. Support tempo curves, half/double-time ambiguity, swing, rubato, live drummers, edits, and non-4/4 material. Preserve alternate grids when ambiguity is real. Never auto-quantize the source recording.

### P3. From four coarse stems to role-aware scene control

Vocal/bass/drums/other are useful interchange categories, not a complete musical ontology. Build toward source-separated and event-grounded roles (lead vocal, backing vocal, kick, snare, percussion, bass line, harmonic bed, hook, ambience), but expose only roles with calibrated evidence. Stem bleed, missing content, phase artifacts, transient smearing, latency, and reconstruction residuals are first-class outputs.

### P4. From recommendation to causal DJ copilot

Measure whether a recommendation changes DJ behavior and improves outcomes. Use randomized or counterbalanced treatment, sham, unrelated, and yoked recommendations. Separate acceptance, execution, perceived usefulness, musical outcome, workload, and agency. Do not treat clicks or acceptance as proof of value.

### P5. From playlist to set trajectory

Model the set as a constrained, multi-objective path through a transition graph. Optimize musical continuity and intentional contrast, energy trajectory, diversity, stylistic context, familiarity, and DJ intent. Avoid collapsing these dimensions into a universal scalar. Offer Pareto alternatives and explain tradeoffs.

### P6. From vendor lock-in to portable musical assets

Treat audio files as user-owned sources and analysis/cues/grids/loops/sets as portable, versioned sidecars. Preserve source hashes, timestamps, sample rates, channel layouts, cue identity, analysis version, rights/provenance, and export mappings. Never claim compatibility without round-trip tests against real library fixtures and supported hardware.

### P7. From "AI magic" to evidence-carrying creative control

Every recommendation, transformation, and automated action should expose what it used, what it changed, what it protected, what remains unknown, and how to replay or undo it. A confidence score without calibration and source provenance is not evidence.

## 3. Product architecture

### Existing UI boundary

The current Leptos app has Listen, Create, Research, Library, and Atlas surfaces and a persistent single-source player. Keep this player authoritative for ordinary listening. Do not turn it into a multi-deck engine.

Introduce a separate Perform route/workspace with subviews:
- Prepare: local library ingestion, analysis status, waveform, beat/tempo hypotheses, structure, cues, loops, stems, source integrity.
- Mix Plan: track/transition graph, set trajectory, alternatives and constraints.
- Performance: deck controls, mixer, cue/loop controls, metering, controller mappings, and live assistance.
- History: session timeline, recordings, action receipts, transition review, and export.

### Runtime boundary

Keep Leptos as presentation. Define a Rust DJ domain/runtime independent of the existing one-source web player:
- DjSession, DjDeck, BeatClock, TempoMap, BeatGridHypothesis;
- CueBank, LoopRegion, MixerGraph, AudioRoute, StemBus;
- ControllerProfile, ControllerEvent, MixPlan, TransitionContract;
- SetTimeline, ActionReceipt, SessionEvidence.

Real-time audio callbacks must not perform blocking I/O, network access, unbounded allocation, model inference, or contended locks. Analysis and inference run off the real-time thread and communicate through bounded, preallocated message paths. Audio failure must fail safe: no surprise transport start, no unapproved routing change, and no loss of the source.

### Incremental delivery

P0: UI shell + local preparation metadata model + deterministic fixtures; no claim of live DJ playback.  
P1: beat/tempo/structure evidence visualization and manual correction.  
P2: transition contract and offline preview/replay.  
P3: two-deck native playback, independent outputs/headphone cue, clock and mixer qualification.  
P4: controller mappings, four decks, stem playback, recording and interoperability.  
P5: opt-in live copilot and causal field studies.

Do not expose a capability as production-ready until its hardware, audio, and failure behavior are qualified.

## 4. Benchmark programme

Create MEL-DJ-001 with separate scorecards. All proposed targets below are aspirational gates for study design, not achieved results. Thresholds must be preregistered against representative datasets and expert review before leaderboard use.

| Track | Primary metrics | Required slices / controls |
|---|---|---|
| Beat/tempo | beat F-measure, continuity, tempo error, downbeat accuracy, calibration | genre, meter, live/rubato, edits, half/double-time |
| Structure | boundary F1 with tolerance, section classification, cue utility | intros/outros, breakdowns, drops, non-EDM |
| Transition | blind pairwise preference, expert rubric, hard-constraint violations | genre, transition type, vocal density, energy |
| Timing | beat/phrase alignment error, drift, correction burden | tempo curves, long blends, live sources |
| Stem/role | SI-SDR plus bleed/fullness, transient/phase metrics, role/event accuracy | dense mixes, backing vocals, acoustic music |
| Audio runtime | xrun/dropout count, callback deadline misses, end-to-end latency, jitter | device, sample rate, buffer, CPU load |
| Hardware | mapping coverage, event latency, reconnect recovery, transport safety | controller families, hot unplug, duplicate/stale events |
| Set planning | sequence preference, trajectory intent attainment, diversity, novelty | genre, set length, cold-start library |
| Copilot | causal outcome improvement, workload, agency, override/undo rate | treatment/sham/yoked, novice/expert |
| Portability | metadata round-trip, cue/loop preservation, loss accounting | Rekordbox/Serato/Traktor/Engine exports where lawful and available |
| Efficiency | quality vs CPU/GPU, RAM, energy, latency, prep time | matched hardware, matched budgets |

### Hard gates

- No transport or routing mutation without explicit authority.
- No silent loss or rewrite of user source files or library metadata.
- No unsupported beatgrid certainty: ambiguous tracks remain ambiguous.
- No transition passes if a required technical invariant is violated, even if preference is high.
- No stem quality claim from SDR alone; report bleed, fullness, reconstruction, phase, and transient behavior.
- No live performance qualification from offline render tests.
- No benchmark pass from skipped/queued jobs or aggregate repository CI unrelated to the exact head.

### Proposed initial target bands (to preregister, not claim)

- Beat/downbeat: report per-slice F-measure and calibrated uncertainty; target >= 0.95 only on a clearly defined supported subset, never as a universal genre-agnostic claim.
- Transition hard constraints: 100% on deterministic known-answer fixtures.
- Transition preference: lower 95% confidence bound above matched incumbent baseline on held-out DJ/track/session cohorts; publish effect size and abstention.
- Stem: beat/phase alignment and residual constraints must pass independently; compare SDR, bleed, fullness, and listening preference.
- Runtime: zero xruns in a preregistered soak cohort on each supported device/configuration; report p99/p99.9 callback and output latency rather than a single mean.
- Controller: 100% deterministic mapping/authority/reconnect tests for declared supported profiles.
- Portability: zero unexplained loss for fields declared supported; unsupported fields explicitly reported.
- Copilot: positive causal improvement with non-inferior agency/workload; acceptance rate alone is not a success metric.

## 5. Statistical design

Use crossed sampling across DJs, source tracks, track pairs, sessions, controllers, venues/monitoring conditions, and raters. The same DJ should perform multiple transitions; the same tracks should appear across multiple DJs; repeated takes and sessions must be modeled rather than treated as independent observations.

Before population intervals are enabled:
1. Build exhaustive tiny known-answer oracles for crossed factors.
2. Simulate null and alternative effects with known truth.
3. Compare candidate estimators on bias, coverage, type-I error, power, and interval width.
4. Include missingness, imbalance, sparse crossing, and influential clusters.
5. Obtain independent statistical review.
6. Keep unsupported designs descriptive-only.

Do not select a bootstrap or cluster-robust estimator by name alone. Validate its assumptions and finite-sample behavior for each metric and sampling design.

## 6. Competitive comparison

The incumbent landscape already includes:
- Rekordbox: library preparation, beat analysis, stem functions, cloud/library workflows, hardware and export ecosystem.
- Serato DJ Pro: real-time and precomputed stem separation, stem effects, hardware control.
- Traktor Pro 4: four decks, flexible beatgrids, sync/nudge, stems, DVS and broad mapping options.
- Engine DJ: standalone ecosystem, library preparation/import, onboard workflows and stems.

Therefore BPM/key detection, cue points, four-stem separation, four decks, waveform display, and ordinary crossfades are parity requirements, not differentiators.

Research references:
- Rekordbox overview: https://rekordbox.com/en/
- Rekordbox stems FAQ: https://rekordbox.com/en/support/faq/stems-7/
- Serato stems: https://support.serato.com/hc/en-us/articles/5700968326927-Stems-Overview
- Traktor Pro 4: https://www.native-instruments.com/products/traktor-pro
- DJ-AI (Audio Mostly 2025): https://research.sabanciuniv.edu/id/eprint/52373/
- MixWeaver (ISMIR 2026): https://ismir2026program.ismir.net/poster_428.html
- Temporal Considerations in DJ Mix Information Retrieval and Generation (TIME 2025): https://drops.dagstuhl.de/storage/00lipics/lipics-vol355-time2025/LIPIcs.TIME.2025.20/LIPIcs.TIME.2025.20.pdf
- Evaluation Platform and Benchmarks for Music Source Separation (TISMIR, Nov 2026): https://reference-global.com/article/10.5334/tismir.366

A particularly important current frontier is MixWeaver (ISMIR 2026): stem-aware transition generation using structured optimization over stem gain trajectories, alignment, overlap, smoothness and energy stability, with more than 2,000 transition pairs reported. We must implement a direct, protocol-matched comparison before claiming superiority. Its own post-review notes acknowledge a missing direct comparison with DJtransGAN; preserve that limitation in our evidence ledger.

## 7. Research hypotheses to falsify

H1: Transition contracts improve hard-constraint compliance and expert preference over fixed crossfade, key/BPM heuristics, and learned black-box baselines.

H2: Uncertainty-aware beat/tempo maps reduce correction burden and failure on live, swung, edited, and ambiguous material.

H3: Stem-aware transitions improve preference only when separation quality and role evidence are sufficient; otherwise abstention or full-mix transitions perform better.

H4: Set-level Pareto planning improves stated DJ intent attainment without reducing diversity or agency.

H5: Explanations and preview/replay improve DJ calibration and error detection, not merely trust or acceptance.

H6: Copilot interventions produce measurable causal improvement under sham/yoked controls and do not increase workload or reduce agency.

H7: Portable sidecars and explicit loss reports reduce migration friction across ecosystems.

## 8. P0 execution plan

1. Inspect current Melothaea UI/backend modules and existing music identity, audio analysis, score, stem, and temporal-grounding artifacts before creating parallel abstractions.
2. Create a route and visual shell for Prepare / Mix Plan / Performance / History without wiring nonexistent playback.
3. Define versioned Rust protocol types for track analysis sidecars and transition contracts; add JSON schemas and deterministic known-answer fixtures.
4. Build a local-library fixture manifest with licensed/public-domain/test-generated assets and source hashes.
5. Implement deterministic transition validator/replay before any generative optimizer.
6. Add a benchmark manifest with dataset license, split identity, DJ/track/session crossing, and leakage controls.
7. Keep native real-time audio as a separately gated implementation with device-specific qualification.

## 9. Non-goals and safety

- Do not copy proprietary vendor databases, reverse-engineer private APIs, or redistribute commercial tracks.
- Do not upload local libraries by default.
- Do not treat generated stem outputs as source truth.
- Do not automate live actions without user authorization.
- Do not claim compatibility from a screenshot or a successful mock.
- Do not hide uncertainty, unsupported metadata, failed analyses, or skipped qualification.
- Do not add a new duplicate programme where an existing authoritative issue already owns a capability; link and coordinate.

## 10. Exit criteria for this research phase

- Architecture reviewed against current UI and backend.
- P0 data contracts and fixtures reviewed.
- Transition known-answer oracle passes deterministic tests.
- Baselines and dataset licensing documented.
- No unsupported real-time or hardware claims.
- Exact-head Rust formatting, compile, tests, and Clippy evidence attached when executable implementation exists.
- Independent statistical review completed before inferential leaderboard claims.
