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


## 11. Integration with existing Melothaea evidence work

This programme is a consumer of the auditory-evidence substrate, not a competing implementation of it. Coordinate with the existing open work before adding low-level audio abstractions:

- **#14227 / MEL-HEAR-011 owns the physics-grounded auditory-scene substrate.** It is responsible for preserving source audio provenance, multi-resolution signal views (including phase-capable evidence), psychoacoustic grouping cues, performance dynamics, source/room/channel hypotheses, identifiability and residuals. DJ preparation and transition planning should consume versioned evidence references from that layer. Do not build a second STFT/ERB/CQT, auditory-grouping or source-cause pipeline inside the DJ workspace.
- **#7230 / MEL-005A owns the canonical symbolic score-state trajectory.** Keep score-derived timing and content distinct from audio-derived event evidence; align them with explicit receipts.
- **#13908 owns property-relative musical identity** and the acoustic-STAY / performance-STAY / musical-FLIP distinctions. This is directly relevant to deciding whether a proposed transition preserves a hook, groove, chord, vocal phrase or rhythmic role despite mastering, EQ, room, instrument or performance changes.
- **#14147 owns the perception / audio-derived reasoning / oracle-symbolic decomposition.** Use it to prevent an oracle-assisted score from being reported as an audio-only result.
- **#14144 owns generic transformation-path algebra** and commutator checks; DJ-specific contracts should specialize it rather than introduce another general path framework.
- **#14151 owns the symbolic identity quotient consumer.** It is not a replacement for acoustic evidence, and must not prune a candidate solely from a learned embedding.
- **#4121 and #12892 remain bounded adjacent primitives** for waveform-window overlap accounting and speech F0/voicing/energy respectively. Neither is a complete music-scene or DJ-tempo solution.

The DJ-specific adapter should retain immutable references to the source audio, analysis configuration/version, time-support intervals, evidence claims, competing hypotheses, uncertainty, residuals, and abstention reasons. A transition receipt should say exactly which evidence versions it consumed. Re-analysis must not silently mutate an already-reviewed plan.

## 12. Benchmark decomposition and updated SOTA protocol

MEL-DJ-001 is the programme umbrella. Split executable scorecards so that gains cannot be hidden by a composite score:

| Suite | Scope | Critical comparison |
|---|---|---|
| MEL-DJ-002 — Transition Contract Oracle | Typed contract validation, authorization, timing, source integrity, replay | Exhaustive deterministic known-answer fixtures; 100% hard-gate compliance |
| MEL-DJ-003 — Musical Property FLIP/STAY | Whether a queried musical property changed under controlled acoustic/performance/production interventions | #13908 cases; MusicListenBench-style FLIP/STAY; real recordings and out-of-domain slices |
| MEL-DJ-004 — Transition Quality | Alignment, structure, stem routing/gains, energy, contrast, phrase fit, perceived quality | Fixed crossfade, BPM/key heuristics, rule-based baseline, MixWeaver reproduction, and DJtransGAN where code/weights/data permit |
| MEL-DJ-005 — Live Runtime and Control | Audio deadline, xruns, monitoring, latency, controller safety, recovery | Device/configuration-specific soak tests and fault injection; no offline-to-live extrapolation |
| MEL-DJ-006 — DJ Copilot Causal Value | Set intent, decision quality, workload, agency, trust calibration | Counterbalanced randomized, sham, and yoked interventions; novice/expert slices |

For MEL-DJ-004, do not claim that reproducing MixWeaver's reported ablation/preference results establishes superiority. Its ISMIR 2026 post-review update explicitly says the study did not directly compare with DJtransGAN; it also narrows the current objective to smooth intro/outro blending, acknowledges that hard cuts/abrupt transitions can be unfairly penalized, and states that its dataset comes from the authors' personal collections rather than a public dataset. Its listening study and objective results are valuable baselines, not a complete DJ-performance evaluation. Reproduce the disclosed method as closely as available, publish deviations, and test transition types its objective does not cover.

For MEL-DJ-003, use paired interventions with opposite expected outcomes. FLIP changes the queried musical property; STAY changes irrelevant sound/performance/production factors while preserving that property. Include same-chord re-strums, room/reverb, hiss/noise, gain, EQ, codec, mastering, instrument substitution, tuning, expressive timing, and real recordings. Evaluate false flips and missed flips separately, calibration, selective risk under abstention, and per-property results. Do not let an audio model use oracle notation in the audio-only track.

For MEL-DJ-004 stem claims, report SI-SDR/SDR and spectral metrics alongside Bleedless and Fullness, reconstruction residual, phase consistency, transient preservation, role/event accuracy, and listening preference. The November 2026 TISMIR MVSep Quality Checker work shows that bleed and missing target content are complementary dimensions that a single SDR number can conceal. Freeze model versions, separation settings, segment support, and test splits; never compare a precomputed high-quality stem to an online low-latency stem without labelling the different compute/latency budgets.

## 13. Competitive execution strategy

The late-2026 incumbent feature set means parity and interoperability need to be treated seriously:

- Rekordbox 7 exposes 3/4-stem controls, Mix Point Link, cloud library workflows, AI vocal-position detection, and personalized cue-point automation.
- Serato exposes real-time or pre-prepared vocal/bass/melody/drum separation and stem-based effects.
- Traktor Pro 4 offers four decks, flexible beatgrids, stem separation, DVS, and controller mapping.
- Engine DJ 5.3.0 (released 5 November 2026) documents playlist/crate import and reimport flows for Serato, Apple Music/iTunes, rekordbox and Traktor, alongside standalone workflows and stem rendering in its ecosystem.

These are source-backed product capabilities, not evidence that all products have equivalent quality or that every feature works on every hardware profile. Maintain a dated capability matrix, record product/version/hardware/subscription context, and validate each interoperability claim with lawful fixture-based round trips.

The defensible route beyond incumbent systems is not a raw feature count. It is a reproducible advantage on: (1) property-relative musical judgment under nuisance transformations; (2) transition success across more than smooth EDM-style blends; (3) calibrated abstention when beat, structure, or stem evidence is ambiguous; (4) set-level intent attainment without homogenizing taste; (5) measurable reduction in DJ workload with preserved agency; and (6) auditable portability with explicit metadata-loss accounting. A claim of state of the art requires protocol-matched data, frozen baselines, held-out DJs and source tracks, uncertainty intervals, and public failure slices. Closed competitors should be compared in controlled user studies or exported artifacts where feasible, not by inventing internal benchmark numbers.


## 14. Proposed target ladder for a genuine SOTA claim

All figures in this section are **research targets, not achieved results**. Freeze the test protocol and comparator versions before measuring them. Report every slice and confidence interval; no overall score may compensate for a failed hard gate.

| Capability | Trustworthy floor | Leadership target |
|---|---|---|
| Contract validator | Exhaustive finite fixture grid: all valid contracts accepted, all known-invalid contracts rejected; zero source-window, authorization, or replay-invariant violations | Mutation testing must show that deleting or weakening each hard validator causes a fixture failure |
| Beat / downbeat | Separate stable-tempo, variable-tempo, swung, live, edited, non-4/4 and half/double-time slices; calibration and abstention reported | On the declared stable electronic subset, target beat/downbeat F-measure >= 0.95 and p95 beat-phase error <= 20 ms; no universal claim outside the supported subset |
| Musical property FLIP/STAY | Per-property FLIP and STAY scores, false-flip and missed-flip rates, real-recording holdout, calibrated abstention | Target >= 92% balanced accuracy on held-out real recordings for each supported property, false flips <= 10%, missed flips <= 10%, ECE <= 0.05 at >= 80% coverage; also measure human performance on the identical cohort |
| Transition preference | Zero hard-constraint violations; matched fixed-crossfade, heuristic, rule-based and MixWeaver baselines | Target >= 60% pairwise preference against each matched baseline with lower 95% confidence bound > 55%, separately by transition type and DJ-experience stratum; power the study using crossed DJ/track/session simulations |
| Stem / role evidence | SDR/SI-SDR plus Bleedless, Fullness, residual, phase and transient diagnostics; reject unsupported role claims | Beat a matched incumbent on the joint quality/latency/compute Pareto frontier, not on a single separation score |
| Live audio runtime | Per-device/configuration deadline, xrun, dropout and latency distributions | Zero xruns in a two-hour soak per declared supported configuration; p99 callback execution <= 50% of the audio-block deadline; publish end-to-end output and headphone-cue latency |
| Controller safety | Full mapping inventory, sequence and authority tests, reconnect and stale-event fault injection | 100% deterministic behavior for declared supported mappings; p99 input-to-control response <= 10 ms under the declared test rig |
| Library portability | Round-trip field inventory and explicit unsupported-field/loss report | Zero unexplained loss for fields declared supported across every declared source/export pair |
| Set planning | Expert rating of intent attainment, novelty, diversity and coherence | At least +5 percentage points in held-out DJ intent-attainment rating over a queue/preference baseline, with no material loss of diversity; test Pareto alternatives, not one scalar |
| Copilot | Treatment/sham/yoked comparison; workload, agency, calibration and override/undo measured | Target >= 0.3 SD improvement in predeclared task outcome with workload and agency non-inferior within 0.2 SD; acceptance rate is not the endpoint |

The numeric leadership targets are intentionally falsifiable. If power simulation, expert annotation reliability, device capability or held-out data do not support a target, downgrade the claim or mark that slice unsupported rather than weakening the test after seeing results.

### Minimum benchmark governance

- Keep source-track, DJ, transition-pair, session, venue/device, and rater identities crossed where possible. No random clip split that leaks tracks or performances across train and test.
- Use at least one real-recording test cohort and one controlled, known-truth cohort. Synthetic data establishes causal ground truth; it cannot substitute for the real-world cohort.
- For initial transition preference studies, plan a pilot with at least 20 DJs, 300 distinct held-out transition pairs and three independent judgments per pair; determine the final sample using a preregistered power simulation and cluster-aware estimator checks.
- Report paired effect sizes, confidence intervals, calibration, coverage, subgroup performance, missingness, and all exclusions. Do not promote descriptive-only or underpowered cohorts to leaderboard claims.
- Freeze analysis model/checkpoint hashes, transition search budgets, separation compute budgets, sample rates, buffers, loudness normalization, preview mastering and playback conditions for each comparison.
- Publish a failure taxonomy: bad beatgrid, wrong phrase boundary, harmonic/melodic clash, vocal collision, transient damage, stem bleed, missing target content, energy discontinuity, wrong set intent, unsafe command, dropout, and portability loss.


## 15. MIREX 2026 beat-tracking baseline and DJ-specific qualification

The 2026 MIREX beat-tracking results give us a more precise public anchor than a generic beat-F1 target. The JSIR experimental jazz set contains 260 recordings. The published Beat This! baseline reports beat F1 94.70 and continuity CMLt 83.50; its downbeat F-measure is 87.27. MIREX also reports GTZAN (999), SMC (217), osu2beat2025 (703 full-length tracks), and Yamaha private cohorts. On the SMC set, the Beat This! row is flagged as trained/tuned on that set and is not ranked; the default CASM system reports F1 57.95 and CMLt 43.30. This variation is exactly why we need per-dataset/per-slice evaluation rather than one pooled average.

References:
- MIREX 2026 beat-tracking results: https://music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking_Results
- MIREX 2026 downbeat results: https://music-ir.org/mirex/wiki/2026:Audio_Downbeat_Estimation_Results
- Beat This! paper (arXiv): https://arxiv.org/abs/2407.21658

### Beat-grid scorecard

Add a dedicated **MEL-DJ-007 — Adaptive Beat-Grid and Tempo-Map Qualification** scorecard (tracked separately from transition quality):

| Dimension | Metric / test | Required comparison |
|---|---|---|
| Beat location | F1 at declared tolerance, Cemgil, P-score | Beat This! and 2026 MIREX submissions where weights/implementation are reproducible; Serato Lucid Beatgrids on the DJ-use cohort where a licensed, reproducible observation protocol is available |
| Continuity | CMLt / CMLc, long-range phase drift, beat insertion/deletion rate | Beat This!; publish accuracy/continuity trade-off |
| Downbeat | F1, precision, recall, meter-confusion matrix | Beat This! / MIREX downbeat results |
| Tempo map | local tempo MAE, slope/turning-point error, half/double-time confusion | constant-tempo tracker, Beat This!, human DJ correction |
| Live adaptation | phase-lock time, reacquisition latency after tempo changes, maximum transient phase error | controlled tempo-ramp / rubato / dropout interventions; matched Serato Lucid Beatgrids comparison on supported material |
| DJ usability | manual corrections per ten minutes, correction time, undo rate, downstream cue/transition failure | current manual workflow and incumbent beatgrid editors, including Serato's Fixed/Flexible modes |
| Calibration | coverage-risk and selective accuracy of alternate grid hypotheses | forced single-grid output versus calibrated multi-hypothesis/abstention |

The target on the JSIR cohort is to exceed the Beat This! reference on beat F1 (>94.70) and continuity (>83.50 CMLt), and to exceed its downbeat F-measure (>87.27), with uncertainty intervals. These are cohort-specific targets, not global thresholds. For DJ-focused electronic material, preregister a stable-tempo subset target of >=0.95 beat F1 and >=0.90 downbeat F1, while preserving CMLt >=0.90. For SMC, rubato, live drummers, tempo ramps, meter changes and edits, report separate distributions and selective coverage; do not force the stable-tempo target onto unsupported material.

MIREX's published GTZAN and SMC test sets are explicitly prohibited from training, validation, model selection, parameter tuning or development. Preserve them as locked test cohorts under their published rules. The osu2beat2025, JSIR and Yamaha test cohorts need their own documented overlap, licensing, annotation, and use review before inclusion. The public leaderboard results are comparison references, not permission to train on test audio.

### The DJ-specific paradigm shift: beat maps as live hypotheses, not static metadata

A beatgrid should be a time-indexed posterior over competing pulse hypotheses, with a confidence and provenance record, not a single static BPM plus a manually placed first beat. Keep independent tracks for:
- pulse/beat phase;
- tempo curve and meter hypothesis;
- phrase/section boundaries;
- expressive timing and microtiming;
- source edit / transient discontinuity;
- confidence, alternative hypotheses and correction history.

The performance runtime consumes a selected, explicitly approved hypothesis. It never overwrites the source or analysis record when the DJ corrects a grid. Reacquisition after a tempo discontinuity is a first-class control problem. Measure the time until the phase estimate returns within the declared tolerance and whether the system falsely reports lock while it remains outside tolerance.

## 16. Dated industry capability snapshot — 2026-11-28

The vendor matrix should be versioned by release, not by product family alone. The currently verified public snapshot is:

| Product | Verified version / recent capability | Implication for Melothaea |
|---|---|---|
| Serato DJ Pro | 5.1.0 is the current download; 5.0 introduced Lucid Beatgrids (automatic tempo tracking) and Open Streaming across multiple services | Variable-tempo following and multi-service support are already industry expectations; test our beat-map hypothesis/phase recovery rather than claiming variable tempo as unique |
| rekordbox | 7.2.20 is the latest release-note entry; 7.2.16 had vocal-position AI detection across subscription plans, Spotify Automix/collection support and hardware additions | The library/streaming/hardware ecosystem is broad; focus on portable, inspectable music intelligence and rights-aware local workflows |
| Traktor Pro 4 | 4.5.1 (2026-07-09) is the current version in Native Instruments' update-status thread; Pro 4 supports four decks, flexible beatgrids and four-stem separation | Four decks, flexible grids and stems are parity; use its beatgrid/manual-correction UX as a comparator |
| Engine DJ | 5.3.0 (2026-11-05) added third-party playlist reimport/one-click import; stem rendering gates export while stems are incomplete | Library interoperation and explicit readiness gating matter as much as generation quality |

Sources:
- Serato DJ Pro downloads / release information: https://serato.com/dj/pro/downloads?download=true&os=win
- rekordbox release notes: https://rekordbox.com/en/support/releasenote/
- Traktor Pro 4 official update status: https://community.native-instruments.com/discussion/43/official-update-status-traktor-pro-4-current-version-4-5-1
- Traktor Pro 4 feature overview: https://www.native-instruments.com/products/traktor-pro
- Engine DJ official downloads/release notes: https://enginedj.com/downloads

Treat this matrix as a dated public-feature inventory. It does not establish equivalent quality, compatibility across all devices, or the internal implementation of any competitor. Re-check release notes when a new comparison is run. The strategic conclusion is unchanged: do not compete on hardware catalog breadth or streaming rights. Build the cross-vendor evidence, musical-property, transition-planning and set-intent layer that a DJ can audit and carry between tools.


## 17. Existing Muse rhythm/audio inventory — ownership boundary

A source inspection of the current `symthaea-muse` crate found useful adjacent components, but did not find an existing waveform-to-beat/downbeat tracker:

- `rhythm.rs` computes a **generative** tempo from cognitive state (arousal, noradrenaline, harmony activation) and provides note-duration/subdivision helpers. It is not an audio beat detector.
- `rhythm_engine.rs` generates meter groupings, groove pockets, syncopation and expressive microtiming. It is not a beat-grid estimator over source recordings.
- `audio_analyzer.rs` uses mel features and an active-inference agent for entropy/surprise curves and peak markers. Its peaks are not annotated beats or downbeats.
- `mixing.rs` is a synthesis/output EQ-compressor-limiter chain, not a DJ transition mixer or source-stem gain planner.
- `streaming.rs` is a PCM synthesis pipeline, not an external-track playback deck or source-audio analysis service.
- `analyst.rs` is an authoritative symbolic composition/structure auditor; its symbolic facts must not be represented as audio-only recognition.

Reuse these components only within their proven semantics. In particular, don't infer an observed beat grid from the generative `compute_tempo`, don't interpret FEP surprise peaks as onsets, and don't use a symbolic score's tempo map as ground-truth audio beat detection. The proposed DJ beat-tracker adapter belongs beside, not inside, those generative systems until a cross-domain interface is reviewed.


## 18. Executable transition-quality replication work

MEL-DJ-004 is now tracked as [issue #14272](https://github.com/Luminous-Dynamics/symthaea/issues/14272). It separates faithful MixWeaver reproduction from any new method. That distinction matters because MixWeaver's public post-review record narrows its current target to smooth intro/outro blending, does not include a DJtransGAN comparison, did not run a MUSHRA study, removed an objective/subjective correlation claim due to only ten pairs, and uses a non-public personal-library dataset. The benchmark will therefore:
1. Reproduce its disclosed candidate construction and objective as closely as available, preserving an explicit deviation ledger.
2. Add matched fixed-crossfade, key/BPM heuristic, and transparent rule-based baselines.
3. Include hard cuts, drop swaps, vocal hand-offs, ambient bridges, variable-tempo blends, and intentional contrast with mode-specific objectives.
4. Measure technical invariants, musical alignment, stem integrity, perceptual preference, set-intent attainment, correction burden, and compute/latency separately.
5. Compare DJtransGAN only when the implementation, weights, license and budgets are available and protocol-compatible.
6. Keep offline transition rendering separate from live device qualification (#MEL-DJ-005).

The key methodological shift is from one universal smoothness score to **intent-conditioned, multi-objective transition quality**. A hard cut that deliberately resets energy should not be scored as a failed smooth crossfade. Preference must also be tested against human DJ transitions, not only algorithmic baselines.


## 19. Copilot and set-trajectory integration with existing programmes

MEL-DJ-006 must be an application layer over existing human/agent-interaction work, not a duplicate causal-study framework:

- **#10088 / MEL-RMI-001** owns reciprocal musical interaction and causal contingency. Reuse its session/event transaction and treatment/yoked-control logic; add DJ tasks (prepare, choose next track, preview transition, respond to live set changes) as a domain-specific cohort.
- **#10190 / MEL-PRE-001** owns predictive transparency and musician anticipation. A DJ copilot preview should be a precommitted forecast object with target execution window, input/evidence digests, uncertainty, alternatives, expiry/revision semantics and eventual outcome receipt. Do not expose private chain-of-thought; show actionable musical commitment and uncertainty.
- **#10133 / MEL-RMD-001** owns longitudinal relational memory and creative non-assimilation. If set-level personalization spans multiple sessions, consume its consent/scope/revision model rather than creating a parallel DJ memory store.
- **#10417 / MEL-BENCH-001** owns the reproducible external SOTA and industry comparison ledger. Add dated DJ product/paper baselines there; this document defines the DJ domain scorecards, not a second global comparison registry.
- **#12190 / MEL-MR-EVAL-042** owns selection-pressure / evaluator-exploitability qualification. Any Pareto or candidate-ranking optimizer must be evaluated as a trajectory across best-of-N, repeated repair, and adaptive search; proxy score increases must be checked against held-out DJ judgments and hard musical obligations.
- **#11297 / MUSE-ORCH-004** is a reusable design precedent for bounded deterministic constraint search and Pareto alternatives. It owns orchestration assignment, not track sequencing or DJ transitions; reuse patterns, not its domain model.
- **#9535 / MUSE-EVAL-MARKET-001** and **#11379 / MUSE-EDITBENCH-001** remain separate composition/editing evaluations. Their outputs can inform music quality and editing, but must not be presented as DJ transition or live-control performance.

The DJ-specific copilot experiment should compare no preview, post-hoc explanation, yoked preview, one precommitted forecast, uncertainty-aware alternatives, and steerable preview. Hold model, input, candidate budget and task schedule fixed across arms. Measure whether a DJ can anticipate, correct, steer or reject a proposed action before execution, and whether this improves intent attainment without degrading agency or increasing workload.

The set-level planner should compose existing candidate and evidence primitives into a DJ-specific transition graph. It returns a Pareto set over stated DJ intent, musical continuity, contrast, novelty, diversity, familiarity and technical feasibility. It must not optimize one universal "good set" scalar or learn a preference profile without an explicit scope/consent and review path.


## 20. MEL-DJ-008 — Pareto set-trajectory planner

The set-level planner is tracked in [issue #14275](https://github.com/Luminous-Dynamics/symthaea/issues/14275). It composes a directed graph: track candidates are nodes; valid transition contracts are edges; a set is a path through that graph under duration, rights, availability, musical and DJ-intent constraints.

Do not optimize a single “good playlist” score. Maintain separate objectives for user-declared energy/density/tension trajectory, continuity and intentional contrast, harmonic/melodic fit where supported, tempo/phrase feasibility, diversity, novelty, familiarity, repetition, vocal/role density, correction burden and technical feasibility. Return multiple non-dominated paths and show their trade-offs. A user preference profile is explicit, scoped, inspectable and revisable; never infer persistent taste by stealth.

The public DJ-AI (Audio Mostly 2025) work is an appropriate track/set planning comparator: its graph-based alignment approach and generated transitions should be evaluated under matched candidate and compute budgets. It is not automatically a direct transition-rendering comparator for all task families. Include expert DJ-curated set plans and a simple queue/nearest-neighbor baseline.

Use #11297 / MUSE-ORCH-004 only as a bounded-search/Pareto design precedent, not as the same domain model. Use #12190 / MEL-MR-EVAL-042 to measure adaptive selection pressure: best-of-N or repeated optimization may increase a proxy score while reducing diversity, human intent attainment, or constraint compliance.

## 21. MEL-DJ-006 — DJ copilot as precommitted and steerable commitment

The DJ copilot is tracked in [issue #14274](https://github.com/Luminous-Dynamics/symthaea/issues/14274). Reuse:
- #10088 / MEL-RMI-001 for reciprocal interaction, event sequencing, and causal intervention design;
- #10190 / MEL-PRE-001 for predictive forecasts committed before execution and musician anticipation;
- #10133 / MEL-RMD-001 for longitudinal memory scope and creative non-assimilation.

The first DJ study should hold model, task, source tracks, candidate count, compute budget and session schedule fixed while comparing no preview, post-hoc explanation, yoked preview, one forecast, uncertainty-aware alternatives and steerable preview. Measure whether the DJ can predict the action before it happens, correct it faster, reach stated intent more reliably, and retain agency. Acceptance/click-through is not a causal outcome. A preview is not authority to execute; the runtime still requires a fresh bounded grant.

## 22. Programme dependency order and exit conditions

Recommended dependency order:
1. #14245 transition contract + schema + deterministic oracle.
2. #14269 beat-grid/tempo-map evidence and correction semantics.
3. #14272 transition-quality baseline harness and intent-specific transition families.
4. #14275 set-trajectory path planner over qualified nodes/edges.
5. #14274 copilot forecasts and DJ-specific RMI/PRE experiment.
6. MEL-DJ-005 live runtime only after the protocol, transitions, and device/audio qualification contracts are stable.

The Perform shell (#14244) is presentation-only and can evolve independently. Do not wire it to mock deck actions. Keep the normal Listen player separate. The DJ engine remains a distinct runtime with a strict real-time callback boundary.


## 23. MEL-DJ-005 — native multi-deck runtime boundary

The native runtime is tracked in [issue #14279](https://github.com/Luminous-Dynamics/symthaea/issues/14279). The existing Muse `live_output.rs` is a useful reference for CPAL device lifecycle and a synthesis-thread-to-callback ring-buffer boundary, but it is not the DJ runtime. It streams `StreamingSynth` output; the instrument `sample_player.rs` loads WAV instrument samples; `mixing.rs` is an output EQ/compressor/limiter. None provides general external-track decoding, independent deck transports, headphone cue routing, a transition mixer graph or controller interoperability.

The existing implementation also exposes exactly the kinds of issues the DJ runtime must qualify rather than inherit blindly: the synthesis worker can sleep/spin while a ring buffer is full, the callback zero-fills when it under-runs, and the observed code does not provide the device-specific underrun/deadline evidence needed for DJ use. These are not necessarily defects for its existing synthesis workload; they simply do not meet the DJ runtime contract.

MEL-DJ-005 therefore uses three explicit planes:
- **Control plane:** decoder loading, source verification, user/controller commands, authorization, state transitions, device negotiation, configuration and recording manifests.
- **Real-time data plane:** bounded preallocated PCM paths, independent deck clocks, resampling/time-stretch, mixer buses, cue/master outputs, and frame-boundary command application. No file/network I/O, decode, allocation, inference, logging, UI or unbounded lock acquisition in the callback.
- **Evidence plane:** source and decoder digests, device/profile settings, callback-time histograms, underruns/overruns, route state, controller receipts, clock drift and recording-integrity/replay evidence.

Start with a deterministic offline two-deck graph and fault injection before opening a hardware device. Then qualify one named device/OS/driver/sample-rate/buffer profile at a time. Four decks, stems, hardware controllers, recording and live copilot are later gates, not implied by a two-deck playback result.


## 24. Transition alignment must bind to selected beat-grid hypotheses

The transition contract now carries source and target beat-grid hypothesis IDs whenever it contains an explicit alignment. Each ID resolves within the analysis evidence manifest attached to that side's source reference. This prevents a transition proposal from retaining only two frame anchors while losing which of several plausible pulse/tempo interpretations produced them.

The resulting evidence chain is:

1. Source audio hash and source-local sample clock.
2. Versioned auditory/analysis evidence manifest from #14227.
3. Versioned beat-grid report and its resolved/ambiguous hypothesis IDs from #14285 / MEL-DJ-007.
4. Transition contract selects one source and one target hypothesis and records the aligned frame anchors.
5. The runtime independently checks current authority, deck state, clock conversion and device constraints before executing anything.

An ambiguous report can still support a DJ-authored choice, but the contract must preserve which hypothesis the DJ selected. The validator checks that the identifiers are present; the verifier that resolves them against their manifests remains a separate evidence-binding check before a contract is treated as fully grounded. Structural validation alone does not prove musical correctness or execution authority.


## 25. Tempo-map interpolation is part of the evidence contract

The beat-grid report contract requires each hypothesis to state how tempo values between map points are interpreted:
- `stepwise_constant`: hold the prior tempo until the next map point;
- `linear_in_frame_time`: interpolate BPM linearly in source-frame time.

A non-empty tempo map without an interpolation declaration is invalid. A report with no tempo map must use `null`. This prevents the same beat-grid points from producing different phase forecasts in the planner, transition preview and real-time clock. It does not imply that either interpolation mode accurately models the performer; the estimator's residual, calibration and out-of-domain status remain separate evidence.


## 26. Decoder, resampler, time-stretch and real-time handoff qualification

The native runtime must select these components by measured end-to-end performance and distribution constraints, not by a crate's feature list or a synthetic demo.

### Candidate component matrix — 2026-11-28

| Role | Candidate | Strengths / constraints | Decision |
|---|---|---|---|
| Source demux/decode | Symphonia 0.6.1 | Pure Rust, MPL-2.0, explicit format/codec feature flags, good metadata and gapless support for several pairs. It does not make every container/codec combination gapless: e.g. MP3/FLAC/ALAC/WAV are documented gapless, while AAC-LC and ISO/MP4 are not. Gapless needs both demuxer and decoder support. | Primary Rust-first candidate for corpus and format-conformance testing. Pin a minimal explicit feature set; do not enable `all` by default. |
| Sample-rate conversion | Rubato 5.1.0 | Rust; fixed-ratio FFT and variable-ratio asynchronous sinc/polynomial modes. Its real-time API uses preallocated buffers; asynchronous modes can compensate for device clock drift. It is a resampler, not pitch-preserving time-stretch. | Primary Rust candidate for output-device conversion and clock-drift correction; evaluate quality/CPU trade-offs. |
| Pitch-preserving time-stretch | Signalsmith Stretch | MIT; C++11 header-only polyphonic pitch/time library with explicit input/output latency. Upstream says time-stretch quality is best over a more modest range, approximately 0.75x–1.5x. A Rust wrapper exists but must be inspected, pinned and qualified. | First quality/latency candidate; FFI and target support are explicit review gates. |
| Pitch/tempo comparator | SoundTouch 2.3.3 | LGPL-2.1; established C++ tempo/pitch processor. | Include in comparative evaluation; decide bundling only after license review. |
| High-quality external reference | Rubber Band 4.0 | GPL or separately licensed commercial distribution. | Optional evaluation comparator; do not silently add it to shipped binaries. |
| SPSC PCM handoff | rtrb 0.4.1 vs existing ringbuf 0.4 | rtrb documents fixed-capacity, wait-free SPSC reads/writes, no post-construction allocation, and explicit full/empty errors. Existing ringbuf is already in the Muse audio output path. Neither is automatically superior for our target load. | Benchmark both with real block sizes, multi-deck traffic, backpressure, cache contention and fault injection before choosing. |
| Device I/O | CPAL 0.18.2 vs repo CPAL 0.17 | CPAL 0.18.2 reports xruns on AAudio/CoreAudio/PipeWire and provides more accurate hardware timestamps/latency; it also contains significant API changes. CPAL repository master has 0.19 API changes that are not the same as a published stable release. | Test the published 0.18.2 migration against our pinned Rust/toolchain and supported backends; do not target unreleased master APIs. |

References:
- Symphonia 0.6.1 feature, codec, gapless and license information: https://docs.rs/crate/symphonia/0.6.1
- Rubato 5.1.0 real-time and resampling modes: https://docs.rs/rubato/5.1.0/rubato/
- Signalsmith Stretch implementation, latency and operating range: https://github.com/Signalsmith-Audio/signalsmith-stretch
- SoundTouch license and tempo/pitch behavior: https://github.com/Tencent-RTC/soundtouch
- Rubber Band license and distribution terms: https://github.com/breakfastquay/rubberband
- rtrb 0.4.1 SPSC guarantees: https://docs.rs/rtrb/0.4.1/rtrb/
- CPAL 0.18.2 release notes: https://github.com/RustAudio/cpal/releases/tag/v0.18.2
- Reproducible DSP benchmark methodology (time-stretch, pitch-shift, SRC): https://github.com/syshlted/timepitch-bench

The `timepitch-bench` project is a useful methodology reference, not a field-wide standard or a sufficient DJ benchmark: it records CPU/real-time factor, memory, pitch error, latency and library/build identities, but its current test corpus is largely synthetic and its own next steps include transient-preservation scoring. Extend that style of reproducibility with polyphonic musical sources, real drum transients, stereo scenes, tempo ramps, audio artifacts, and listener studies.

### Real-time graph boundary

The proposed pipeline is:

1. **Decoder worker:** source-file I/O, demux, decode, metadata/gapless handling and source digest verification.
2. **Per-deck DSP worker:** resampling, pitch-preserving time-stretch, channel conversion and bounded pre-render of upcoming PCM. All algorithm state and input/output buffers are allocated before processing.
3. **Bounded SPSC handoff:** one explicitly sized PCM queue per deck (or a benchmark-justified topology); no worker waits by spinning inside the device callback. Full and empty are observable, counted and fault-injected.
4. **Device callback:** consumes ready frames, applies already-validated sample-accurate transport/gain/routing commands, mixes preallocated buses and writes the device buffer. No decode, filesystem access, stretch model, allocation, blocking lock, logging, UI callback or unbounded work.
5. **Evidence collector:** outside the callback, records queue occupancy, underrun/overrun/xrun events, command age, phase error, decoder faults, worker timing and device timestamps.

The DSP worker's look-ahead buffer is part of the user-facing latency budget. A high-quality offline stretch render does not qualify a live tempo adjustment. Commands must declare an effective frame/time, processing horizon and cancellation/supersession behavior; late commands are rejected or explicitly deferred rather than applied at an arbitrary future point.

### Required decoder qualification

- Explicit supported format/codec matrix with version and feature flags; test each pair, not just each file extension.
- Gapless priming/padding, encoder delay, VBR seeking, container timestamps, long files, corrupt/truncated frames and metadata/cover-art bounds.
- Sample-count agreement to the declared source clock; channel order/layout; 16/24/32-bit integer and float conversions; NaN/Inf/clipping handling.
- Decode throughput as real-time factor plus p50/p95/p99 chunk time, first-audio latency, cold/warm seek latency and peak memory per concurrent deck.
- Reference decoding comparisons (e.g. FFmpeg) for licensed fixture corpus with explicit tolerance and version pin.
- Decoder errors, unsupported feature and corrupt frame remain typed outcomes; no silent substitute codec, duration estimate or false gapless claim.

### Required DSP qualification

Test resampling and pitch-preserving time-stretch separately:

| Layer | Signal / task metrics | Live/runtime metrics |
|---|---|---|
| Sample-rate conversion | passband ripple, stopband attenuation, alias rejection, channel match, sample-count and group-delay error | p95/p99 worker time, fixed/variable-ratio response, drift correction and buffer occupancy |
| Time-stretch | pitch error in cents, tempo-ratio error, transient timing/shape preservation, phase coherence, stereo image, formant/timbre stability, modulation artifacts | algorithmic input/output latency, p95/p99 block time, worst-case time under tempo ramps, queued look-ahead and audible response time |
| Pitch-shift | tuning error, spectral residual, formant deviation, transient degradation, stereo coherence | processing budget and automation response across the declared pitch range |
| Integrated deck | beat/phrase phase drift, discontinuity/click energy, loudness and clipping, recovery after seek/rate change | underruns/xruns, callback deadline, command-to-audible latency, CPU/RAM/energy under 2- and 4-deck load |

Use synthetic known-truth material (sines, impulses, chirps, multitone, noise, controlled transients) and licensed polyphonic material (drums, bass, sustained harmony, lead vocals, dense mixes, live/rubato). Evaluate both static ratios and ramps, with primary declared range 0.75x–1.5x plus explicit stress tests outside it. Normalize loudness for listening, but keep original and output levels available for diagnostic metrics. A single spectral distance or one pitch-error score must not hide damaged transients or stereo cancellation.

### Decision rule

No library is selected for production by reputation alone. A candidate must pass:
- license and distribution review;
- deterministic offline quality tests;
- no-allocation/preallocated processing audit for its real-time path;
- CPU/RAM/latency tests on target hardware;
- failure and recovery tests;
- comparative listening on matched, held-out, licensed material.

Maintain at least one permissively licensed production candidate and one external reference comparator. If no candidate meets the declared quality/latency/compute envelope, the capability remains unsupported rather than weakening the gate.


## 27. Verify transition alignment against the selected pulse hypotheses

The transition contract now includes `max_phase_error_milli` (0–500 milli-beats). This is the declared maximum circular beat-phase difference between source and target alignment anchors. It is not a probability or a quality score.

The stacked MEL-DJ-007 implementation adds `verify_transition_grid_binding`:
1. Revalidates the transition contract and both beat-grid reports.
2. Requires each report's exact source reference and analysis manifest to equal the corresponding TrackWindowRef in the contract.
3. Resolves the selected source and target hypothesis IDs within those reports.
4. Maps each alignment frame to a beat position using only observed pulse anchors.
5. Compares circular phase error modulo one beat against the contract's declared tolerance.

The verifier interpolates only between adjacent beat ordinals. It accepts an exact observed pulse anchor directly, but does not interpolate through a multi-beat detection gap or extrapolate outside observed pulse support. Such cases return a typed abstention/violation instead of manufacturing a phase estimate. The verifier establishes a structural link between declared evidence and the transition's beat-phase constraint; it does not prove detector accuracy, calibrated confidence, phrase semantics, perceived transition quality, or playback authority.


## 28. Exact JSON integer boundary for source clocks and beat indices

The Rust protocol uses integer frame coordinates and signed beat/bar ordinals so that source clocks are deterministic and no floating-point drift is introduced by the wire representation. However, ordinary browser JavaScript consumers parse JSON numbers as IEEE-754 Number values and cannot exactly represent every 64-bit integer.

The protocol therefore caps every numeric field that represents a frame position/count or beat/bar ordinal to:

`MAX_SAFE_JSON_INTEGER = 9,007,199,254,740,991` (`2^53 - 1`).

The shared transition validator rejects source frame totals and transition-local beat ordinals above this boundary. Its v1 JSON Schema also caps source/target sample-window frames, source frame totals, alignment anchors and gain-point frame coordinates. The beat-grid validator applies the same constant to source frame totals, pulse/downbeat/tempo frame positions (through the covered source window), signed pulse/downbeat beat ordinals, bar indices and meter segment beat positions; the beat-grid schema encodes the same bounds.

This is a wire-compatibility bound, not a physical audio limit. At 48 kHz it is many orders of magnitude beyond any realistic track duration. It prevents a browser or JSON intermediary from silently rounding a valid Rust coordinate to a different frame and preserves deterministic comparison across native and WASM clients. Any future need for larger counters must introduce a new schema version with decimal-string or explicitly structured wide-integer encoding; it must not quietly widen the JSON integer fields.

The current PR stack is:
- #14267 transition contract, exact head `1f8b30267576d7fade2c2abdb977a1efc5ea80b1`.
- #14302 beat-grid contract and phase verifier, exact head `4c2f4f166a0e382cb51e5a26883685f44ff6161e`, based on #14267's exact head.

At the latest check, #14267 had CI and PR Governance queued, Showroom Integrity pending, and five skipped entries; #14302 had CI and governance queued and two skipped entries. These are not passes. The local execution container still lacks `rustc` and `cargo`, so local formatting, compilation, tests and Clippy remain unrun.


## 29. MIREX 2026 benchmark rules and score interpretation

The MIREX 2026 task page makes the public comparison protocol more precise:

- The SMC and GTZAN test splits are explicitly prohibited for training, validation, model selection, parameter tuning, or any other development purpose. These are locked test sets, not a source of weak labels or convenient tuning clips.
- Other collections may be used, but participants must clearly state dataset usage and, when applicable, the collection process in the extended abstract.
- MIREX beat F-measure uses a 70 ms tolerance window; Cemgil uses a 40 ms Gaussian standard deviation. CMLc/CMLt and AMLc/AMLt capture continuity over longer tracked sections and must be reported alongside local accuracy.
- The 2026 JSIR experimental set has 260 jazz recordings. Its beat labels were pre-annotated using madmom and hand-corrected; downbeats were manually marked on those beat sequences. Seven recordings overlapping the Beat This!/CASM training data (via JAAH, Hainsworth and SIMAC) were removed. Metrics are computed on annotated valid segments and weighted by duration.

References:
- MIREX 2026 beat task protocol and locked-test policy: https://music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking
- MIREX 2026 beat results: https://music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking_Results
- MIREX 2026 downbeat results: https://futuremirex.com/mirex/wiki/2026:Audio_Downbeat_Estimation_Results

The JSIR Beat This! comparison remains beat F1 94.70 / CMLt 83.50 and downbeat F-measure 87.27. These are cohort- and protocol-specific references, not a global universal threshold. The new MEL-DJ-007 harness must preserve the official evaluation definitions (including valid-segment handling) when producing a MIREX-compatible scorecard; DJ-specific tempo-ramp and phase-recovery measures are additional columns, not replacements for the standardized metrics.

### Dataset governance rules for the Melothaea benchmark

1. GTZAN and SMC remain test-only under MIREX rules. They may not appear in any train, validation, model selection, hyperparameter search or debugging flow.
2. osu2beat2025 and JSIR are not automatically free development sets merely because their annotations are public/visible. Record source, audio rights, annotation provenance, overlap audit, split role and allowed use before adding any example.
3. Yamaha private cohorts are comparison references only where the data owner authorizes access and publication; no private audio or labels enter the public test fixtures.
4. Keep source master identity, remix/edit identity and segment overlap in the split keys. A different encoding or cut of the same master is not independent test data.
5. Publish the full metric vector and cohort counts; do not hide continuity/coverage losses behind a high beat F1.
6. Record any human correction, retiming, or DJ beatgrid adjustment as a separate treatment/outcome. Do not overwrite ground truth with the algorithm's own corrections.

## 30. Joint phase and tempo verification for a beatmatched transition

The latest protocol revision requires three independently declared limits on any explicit beat alignment:

- `max_phase_error_milli`: circular difference between fractional beat phases at the two selected source-local frame anchors.
- `max_tempo_delta_milli_bpm`: maximum absolute difference between the source and target tempo-map values evaluated at their respective anchor frames.
- `max_tempo_ratio_error_ppm`: maximum relative tempo difference, normalized by the slower tempo, in parts per million.

The absolute and relative constraints are both necessary. A fixed absolute BPM tolerance behaves differently for slow and fast material. A fixed percentage can permit a musically unacceptable absolute delta in some use cases. The dual contract makes the DJ's allowed trade-off explicit and machine-checkable.

The dependent implementation [PR #14317](https://github.com/Luminous-Dynamics/symthaea/pull/14317), stacked on transition-contract PR #14267, now evaluates each selected tempo map at the source-local anchor using its declared interpolation mode. It rejects missing tempo support, an excessive absolute delta, or an excessive slower-normalized ratio error. Its known-answer tests include:
- Same fractional phase but 120 BPM against 90 BPM: reject.
- 60 BPM against 60.5 BPM under a 1% relative tolerance and 1 BPM absolute tolerance: accept.
- Missing tempo-map support on either side: reject.
- Stepwise-constant and linear-in-frame-time tempo interpolation: produce distinct, deterministic values.

The transition-local ordinal is named `transition_beat_ordinal_milli`; it is a DJ-set timeline label, not an observed track-local beat index. The actual track-local positions are resolved from the selected hypotheses and source-local pulse anchors.

This verifier establishes only the consistency of declared reports and transition constraints. It does not establish that the underlying beat/tempo estimator is accurate, its confidence is calibrated, the selected phrases are musically compatible, or the transition sounds good. Those are separate MEL-DJ-007 and MEL-DJ-004 qualification questions.

## 31. Competitive strategy update — what is already parity

The late-November 2026 public capability snapshot confirms that the following are not differentiators:
- Variable tempo beatgrids and manual warp markers (Serato Lucid Beatgrids / Fixed and Flexible modes).
- Flexible beatgrids and stems in Traktor Pro 4.
- Broad controller/hardware ecosystem, cloud/library integrations, and Spotify/streaming workflow features in rekordbox.
- Third-party playlist/crate reimport and one-click import in Engine DJ 5.3.

Official release notes confirm rekordbox 7.2.20 (19 Nov 2026), Engine DJ 5.3.0 (5 Nov 2026), and Traktor Pro 4.5.1 (9 Jul 2026). Serato's current support article describes moving-tempo beatgrids with Beat Warp Markers and preserving older beatgrid data until explicit reanalysis.

The opportunity is not to reproduce each vendor's hardware/streaming surface. It is to outperform their observable workflows in a controlled study through:
- Multi-hypothesis beat/tempo evidence with calibrated abstention and correction provenance.
- Joint phase and tempo feasibility with explicit user-controlled tolerances.
- Transition objectives conditioned on declared intent (smooth blend, hard cut, drop swap, vocal handoff, ambient bridge, deliberate contrast).
- Portable source/analysis/transition evidence that can be consumed by more than one player.
- Set-level Pareto trajectories and causal copilot forecasts that preserve DJ steering and veto.

Compare to the vendors as products, but only make technical superiority claims on matched data, named software/hardware versions, frozen inputs, equal candidate/compute budgets, blinded DJ cohorts and pre-registered outcomes.
