# Melothaea Instrument, Orchestration & Style Identity Protocol

**Protocol:** MEL-MR-STYLE-001  
**Issue:** #9397  
**Status:** preregistration / design specification — no benchmark result claimed  
**Date:** 2026-10-31

## 1. Research objective

Melothaea should produce instrumental music whose identity comes from the actual musical grammar and its performance—not merely from a genre label, an instrument preset, a SoundFont or a mastering chain.

The qualification problem is to measure three interacting but separately attributable layers:

- **Composition grammar:** form, phrase rhetoric, melody, motif development, harmony, rhythm, counterpoint and structural obligations.
- **Performance and orchestration:** instrument assignment, register, technique, articulation, dynamics, phrasing, balance, doubling and microtiming.
- **Rendering and production:** synthesizer/build, SoundFont or sample bank, controllers, effects topology, spatialization, sample rate and mastering.

A listener's ability to identify a style from its timbre alone is not proof of compositional identity. A plausible instrument patch is not proof of idiomatic performance. A clean render is not proof of either.

## 2. Main repository facts at protocol creation

The following were checked on the default branch while preparing this protocol:

| Area | Current evidence | Interpretation |
|---|---|---|
| Instrument vocabulary | 42 `Instrument` enum variants in `symthaea-muse/src/instruments.rs` | A vocabulary of timbre models, not 42 equally complete sampled instruments |
| Instrument realization | Partial spectra, velocity response, sustain/decay classification, attack noise, vibrato, inharmonicity, FM / Karplus–Strong routing, GM programs and name mapping | Several tables are derived/estimated or fall back to shared tables; each model needs an evidence label |
| Sampled instrument path | `vcsl.rs` indexes VCSL and VSCO2-CE, with lazy decode/cache, velocity-layer selection and deterministic round-robin; `synth.rs` tries sampled playback before synthesis fallback | Current source maps 19 enum variants to sustain banks (Piano/PianoPP share one bank) and nine variants to short-articulation banks; activation is optional and installed file/pitch/velocity coverage is not yet proven by a runtime receipt |
| Ensemble selection | Seeded, style-specific pools in `theory_realize.rs::instruments_for` | Deterministic variety exists, but it is not yet a complete score-conditioned orchestration decision |
| Style vocabulary | 29 currently composer-backed `Style` variants | These are the actual engine routes to qualify first |
| Style catalog | 152 target entries, 19 constellations; 29 composer-backed and 123 research-only according to the catalog document | Research-only entries must remain unavailable for composition until implemented and qualified |
| MIDI output | Multi-track export, per-voice CC7/CC10, and style-gated channel-10 percussion already exist | Do not redo basic balance/panning/percussion; investigate expression and articulation beyond them |
| Performance dialects | Existing ClassicalRubato, DanceLocked, ProcessExact, FolkLift and JazzLaidBack mechanisms | Audit coverage and style assignment; do not replace existing distinctions with a single new generic humanizer |

The repository's older diversity review remains useful historical evidence, but its claims must be rechecked against current code before being treated as current defects. This protocol does not assume that all previously reported issues remain unfixed.

## 3. Falsifiable hypotheses

**H1 — Grammar survives timbre substitution.** A held-out listener or render-neutral evaluator can identify a supported style above a preregistered chance/control level when instrumentation is held constant across styles.

**H2 — Timbre cannot impersonate grammar.** Changing only instruments or SoundFonts must not cause the symbolic grammar verifier to report a different style contract or an altered score.

**H3 — Score-conditioned orchestration improves fit.** A planner using score register, density, role, sustain and articulation requirements reduces range violations and masking relative to the current seeded-pool baseline without degrading style identity or within-style diversity.

**H4 — Expressive controls are renderer-specific.** Controller curves and articulation policies produce the declared effect only on the exact tested renderer/patch combinations; other combinations abstain or use a separately qualified fallback.

**H5 — More variety need not mean style drift.** A family of compositions can be measurably diverse in motifs, phrase development, rhythmic cells, harmony, form and orchestration while retaining its frozen identity contract.

The hypotheses are to be tested, not assumed true.

## 4. Data model

### 4.1 Instrument capability profile

Reuse and extend the existing source-backed instrument capability inventory in PR #7464 rather than creating a competing profile schema. Its canonical profile should continue to separate *what an instrument is*, *what the model can do*, *what sample assets are declared*, and *what a renderer actually supports*. This protocol defines the missing measurement/qualification fields and expects them to be added through #7464 / #7554.

The canonical qualification workflow is the four-tier contract in #7554: deterministic parser/receipt tests (Tier A), decoded asset coverage (Tier B), actual render matrices (Tier C), and separate acoustic/perceptual evidence (Tier D). A sample path in the inventory must never be treated as proof of a rendered articulation.

Minimum fields:

- stable instrument ID, display/canonical names, family and subfamily;
- transposition and documented playable range / tessitura with source and confidence;
- polyphonic or monophonic behavior and role suitability;
- sustain, decay, attack and velocity behavior;
- implemented articulation/technique set, with each technique linked to its implementation;
- synthesis method: additive measured / additive derived / estimated, FM, Karplus–Strong, SoundFont program, SFZ or mapped sample;
- renderer-specific program, bank, keyswitch, controller and effect support;
- declared sample-bank mapping, exact installed-asset availability, and sample coverage by pitch, velocity layer, articulation, round-robin and release sample where applicable;
- source URI, license, acquisition version, content digest and attribution;
- evidence status: measured, source-cited model, derived, estimated, code-declared mapping, runtime-verified or unverified.

A source-code mapping to a sample directory is not itself a coverage claim. Runtime receipts must distinguish a selected recording from a synthesis fallback, and must state pitch/velocity window fallbacks explicitly.

Do not fill unknown ranges or techniques with plausible-looking defaults. Unknown must remain unknown. A fallback instrument may be offered, but its use must be visible in the orchestration decision and render receipt.

### 4.2 Renderer capability profile

A control is not a property of an instrument name alone. It is a property of a particular renderer, patch/program, bank and configuration.

A `RendererPatchCapability` should bind:

- renderer executable/build identity;
- SoundFont/SFZ/sample-bank digest;
- bank and program (and keyswitch map if relevant);
- sample rate, channels and output format;
- effects, modulation and route configuration;
- validated controller ranges and observed control response;
- supported articulations and any patch-specific restrictions.

CC1, CC2, CC11 and CC64 are candidates for experiments, not universal guarantees. CC7 and CC10 are already emitted by current theory MIDI export. Validate a proposed controller through paired renders and retain the measured result. A controller that does nothing on a patch must not be treated as expressive capability.

For compatible targets, add an optional **MPE** export profile for per-note pitch, timbre and pressure/expression. MPE uses per-note MIDI channels to avoid applying channel-wide controls to every note in a chord (MIDI Association, https://midi.org/mpe-midi-polyphonic-expression). Keep ordinary GM MIDI as a compatible fallback; never claim MPE support without an explicit target-device/renderer profile.

### 4.3 Style contract

Each engine-backed style should have a versioned `StyleIdentityContract`:

- required and optional grammar features;
- phrase rhetoric and melodic-development expectations;
- rhythmic cell / meter / accent behavior;
- harmonic syntax and cadence behavior;
- form and long-range obligations;
- allowed instrumentation / role substitutions;
- performance dialect and technique expectations;
- anti-features (e.g. a required drone must not become a changing chord progression);
- culturally scoped terminology, provenance, expert review and attribution rules;
- deterministic feature extractor and contract version.

A named genre is not a single universal recipe. Contracts should describe a particular tradition, region, period or substyle where required. Do not promote styles requiring cultural review based only on generated examples or model-classifier confidence.

### 4.4 Orchestration plan

An `OrchestrationPlan` should be computed from the score and intent, not solely from `seed % pool.len()`.

Candidate constraints and objectives:

**Hard constraints**
- instrument's documented pitch range and transposition;
- monophony/polyphony limitations;
- supported technique / articulation;
- voice-role and required part coverage;
- renderer/patch capability, if the selected output target is known.

**Soft objectives**
- register separation and masking risk;
- ensemble blend and timbral contrast;
- score density and sustained-energy budget;
- idiomatic style fit;
- dynamics and spatial balance;
- resource footprint and render latency;
- within-style variation.

The planner must be deterministic for a frozen input/seed. It emits the chosen assignment, rejected alternatives, hard-constraint checks, objective contributions, evidence references and explicit abstentions. Hard constraints cannot be traded away for a higher diversity or novelty score.

## 5. Factorial identity experiment

Use a crossed design that isolates causes rather than confounding style, instrument and renderer.

| Arm | Composition grammar | Ensemble / performance | Renderer |
|---|---|---|---|
| A. Native-style baseline | Style-specific | Style-specific | Fixed reference renderer |
| B. Instrument-controlled | Style-specific | Same common ensemble across styles | Fixed reference renderer |
| C. Ensemble-swap | Fixed source grammar | Swap among qualified ensembles | Fixed reference renderer |
| D. Renderer-swap | Fixed MIDI/performance | Fixed | Multiple SoundFonts / renderers |
| E. Grammar-swap | Swap style grammar | Same ensemble | Fixed reference renderer |
| F. Full-system | Style-specific | Score-conditioned plan | Target renderer |

The minimum analysis estimates:

- style-grammar main effect;
- instrumentation main effect;
- renderer main effect;
- grammar × instrumentation interaction;
- grammar × renderer interaction;
- instrumentation × renderer interaction.

Do not infer style identity from Arm A alone: it confounds all three layers.

### Renderer strata

For the synthetic SMEBench-compatible work, test the available five SoundFonts named in EVAL-030: Arachno 1.0, FluidR3_GM, GeneralUser GS, General Montage and TimGM6mb. Check acquisition conditions, exact file identity and licensing before using each bank.

Use identical source MIDI, performance events, sample rate, channel configuration and effects settings wherever the renderer allows. Retain the full configuration and audio digest. Different renderers are not expected to produce identical waveforms. The cross-render comparison is about retained musical properties and perceptual identity, not raw PCM equality.

Compare NativeMuse and FluidSynth as distinct renderer strata. Do not pool their results without renderer-specific analysis.

## 6. Benchmark battery

### B1 — Blind style recognition

Ask listeners to identify style from randomized audio, then repeat with a common instrumentation and with renderer cues controlled. Report per-style recall, macro-F1, balanced accuracy, confusion matrix, listener confidence, and human expertise. The primary result is the controlled-instrumentation result, not the easiest full-mix classification.

### B2 — Grammar-only identity

Evaluate score-derived/render-neutral descriptors and frozen contracts:
- interval and melodic-contour vocabulary;
- phrase lengths, boundaries, antecedent/consequent structures;
- rhythmic-cell / accent / syncopation behavior;
- harmonic vocabulary, progression and cadence;
- motif recurrence/development;
- form and long-range obligations;
- contrapuntal entry, part identity and texture.

This is not a claim that feature extraction fully captures music. It is a reproducible objective layer that can be checked against listening.

### B3 — Instrument and orchestration fit

Report:
- range/tessitura violations and transposition errors;
- role coverage and inappropriate doubling;
- register overlap and masking proxy;
- instrument-family recognition under renderer swap;
- controller response coverage;
- sample coverage holes and fallback incidence;
- listener ratings of idiomaticity, balance and suitability.

### B4 — Articulation and expression

For every claimed technique, create paired fixtures:
- intended technique present;
- neutral/no-op control;
- wrong technique;
- wrong voice/region;
- unsupported renderer/patch;
- accidental collateral change.

Measure attack, sustain and release envelopes; onset/offset timing; velocity/dynamic trajectories; controller response; acoustic tail; and blinded listener judgments. Technique claims must be scoped to the patches where they were verified.

### B5 — Within-style diversity and identity retention

Measure variation in motif, interval contour, rhythm, harmony, phrase rhetoric, form, register, ensemble and production separately. Report nearest-neighbor distances, exact/near duplicate rates, and whether the identity contract still passes. Do not reward novelty that violates the style.

### B6 — Renderer robustness

For each supported style/instrument/technique, report:
- style-recognition and instrument-recognition changes across renderers;
- preservation of symbolic/behavioral contracts;
- controller effect sizes and confidence intervals;
- failed, unsupported and abstained renderer/patch combinations;
- rank stability with uncertainty;
- p50/p95 latency and resource use.

No universal cross-renderer waveform similarity threshold is valid.

### B7 — Blinded perceptual study

Randomize candidate order and blind system, style label and renderer where possible. Retain per-listener judgments for:
- style recognition;
- instrumental identity;
- idiomaticity;
- arrangement balance;
- musical coherence;
- perceived artifacts;
- emotional-intent agreement;
- preference and confidence.

Use a pilot to estimate variance and sample size. Cluster uncertainty by composition lineage and listener, not by excerpts or repeated ratings treated as independent. Report trained-musician and non-expert strata separately.

## 7. Baselines and external comparison

Internal baselines:

1. current seeded ensemble selection;
2. common fixed ensemble;
3. random valid ensemble;
4. score-conditioned orchestration planner;
5. no-expression MIDI baseline;
6. renderer-qualified articulation/expression path;
7. native renderer;
8. FluidSynth reference path.

External comparisons:

- **SMEBench:** use as a controlled edit and paired-target reference where operations overlap; do not mislabel it as a genre-recognition benchmark.
- **MMAE:** compare editing instruction-following only under its official task scope and frozen evaluator; do not claim that composition quality or instrumental idiom is captured by MMAE.
- **Human instrumental references:** curate legally usable, source-attributed, instrument/technique-labeled corpora for the exact instrument/style question. Compare with commercial sample libraries only when license and evaluation access permit it.
- **Music generation systems:** compare only on matched prompt, duration, output modality, human study and task. A symbolic score engine and an audio foundation model do not share the same input/output scope by default.

This track's proposed differentiator is *controllability, compositional identity, transparent orchestration and verifiable preservation*, not a generic claim to beat every audio model at open-ended generation.

## 8. Qualification policy

The following are proposed gates, not measured results:

- 100% reject known out-of-range notes, transposition errors, impossible role assignments and unsupported controller claims in deterministic fixtures.
- 100% preserve frozen score invariants in renderer-only and ensemble-only counterfactuals.
- Every instrument profile exposes measured / derived / estimated / unknown status and renderer scope.
- Every technique claim has at least one positive, no-op and wrong-voice/wrong-region control.
- No style can become Foundation without held-out blind listening and any expert review required by its catalog entry.
- Style recognition must remain discriminable with instrumentation held constant; establish confidence intervals first, then freeze a threshold on the development split.
- Every report includes worst-style and worst-renderer strata, abstention, fallback and error reasons.
- No SOTA or cultural-authenticity statement without matched baselines, source lineage, qualified listeners and independently reviewable evidence.

## 9. Data and licensing

Use asset manifests, SHA-256 digests, provenance and license records before sample ingest or model fitting.

Candidate sources:

- **VCSL** (official project: https://github.com/sgossner/VCSL) declares CC0. Its upstream README documents readable instrument/articulation/velocity/round-robin naming and notes that most articulations have only one round-robin and two to three velocity layers. It is suitable for sample-backed instrument and percussion coverage, but does not imply a deep commercial-library level of round-robin or dynamic sampling. Pin the acquired release and digest every file.
- **VSCO 2 Community Edition** (official project: https://github.com/sgossner/VSCO-2-CE) is already used by the current Muse sample mapper. Its upstream README permits use for any purpose but asks users not to sell the raw samples directly and encourages credit/sharing improvements. Verify the exact license file/package terms at the pinned revision before bundling or redistribution.
- **NSynth** (official dataset: https://magenta.tensorflow.org/datasets/nsynth) contains 305,979 four-second, monophonic, 16 kHz notes from 1,006 instruments, with pitch/velocity sweeps and source/family/sonic-quality annotations under CC BY 4.0. Use it for instrument-family recognition, timbre descriptors and pitch/velocity-response probes; it is not a polyphonic arrangement, phrase-performance or genre-grammar corpus.
- **MAESTRO** (official dataset: https://magenta.tensorflow.org/datasets/maestro) provides aligned piano audio and MIDI useful for piano timing/expressive-performance experiments. Its published CC BY-NC-SA 4.0 terms are non-commercial; do not treat it as unrestricted commercial training data. It is not a multi-instrument coverage corpus.
- Additional sample libraries must be reviewed asset-by-asset for license, provenance, instrument naming, articulation metadata and redistribution terms.

The official VCSL instrument catalog also lists **Harmonica Special 20 (C/F) and Super 64**, plus dedicated Claves, Cowbells, Cabasa, Tambourines, Bongos, Conga/Quinto/Tumba, Cajon, Claps, Agogo, Guiro, Shaker and Frame Drum entries. These are concrete candidates for sample-backed expansion, not evidence that a local checkout has exact runtime coverage. The sample percussion adapter is tracked in #9406; the Blues harmonica instrument integration is tracked in #9409. See the [official VCSL catalog](https://versilian-studios.com/vcsl/).

Keep source audio private and restricted where licenses or contributor permissions require it. A user uploading a recording does not automatically grant global training, derivative or redistribution rights.

## 10. Implementation order

1. Code-grounded inventory and baseline receipts for all 42 instrument variants and 29 composer-backed styles.
2. Capability profile schema plus complete machine-readable inventory; record gaps honestly.
3. Style identity contracts for the 29 engine-backed styles; catalog research entries remain disabled.
4. Renderer-patch capability probes for CC11, CC1, CC64, CC2 and relevant program/keyswitch behaviors.
5. Deterministic score-conditioned orchestration planner with fail-closed constraints and replayable plan receipt.
6. Per-instrument articulation and expressive performance profiles; add one technique/control at a time with positive and negative fixtures.
7. Crossed factorial identity campaign, then pilot listening study and held-out confirmation.
8. Promote only instrument / technique / style / renderer combinations whose evidence supports the claim.

## 11. Related work and non-duplication

- EVAL-030 #9284 — SMEBench / MMAE compatibility.
- EVAL-031 #9300 / PR #9303 — benchmark source and composition lineage isolation.
- EVAL-032 #9306 / PR #9312 — judge reliability / human calibration.
- EVAL-033 #9307 — contract-valid equivalence.
- EVAL-034 #9375 / PR #9380 — realized Score contract verification.
- EVAL-035 #9337 / PR #9352 and #9381 / PR #9383 — topology-specific tail and replayable temporal support.
- EVAL-036 #9355 — operation-specific symbolic counterfactual validation.
- MUSE-STYLE-QUAL-001 #7554 — four-tier qualification of style/sample selection, decoded assets, render cells, and acoustic/perceptual evidence.
- #7464 / #7551 / #7555 — existing instrument capability inventory, per-note sample plans and SFZ zone inspection; extend these rather than duplicating their schemas.
- #7422 / PR #7418 — measured, fixed-grid causal response curves for style controls; #9438 adds the constrained register-feasibility follow-up for the Fugue path.
- #426 — the listening-first preference/quality loop and human observation capture.
- MEL-MR-PERC-SAMPLES-001 #9406 — dedicated semantic hit-bank qualification for unpitched world percussion.
- MEL-MR-INSTR-002 #9409 — sample-backed harmonica and a Blues lead option.

MEL-MR-STYLE-001 consumes their verified evidence where relevant and does not weaken their gates.

## 12. Claim boundary

This protocol does not claim that the 42 instrument variants are sample-realistic, the 29 engine-backed styles are already distinguishable, the 152-style catalog is implemented, or any controller/technique works across all SoundFonts. It is a preregistered plan to generate that evidence. CI, compilation, benchmark execution and listener validation remain separate gates.


## 9. Competitive frontier and human-preference protocol (2026-10-31)

### 9.1 Public instrumental reference

The current Artificial Analysis AA-Music-Instrumental v1.1 page was checked on 2026-10-31. It lists Suno v6 at Elo 1148 (95% interval 1126–1170; 2,208 samples), Suno v6-mini 1114 (1093–1135; 2,332), Mureka V9.5 1084 (1063–1105; 2,297), Mureka V9 1083 (1062–1104; 2,322), Lyria 3 Pro 1080 (1059–1101; 2,329), and MusicGen 868 (845–891; 2,130). This is a live external leaderboard snapshot, not a direct head-to-head result for Melothaea.

Its methodology uses blind same-prompt pairwise comparison, a minimum 10-second listen per sample, Bradley–Terry maximum-likelihood ratings, 95% confidence intervals, genre slices, and audio normalization (static gain toward -16 LUFS, capped at -1 dBTP, then 44.1 kHz / 320 kbps MP3). The default study uses one output per prompt and seed 42 where exposed; model-specific settings are held constant within each provider.

Primary sources:
- https://artificialanalysis.ai/music/leaderboard/instrumental
- https://artificialanalysis.ai/methodology/music

### 9.2 Current research frontiers

- Grötschla et al., ICASSP 2025, report 6,000 songs from 12 systems, 15,600 pairwise comparisons and more than 2,500 participants; their work finds significant disagreement between objective metrics and human preference. https://arxiv.org/abs/2506.19085
- The Qwen-Music report (2026-07-13) describes a semantic-token LLM + generative stereo renderer pipeline, Melody-CoT planning, >5 million hours of training data, and SFT → offline DPO → online GSPO alignment. Its authors report 13/16 best objective results and a professional-rater blind A/B study using 50 experts and three judgments per pair on 600 Chinese/English prompts; they report 50.3% vs 49.7% against Suno V5.5 and meaningful genre differences. This is an author-reported frontier reference, not independent replication, and must not be conflated with the instrumental-only Arena. https://arxiv.org/abs/2607.11699
- MIREX 2026 instrument-recognition results report SSR macro-F1 0.801 on the OpenMIC public test but 0.488 on a hidden MedleyDB set (the baseline is 0.497 on that hidden set). This is evidence that instrument recognition is distribution-sensitive; classifiers are a diagnostic, not an authority on timbral authenticity. https://music-ir.org/mirex/wiki/2026:Audio_Instrument_Recognition_Results

### 9.3 What this means for Melothaea

Do not compete only on a single prompt-to-audio Elo. Foundation systems have large-scale semantic-token generation and generative stereo renderers; the reported Qwen system trains on >5 million hours. That is a different scaling regime from a symbolic, renderer-grounded Rust system.

Melothaea's credible competitive wedge is **directable instrumental composition with inspectable musical structure**:
1. generate multiple *symbolic identities* before paying for full arrangements;
2. reject invalid candidates using independent musical constraints;
3. select and score-condition instruments from measured capabilities;
4. render through qualified, replayable asset/renderer paths;
5. let a musician request localized transformations while immutable score fields remain unchanged;
6. use listener preference to rank valid alternatives, never as a substitute for musical validity or human authorship.

This is a two-loop architecture: a composition/constraint loop and a human preference loop. The preference learner can rank and suggest; it cannot silently rewrite the authoritative score, invent a technique, or override a hard failure.

### 9.4 Competitive benchmark minimum

The new [MUSE-EVAL-IND-001 #9453](https://github.com/Luminous-Dynamics/symthaea/issues/9453) specification defines six separate scorecard tracks:
- blind instrumental preference, with model/version/prompt/output provenance;
- requested intent and hard-constraint fidelity;
- grammar × ensemble × renderer factorial identity;
- long-range musical structure;
- localized edit preservation and musician agency;
- latency, real-time factor, memory and reliability.

The 150-prompt confirmatory set is stratified over at least 10 instrumental/style families and three density levels. Pilot data must not leak into confirmatory evaluation. Reference-provider access and supported parameter settings are recorded per run; missing access is disclosed rather than simulated.

Proposed (not measured) success gates:
- 100% of requests accounted for; 0 silent fallback; 0 unsupported instrument/technique claims.
- Every hard symbolic invariant either passes or produces an explicit abstention.
- ≥95% exact preservation of non-target events on localized edits, and 100% preservation of fields explicitly declared immutable by the operation contract.
- 0 NaN/Inf, unintended clipping or silent output for a result claimed as a successful render.
- Against the leading accessible instrumental baseline, a confirmatory pairwise win rate whose 95% interval has a lower bound >50%; ≥55% on a separately held-out prompt set is a stretch target. If the interval overlaps parity, claim parity/uncertainty, not superiority.
- Human musicality features must deliver a preregistered, statistically supported improvement in at least one desired listening dimension without unacceptable regressions in the others.

Use Bradley–Terry or an equivalent paired-comparison model with prompt and rater structure, not a raw win percentage alone. Keep expert and non-expert cohorts separately reportable. Include blinded duplicates, hidden reference pairs, fixed prompt cards, exact audio digests, and analysis code. Metrics such as FAD, CLAP similarity, aesthetic predictors, instrument classifiers and symbolic scores must be calibrated to the human corpus; no single automated metric becomes a quality authority.

