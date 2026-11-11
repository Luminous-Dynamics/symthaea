# Creative Intent Contracts v1

Status: additive engineering prototype; no claim that the contract is connected to production selection or feedback.

## Why this tranche exists

The creative-quality audit in CREATIVE_METRIC_VALIDITY_AUDIT_2026-07-31.md records a concrete failure: the current melodic-coherence proxy prefers a one-pitch drone over a more varied generated composition, while the current form proxy can reward a concentration of notes followed by silence. These are not repaired by adding more arbitrary weights.

The underlying problem is category error. A repeated-pitch drone may be exactly right for an ambient brief and a failure for a melody brief. A single scalar with no declared intent cannot tell those cases apart.

## Paradigm shift: intent contract first, measured features second, taste evaluation separately

The first vertical slice adds CreativeIntentContractV1 in symthaea-muse. A caller explicitly declares mechanical constraints; the evaluator reports observations and contract violations. It does not infer that the candidate is beautiful, emotionally successful, original, culturally grounded, or fun.

The contract initially measures:
- valid note count;
- distinct rounded MIDI pitches;
- pitch range in semitones;
- onset coverage across equal-duration windows;
- share of notes occupying the dominant pitch.

Malformed contracts, durations, and notes fail closed. Invalid notes are reported by index and cannot be silently discarded to make a candidate pass. Zero-velocity note events are rejected as inaudible in this synthesis representation, so they cannot satisfy note-count, pitch-diversity, or temporal-support constraints. The module deliberately leaves the legacy CreativeQualityScore and live reward loop unchanged until the new seam has been calibrated and integrated in a separately reviewed change.

## How this generalizes to games and cross-modal creative work

A future shared project contract should declare, by modality:
- intention and audience;
- immutable requirements and editable degrees of freedom;
- machine-verifiable constraints;
- observed feature vector, with no universal aesthetic scalar;
- provenance, seed, build and generator identity;
- human review questions and failure cases.

For a game, hard constraints might cover launchability, reachability of an ending, controller/input coverage, save/replay integrity, and content restrictions. Separate feature descriptors could capture navigation density, risk/reward profile, mechanics, pacing, visual palette, and sound dynamics. A candidate can be mechanically admissible but artistically weak; those outcomes must not be collapsed into one label.

For music, candidate search should retain a set of outputs along different useful axes instead of repeatedly selecting the current scalar maximum. An ambient drone, a lyrical melody, a percussion-led groove, and an experimental piece should be judged against their respective briefs. The system should preserve surprising but admissible candidates rather than allowing the current taste model to erase all alternatives.

## Benchmark programme

All numbers below are proposed gates, not achieved results.

### Gate A — contract correctness
- 100% of seeded invalid-contract and invalid-note cases are rejected.
- 100% of seeded task-constraint violations are detected by the independent test oracle.
- Identical input and contract produce identical reports.
- No malformed candidate is accepted because its invalid fields were dropped before scoring.

### Gate B — generator usefulness
Compare, on matched briefs and held-out seeds:
1. current generator without candidate filtering;
2. random candidates filtered only by the contract;
3. generator plus contract;
4. a quality-diversity search that maintains alternatives across declared feature dimensions.

Report acceptance rate, runtime, memory, contract-specific feature coverage, pairwise novelty, and every rejection reason. Do not compare best-of-N selection with one-shot generation without reporting candidate budget and both results.

### Gate C — human evaluation
- Blind, randomized pairwise ratings on held-out briefs and evaluators.
- Target: more than 60% preference over the strongest matched baseline, with the 95% confidence interval's lower bound above 50%, replicated on a separate set of briefs before any superiority claim.
- Independently rate intent adherence, coherence, novelty, diversity, playability/listenability, and overall preference. Preserve disagreements; do not merge all dimensions into one score.
- Report within-system and across-system diversity. A system can make each individual user's output more pleasing while narrowing collective diversity.

#### Proposed staged evaluation protocol

Use a discovery pilot and a separate confirmatory study; do not tune descriptors on the confirmatory set.

- **Pilot only:** approximately 20 briefs spanning the five intent families, with multiple listeners per brief. Use this stage to find broken instructions, audio-level confounds, unusable rating scales, and prompt/rater variance. Treat all effect estimates as exploratory.
- **Confirmatory planning target:** begin planning around 100 held-out briefs (about 20 per family) and five independent judgments per pair comparison. This is a starting design, not a guaranteed-power claim: repeated ratings share prompt and listener effects, so finalize sample size with a preregistered mixed-effects power simulation informed by the pilot before collecting confirmatory labels.
- **Matched candidate budgets:** generate exactly eight candidates per system per brief where feasible, record all candidate IDs, and freeze the selection procedure in advance. If systems use different budgets, show generation cost and report a separate budget-matched comparison rather than hiding best-of-N selection.
- **Blinding and playback:** hide system identity; randomize A/B position and brief order; use the same player/rendering chain; disclose loudness matching; check pre-normalization RMS/peak and near-silence so gain normalization cannot turn a silent render into a misleadingly comparable sample.
- **Separate labels:** obtain overall pair preference plus separate ratings for intent adherence, coherence, structure, novelty, and listenability/musicality. Record ties or “no preference” separately, and report disagreement and rater-level effects rather than collapsing them into one scalar.
- **Frozen analysis:** report pairwise preference with a 95% interval, prompt-family breakdowns, inter-rater reliability, and the best matched baseline. Calibrate automatic evaluators on training/pilot labels only, then report held-out accuracy, balanced accuracy, calibration and slice metrics. Do not promote an evaluator metric into a live reward loop solely because it correlates on the same prompts used to tune it.

This is a prospective protocol, not a claim that the pilot or confirmatory study has run. The scale is deliberately staged: MIREX's symbolic continuation task provides a useful fixed-eight-samples/blind-rating precedent, while the 2025 human-preference study illustrates the scale needed to study metric-to-human correlations. Sources: https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation and https://arxiv.org/abs/2506.19085.

### Gate D — cross-modal integration
For a playable prototype, require:
- all critical interaction paths pass deterministic smoke tests;
- sound, visuals, mechanics, and narration remain bound to one versioned project intent;
- at least 90% of blinded reviewers can identify the declared high-level intent from the assembled experience;
- revision improves a preregistered human-rated dimension without worsening hard validity or collapsing diversity;
- every claim is tied to the exact source/build and evaluation corpus.

The 90% intent-recognition target is an engineering aspiration, not a literature standard. Calibrate the task and reviewer protocol before confirmatory trials.

## Additive temporal-support measurements

The separate module `src/creative_temporal_support.rs` addresses a known blind spot without changing `CreativeIntentContractV1` acceptance semantics or the existing QD archive:

- **Onset-window coverage**: how many windows contain attacks.
- **Sustain-window coverage**: how many windows overlap at least one sounding note.
- **Occupied-time ratio**: the union of sounding note intervals divided by clip duration, so overlapping notes are not double-counted.
- **Maximum silent gap**: the longest silent interval, including leading and trailing silence.

This separates a single sustained drone from a sequence of staccato attacks that happens to touch every window. Notes are fully validated; note tails crossing the declared clip end are clipped only for these measurements. Empty scores are representable and report zero occupancy with the full clip duration as silence. Excessive window counts fail before allocation.

The measurements are descriptive and are **not** currently used to accept/reject V1 candidates, rank outputs, or modify production selection. They should become QD dimensions only after calibration against human judgments. This feature is descriptive only, not yet used for archive bins or candidate ranking. It measures symbolic note timing/velocity validity rather than inspecting rendered audio energy; a later audio-level benchmark must reject empty or near-silent renderings and bind those measurements to the exact renderer, instrument/sample-bank identity, and output digest. The tests cover long sustain, short attacks, interval union, clip-edge clipping, malformed notes, empty scores, window limits, note-order invariance, and duplicate-note invariance.

## Waveform-level integrity audit

Run the standalone renderer/negative-control probe with:

    cargo run -p symthaea-muse --example creative_audio_audit -- <source-revision-claim>

The JSON receipt covers an actually rendered phrase, an empty-score render, a zero-velocity-note render, and a deliberately corrupted synthetic buffer. It includes explicit expected observations, per-case pass/fail, and a SHA-256 over the exact typed `AudioData` sample representation (this is not a hash of exported WAV bytes). The optional source revision is caller-supplied metadata, not an authenticated checkout identity.

`src/creative_audio_audit.rs` audits the actual `AudioData` buffer after rendering, separately from note-level contracts. It reports normalized sample peak and RMS, per-channel DC offsets, non-finite and out-of-range sample counts, near-ceiling and near-zero sample ratios, active-window coverage, longest contiguous near-silent run, and an explicitly thresholded effectively-silent flag. The caller's thresholds are included in the config and are not presented as universal musical-audibility cutoffs.

The first gate is intentionally narrow:

- Empty buffers and invalid measurement configuration return explicit errors.
- Non-finite samples are counted, not hidden; aggregate arithmetic substitutes zero only so the other observations remain inspectable.
- Samples outside the normalized `[-1, 1]` range are reported, not silently clipped by the audit.
- A zero-RMS render is distinct from a sparse but nonzero render. Near-silence and active-window coverage remain thresholded diagnostic measurements.
- Mono i16, mono f32, and stereo f32 buffers are handled without re-rendering or normalizing them.
- A test renders a real note using Muse's existing renderer and audits the result.

This does **not** implement true-peak measurement, broadcast-loudness compliance, timbral quality, or perceived audibility. Muse already has `auto_master::measure_lufs`; that separate algorithm should be validated against the EBU Tech 3341 minimum-requirement signals before claiming EBU meter conformance. EBU's `-23 LUFS` recommendation is a broadcast normalization target, not a universal creative-music quality target. Sample peak is also not true peak, so the new audit deliberately does not claim to detect every inter-sample peak.

Research and benchmark framing:
- MIREX 2026 Music Audio Generation evaluates melody, harmony, key consistency, style consistency, and arrangement/production separately; its auxiliary acoustic descriptors are described as diagnostic rather than ranking scores: https://www.music-ir.org/mirex/wiki/2026%3AMusic_Audio_Generation
- EBU Tech 3341 specifies loudness-meter behaviors and a defined test-signal suite for momentary, short-term, integrated loudness, and true-peak checks: https://tech.ebu.ch/publications/tech3341
- MIREX 2026 Symbolic Music Generation uses blind subjective ratings and a fixed eight-candidate choice budget: https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation

Proposed next gates (not yet achieved): zero non-finite samples, zero out-of-range samples, non-empty duration, and no effectively-silent waveform for briefs that require sound. Report peak/RMS/DC/silence diagnostics for every output, but do not impose a universal RMS or loudness target across ambient, sparse, percussion, and full-arrangement intents. Add audio-rendered positive/negative fixture pairs and repeat them across output formats, sample rates, instrument fallbacks, and optional sample-bank paths. Tie each receipt to the exact source revision, renderer configuration, sample rate, sample-bank identity, and output-byte digest before making reproducibility claims.

## Research basis

- Khalifa et al., The Procedural Content Generation Benchmark: An Open-source Testbed for Generative Challenges in Games (FDG 2025). It explicitly separates quality, diversity and controllability over 12 game-content problems: https://arxiv.org/abs/2503.21474
- MIREX 2026 Symbolic Music Generation uses double-blind subjective ratings for prompt coherence, creativity, structuredness and overall musicality, and separates an eight-candidate generation/selection budget: https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation
- Ding et al., Quality Diversity through Human Feedback (ICML 2024). It explores inferring diversity dimensions from human similarity judgments instead of relying only on hand-authored descriptors: https://proceedings.mlr.press/v235/ding24h.html
- Bradley et al., Quality-Diversity through AI Feedback (ICLR 2024). It studies combining AI-based critique with diversity-driven search for qualitative generation tasks: https://proceedings.iclr.cc/paper_files/paper/2024/hash/5b9bef4eae0f574cedbf9f4bf29d8ae7-Abstract-Conference.html
- Doshi & Hauser, Generative AI enhances individual creativity but reduces the collective diversity of novel content (Science Advances, 2024): https://doi.org/10.1126/sciadv.adn5290

## Non-claims and follow-up

This prototype is not a creativity test, aesthetic critic, consciousness measurement, music-theory validator, player-experience test, or proof that an output is good. It is a fail-closed mechanical contract boundary. Before production integration, separately audit and repair the known legacy score defects, add human-calibrated criteria, and run the benchmark matrix above against declared baselines.


## Executable contract / descriptor-diversity slice

The first frozen corpus and runner are available as a deterministic benchmark fixture:
- Corpus: crates/domains/symthaea-muse/fixtures/creative_intent_contract_v1.json
- Rust runner: crates/domains/symthaea-muse/examples/creative_qd_benchmark.rs
- Reusable receipt/archive implementation: src/creative_qd_bench.rs

Run it with:

    cargo run -p symthaea-muse --example creative_qd_benchmark -- <source-commit-sha>

The corpus has five task families and 17 fixed candidates: ambient drone, lyrical melody, rhythmic groove, formal development, and experimental texture. It contains adversarial controls for one-pitch melodies, clustered events, invalid note values, and onsets at the duration boundary, and a zero-velocity silent-event trap. Each candidate has a preregistered expected acceptance value. The runner emits a SHA-256-bound JSON receipt, candidate-level audits, per-family acceptance counts, and a simple descriptor archive.

The archive uses pitch-diversity, temporal-coverage, onset-density, and inter-onset-interval-CV bins. The benchmark runner receipt is now schema version 2 (the frozen input corpus remains schema version 1). Every candidate receipt includes a temporal-support outcome: measured onset coverage, sustain coverage, occupied-time ratio, and maximum silent gap, or an explicit error when the notes are malformed. This evidence is recorded side-by-side with contract acceptance. It does not yet affect the contract result or the archive's four descriptor bins. For the frozen steady ambient drone, the receipt can show 100% onset-window coverage and 100% sustain-window coverage while occupied time is only 40%; this is precisely why attack coverage is not interchangeable with sustained support or time occupancy. The dense-drone fixture also has 100% onset/sustain coverage, but 80% occupied time and a 0.05-second maximum silent gap, versus the steady drone's 40% occupancy and 0.30-second gap. These are descriptive distinctions, not quality judgments. It keeps the lexicographically smallest candidate ID in a colliding cell so archive membership is independent of candidate enumeration order. It does not score aesthetic quality and does not wire into production selection. The onset-interval CV is only a statistical feature; it cannot tell whether an irregular pattern is intentional syncopation because meter, beat phase, and instrument role are not in this contract.

Proposed first gates: all preregistered fixture outcomes match; unknown corpus fields fail closed; receipts bind the exact fixture bytes; and reversing candidate enumeration leaves occupied cells and representatives unchanged. The 17 fixtures are a synthetic correctness corpus, not a human preference study, generator benchmark, or measured creative-quality result.
