# SYNTH-VOCAL-001 — synthetic vocal acoustic plant and known-answer benchmark

**Status:** implementation draft; 30 authored synthetic tests across the plant, observer and response-profile compiler; Rust compilation, tests and exact-head CI are pending.

**Policy:** synthetic-only. No performer recordings, voice identities, external corpus assets, participant recruitment or permission requests are part of this track.

## 1. Research objective

Muse should be evaluated as a musical-performance compiler and controlled editing system, not only as a waveform generator.

The target execution contract is:

**intent → resolved performance graph → capability-qualified renderer → independent observation → bounded correction proposal → re-render → preservation verification → replayable evidence**

The seven adjacent research tracks already provide evaluators and assurance contracts for correction trajectories, pitch tracking, task-aware edits, session history, submission export, control allocation and rights-aware singing evaluation. Those crates intentionally consume observations; they do not themselves create independently measured acoustic observations from a rendered waveform.

SYNTH-VOCAL-001 fills that bridge with a small, deterministic acoustic plant and a known-answer fixture suite. It is linked to #11402 and is designed to feed the existing VCRB/VIA/VCCR/VMEA/VELB work rather than define replacement scorers.

## 2. Paradigm shifts

### 2.1 From prompts to a compiled performance graph

A natural-language instruction should not remain the only specification of a performance. Compile it to stable, typed events:

- lyric/grapheme spans and pronunciation hypotheses;
- phoneme and syllable events;
- note, rest, tie, grace-note and ornament events;
- timing, pitch and expression trajectories;
- renderer capability claims;
- target and protected regions;
- dependency edges and edit obligations.

The current #10876 VocalPerformance v3 proposal already identifies the one-to-many and many-to-many alignment problem. Preserve that as the architectural owner. Do not create a second graph or another lyric-to-score schema in this tool.

SegTune (ACL 2026) is a useful external design reference: it aligns local musical descriptions to song segments while retaining global prompt context. Muse's stronger implementation opportunity is to express those scopes as stable graph references and explicit constraints, rather than as untyped prompt text. [SegTune paper](https://aclanthology.org/2026.acl-long.586/)

### 2.2 From "generate audio" to proof-carrying render transactions

Every edit should become an immutable transaction containing:

- exact parent output digest;
- graph version and target event IDs;
- requested operation and accepted tolerance;
- protected event IDs and allowed dependency cone;
- selected renderer capability and artifact digest;
- measurement versions and calibration receipts;
- action proposal and action bounds;
- before/after output digests;
- terminal result: compliant, noncompliant, indeterminate, unsupported or abstained.

A renderer does not get to redefine the request. An evaluator does not get to manufacture a reference. An edit is not accepted because the audio changed or because it sounds plausible. It is accepted only when the requested target changed in the right direction and the protected obligations remained within their declared limits.

### 2.3 From opaque model behavior to an independently measured synthetic plant

Synthetic generation gives us exact ground truth without another person's recording. It permits controlled factorial experiments across pitch, formants, bandwidth, energy, vibrato, durations, sample rates, noise, reverberation and seam position.

The crucial separation is:

1. The generator knows the control values it created.
2. The renderer receives the control values.
3. The observer receives only PCM and its declared configuration.
4. The evaluator receives the independent observations and their provenance.

A measurement must never certify itself by re-reading the generator's target values.

### 2.4 From one-pass editing to counterfactual control

The correction loop should compare at least these policies on exactly the same starting states:

- do nothing / abstain;
- fixed-step correction;
- greedy single-control correction;
- damped pseudoinverse;
- regularized bounded least squares;
- VCCA finite-candidate allocation;
- clairvoyant best candidate, as an upper bound only.

Each candidate should be assessed against target convergence, collateral violations, false-safe rate, iterations, render count, compute cost, and abstention quality. VCCA must remain proposal-only until the response profile has passed calibration and held-out intervention checks.

### 2.5 From one benchmark score to adversarial, multi-axis qualification

A single composite score can conceal a dangerous trade-off. Keep instruction realization, preservation, acoustic accuracy, uncertainty calibration, resource cost and evidence integrity as separate primary result families. Publish the failure strata and denominators.

## 3. Current implementation

The tool lives under tools/synthetic-vocal-plant/ and uses the existing public API:

symthaea_vocal_tract::speech::vocoder::synthesize

It constructs FormantFrame blocks from explicit synthetic parameters, renders each segment independently and joins the blocks using a fixed 5 ms overlap. The overlap is part of the declared dependency cone; it is not falsely counted as unchanged audio.

The renderer is explicitly identified as:

- backend: symthaea_vocal_tract::speech::vocoder::synthesize
- class: legacy_formant_source_filter
- claim ceiling: legacy source-filter test plant only

The source documentation distinguishes this lightweight renderer from the recovered Series 23 physical articulatory renderer. This work does not claim Series 23 qualification, natural human voices, native phonology or perceptual naturalness.

The fixture phrase contains five abstract vowel-like proxy events. The proxy IDs are not IPA annotations and are not a language corpus.

### Generated cases

| Case | Intervention | Required interpretation |
|---|---|---|
| baseline | None | Frozen reference |
| no_op | No change | PCM digest must equal baseline |
| pitch_up_100c | +100 cents on one segment | Independently estimated target F0 should rise |
| wrong_direction_pitch_down_100c | −100 cents | Negative control for a requested upward edit |
| formant_scale_110pct | Scale F1–F3 on one segment | Orthogonal-control probe; declared F0 remains unchanged |
| collateral_corruption | Edit target and a protected segment | Locality evaluator must reject |

The manifest also carries a fail-closed rejection receipt for an unsupported vibrato-phase control. Silence and unusable measurement windows remain unmeasured/indeterminate; they are not assigned a fabricated F0.

Each output includes float WAV, sample count, PCM SHA-256, segment ranges, target and protected scope, measured target F0 where estimable, and a content-addressed manifest. The manifest explicitly states that no human recordings, external corpora or voice identities were used.

### Measurement separation

- F0: a separate PCM-only normalized-autocorrelation estimator with parabolic lag interpolation. It never reads FormantFrame.f0.
- Locality: sample-bit comparison outside the declared dependency cone.
- Ground truth: source parameter vector and known edit operation, recorded separately from observed measurements.
- No-op: required to be bit-identical.
- Collateral negative control: required to produce a nonzero changed-protected-sample count.

The autocorrelation estimator is intentionally narrow. It is a first independent observation path for clean synthetic monophonic signals, not a general singing-pitch estimator and not a replacement for VPTB-001.

## 4. Benchmark protocol

### 4.1 Split design without human data

Generate independent parameter families and deterministic seeds:

- **Calibration:** repeat identical parameter vectors, repeated PCM renders, and repeated analyzer runs. Estimate renderer/analyzer repeatability and freeze tolerances.
- **Development:** tune the allowed edit and dependency-cone rules on known synthetic conditions.
- **Confirmatory:** hold out combinations of F0, formant vectors, duration, vibrato, bandwidth, energy and segment position.
- **Adversarial:** deliberately malformed controls, missing segments, wrong directions, duplicated IDs, out-of-range values, non-finite samples, clipping, silence, unexpected cross-coupling, and collateral edits.

Do not split random windows from the same generated phrase across calibration and test. Split by parameter-family/seed group so the confirmatory cases remain genuinely held out.

### 4.2 Factorial dimensions

Minimum initial design:

| Dimension | Proposed levels |
|---|---|
| F0 | 80, 110, 146.83, 220, 293.66, 440, 600 Hz |
| Formant configuration | Low/mid/high synthetic vectors with valid F1 < F2 < F3 |
| Vibrato | 0, 4.5, 5.5, 6.5 Hz; 0, 12, 25, 40 cents |
| Energy | 0.1, 0.3, 0.6, 0.9 |
| Duration | 40, 80, 160, 320, 640 ms |
| Sample rate | 16, 24, 48 kHz |
| Perturbation | F0, F1/F2/F3, bandwidth, energy, timing, seam, silence |
| SNR / effects | clean first; then controlled noise and synthetic reverb as separate strata |
| Edit locality | initial, middle, final segment; one and multiple protected neighbors |

This matrix should be generated in budgeted deterministic batches. The first commit uses a compact known-answer suite, not the full factorial matrix.

### 4.3 Primary metrics

| Family | Metric | Initial synthetic qualification target |
|---|---|---:|
| Operation | Supported edit exact-match | 100% |
| No-op | PCM identity | 100% |
| Direction | Correct sign of measured target change | 100% on clean supported controls |
| Pitch | Median absolute F0 error | <= 5 cents on clean known-answer signals |
| Pitch | F0 estimates within 10 cents | >= 99% of eligible clean voiced windows |
| Locality | Changed samples outside dependency cone | 0 |
| Negative controls | No-op / wrong-direction / collateral detection | 100% on declared fixture cases |
| Uncertainty | False-safe decisions | 0 on deterministic synthetic boundary and adversarial cases |
| Correction | Convergence on held-out supported perturbations | >= 95% within 3 interventions, as a later target |
| Evidence | Required provenance and report coverage | 100% |

These are proposed targets, not results. The initial 16 authored tests are contract tests. Rust compilation, test execution, audio fixture generation, and CI qualification are still pending.

### 4.4 Controller experiment

For each calibration profile, fit a local response matrix using one-factor-at-a-time and selected factorial interventions. Freeze the profile, then evaluate on held-out seeds and parameter regions.

Compare do-nothing, fixed-step, greedy, damped pseudoinverse, regularized bounded least squares and VCCA. Record:

- target interval distance before/after;
- collateral interval distance and hard-constraint failures;
- false-safe and false-abstain counts;
- median and p95 number of renders/interventions;
- wall-clock and CPU cost;
- calibration prediction error on held-out interventions;
- behavior when response is nonlinear, out of trust region, or unidentifiable.

Do not fit and score a response model on the same interventions. Invalidate a profile when renderer, renderer parameters, sample rate, or analyzer version changes.

## 5. External comparison and SOTA strategy

### 5.1 General audio editing: MMAE / ICASSP 2027

The official challenge evaluates Instruction Following Rate (IFR), Consistency Rate (CR), and Exact Match Rate (EMR); EMR requires every required instruction and preservation rubric to pass. Its two tracks are ranked independently. The official site states that each hidden challenge track has 500 previously unreleased examples and that registration closed on October 8, 2026. The test set and SDK were released November 10; final submission is November 25, 2026. [Official challenge protocol and schedule](https://audio-editing-challenge.github.io/)

The current public cards quoted in this research cycle show 7.60% EMR for the single-model card and 8.00% for the agent card. These are separate tracks and must not be treated as a controlled ablation or a direct model-versus-agent comparison. The local VMEA rubric scorer is not the official private judge. Do not publish a local score as an MMAE score.

**SOTA route:** first reproduce the public development protocol, then use an eligible official submission only if the team was registered and has valid organizer credentials. Otherwise use MMAE only as a design/reference source and publish clearly labelled local metrics.

### 5.2 Singing-specific editing: YingMusic-Singer-Plus / LyricEditBench

YingMusic-Singer-Plus explicitly targets controllable singing synthesis with lyric manipulation and melody guidance. Its public project describes LyricEditBench as a melody-preserving lyric-edit benchmark, with lyric and melody measures including PER and F0-CORR. Its architecture includes an IPA tokenizer and melody extraction. [Project and benchmark description](https://github.com/ASLP-lab/YingMusic-Singer-Plus)

This is the closest task-specific model comparison, but any external reproduction must use matched source audio, split, operation, language and reference conditions. The dataset's licensing and underlying recording rights are separate constraints. This permission-free track does not download or reuse that corpus.

**SOTA route:** first compare Muse against the published YingMusic-Singer-Plus/Vevo2 baseline only where the exact benchmark split, metrics and model version are reproducible. The synthetic plant is not a substitute for LyricEditBench.

### 5.3 Fine-grained song control: SegTune

SegTune (ACL 2026) makes segment-scoped musical description a first-class control mechanism and retains global style context. This supports Muse's stable phrase/phone/note graph and local edit scope. The opportunity is to add explicit causal dependencies and independent preservation checks, which are not established merely by segment prompting. [ACL Anthology](https://aclanthology.org/2026.acl-long.586/)

### 5.4 Singing quality evaluation: MusicJudge

MusicJudge (Interspeech 2026) couples block-aligned lyric correctness with pitch and rhythm evaluation and reports agreement with expert judgments. It is a useful design reference for a future evaluator ensemble, but it is not proof that an edited region was local or that untouched samples were preserved. Because this programme is synthetic-only, we cannot reproduce human expert agreement or listener preference. [Samsung Research publication](https://research.samsung.com/research-papers/Listening-Like-a-Judge-A-Music-Aware-Framework-for-Automatic-Singing-Performance-Evaluation)

### 5.5 Speech-singing transitions: UniVocal

UniVocal (ACL 2026) studies speech-to-singing mode transitions and introduces SCSBench. That task is distinct from switching between isiZulu and English while continuing to sing. Muse should keep mode-switching and language-switching as separate graph annotations and benchmark strata. [ACL Anthology](https://aclanthology.org/2026.acl-long.1452/)

### 5.6 Independent pitch estimation

Pitch Detection Benchmark v2 compares 19 monophonic trackers across 10 corpora and includes clean and perturbed audio. Its published aggregate pitch F1@50c numbers include SwiftF0 0.778, RMVPE 0.768 and FCPE 0.728. This is a mixed-corpus benchmark, not a singing-only ground truth and not a score-conditioned singing test. [Benchmark report](https://github.com/lars76/pitch-benchmark)

The synthetic estimator should be compared with pYIN/Praat and neural trackers on identical generated PCM, sample rate, voiced/unvoiced policy, thresholds and runtime hardware. A perfect result on a clean sine or the tool's own renderer is only a unit test; it is not evidence of superiority on singing.

## 6. How to beat SOTA without gaming the benchmark

1. **Match the task.** Same source files, instructions, reference alignment, partition, score definitions, model restrictions, and compute budget.
2. **Use a frozen hidden-like test split.** Keep development and confirmatory seed families distinct. Publish exact denominators and subgroup coverage.
3. **Report paired uncertainty.** Bootstrap at the source-work/session family level when source families exist; on the synthetic side, cluster by generated phrase and parameter family, not by correlated audio frames.
4. **Win on the joint objective.** Increase target success without sacrificing protected-region consistency. EMR-like joint success is the decisive metric; IFR-only improvement can be achieved by damaging unrelated content.
5. **Make abstention useful.** An honest unsupported or indeterminate result is preferable to a false-safe edit. Report coverage-risk curves and false-safe rate alongside success.
6. **Charge for computation.** A quality gain that costs many re-renders may be inferior to a fast bounded controller. Report wall-clock latency and total rendered seconds.
7. **Red-team the evaluator.** Include positive controls, no-op controls, wrong-direction edits, protected-region corruption, stale digests, calibration mismatch and deliberate analyzer disagreement.

A claim of SOTA should require a matched external benchmark and a paired confidence interval showing an improvement on its primary metric without a material regression in preservation or coverage. Synthetic-only scores cannot be called SOTA for singing synthesis or human listening.

## 7. Permission-free expansion plan

The following work remains possible without contacting any performer or rights-holder:

1. Expand the procedural plant to consonant classes, stop closures/releases, fricative noise, nasality, vowel transitions, coarticulation, melisma and rests. Keep its phonetic labels abstract until the mapping is grounded in licensed references.
2. Add independent spectral/formant measurements from PCM, with a separately calibrated analyzer and reference windows; do not reuse the synthesizer's internal controls.
3. Build a many-to-many symbolic alignment known-answer suite from generated syllable and note events, exercising #10876 without claiming natural phonology.
4. Export versioned observations and calibration receipts to the VCRB-001 schema, then consume the VCCA proposal and VCCR trajectory contracts on held-out synthetic interventions.
5. Add adversarial edit sequences and multi-turn instruction supersession; reuse MMAE-TRACE rather than create a new ledger.
6. Add backend conformance fixtures. The same event graph should be rendered by each explicitly supported backend, with unsupported controls remaining unsupported rather than silently approximated.
7. Preserve the SA-SING-001 rights programme as an optional future human-evaluation track. It is not a blocker or a prerequisite for this synthetic programme.

## 8. Explicit non-goals and claim ceiling

This work does not:
- use public singing recordings, cloned voices, voicebanks or artist likenesses;
- infer human pronunciation quality from abstract proxy tokens;
- establish isiZulu/isiXhosa/Sesotho/Afrikaans singing competence;
- assess singer identity continuity or listener preference;
- claim the legacy renderer is the Series 23 physical renderer;
- claim real-audio correction, naturalness, official MMAE scores or SOTA performance;
- change production renderer capability gates.

## 9. Validation record

The branch currently contains a standalone crate, deterministic fixture generator, 15 authored tests, README and exact-head workflow. The workflow is designed to pin Rust 1.96, check formatting, run tests and Clippy, generate WAV fixtures, and verify the synthetic-only manifest.

No local Cargo/rustc toolchain was available during authoring. Therefore no Rust compile, rustfmt, unit-test, Clippy, WAV generation or CI pass is claimed. The workflow's generated standalone lockfile is a CI-time lock, not a committed lockfile; a later hardening step should commit and digest a reviewed lockfile once a Rust toolchain is available.

## 10. Related implementation

- New tracker: #11565
- Parent observation integration: #11402
- Performance alignment graph: #10876
- Edit locality: PR #11354
- Control response: PR #11401
- Interval assurance: PR #11461
- Correction trajectory: PR #11488
- Independent pitch evaluator: PR #11491
- Task-aware edit assurance: PR #11506
- Multi-turn ledger: PR #11517
- Conservative allocator: PR #11544


## 11. SYNTH-OBS-001 — independent acoustic observation bridge

The follow-on work in #11575 adds `tools/synthetic-vocal-plant/src/observer.rs` and extends the CLI output with:

- `observations.jsonl`: per-segment F0, F1/F2/F3, relative RMS energy and voicing coverage.
- `calibration.json`: repeated deterministic render/no-op observations on a distinct calibration parameter family.
- `observer-manifest.json`: analyzer IDs/versions, configuration digest, exact artifact hashes, synthetic-only flags and split identities.

### Two-observer F0 cross-check

The pipeline retains two separate F0 measurement paths:

1. The synthetic plant's own normalized-autocorrelation PCM estimator.
2. The existing `symthaea_vocal_tract::formant_extraction::extract_formant_frames` LPC/autocorrelation path, which also returns LPC formants, relative RMS energy and voicing.

The two values are never averaged to conceal disagreement. Their difference is recorded in cents, with a 50-cent diagnostic threshold. When the signal is silent, insufficiently voiced, or too short for either path, the relevant result is `INDETERMINATE`, not zero or success.

Every metric record includes:
- stable metric ID and exact unit;
- analyzer ID and version;
- PCM SHA-256;
- exact sample window and eligible frame count;
- observed value or explicit indeterminate disposition.

Analysis windows exclude the segment boundary and seam guard. The metric IDs are suitable for the existing VCRB observation layer; a narrow adapter exports only observed values and does not implement a parallel control-response scorer.

### Calibration split

The repeatability calibration uses a distinct source-case identity and a different synthetic parameter family from the confirmatory phrase. It renders the calibration source twice and records exact PCM repeatability plus per-metric maximum absolute repeat differences.

This establishes deterministic measurement plumbing for the named legacy renderer. It does not estimate uncertainty for stochastic neural backends, human variability or population-level acoustic distributions. Neural backends will need repeated stochastic seeds and a separately estimated calibration distribution before collateral tolerances can be frozen.

### Claim ceiling

The observer's LPC values are independent observations of generated PCM, not human-annotated formant ground truth. Its F0 cross-check is an internal observer disagreement diagnostic, not a third-party benchmark. No human-naturalness, phonetic correctness, language competence, singer identity or SOTA claim follows.

### Current engineering state

The stacked branch contains the observer implementation, eight additional authored tests, a VCRB-shaped adapter record, calibration/confirmatory split metadata, and workflow assertions for observation/calibration artifact generation. Total authored tests in the two slices: 24. The Rust workflow must still compile, format, test and run the fixture generator on exact PR head before any pass is claimed.


## 12. Competitive landscape refresh — 11 November 2026

The industry has at least three distinct evaluation frontiers. They should be reported separately; a score on one is not a proxy for another.

### 12.1 End-to-end vocal music generation

Artificial Analysis AA-Music-Vocal v1.1 is a blind pairwise-preference leaderboard for prompt-to-music generation with vocals. At the 11 November 2026 inspection, the public leaderboard reported Suno v6 at Elo 1144 ±18 (95% interval 1126–1162; approximately 3,063 samples), Suno v6-mini at 1132 ±18, Mureka V9.5 at 1101 ±18, and Lyria 3.5 at 1050 ±18. The Arena normalizes loudness and output delivery format before blind comparisons.

This is a strong industry reference for full-song generation preference. It is **not** a local lyric-edit benchmark, not a control-response test, and not evidence of instruction preservation. Muse should not claim to beat these systems on AA-Music-Vocal unless it is actually evaluated through that protocol with comparable outputs. The more relevant use is to establish a separate future full-song quality frontier.

Sources:
- https://artificialanalysis.ai/music/leaderboard/vocals
- https://artificialanalysis.ai/methodology/music

### 12.2 Score-conditioned / zero-shot singing synthesis

SoulX-Singer (February 2026) is a high-priority open research baseline for zero-shot singing synthesis. Its report describes over 42,000 hours of vocal training data, score/MIDI or melodic conditioning, and support for Mandarin, English and Cantonese; SoulX-Singer-Eval emphasizes strict training/test disentanglement. This is a scale and synthesis-quality comparison, not an editing-preservation result. It uses substantially more data than our permission-free synthetic-only plant and should be treated as an external reference, not a directly matched system today.

VITA-QinYu (May 2026) is a multimodal spoken-language model with role-playing and singing generation, trained using a reported 15.8K hours of generated/curated speech, role-playing and singing data. Its paper reports a 0.13-point MOS improvement over peer SLMs on singing. It is a useful expressive-generation comparator, but not a local-edit transaction benchmark.

Sources:
- https://arxiv.org/abs/2602.07803
- https://arxiv.org/abs/2605.06765
- https://github.com/Soul-AILab/SoulX-Singer-Eval

### 12.3 Melody-preserving lyric editing

YingMusic-Singer-Plus is the closest task-aligned baseline for controllable lyric editing. LyricEditBench describes 7,200 test instances from GTSinger, six lyric modification scenarios, and balanced sampling across singer gender, language, modification type and singing technique. The model uses melody-providing singing audio plus modified lyrics and can optionally use a timbre reference. Its task includes partial/full replacement, insertion, deletion, translation and code-switching.

This is much closer to Muse's lyric-editing claim than a prompt-to-song Elo ranking. Any future comparison must match the exact split, edit operation, source audio, language and reference conditions, and honour the dataset/recording licence. The current synthetic plant does not use this corpus.

Sources:
- https://github.com/ASLP-lab/YingMusic-Singer-Plus
- https://huggingface.co/datasets/ASLP-lab/LyricEditBench

### 12.4 Independent pitch tracking

Pitch Detection Benchmark v2 reports 19 monophonic trackers over 10 corpora and nine signal versions per clip (clean plus eight background/reverb/microphone perturbations). Its README reports pitch F1@50c of 0.778 for SwiftF0 and 0.768 for RMVPE, with materially different real-time speed figures. The benchmark's corpus mix means these numbers must not be described as singing-only results.

The VPTB-001 plan should include SwiftF0 and RMVPE as high-priority external trackers, plus FCPE, pYIN/Praat and a clearly documented classical baseline, all on matched PCM/sample-rate/voicing-policy conditions. For the synthetic-only path, use generated known-answer F0 and adversarial octave/subharmonic controls; keep those scores separate from the benchmark's mixed-corpus aggregate.

Source:
- https://github.com/lars76/pitch-benchmark

### 12.5 Structured control and multimodal evaluation

- SegTune (ACL 2026) provides segment-level musical descriptions plus global prompt context. Muse's opportunity is event-level scope and dependency-aware preservation, not another segment prompt field: https://aclanthology.org/2026.acl-long.586/
- MusicJudge (Interspeech 2026) couples lyric-block correctness with pitch/rhythm assessment and reports agreement with expert judgments. It is an independent evaluation reference, not a waveform-locality witness: https://research.samsung.com/research-papers/Listening-Like-a-Judge-A-Music-Aware-Framework-for-Automatic-Singing-Performance-Evaluation
- UniVocal (ACL 2026) evaluates speech↔singing code-switching. That is distinct from switching natural languages while remaining in singing mode: https://aclanthology.org/2026.acl-long.1452/
- MMAE/ICASSP 2027 measures general-purpose audio editing across modalities and multi-turn complexity. Its hidden test tracks and private rubrics are separate from local VMEA/VCCR scores: https://audio-editing-challenge.github.io/

### 12.6 Strategic interpretation

We should maintain four independent result boards:

1. **Edit correctness and preservation:** MMAE, LyricEditBench-style operations, VMEA, VELB, VCCR and multi-turn integrity.
2. **Musical performance control:** score adherence, phone/note timing, vibrato/dynamics response, cross-coupling and controller efficiency.
3. **Singing synthesis quality:** naturalness, pronunciation, pitch/rhythm fidelity, identity/voice consistency and blind preference.
4. **Evidence integrity:** provenance coverage, repeatability, calibration validity, false-safe rate, reproducibility and correct abstention.

The synthetic-only work can substantially advance boards 1, 2 and 4 without permissions. It can provide signal-processing baselines for board 3, but it cannot qualify human naturalness, singer identity or listener preference. The immediate win condition is a reliable, reproducible editing/control instrument that can later host better renderers, not an unsupported claim to have surpassed 42K-hour or 15.8K-hour trained generation systems.


## 13. SYNTH-CAL-001 — measured response-profile compiler for VCCA

The third slice (#11578) now fits a local control-response profile from actual synthetic PCM interventions. It is the missing producer between SYNTH-OBS-001 observations and the existing VCCA-001 allocator inputs.

### One-factor calibration design

Three input controls are perturbed independently and in both directions:

| Control | Unit | Measured calibration amplitudes |
|---|---|---|
| Pitch shift | cents | −50, −25, +25, +50 |
| Formant scale | unitless scale delta | −0.10, −0.05, +0.05, +0.10 |
| Energy scale | unitless scale delta | −0.20, −0.10, +0.10, +0.20 |

The same control amplitudes run over two distinct synthetic source families. For every baseline/intervention pair, the runner measures F0, LPC F0, F1/F2/F3, relative energy and voicing coverage using the PCM-only observer. The compiler records each control's off-diagonal effects as well as its intended target effect.

### Local fit and uncertainty

For metric m and control i, the initial local slope is fitted through the origin:

~~~text
J[m,i] = Σ(u[i] × Δy[m]) / Σ(u[i]²)
~~~

Coefficient uncertainty is bounded by the largest calibration residual ratio:

~~~text
epsilon_J[m,i] = max |Δy[m] − J[m,i] × u[i]| / |u[i]|
~~~

A separate no-op repeatability family supplies each metric's residual envelope. Both are derived from rendered/observed PCM, not from parameter values. They are engineering bounds for this deterministic synthetic backend, not confidence intervals or a guarantee of global linearity.

The output includes a VCCA-shaped adapter with complete control bounds, metric residuals and sensitivity cells, but no duplicated allocator implementation. VCCA remains proposal-only and downstream capability admission and fresh post-render measurements remain mandatory.

### Held-out promotion gate

A third synthetic phrase family supplies a confirmatory baseline and one unseen intervention per control. The compiler compares each measured metric delta against the prediction interval derived from calibration coefficients and residual bounds.

The profile is marked promotable only if:
- all metric × control cells are present and finite;
- each supported control has positive and negative excitation;
- the calibration and confirmatory parameter digests/identities are disjoint;
- no-op PCM repeatability and metric repeatability are complete;
- every held-out metric response falls inside its predicted interval.

A profile can be generated while remaining `promotable=false`. That is a fail-closed finding of model inadequacy, not a CI failure and not a reason to widen the interval after seeing the result. No claim of VCCA superiority or measured real-performer convergence is made.

### Current source and evidence state

The stacked branch now includes the original plant, two independent PCM observation paths, a no-op calibration report, metric-level VCRB-shaped observations, a response-profile compiler and a VCCA-shaped profile adapter. Total authored test count is 30 across the three slices. Rust execution remains pending until exact-head CI runs; no compilation, test, Clippy or generated-WAV pass is claimed from the local authoring environment.


A further fail-closed support gate checks that each declared control produces a positive response in its designated target metric (pitch→measured F0, formant scale→measured F1, energy→measured relative RMS). If the renderer accepts but ignores a control, its VCCA adapter entry is marked unsupported with zero-only bounds and the whole response profile cannot be promoted. Matrix completeness alone is not treated as proof that a control works.
