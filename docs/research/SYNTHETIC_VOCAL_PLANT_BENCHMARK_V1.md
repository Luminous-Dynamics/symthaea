# SYNTH-VOCAL-001 — synthetic vocal acoustic plant and known-answer benchmark

**Status:** implementation draft; Rust compilation, tests and exact-head CI are pending.

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
- F1–F3: a separate PCM-only LPC formant-analysis path. Each scenario records observer identity, full extraction configuration, analysis/voiced/ordered-window counts, median estimates and median absolute deviations. It does not read target formant controls. This is a separate algorithm in the same crate, not an external verifier; its fallback behavior and bias must be calibrated before qualification thresholds are frozen.
- Locality: sample-bit comparison outside the declared dependency cone.
- Ground truth: the complete source parameter vector is embedded in the manifest and accompanied by a SHA-256 digest, so reviewers can recompute the digest and verify the exact fixture inputs.
- No-op: required to be bit-identical.
- Collateral negative control: required to produce a nonzero changed-protected-sample count.

The autocorrelation estimator is intentionally narrow. It is a first independent observation path for clean synthetic monophonic signals, not a general singing-pitch estimator and not a replacement for VPTB-001. The LPC formant observer is also a first measurement path: it consumes PCM, but shares a codebase with the renderer and uses coarse LPC-envelope peak picking. Treat it as a sensor under calibration, not an oracle.

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
| Formants | F1–F3 median absolute percentage error | <= 10% on clean, calibrated synthetic resonances (proposed target; not yet measured) |
| Observer coverage | Qualified, non-fallback F1–F3 frame coverage | >= 90% of voiced frames on supported clean strata; report fallback-sentinel and abstention rates separately (proposed target) |
| Locality | Changed samples outside dependency cone | 0 |
| Negative controls | No-op / wrong-direction / collateral detection | 100% on declared fixture cases |
| Uncertainty | False-safe decisions | 0 on deterministic synthetic boundary and adversarial cases |
| Correction | Convergence on held-out supported perturbations | >= 95% within 3 interventions, as a later target |
| Evidence | Required provenance and report coverage | 100% |

These are proposed targets, not results. The initial 19 authored tests are contract tests. Rust compilation, test execution, audio fixture generation, and CI qualification are still pending.

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

### 5.7 Commercial singing-editor baseline: Synthesizer V Studio 2

Synthesizer V Studio 2 is a strong commercial baseline for fine-grained vocal authoring, not a straw man. Its 2.3.0 update (2 November 2026) introduced a language-independent X-SAMPA option with 150 phonetic symbols, an updated phoneme system, a vocal-fry phoneme, phoneme candidate assistance, cross-lingual synthesis support for French, German, Brazilian Portuguese and Italian with updated voices, and faster/more reproducible rendering. Dreamtonics reports 30–50% faster rendering on modern CPUs with four or more cores. [Official 2.3.0 release notes](https://dreamtonics.com/synthesizer-v-studio-2-pro-2-3-0-update-and-improvements/)

This changes the competitive question. Muse should not try to win merely by exposing note-level pitch curves, phoneme overrides, AI retakes or a better editing surface. Commercial tools already offer extensive authoring controls. Muse's differentiator should be the **verified execution layer** underneath any editor: a backend-neutral performance graph, explicit capability declarations, typed dependency cones, independently measured output, preservation receipts and replayable multi-turn constraints.

X-SAMPA availability is not evidence of a language-specific G2P, lexical-tone, click-gesture or singing-phonology capability. Each language profile and each articulatory event family needs its own provenance and qualification stratum. The current synthetic plant and ZA-CLICK-001 are foundations for those contracts, not language competence.

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

1. Expand the procedural plant to consonant classes, explicit stop closure/release, fricative noise, nasality, vowel transitions, coarticulation, melisma and rests. Do not force clicks into the legacy single-source renderer; use [ZA-CLICK-001 / PR #11588](https://github.com/Luminous-Dynamics/symthaea/pull/11588) for symbolic multi-source click gestures and keep acoustic click rendering a separate unqualified capability.
2. Calibrate the new PCM-only LPC F1–F3 observer against independent known-resonance sweeps, including the estimator's documented default-value fallbacks. The current code reports fallback sentinels and widens its voiced-pitch gate to the plant's 700 Hz ceiling; its bias and coverage are not yet measured.
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

The branch currently contains a standalone crate, deterministic fixture generator, 19 authored tests, README and exact-head workflow. The workflow is designed to pin Rust 1.96, check formatting, run tests and Clippy, generate WAV fixtures, and verify the synthetic-only manifest.

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
- Synthetic click gestures (separate symbolic track): issue #11586 / PR #11588
- Renderer-neutral typed local-edit receipts: issue #11356
- Independent score-to-audio conformance: issue #11389
