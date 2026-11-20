# MUSE Paired Counterfactual Edit Evidence Benchmark v1

**Status:** protocol and wire contracts drafted; dataset, scorer and listening study not yet executed. All sample counts and thresholds below are prospective targets, not measured results.

**Tracking:** [MUSE-EDIT-EVAL-001 / #12949](https://github.com/Luminous-Dynamics/symthaea/issues/12949)  
**Related:** render-attempt integrity [#12927](https://github.com/Luminous-Dynamics/symthaea/pull/12927), score-native edit transaction [#11332](https://github.com/Luminous-Dynamics/symthaea/pull/11332), Part-aware MIDI realization [#11541](https://github.com/Luminous-Dynamics/symthaea/pull/11541).

## 1. Why a new benchmark is needed

MUSE must establish more than plausible output audio. A successful edit has to realize the requested intervention, preserve unrelated musical content, remain identifiable against a frozen source, and make the intended change perceptible. These are distinct facts and require distinct evidence.

Current public results motivate this split:

- **YuE2 has separate generation and editing evidence.** Its current upstream WildSongBench generation comparison covers 192 prompts and 17 system settings; the reported best-of-8 lead is a generation-quality result with an eight-candidate selection budget, not an edit-fidelity result. The same benchmark report describes a separate paired editing study on 10 original works, two seeds and 380 full-song recordings: local changed-note melody attainment rises from 0.0083 to 0.9375, while changed-duration harmony attainment rises from 0 to 0.8313. These are different operation-specific units and must not be combined. The authors report outside-edit content close to unedited regeneration, not bit-identical waveform preservation. Reproduce this 10-work edit study separately from WildSongBench generation quality; neither is a matched MUSE result. [YuE2 benchmark report](https://github.com/dsmpd/YuE2/blob/main/docs/benchmarks.md).
- **SMEBench** evaluates paired source/target edits across 12 operations, with 3,600 synthetic paired examples and a separate 416-example real-audio complement. Its human validation found several synthetic reference operations were not reliably perceived as following the instruction. A symbolic target is therefore not automatically a valid perceptual target. [SMEBench paper](https://arxiv.org/abs/2610.36715).
- **ICASSP 2027 Audio Editing Challenge** measures instruction-following, preservation/consistency and exact sample completion separately. Its official page reports existing MMAE systems below 5% Exact Match Rate (EMR), despite plausible-sounding outputs. This is a broader, harder multimodal task than MUSE's first benchmark slice, so scores are not directly comparable. [Challenge protocol and timeline](https://audio-editing-challenge.github.io/).
- **MIREX 2026 CMI-RewardBench** reports 73.9% best pairwise preference accuracy on its 249-pair core, but 60.9% on the vocal extension and 72.6% on the instrumental extension. A general music judge can have material domain gaps. [MIREX results](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results).

The key methodological shift is to treat each case as an executable *counterfactual contract*: frozen source + explicit instruction + exact edit scope + expected target relation + protected invariants + independently checked render + perceptual confirmation.

## 2. Evaluation dimensions

Never collapse these into one weighted average. A strong result on one dimension must not compensate for failure on another.

| Dimension | Question | Primary evidence |
|---|---|---|
| Target adherence | Did the system perform the requested change? | Operation-specific symbolic/audio metrics |
| Preservation | Did unrelated events, parts and regions remain stable? | Protected event and acoustic-region metrics |
| Correspondence coverage | How much of the source/target content was actually matched? | Explicit event map, unresolved count |
| Instruction compatibility | Did it follow the instruction rather than a related caption? | Frozen rubric evaluator + human review |
| Perceptibility and musical validity | Can listeners hear and recognize the intended change, and is it musically valid? | Blinded listener panel |
| Evidence integrity | Are the source, candidate, render and observer records joined to this exact case? | Content-addressed execution/evaluation receipts |
| Efficiency | What quality was achieved per candidate, second and compute unit? | Matched-budget curves |

### 2.1 Target adherence

Each operation family receives its own metrics and denominator.

- Pitch/melody: target-pitch accuracy, interval accuracy, contour agreement, register correctness.
- Rhythm/onset/duration: onset accuracy under a preregistered tolerance, duration accuracy, local interval accuracy, event order and beat-phase agreement.
- Harmony: root/chord-tone agreement, progression order and duration-weighted agreement.
- Tempo/key: tempo error and target-beat accuracy; tonic/mode agreement with explicit partial-credit policy.
- Velocity/gain/pan: numeric target error plus clipping, loudness and stereo placement constraints.
- Instrumentation/timbre: source/target instrument identity or allowed-set accuracy; do not treat a broad caption similarity score as proof of instrument replacement.
- Add/remove/part-level edits: event-level precision/recall, exact scope and no unintended duplicate/missing parts.

For all inferred event metrics, publish both accuracy and correspondence coverage. Unresolved correspondences count as unresolved failures for strict scoring; they are never silently discarded from the denominator.

### 2.2 Preservation

Use source event identity and explicit protected scopes. At minimum report:

1. protected-event agreement;
2. outside-scope event agreement;
3. collateral-change rate (unauthorized changes / protected or eligible unchanged events);
4. protected-audio similarity on aligned time/frequency regions;
5. source identity retention when the task requires the original performance or recording to remain recognizable.

Do not require bit-identical audio when an operation regenerates an entire mix. Use multi-resolution spectral distance and source-separated metrics where appropriate, and label them as proxies. A spectral similarity score is not a substitute for musical or human judgement.

### 2.3 Correspondence

Mappings must be explicit or independently verified. In the v1 symbolic case schema, source and target note ordinals are scoped to their exact Score digests; they are not persistent IDs. A nearest-pitch/nearest-onset heuristic may propose matches, but cannot silently declare event identity.

Each mapping records source note ordinals, target note ordinals, relation (preserve, modify, split, merge, delete, insert) and evidence provenance. Report coverage as:

\[
C = \frac{N_{\mathrm{mapped}}}{N_{\mathrm{expected}}}
\]

with unresolved mappings visible. A result with high accuracy over a small matched subset but poor coverage must not rank highly.

### 2.4 Strict sample exact match

The primary benchmark completion measure is conjunctive:

\[
\mathrm{StrictEMR} =
\frac{\#\{i: T_i \land P_i \land C_i \land I_i\}}
{N}
\]

where target adherence (T), preservation (P), required correspondence coverage (C), and instruction compatibility (I) all pass the frozen per-case rubric. No compensation between dimensions is allowed.

Human perceptibility is reported as a separate gate and secondary exact-match tier. This prevents an automatic scorer from hiding an inaudible or musically invalid change.

## 3. Case contract and evidence artifacts

The first wire contracts are:

- [muse-edit-evidence-case-v1.schema.json](schemas/muse-edit-evidence-case-v1.schema.json): rights, source artifacts, instruction, edit scopes, target relation, event correspondence, render profile and evaluation plan.
- [muse-edit-evidence-result-v1.schema.json](schemas/muse-edit-evidence-result-v1.schema.json): disaggregated per-case metrics, coverage, human results, integrity and strict outcome.
- [Synthetic case fixture](fixtures/muse-edit-evidence-case-v1.synthetic.json) and [synthetic result fixture](fixtures/muse-edit-evidence-result-v1.synthetic.json): protocol-shape examples only; hashes are sentinels and do not refer to real audio.

Every case must pin the exact source audio/score and record the rights basis. Every run must bind candidate audio/score hashes to a render receipt. Every audio observer and instruction evaluator must identify its version and output receipt. Human ratings must retain anonymous rater IDs, question version, randomized order and raw responses under the study's consent/data policy.

The benchmark does not assume that a target is unique. Some tasks allow multiple valid musical realizations. Such cases must declare a valid-target set or score a semantic rubric; they must not pretend a single waveform is the only correct answer.

## 4. Dataset design

### Prospective v1 target

- **At least 400 rights-cleared source works** spanning vocal, instrumental and mixed material, multiple meters/tempi, dynamics, genres, polyphony levels and instrument families.
- **At least 4,800 eligible paired cases** across 12 edit families. Eligibility is established before system outputs are seen; unavailable operations are not padded with synthetic trivialities.
- Data-mode mix target: 50% deterministic symbolic/renderer-grounded cases, 25% real multitrack or MIDI-plus-stems cases, 25% real audio cases with manually reviewed target and preservation references.
- Every case receives operation strength and salience strata. Include low-salience and difficult cases, not only edits chosen to be obvious.
- Use at least two renderer/patch configurations for applicable symbolic cases; reserve a renderer/patch family entirely for held-out domain-shift evaluation.

Suggested operation families: velocity/dynamics, gain, pan, melody/pitch, rhythm/onset, duration/articulation, tempo, key, harmony, instrumentation/timbre, add/remove part, and composite multi-part edits.

### Splits and leakage prevention

Split by source work and composer, never by crops or render variants. All stems, arrangements, excerpts, alternate renderings and edit strengths derived from a source work must remain in one split. Add separate stress tests for held-out composer, genre/style, instrument, renderer, sound bank and recording domain.

Initial split target: 60% train, 20% development, 20% private test by source work, subject to eligibility. Publish hashes and counts for public assets; keep private-test targets/rubrics inaccessible to system developers until evaluation is frozen.

### Human study

Run a stratified blinded panel on at least 1,200 cases, with at least three independent raters per case initially. Use more raters for low-agreement or ambiguous targets under a preregistered adjudication rule. Randomize candidate order and anonymize system identity. Questions should separately ask:

- Was the intended edit perceptible?
- Did the output follow the instruction?
- Was the musical result valid/coherent?
- Did unrelated content change?
- Which version is preferred, when preference is meaningful?

Report raw agreement, Krippendorff's alpha or an appropriate ordinal agreement statistic, confidence intervals, subgroup counts and annotator disagreement. Do not tune an automatic judge on the final listener panel.

## 5. Negative controls and adversarial cases

Every benchmark release must include controls that catch superficially attractive but incorrect systems:

- no-op identity (should preserve the exact symbolic score and, where applicable, exact audio);
- wrong operation and wrong direction;
- under-edit and over-edit;
- edit outside the authorized scope;
- edit that satisfies the numeric target but is inaudible;
- perceptible but musically invalid edit;
- high-quality audio with the wrong requested edit;
- correct target with collateral damage in a protected part;
- alternate valid realization that differs from the single reference waveform;
- event-map omission, duplicate, split/merge ambiguity and low correspondence coverage;
- stale source hash, wrong candidate hash, renderer/asset substitution and observer replay.

The no-op and wrong-direction controls must be scored in the same evaluation run as positive cases. A system should not pass by always returning the source or always editing.

## 6. Baselines and matched comparison

Reproduce published metrics before claiming improvement. Baseline groups should include:

1. deterministic symbolic/DSP edits as an upper bound for operations that are exactly specified in score space;
2. YuE2's published and available score-editing path with native metric definitions and denominators;
3. SMEBench-supported paired edit baselines for overlapping operations;
4. open-source audio-editing models/agents from the MMAE/ICASSP 2027 evaluation ecosystem;
5. human-expert edits on a stratified subset;
6. an identity/no-op baseline and random/incorrect-operation controls.

For comparisons, freeze identical source cases, target definitions, renderer set, output length, candidate count, inference budget and evaluator versions. Compare Best-of-1/2/4/8 under equal compute budgets, not merely the best of a larger sample against one-shot baselines. Report latency, GPU/CPU time, memory and render cost alongside quality.

### YuE2 reproduction boundaries

Do not quote the 192-prompt / 3,844-recording material as the upstream YuE2 editing result: the current upstream report identifies the 192-prompt / 17-setting WildSongBench comparison as generation quality, and documents editing evidence separately on 10 works, two seeds and 380 recordings. The separate editing values are local changed-note melody attainment 0.9375 (from 0.0083) and changed-duration harmony attainment 0.8313 (from 0). The metrics have different denominators and meanings, so they cannot be collapsed or compared to MUSE's per-event accuracy. The best-of-8 generation setting also uses a larger candidate-selection budget than a single render. MUSE should publish two distinct comparisons: exact matched symbolic/score edit adherence, and matched rendered/full-song audio editing with outside-edit preservation. Treat the 10-work set as a useful reproduction target, not a broad statistical proof.

### SMEBench replication boundaries

SMEBench's held-out synthetic test split contains 3,600 balanced cases (300 per operation); its separate SME-Real complement contains 416 test cases from 107 multitrack recordings. Its 17-participant listener study yielded 236 trials: mean instruction alignment was 3.99/5, and Legato, Remove_Instrument and Swap_Drums did not reliably exceed the midpoint. These operations remain reportable but should not be hidden inside a headline perceptual aggregate.

Two design consequences follow. First, MUSE must preregister per-operation perceptibility eligibility and report both symbolic-target correctness and listener recognition; a correct MIDI transformation can still be an inaudible or poorly perceived edit. Second, SMEBench implements crescendo/decrescendo with audio-domain gain envelopes because changing MIDI velocity alone gave poor perceptual results. MUSE's initial velocity-scale evaluator is therefore a symbolic correctness slice, **not** a matched reproduction of SMEBench's audio-domain dynamics operation.

SMEBench also demonstrates evaluator ranking reversals: under caption conditioning, CLAP-Text favored MusicGen while paired-target audio metrics favored SteerMusic; under instruction conditioning, CLAP-Text favored VCI while target metrics favored ZETA. Baseline reports must preserve separate source-preservation, target-agreement and text-compatibility axes rather than collapsing them into one reward score.

### Prospective engineering targets (not current results)

| Metric | Deterministic symbolic slice | Rendered/real-audio slice |
|---|---:|---:|
| Target adherence | ≥99.5% | ≥95% |
| Protected-content preservation | ≥99.9% | ≥98% |
| Event correspondence coverage | ≥99.9% | ≥99% |
| Strict sample exact match | ≥99% | ≥85% |
| Human edit recognition | Not applicable | ≥90% of eligible high-salience cases |
| Human musical-validity agreement | Not applicable | ≥85% positive ratings |

These targets are intentionally ambitious and must be adjusted only through a documented preregistration, not after seeing test outputs. The rendered/real-audio thresholds are not directly comparable with broad MMAE EMR because the task distribution and rubric scope differ.

To claim a meaningful improvement over a reproduced baseline, require at least a 5 percentage-point gain in StrictEMR on the same held-out cases, source-clustered 95% confidence intervals excluding zero, and no material regression in any primary operation family or major subgroup. Publish failures and per-family gaps even when aggregate performance improves.

## 7. Statistical analysis

- Bootstrap by source work and listener, not by individual note or audio crop.
- Use paired comparisons on identical cases and report 95% confidence intervals for every primary metric.
- Correct multiple comparisons across operation families; preregister primary families and subgroup analyses.
- Report exact numerators and denominators. Do not average percentages from groups with different denominators without a declared weighting scheme.
- Publish per-operation, vocal/instrumental/mixed, renderer, instrument, style and difficulty strata.
- For preference judges, report pairwise accuracy, calibration/Brier score, rank correlation, abstention and subgroup gaps. MIREX CMI-RewardBench results show why vocal/instrumental calibration must be separate.
- Keep the human panel as an independent confirmation layer; reward models produce evidence, not authorization.

## 8. Roadmap

1. Validate both JSON Schemas and fixtures with a pinned validator; add schema conformance and malformed-field tests.
2. Implement a deterministic score/event metric scorer that uses source-bound event correspondence and emits the result schema.
3. Add audio observers with known-answer synthetic fixtures, calibration sets and explicit abstention on poor correspondence.
4. Reproduce YuE2 and SMEBench subsets; publish exact settings, versions, denominators and deviations.
5. Construct rights-cleared cases and operation eligibility; freeze source-level splits before training/tuning.
6. Run the blinded listener study and preregister its analysis.
7. Publish a reproducibility bundle and matched-baseline report; only then make a SOTA claim.

## 9. Current status

**Implemented here:** case/result schemas, synthetic fixtures, protocol, and an initial deterministic Rust scorer/CLI for the one-to-one velocity-edit slice.

**Not yet implemented or measured:** full schema-validation pipeline, general multi-operation scorer, calibrated instruction evaluator, acoustic observers, real dataset, human panel, matched baseline reproduction, held-out evaluation and all prospective targets. Synthetic fixtures and authored tests are not benchmark performance evidence.


## 10. Executable scoring increment — 2026-11-20

The first scorer is intentionally narrow and fail-closed. It currently supports deterministic symbolic velocity scaling, bounded pitch transposition, and exact rational note-duration scaling (with onset and global Score metadata held fixed):

- Accepts typed `symthaea_music_theory::Score` JSON for the frozen source and candidate. Before deserialization, closed-field checks reject unrecognized properties in the case contract and Score/ScoreNote/rational-time objects; MIDI pitch bounds and rational timing/dynamic ranges are checked after deserialization.
- Recomputes the Score content digest using the same `serde_json::to_vec(typed_score)` followed by SHA-256 convention used by MUSE score provenance, and rejects a stale/mismatched source binding.
- Supports velocity scaling, fixed-semitone pitch transposition, and rational note-duration scaling with explicit one-to-one source/target note correspondence and `preserve` / `modify` relations. Pitch transposition refuses MIDI-boundary saturation. Duration scaling changes only target note lengths: onset, tempo, key, meter and total Score length remain invariant in this slice; it does not claim articulation, note-tail or audible legato correctness. Note ordinals are interpreted only against their bound source/candidate Score digests.
- Uses bounded rational beat inputs and PartId scopes; requires each source/candidate note event to be wholly contained in its target/protected interval (boundary-crossing events abstain), checks target/protected overlap and declared event relations, and keeps every non-target note attribute and global Score metadata unchanged.
- Checks the requested rational velocity scale with the case's MIDI-velocity tolerance, exact semitone transposition without clamping, or exact rational duration scaling. Protected and outside-scope agreement and collateral-change rate remain separate result metrics.
- Produces the v1 result shape with canonical case SHA-256, typed source Score SHA-256, candidate Score/audio hashes, and a domain-separated evaluation receipt seal. The case digest uses recursive lexicographic object-key normalization followed by compact serde_json serialization, domain-separated as `muse-edit-evidence-case-serde-json-sha256-v1`; the result receipt uses the same recursive normalization. This is a versioned project recipe, not an RFC 8785 claim. A public seal-verification function recomputes the result receipt. An optional candidate WAV is byte-hashed and checked against the exact/acceptable audio hashes only for the exact-score-and-audio target kind.
- Exposes the evaluator through a `theory`-gated CLI:

~~~sh
cargo run -p symthaea-muse --features theory --bin muse_edit_evaluator -- --case case.json --source-score source-score.json --candidate-score candidate-score.json --output result.json
~~~

**Deliberate non-claims:** this slice does not evaluate free-form instruction compatibility, audio perceptibility, musical validity, acoustic preservation, articulation/legato, or renderer execution integrity. It emits `not_run` for instruction and integrity evidence; consequently `strict_sample_exact_match` remains false even when the symbolic operation and preservation sub-gates pass. Unsupported edit families, target kinds, and split/merge/insert/delete correspondence relations are refused rather than heuristically scored.

Twenty-eight unit tests are authored across positive velocity, pitch and duration edits; inverse pitch and duration editing; explicit event-map reordering; wrong velocity/pitch/duration targets; protected-note mutation; duplicate correspondence; stale source digest; unsupported-operation abstention; malformed/boundary-crossing rational timing, including candidate note-off boundary rejection; object-key-order-independent case/result digest checks; unknown case/Score fields; empty/omitted metric rejection; out-of-range MIDI pitch; no-op scale rejection; inserted candidate notes; empty protected-scope rejection; MIDI-boundary pitch saturation; and result-seal tampering. They have **not** been compiled or executed: no local Cargo/rustc/rustfmt is available, and no exact-head CI pass is claimed. The full JSON Schema validator remains outstanding; these tests are source additions, not test results.
