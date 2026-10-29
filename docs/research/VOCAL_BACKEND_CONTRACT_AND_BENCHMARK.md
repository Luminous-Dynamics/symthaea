# Vocal backend contract and benchmark plan

Status: research/engineering plan. This document does not claim a backend has passed, new audio has been rendered, or an external candidate has been tested locally.

## Goal

Improve Symthaea vocals by resolving the checked-in renderer contract first, preserving existing evidence, and comparing viable backends without mistaking technical metrics for perceptual quality.

## Exact-head source audit

The Git tree for `main` at base revision `b9af191f380d635f0459b24284c1d1fd6a5516aa` was inspected directly.

- `src/voice/diffsinger.rs` implements a version-1 JSONL bridge. It forwards returned samples and the worker's declared sample rate to `VocalStem` and calls `stem.validate()`; it does not resample.
- `scripts/diffsinger_worker.py` declares `SAMPLE_RATE = 44100`, uses that rate to derive frame timing, and returns it with the synthesized waveform.
- At the base revision, `examples/diffsinger_worker_smoke.rs` asserted `stem.sample_rate == 24_000`, conflicting with the worker's declared `44100`. This PR aligns the smoke assertion to `44_100` for interface consistency. That does not establish the external ONNX voicebank's true native output rate: verify its pinned config or an observed render before treating the backend as validated. Do not change WAV metadata to disguise a mismatch.
- At the base revision, `docs/MUSE_SINGING_PRODUCTION.md` described a pinned OpenVPI/DiffSinger v2.5.1 path and commands under `communication/singing/` and `communication/worker/`. Those paths are absent from the complete `main` tree at the revision above, so a clean checkout could not follow that earlier guide as written. This PR corrects the guide to the in-tree harness; the path discrepancy is recorded as a finding, not proof that the referenced external assets never existed elsewhere.
- The in-tree worker itself calls its voice a placeholder and points to voicebank-specific terms. It is not evidence of a cleared or distinct Symthaea voice identity.
- `examples/muse_vocal_release_gate.rs` exists and already defines 20 phrase challenges crossed with three vocal ranges/styles (60 cases), concealed clip IDs, objective diagnostics, and a human-rating template. Build on this rather than creating a competing miniature benchmark.
- The prior DiffSinger + RVC evidence bundle shows a completed historical experiment, but explicitly says that speaker identity, commercial clearance, broader generalization, and perceptual preference were not established. Retain those limitations with the evidence.

## Immediate priorities

### P0 — reconcile the published renderer path

1. Decide which DiffSinger integration is canonical: the checked-in legacy ONNX JSONL worker, or the production path described in the guide.
2. Either restore the guide's referenced scripts and lockfiles with a reproducible, reviewed commit, or rewrite the guide to document only files that exist at the pinned repository HEAD.
3. Record external weights/runtime separately from source assets. A clean checkout must fail with a precise missing-prerequisite message rather than implying the backend is ready.
4. Add a path-existence / documentation consistency check so commands in the guide do not silently rot.

### P0 — make sample-rate truth testable

The worker's declared rate, the vocoder's configured native output rate, any actual decoded WAV rate, and any resampling stage are distinct facts. Record all four when applicable. Add a test that compares metadata to the authoritative pinned model configuration or worker handshake; do not hardcode an expected rate without tracing its provenance. A resampling operation must be explicit, measured, and recorded.

### P1 — protect benchmark validity

Reuse the existing 60-case gate as the primary corpus. Freeze its source revision, lyrics, score generation, voice/model IDs, category assignment, and output-processing rules before comparing candidates. Expand it only when a specific uncovered failure mode is identified; retain the 20 phrase challenges as traceable case IDs.

The current `singing_quality.rs` keeps its target-conditioned pitch metric (`estimate_f0_near`, a ±300-cent search) for continuity with the existing release gate. A new `score_blind_*` metric family adds a separate 55–1,000 Hz autocorrelation search that does not use the expected pitch as a search prior. The paired regression fixture synthesizes 500 Hz against a 220 Hz target: the legacy score-guided error remains window-censored, while the score-blind diagnostic is expected to expose a mismatch above 1,000 cents. These are distinct signals; the score-blind routine is an in-tree diagnostic, not a validated singing tracker and not yet part of `objective_pass` or `release_pass`. Inspect per-case voiced coverage, voiced/unvoiced disagreement, median/p95 error, octave mistakes, and undefined estimates; do not map undefined values to zero or treat the new diagnostic as release authority.

Use the existing concealed IDs and human-rating workflow. Keep technical checks separate from ratings for naturalness, intelligibility, melodic fidelity, emotional fit, artifacts, and identity. Compare loudness-matched dry captures; human anchors cannot count toward the synthetic-candidate pass threshold. The release gate must reject ratings with unknown clip IDs, blank listener IDs, duplicate listener/clip pairs, or a lyric-word denominator that differs from the frozen manifest. These checks prevent extra or malformed rows from contaminating the aggregate release score. In addition to overall means, each synthetic clip must individually meet the same 4.0 mean floors for naturalness, emotional fit, identity consistency, and artifact-free quality plus 0.95 lyric comprehension; a low-quality phrase/register cannot be masked by stronger cases elsewhere. The private `human_case_report.json` stores per-clip summaries and `release_status.json` stores a failing-case count. Each ratings-only rerun archives the previous report/status to a unique `*.review-*.json` file before updating the latest copy; only these recognized history files are allowed in the private run-state directory. The shareable `blind_manifest.json` must contain only listener-needed fields and omit backend, case ID, category, and challenge labels. Store the answer key, objective report, human case report, and release status in a sibling `.private` directory (Unix mode `0700`); never pass it to raters. Require fresh, empty public/private output directories per run to fail closed on stale audio or mappings. Never promote on a single aggregate score.

## Output and provenance contract

The release harness separates two output trees: the shareable listening pack (`--output`) contains WAVs, a minimal listener manifest, and a rating template; the organizer-only sibling (`<output>.private`) contains the backend answer key, objective report, per-clip human case report, and release status. Unix permissions are set to `0700`; on other platforms the operator must configure and verify suitable ACLs. Both output paths must be non-symlink directories, must canonicalize to separate non-nested trees, and must be empty for a fresh run; listeners must never receive the private sibling. The legacy-artifact migration path moves old answer-key, objective-report, per-clip human-report, and release-status files out of an existing listening pack before failing closed so the operator can archive them safely; filename collisions are retained under distinct legacy names instead of overwriting prior evidence.

Every backend invocation should produce a machine-readable run manifest containing:

- Exact source HEAD, backend name/version, invocation/configuration, and checkpoint digest.
- Voicebank/reference identity, authorization and license status, and provenance of any voice prompt.
- Normalized lyrics/score input and hashes; declared language and supported-control mode.
- Declared native rate, observed decoded rate, channel count, format, sample count, duration, peak/RMS, and whether samples were generated, resampled, trimmed, padded, or normalized.
- Exit status, stdout/stderr/log reference, elapsed time, execution device/provider, and resource profile when available.
- Output digest, immutable run ID, metric versions/configuration, per-case results, and explicit limitations.

Fail closed on empty or non-finite audio, invalid sample rate, unexpected channel layout, model/voice identity absence, malformed protocol, or digest mismatch. Unknown evidence fields must be marked unknown. Do not silently convert missing metrics to zeros or pass results.

## Evaluation dimensions

### Integrity and reproducibility
- Valid audio container and decoded samples; no unexpected silence, clipping, DC offset, or discontinuities.
- Declared versus observed rate/channel consistency and complete configuration/checkpoint/input/output hashes.
- Repeated-run variability for stochastic backends, with seed and determinism limits recorded.

### Musical and linguistic fidelity
- Independent F0 tracking and score-aligned measurements with voiced coverage, octave mistakes, missing-frame handling, and predeclared tolerances.
- Note onset/duration error, melisma handling, syllable-to-note alignment, and phrase-release behavior.
- Singing-robust ASR as a diagnostic—not ground truth—with manually checked transcripts for a fixed subset.
- For expressive alternatives, distinguish acceptable vibrato/portamento from wrong notes instead of requiring samplewise waveform equality.

### Perceptual quality and generalization
- Singing-specific MOS models are screening signals, not a release authority. Speech-trained MOS scores remain diagnostic only.
- Use concealed, loudness-matched dry audio and stable rating instructions. Averages cannot hide a catastrophic phrase or register failure.
- Hold out complete phrases/songs or recording sessions where relevant; report case-level results, language, genre, register, hardware, and prompt compatibility.
- Human evaluation remains necessary at promotion milestones. A successful build/render or a high pitch score alone is not a quality pass.

## Candidate comparison

- **Current in-tree candidates:** Kokoro speech-to-singing baseline, DiffSinger worker (after rate resolution), and the functional singing/vocal-tract experiment. These already have different tasks and must be characterized rather than assumed equivalent.
- **SoulX-Singer:** a high-priority external comparison candidate because its official repository describes both melody-conditioned and score/MIDI-conditioned singing synthesis. Before integration, pin a specific source/model revision, verify the published model and dataset terms for the intended use, test English inputs and controls, and record its hardware/memory needs. A public paper or project claim is a reason to test, not evidence it outperforms Symthaea locally.
- **SoulX-Singer-Eval:** a useful evaluation reference because its suite separates singing aesthetics (SingMOS-Pro/Sheet-SSQA), spectral quality, ASR WER/CER, speaker similarity, and melodic precision (FFE/GPE/VDE). Reuse the taxonomy first; adopt the heavyweight dependency only after checking licenses, model costs, and whether its inputs fit this corpus.
- **SingMOS-Pro predictor:** the project publishes a dedicated singing MOS predictor; its benchmark includes 7,981 clips from 41 models across 12 datasets, each with at least five ratings. Evaluate it as an auxiliary screening signal with a pinned predictor/checkpoint and fixed sample-rate/cropping pipeline; it is not a substitute for a blinded panel. Predictor repository: https://github.com/South-Twilight/SingMOS
- **SwiftF0:** a promising score-blind external cross-check. The project's [current benchmark](https://github.com/lars76/pitch-benchmark) reports a pitch F1 of 0.781 (95% CI 0.768–0.795) across its ten-corpus, eight-recording-condition suite, versus 0.768 (95% CI 0.752–0.783) for RMVPE; it explicitly says this difference is not statistically resolved. The SwiftF0 project README currently reports roughly 180× real-time on one laptop CPU core (benchmark details and machine differ by version), 16 ms frame outputs plus confidence, and a documented 46.875–2,093.75 Hz range. This benchmark is not singing-synthesis-specific; test a pinned version on frozen Symthaea stems, review model/code/data licensing separately, and compare it against the in-tree diagnostic before using output in a release decision. The benchmark evaluation dataset is marked CC BY-NC-SA 4.0 for research use, so do not bundle or redistribute it as if the dataset shared the code license.
- **MusicJudge (Interspeech 2026):** a promising later evaluation experiment because it attempts joint lyric correctness and pitch/rhythm assessment at musically meaningful lyric blocks. Its September 27, 2026 publication is a research lead, not independent proof of local performance. Do not make it a gate dependency until implementation, model availability, licensing, and failure behavior have been independently inspected.
- **VocalRender:** retain as a separate score-native research candidate only for cases where language, prompt-conditioning, checkpoint license/provenance, and hardware match. English quality must be measured rather than inferred from Mandarin-focused results.
- **Whole-song music generators:** a separate task track. Do not score them as equivalent to score-conditioned vocal synthesis unless controllability, lyric requirements, and inputs are matched.

Research links:
- SoulX-Singer official implementation: https://github.com/Soul-AILab/SoulX-Singer
- SoulX-Singer evaluation suite: https://github.com/Soul-AILab/SoulX-Singer-Eval
- MusicJudge paper: https://arxiv.org/abs/2606.26451
- VocalRender implementation: https://github.com/pymaster17/VocalRender

## Execution sequence

1. Keep the production guide synchronized with the in-tree release harness and the exact source HEAD; the guide was updated in this PR.
2. The performance contract now rejects non-finite pitch-curve values, malformed melisma timing/velocity, invalid phoneme durations, out-of-range expression controls, overflowing note endpoints, and syllables mapped outside their declared phrase; regression tests were added for representative malformed inputs.
3. The standalone waveform-cleanliness gate now rejects empty/degenerate signals and rates below 8 kHz rather than allowing vacuous finite/zero-delta metrics to classify absent audio as clean. Regression tests were added for empty samples, one-sample input, and invalid rate.
4. Align the DiffSinger smoke assertion with the worker's declared 44.1 kHz interface, but verify the actual pinned voicebank/vocoder native rate before promoting the path. Interface alignment is not native-rate evidence.
5. The release-gate rating parser validates listener identity, known manifest clip IDs, exact lyric-word totals, rating ranges, and duplicates before aggregation. The renderer boundary calls `VocalStem::validate` before writing audio. Regression tests cover malformed ratings, per-clip vetoes, saved-run tampering, and rerating history; current-head execution remains required.
6. The score-blind diagnostic represents unavailable estimates as optional values and serializes them as JSON `null`, not zero. New reports record the eligible pitch-error frame count; deserialization treats that count as absent only for historical reports written before the field existed. Regression coverage checks a too-short signal, the serialized null contract, and compatibility with a prior report missing the score-blind fields. These metrics remain outside the objective and release gates.
7. When pinned assets and runtime are available, run the smallest real renderer smoke test and save the prerequisite inventory, logs, observed decoded rate, and output digest.
8. Run the existing frozen 60-case corpus against viable in-tree backends; preserve each WAV, minimal listener manifest, ratings template, and private evidence sibling separately. Use fresh output paths and never hand the private sibling to listeners.
9. After collecting ratings, invoke the dedicated `--ratings-only --ratings <file>` phase against that saved run. Require the ratings file to be outside both output trees, validate public/private manifests and every WAV identity, and compute per-clip plus overall release checks without rerendering. Revisions archive the prior private human-case report and status rather than destroying prior review evidence.
10. Compare the in-tree score-blind diagnostic with score-guided metrics and human ratings; inspect per-case disagreements before predeclaring any new threshold.
11. Run a pinned SwiftF0 shadow evaluation on the same WAV digests and compare it with at least one independent tracker. Do not gate on it until confidence handling, out-of-range behavior, model/runtime pinning, and licensing are reviewed.
12. Add an external singing-generation backend adapter only after capability mapping, licensing, language coverage, model pinning, and host resource budget are recorded.

## Human communication and spoken speech (separate evaluation track)

The existing 60-case suite is a sung-lyrics benchmark; it is **not** evidence that everyday spoken communication is natural, intelligible, or conversationally competent. The current repository describes the speech/singing backends as experimental, and no current-head audio has been supplied for a direct perceptual critique. Keep any statement about audible quality provisional until actual output is reviewed.

A source audit found two pronunciation-integrity defects and one language-contract limitation:

- The functional tract singing adapter mapped any unrecognized IPA symbol to `AX`, silently substituting a schwa-like vowel. It now returns an explicit error for a phoneme outside its declared English inventory.
- The DiffSinger worker had the same unknown-IPA-to-`ax` fallback. The mapping now lives in a small dependency-light module and rejects unsupported symbols rather than producing plausible but incorrect audio.
- The DiffSinger adapter always constructed `en/<phone>` tokens and used language ID 0 for an unknown language. The worker now requires an explicit English (`en`) mapping and rejects non-English requests until language-specific G2P, phone inventory, and model tokens are implemented together. This is an honest capability limit, not a claim of multilingual support.

The DiffSinger score builder also now divides a composite phone's duration among all of its output tokens. Melisma notes sustain the syllable's last explicitly mapped vowel rather than its final phone (which may be a consonant); empty or consonant-only syllables cannot acquire an implicit schwa to make that work.

The source-alignment audit also found that the existing letter fallback emits the composite IPA token `ks` for `x`. The functional renderer now expands that token into ordered `K` then `S` spans, rather than either rejecting a valid G2P output or losing one articulation. A Rust regression walks the current G2P dictionary and alphabet fallback; a dependency-light Python fixture checks the DiffSinger map. These tests protect mapping coverage, not acoustic pronunciation correctness.

These checks prevent a subset of hidden pronunciation corruption; they do not prove improved naturalness or speech intelligibility. A future non-English path should use language-specific grapheme-to-phoneme rules, a declared phone inventory, and exact token coverage rather than reusing English phone labels.

### Evaluation dimensions for spoken communication

Keep this evaluation separate from sustained-note singing:

1. **Message fidelity:** preserve negation, names, acronyms, dates, numbers, units, and user-supplied wording. Use transcript round-trips (WER/CER) as alarms, then exact checks for critical tokens; ASR output is a noisy proxy, not ground truth.
2. **Diction and phoneme coverage:** test minimal pairs and confusable consonants, clusters, multisyllabic words, unstressed syllables, and phrase-final consonants. Report unsupported phones explicitly; never convert unsupported phones to silence or a default vowel.
3. **Listener-oriented prosody:** control phrase boundaries, sentence-final questions versus statements, contrastive emphasis, information focus, uncertainty, reassurance, and urgency. Test whether emphasis lands on the words that carry the message, not just whether F0 varies. Conversation research suggests that prosodic prominence and word predictability are related to intelligibility-oriented pressures ([Clark et al., 2025](https://doi.org/10.1111/cogs.70134)).
4. **Emotional congruence:** compare intended conversational function with acoustic delivery (rate, pause length, pitch range, intensity, and voice quality). Do not infer a listener's emotional state from prosody alone or force an affect label onto every utterance.
5. **Timing and turn-taking:** if used interactively, measure time-to-first-audio, real-time factor, chunk gaps, end-of-turn delay, stop/cancel latency, and interruption handling at p50/p95. A fluent offline WAV is not necessarily a usable conversational voice system.
6. **Listening comfort and identity:** check loudness stability, sibilance, breath noise, pitch/timbre consistency, and fatigue across long passages. Speech and singing should share identity constraints where desired, but singing-specific vibrato and sustained vowels must not be forced into ordinary speech.

### Spoken-passage comprehension fixture (implemented, audio integration pending)

The dependency-free `scripts/voice_comprehension_gate.py` and its original case bank provide a first structured comprehension layer beyond WER. The 16 synthetic passages cover negation, quantities and amounts, dates/times, names, units, sequencing, conditions, and safety instructions (32 four-choice questions).

- `prepare --output <pack>` writes listener-facing multiple-choice questions to the public directory, while the exact source text, SHA-256 digests, category labels, and answer key are written to a sibling private directory with restrictive Unix permissions. Public questions intentionally omit source text, answer keys, and category labels.
- `score --output <pack> --responses <outside-file>` requires one answer for every question and fails closed on missing/duplicate rows, unknown IDs, invalid choices, or coverage drift. It reports total and per-category accuracy plus the fraction of cases with all answers correct.
- It deliberately sets no release threshold: the fixture has no validated listener baseline or population norms. Reports say so explicitly. This is a deterministic scoring and artifact-integrity layer, not proof of speech intelligibility.
- Audio rendering is linked through `examples/voice_roundtrip_wer.rs --case-manifest <pack.private/render_manifest.json> --save-wavs <fresh-audio-dir> --report <private-report.json>`. The renderer validates manifest schema, English language support, unique safe clip IDs, non-empty text, question IDs and each source-text SHA-256; it emits `<clip_id>.wav`, records source and WAV digests and WER by clip ID. The renderer refuses stale/non-empty WAV output directories and report overwrites. The `voice-tts` CI feature leg runs seven manifest-contract unit tests for digest drift, path-like/duplicate IDs, unsupported language, malformed/empty question IDs, empty case sets, unsupported schema versions and standard SHA-256 output. The result report also records the SHA-256 of the exact case-manifest bytes. Actual synthesis and listener comprehension still must be run; no audio quality or comprehension result is implied by adding the path.

### Automated-first validation sequence

- Build a deterministic 60–100 prompt spoken suite spanning short answers, long explanations, questions, contrastive stress, numbers/dates, names/acronyms, negation/minimal pairs, low-energy delivery, and multi-clause sentences.
- Freeze expected text, language, phoneme/word coverage, prompt SHA, backend/model SHA, sample-rate path, and output WAV digests.
- Compute ASR WER/CER with a pinned recognizer, phoneme/word timing coverage where alignment is available, pause and speaking-rate distributions, clipped/silent duration, and synthesis latency. Any ASR mismatch is a review trigger; it is not by itself proof the audio is wrong. The existing `examples/voice_roundtrip_wer.rs` now records requested/rendered counts and every synthesis failure, writes an incomplete report when requested, and exits nonzero if any requested sentence has empty or failed synthesis. A partial corpus must never be presented as a complete WER result. It also refuses a non-empty or symlinked WAV directory, refuses an existing report path, and keeps report/audio output paths separate, preventing stale clips and silent replacement of earlier evidence.
- Add deterministic stress fixtures for unsupported symbols, unknown language IDs, missing dictionary words, and malformed audio. The engine should return a typed/clear error and must not quietly emit a substitute pronunciation.
- Reserve owner listening/capture review for meaningful audible milestones, not as a substitute for automated coverage or a requirement for every code change. Do not claim an improvement until both exact-head automated results and a recorded audio review support it.

## Promotion rule

A candidate is not “better” because it builds, has more features, follows pitch under a score-guided tracker, or scores well on a speech-trained estimator. Promotion requires no contract/integrity failures; documented lyric and musical fidelity; improved blinded perceptual outcomes on relevant cases; no unacceptable identity/language/register regressions; reproducible artifacts and model provenance; and a rights review appropriate to intended deployment.

## Evidence record template

For each run retain at least `run_id`, `git_sha`, `backend`, `backend_version`, `checkpoint_sha256`, `voicebank_id`, `input_sha256`, `declared_sample_rate_hz`, `observed_sample_rate_hz`, `resampling_provenance`, `channels`, `samples`, `duration_s`, `output_sha256`, `integrity_status`, `independent_f0_metrics`, `score_guided_f0_metrics`, `lyric_metrics`, `singmos_metrics`, `listening_notes`, `license_status`, and `limitations`.

Unknown fields are explicitly marked unknown; queued work is not passed work.
