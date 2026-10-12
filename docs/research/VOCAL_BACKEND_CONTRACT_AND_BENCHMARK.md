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

Use the existing concealed IDs and human-rating workflow. Keep technical checks separate from ratings for naturalness, intelligibility, melodic fidelity, emotional fit, artifacts, and identity. Compare loudness-matched dry captures; human anchors cannot count toward the synthetic-candidate pass threshold. The release gate must reject ratings with unknown clip IDs, blank listener IDs, duplicate listener/clip pairs, or a lyric-word denominator that differs from the frozen manifest. These checks prevent extra or malformed rows from contaminating the aggregate release score. The shareable `blind_manifest.json` must contain only listener-needed fields and omit backend, case ID, category, and challenge labels. Store the answer key, objective report, and release status in a sibling `.private` directory (Unix mode `0700`); never pass it to raters. Require fresh, empty public/private output directories per run to fail closed on stale audio or mappings. Never promote on a single aggregate score.

## Output and provenance contract

The release harness separates two output trees: the shareable listening pack (`--output`) contains WAVs, a minimal listener manifest, and a rating template; the organizer-only sibling (`<output>.private`) contains the backend answer key, objective report, and release status. Unix permissions are set to `0700`; on other platforms the operator must configure and verify suitable ACLs. The gate rejects stale files in either directory, and listeners must never receive the private sibling. The legacy-artifact migration path moves old key/report/status files out of an existing listening pack before failing closed so the operator can archive them safely.

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
- **SwiftF0:** a promising score-blind external cross-check. The project's [current benchmark](https://github.com/lars76/pitch-benchmark) v2.1 reports the highest pitch F1 among 19 monophonic trackers on its ten-corpus suite (0.781 vs. 0.768 for RMVPE), while explicitly stating that the difference is not statistically resolved. SwiftF0 reports 16 ms frame estimates and confidence, with a documented 46.875–2,093.75 Hz range. This benchmark is not singing-synthesis-specific; test a pinned version on the frozen Symthaea stems, review model/code/data licensing separately, and compare it against the in-tree diagnostic before using its output in a release decision. The benchmark evaluation dataset is marked CC BY-NC-SA 4.0 for research use, so do not bundle or redistribute it as if the dataset shared the code license.
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
5. The release-gate rating parser now validates listener identity, known manifest clip IDs, exact lyric-word totals, rating ranges, and duplicates before aggregation; regression tests cover unknown IDs, blank listeners, count mismatch, and valid rows. The renderer boundary also calls `VocalStem::validate` before writing audio. These tests are authored but still require execution.
6. When pinned assets and runtime are available, run the smallest real renderer smoke test and save the prerequisite inventory, logs, observed decoded rate, and output digest.
7. Run the existing frozen 60-case corpus against viable in-tree backends; preserve each WAV, minimal listener manifest, ratings template, and private evidence sibling separately. Use fresh output paths and never hand the private sibling to listeners.
8. Compare the new score-blind in-tree diagnostic with score-guided metrics and human ratings; inspect the per-case disagreement before predeclaring any new threshold.
9. Run a pinned SwiftF0 shadow evaluation on the same WAV digests and compare it with at least one independent tracker; do not gate on it until confidence, out-of-range behavior, model/runtime pinning, and licensing are reviewed.
10. Add an external singing-generation backend adapter only after capability mapping, licensing, language coverage, model pinning, and host resource budget are recorded.

## Promotion rule

A candidate is not “better” because it builds, has more features, follows pitch under a score-guided tracker, or scores well on a speech-trained estimator. Promotion requires no contract/integrity failures; documented lyric and musical fidelity; improved blinded perceptual outcomes on relevant cases; no unacceptable identity/language/register regressions; reproducible artifacts and model provenance; and a rights review appropriate to intended deployment.

## Evidence record template

For each run retain at least `run_id`, `git_sha`, `backend`, `backend_version`, `checkpoint_sha256`, `voicebank_id`, `input_sha256`, `declared_sample_rate_hz`, `observed_sample_rate_hz`, `resampling_provenance`, `channels`, `samples`, `duration_s`, `output_sha256`, `integrity_status`, `independent_f0_metrics`, `score_guided_f0_metrics`, `lyric_metrics`, `singmos_metrics`, `listening_notes`, `license_status`, and `limitations`.

Unknown fields are explicitly marked unknown; queued work is not passed work.
