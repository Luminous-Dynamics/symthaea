# Vocal backend contract and benchmark plan

Status: research/engineering plan; not a claim that a backend has passed or that new audio has been rendered.

## Goal

Identify the best usable singing backend for Symthaea's target use cases, separate generator defects from control-layer defects, and prevent technical metrics from being mistaken for perceptual quality.

## Exact-head observations

- `src/voice/kokoro_singing.rs` adapts speech audio into sung notes. Keep it as a baseline, not the presumed production winner.
- `src/voice/diffsinger.rs` and `scripts/diffsinger_worker.py` implement a JSONL worker path.
- At the inspected main revision, `scripts/diffsinger_worker.py` declares `SAMPLE_RATE = 44100`; `examples/diffsinger_worker_smoke.rs` asserts that the returned `VocalStem` is 24,000 Hz. The Rust bridge passes through the worker's `sample_rate` field and validates the stem, but does not resample it. This is a real source-level contract discrepancy. Do not change either value by assumption: inspect the voicebank/vocoder's actual emitted rate and make the authoritative contract explicit.
- The current production guide references a substantial `communication/singing/` and `communication/worker/` provisioning path. These assets do exist on current main; an earlier tree listing was stale/incomplete. The remaining question is whether a clean checkout plus the documented external provisioning steps can reproduce a render.
- The Python worker's module documentation calls its voice a placeholder and requires auditing the voicebank terms. Do not treat this as the final Symthaea identity.
- Prior evidence reports quality problems in speech-to-singing experiments. Keep those results visible as baselines.

## Contract: audio output

Every backend invocation must return or record:

- Exact backend name/version and model/checkpoint digest.
- Exact voicebank/reference identity and licensing/provenance record.
- Input score and lyrics digest; normalized input representation.
- Actual decoded sample rate, channel count, sample format, sample count, duration, peak/RMS.
- Whether output was generated, resampled, trimmed, padded, or otherwise post-processed.
- Exit status, stderr/log path, wall-clock time, and resource profile when available.
- Output WAV digest and immutable run identifier.

Fail closed if the observed sample rate differs from the backend's declared rate, if audio is empty/non-finite, if the channel layout is unexpected, or if a model/voicebank identity is missing. Never fix a sample-rate mismatch by merely changing WAV metadata. Resampling must be an explicit, recorded transform with its own test.

## Minimum reproducible phrase pack

Start small, deterministic, and rights-cleared. Use equivalent symbolic inputs wherever backend contracts permit:

1. Mid-range legato phrase with ordinary English phonemes.
2. Sustained open vowel with a controlled dynamic arc.
3. Consonant-dense phrase (fricatives, plosives, and clusters).
4. One syllable across multiple notes (melisma).
5. Large melodic leap and register transition.
6. Phrase ending with controlled release.
7. Low/intimate delivery.
8. High/intense delivery.

Do not compare models on a phrase their language, score representation, or conditioning setup cannot meaningfully support. Record exclusions and reasons instead of silently dropping difficult cases.

## Evaluation dimensions (never collapse into one score)

### Integrity and reproducibility
- Valid WAV, finite samples, no unexpected silence, clipping, or discontinuities.
- Actual versus declared sample rate and channel count.
- Configuration, checkpoint, voice, input, and output hashes.
- Repeated-run variability, if the backend is stochastic.

### Musical and linguistic fidelity
- Independent F0 tracker with voiced-frame coverage, octave errors, gross pitch error, and missing-frame accounting.
- Note-level pitch and timing errors, with score-aware tolerance documented before evaluation.
- Singing-robust ASR as a diagnostic, not ground truth; manually verified transcripts for a small fixed reference pack.
- Melisma and syllable-to-note assignment checks.

### Perceptual quality
- Singing-specific MOS estimator (e.g. SingMOS-Pro or a compatible maintained evaluation suite) as a screening metric only.
- Direct listening to loudness-matched, dry captures at defined milestones.
- Separate judgments for naturalness, timbre/artifacts, intelligibility, melodic fidelity, expression, and identity consistency.
- No promotion based solely on automated metrics. Speech-only MOS metrics remain diagnostic, not singing release gates.

### Generalization
- Hold out phrases, not just random frames from the same song.
- Report each case and aggregate summaries; averages must not conceal a catastrophic phrase or register failure.
- Keep language, genre, hardware, and voice-prompt applicability explicit.

## Backend comparison

- Current Kokoro speech-to-singing path as baseline.
- Exact current DiffSinger worker/voicebank after output-rate contract reconciliation.
- VocalRender as a research candidate only where prompt, language, license, and hardware constraints are appropriate; its released checkpoints primarily target Mandarin, so English quality is unproven.
- Functional vocal tract as an interpretable experimental backend, not assumed production-quality.
- Whole-song music generators in a separate track; do not score them as direct substitutes for score-conditioned singing unless task and controls are equivalent.

For every candidate, capture exact revisions and model hashes. Do not infer current model quality from a paper's aggregate results.

## Execution sequence

1. Inspect exact-head code, docs, scripts, and tests; identify the actual vocoder output-rate contract.
2. Add a focused contract test that validates the authoritative output-rate declaration and rejects mismatches.
3. Verify the documented provisioning chain against current paths and pinned revisions; distinguish repository assets from external model weights and runtime prerequisites.
4. Run the smallest local smoke test when dependencies/checkpoints are available. If not, record the exact missing prerequisite and complete static validation.
5. Render the fixed phrase pack for each viable backend; store immutable WAVs and manifests.
6. Evaluate technical fidelity and perceptual proxies separately; inspect captures directly before selecting a winner.
7. Make a targeted follow-up change based on the dominant evidenced failure mode.

## Promotion rule

A candidate is not "better" because it builds, has more features, passes pitch checks, or scores well on a speech-trained estimator. Promotion requires:
- no contract or integrity failures;
- documented lyric and musical fidelity;
- improved perceptual results on relevant cases;
- no unacceptable regressions in identity, language, or register;
- reproducible artifacts and model provenance;
- a rights review appropriate to intended use.

## Evidence record template

For each run, retain a JSON manifest with `run_id`, `git_sha`, `backend`, `backend_version`, `checkpoint_sha256`, `voicebank_id`, `input_sha256`, `declared_sample_rate_hz`, `observed_sample_rate_hz`, `channels`, `samples`, `duration_s`, `output_sha256`, `integrity_status`, `f0_metrics`, `lyric_metrics`, `singmos_metrics`, `listening_notes`, `license_status`, and `limitations`.

Unknown fields must be explicitly marked unknown; never fabricate a result or mark queued work as passed.
