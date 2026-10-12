# Muse singing: current implementation and reproducible evaluation

Status at repository base `b9af191f380d635f0459b24284c1d1fd6a5516aa`: the in-tree singing paths are experimental. This guide intentionally describes only assets present in that source tree; it does not claim a production-ready DiffSinger deployment or a distinct, cleared Symthaea voice identity.

## What exists in this checkout

- `examples/muse_vocal_release_gate.rs` is the current in-tree blind comparison harness. It builds 20 lyric/phonetic challenge phrases and crosses each with three style/register conditions for 60 cases. It includes Kokoro, functional singing, and optional DiffSinger comparison, salted clip IDs, objective diagnostics, and a human-rating workflow.
- `src/voice/singing_engine.rs` defines the backend-neutral `VocalPerformance` contract.
- `src/voice/kokoro_singing.rs` adapts speech synthesis to a target melody. Treat it as a baseline, not presumed singing-quality winner.
- `src/voice/functional_singing.rs` is an experimental interpretable alternative; do not equate architectural interpretability with realism.
- `src/voice/diffsinger.rs`, `scripts/diffsinger_worker.sh`, and `scripts/diffsinger_worker.py` form an optional local ONNX/JSONL worker path. The Python module explicitly describes its voice as a placeholder and requires an externally provisioned voicebank.
- The previously documented `communication/singing/` and `communication/worker/` OpenVPI/DiffSinger v2.5.1 deployment path is not present in this `main` tree. Commands targeting those paths are not reproducible from a clean checkout and are intentionally not shown as executable instructions here.

## Run the in-tree comparative gate

From the repository root with Nix and the required Rust features/dependencies available:

```bash
nix develop -c cargo run --example muse_vocal_release_gate \
  --features "singing,voice-tts" -- \
  --output audio_output/muse_vocal_gate-run-001 \
  --voice af_bella
```

This uses the in-tree Kokoro and functional-singing paths. It renders the 60 fixed cases and writes WAVs, a listener manifest, and a ratings template to the shareable `--output` directory. The listener manifest includes only `clip_id`, `lyrics`, and `audio`; it omits backend, case ID, category, and challenge labels.

Organizer-only artifacts are written to the sibling `audio_output/muse_vocal_gate-run-001.private/`: the internal manifest, backend answer key, objective report (which includes backend/case mappings), per-clip human case report, and release status. On Unix this directory is restricted to owner access (`0700`). **Share only the non-private `--output` directory with listeners; never share the `.private` sibling.** On non-Unix systems, set and verify restrictive ACLs on the private directory before sharing anything.

Each new render run requires fresh, empty `--output` and `.private` directories. The gate rejects symlink output paths, checks that the canonical directories are separate and non-nested, and fails closed if either contains files, preventing stale WAVs/manifests from mixing across runs. Archive earlier run directories rather than reusing or deleting them. When upgrading an older output directory, legacy answer-key, internal-manifest, objective-report, per-clip-report, or release-status files are moved to the private sibling without overwriting existing evidence; the run then stops because the private directory is non-empty. Archive that prior run and select a fresh run-specific path. The dedicated `--ratings-only` review phase described below intentionally reuses a completed run without rendering again.

## Review completed blind ratings without rerendering

After listeners complete the generated template, save the filled ratings JSON **outside both** the public listening-pack directory and the `.private` sibling. Then run the ratings-only phase against the same saved run:

```bash
nix develop -c cargo run --example muse_vocal_release_gate \
  --features "singing,voice-tts" -- \
  --output audio_output/muse_vocal_gate-run-001 \
  --ratings /absolute/path/ratings.completed.json \
  --ratings-only
```

This phase does not initialize a voice backend or render audio. It validates the saved public manifest against the private internal manifest, answer key, objective report, and referenced regular WAV files; rejects unexpected/stale files or mismatched identities; and validates listener ratings. The latest results are written to private `human_case_report.json` and `release_status.json`; the previous versions are preserved as uniquely named `*.review-*.json` history files before replacement. It requires the ratings file to live outside both output trees. You can rerun it with revised ratings without changing the blind IDs or rerendering; it preserves the exact audio/run mapping and prior review evidence. Never distribute the private sibling.

To include the optional DiffSinger worker, provision a voicebank outside the repository and point `SYMTHAEA_DIFFSINGER_VOICEBANK` at its root. The worker expects the ONNX stages and metadata under `dsmain/`, `dsvariance/`, and `dsvocoder/`; it does not download models or validate voicebank licensing for you. Then run:

```bash
SYMTHAEA_DIFFSINGER_VOICEBANK=/absolute/path/to/voicebank \
nix develop -c cargo run --example muse_vocal_release_gate \
  --features "singing,voice-tts" -- \
  --output audio_output/muse_vocal_gate-run-002 \
  --voice af_bella \
  --diffsinger-worker scripts/diffsinger_worker.sh
```

**Important known blocker:** `scripts/diffsinger_worker.py` declares and returns `44100` Hz, and this branch aligns `examples/diffsinger_worker_smoke.rs` to that declared rate. The Rust bridge forwards the worker's rate without resampling. This assertion checks interface consistency only; the actual native rate of the externally provisioned ONNX voicebank has not been verified from its pinned config or an observed render. Until that verification exists, treat DiffSinger results as experimental and do not promote them based on the harness output alone. Never fix a mismatch by merely relabeling WAV metadata; identify the actual rate and record any explicit resampling.

## Interpreting the gate

Keep each backend/case result—not just an average. The existing target-conditioned F0 diagnostic uses expected score pitch as a ±300-cent search prior. This branch also emits `score_blind_*` metrics using a broad 55–1,000 Hz autocorrelation search without that pitch prior; a wrong-note synthetic fixture is designed to expose cases whose score-guided error saturates near the search boundary. These new metrics are cross-checks only: they do not participate in the current objective/release pass decision, and the in-tree autocorrelation method has not been validated as a production pitch tracker. Compare them per case against a pinned external score-blind tracker (SwiftF0 is one candidate) before deciding whether a new metric belongs in the release gate.

The harness supports concealed listening and a complete listener-by-clip rating matrix. Use dry, loudness-matched audio and rate naturalness, emotional fit, identity consistency, artifact-free quality, and lyric comprehension separately. Keep human reference anchors concealed and exclude them from synthetic pass-rate calculations. The rating parser rejects unknown clip IDs, blank listener IDs, duplicate listener/clip pairs, invalid score ranges, and lyric-word totals that differ from the frozen manifest. In addition to overall means, every synthetic clip must individually reach listener-mean scores of at least 4.0 for naturalness, emotional fit, identity consistency, and artifact-free quality, plus at least 0.95 lyric comprehension. One failing synthetic clip blocks release even if the overall means pass. Private `human_case_report.json` records the anonymized per-clip summaries and `release_status.json` reports the failing synthetic-clip count. The render gate validates each stem before saving it so malformed backend output cannot enter the listening pack. A build, successful render, or pitch metric is not itself a release pass.

Before comparing revisions, preserve the source SHA, backend/provider, voicebank/model hash, voicebank authorization/license, exact input case IDs, output hashes, observed sample rate/channels/duration, metric configuration, and all missing fields. Treat the listener manifest as shareable and the `.private` answer-key/report directory as organizer-only. Prefer a fresh secure blind key for each evaluation; use a fixed seed only for diagnostic reproduction. Archive both output directories together as one run, but never distribute the private sibling with the listening pack.

## Existing historical evidence and limitations

The evidence bundle at `docs/research/evidence/singing-voice-rvc-2026-07-26/` records a completed DiffSinger + RVC research run with logs, configs, metrics, and output/checkpoint hashes. It explicitly does **not** establish speaker-identity similarity, a distinct Symthaea identity, commercial clearance, broad language/register generalization, or human preference for a particular checkpoint. Artifacts such as training data, virtual environments, and checkpoint weights are not all preserved in the repository; see the bundle's `REPRODUCE.md` before assuming a clean reproduction is possible.

Do not train on scraped voices or pitch-shifted Kokoro output. Any future identity training requires explicit voice-identity and training authorization, documented dataset provenance, held-out splits by complete recording session, validated alignments, and licensing review for the intended deployment.

## Research candidates (not yet tested here)

- **SoulX-Singer** is worth an external comparison because its official implementation describes melody-conditioned and score/MIDI-conditioned synthesis: https://github.com/Soul-AILab/SoulX-Singer
- **SoulX-Singer-Eval** gives a useful evaluation taxonomy spanning singing MOS, spectral quality, ASR, speaker similarity, and melodic precision: https://github.com/Soul-AILab/SoulX-Singer-Eval
- **MusicJudge** is a research lead for joint lyric and pitch/rhythm evaluation, not a gate dependency until implementation and licensing have been reviewed: https://arxiv.org/abs/2606.26451
- **VocalRender** is score-native and reports 48 kHz output, but requires prompt audio and a substantial checkpoint; test only where language and model/license provenance are suitable: https://github.com/pymaster17/VocalRender

For external models, pin source and checkpoint revisions, check data/model licenses separately, record the required hardware/runtime, and use the same fixed cases where the input/control semantics are genuinely comparable. Do not infer superiority from an external paper or benchmark headline.

## Promotion requirements

Promote a backend only after:

- contract and audio-integrity checks pass at the exact source/model head;
- sample rate is verified against the actual model output or explicit resampling;
- lyrics, note timing, melisma, register transitions, and voice identity are evaluated separately;
- independent musical metrics and concealed human ratings improve on the cases that matter;
- per-case failures, reproducibility limits, model provenance, and licensing status are recorded.

Queued CI, a clean compile, a successful single render, or a high automated score is not a passed singing-quality gate.
