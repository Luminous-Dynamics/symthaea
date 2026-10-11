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
  --output audio_output/muse_vocal_gate \
  --voice af_bella
```

This uses the in-tree Kokoro and functional-singing paths. It renders the 60 fixed cases and writes audio, objective diagnostics, a concealed listening manifest/key, and a ratings template. It is a test harness, not evidence of passing quality by itself.

To include the optional DiffSinger worker, provision a voicebank outside the repository and point `SYMTHAEA_DIFFSINGER_VOICEBANK` at its root. The worker expects the ONNX stages and metadata under `dsmain/`, `dsvariance/`, and `dsvocoder/`; it does not download models or validate voicebank licensing for you. Then run:

```bash
SYMTHAEA_DIFFSINGER_VOICEBANK=/absolute/path/to/voicebank \
nix develop -c cargo run --example muse_vocal_release_gate \
  --features "singing,voice-tts" -- \
  --output audio_output/muse_vocal_gate \
  --voice af_bella \
  --diffsinger-worker scripts/diffsinger_worker.sh
```

**Important known blocker:** `scripts/diffsinger_worker.py` declares and returns `44100` Hz, and this branch aligns `examples/diffsinger_worker_smoke.rs` to that declared rate. The Rust bridge forwards the worker's rate without resampling. This assertion checks interface consistency only; the actual native rate of the externally provisioned ONNX voicebank has not been verified from its pinned config or an observed render. Until that verification exists, treat DiffSinger results as experimental and do not promote them based on the harness output alone. Never fix a mismatch by merely relabeling WAV metadata; identify the actual rate and record any explicit resampling.

## Interpreting the gate

Keep each backend/case result—not just an average. Objective checks cover score-guided pitch/timing/voicing and waveform cleanliness, but the current F0 estimator uses the expected score pitch as a search prior. It is useful diagnostically, yet can bias estimates toward the target; report a score-independent pitch track alongside it before making strong musical-accuracy claims.

The harness supports concealed listening and a complete listener-by-clip rating matrix. Use dry, loudness-matched audio and rate naturalness, emotional fit, identity consistency, artifact-free quality, and lyric comprehension separately. Keep human reference anchors concealed and exclude them from synthetic pass-rate calculations. A build, successful render, or pitch metric is not itself a release pass.

Before comparing revisions, preserve the source SHA, backend/provider, voicebank/model hash, voicebank authorization/license, exact input case IDs, output hashes, observed sample rate/channels/duration, metric configuration, and all missing fields. Prefer a fresh secure blind key for each evaluation; use a fixed seed only for diagnostic reproduction.

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
