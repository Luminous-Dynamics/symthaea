# Muse singing production path

Muse now plans vocals through `VocalPerformance` v2 and can render the identical
performance through Kokoro v2 or a local singing-native worker. Kokoro remains
the zero-provision fallback; DiffSinger is the intended release backend.

Version 2 represents phrase-level expression arcs, breath placement,
consonant anticipation, articulation, slurs, and real melisma (one syllable
carried across multiple notes). The identity target is defined in
`communication/singing/SYMTHAEA_VOICE_BIBLE.md`.

## Run the Kokoro baseline

```bash
nix develop -c cargo run --bin symthaea-repl \
  --features "demo,audio,singing" -- \
  --voice --voice-model af_bella --singing-backend kokoro
```

Use `/sing <lyrics>` inside the REPL.

## Provision the pinned singing-native backend

The reviewed source graph is locked in
`communication/singing/openvpi.lock.json`. Provision or verify it outside the
repository:

```bash
python communication/singing/provision_openvpi.py /srv/luminous-dynamics/.local/openvpi
python communication/singing/provision_openvpi.py /srv/luminous-dynamics/.local/openvpi --verify-only
```

This checks out DiffSinger v2.5.1 at commit
`323a5693245b827caba0ab1a7371fe5ca885b061` plus locked MakeDiffSinger, GAME,
and SingingVocoders revisions. It does not download weights.

Provision and audit the isolated Python/CUDA runtime separately:

```bash
python communication/singing/provision_openvpi_runtime.py \
  /srv/luminous-dynamics/.local/openvpi/runtime
python communication/singing/provision_openvpi_runtime.py \
  /srv/luminous-dynamics/.local/openvpi/runtime --verify-only --require-cuda
```

Runtime never downloads code or model weights. The reviewed concrete wrapper
is `communication/worker/diffsinger_v251_infer.py`; it accepts:

```text
--input-json DIFFSINGER_INPUT.json
--output-wav VOCAL.wav
--model-path MODEL
--device cuda|cpu
--expected-commit COMMIT
```

`openvpi_v251_adapter.py` validates the checkout and translates the versioned
`VocalPerformance` contract into phoneme/duration/note/slur sequences plus F0
and expression curves. The wrapper converts these to DiffSinger frame curves,
maps normalized expression into native dB/logit domains, invokes the pinned
acoustic CLI, and validates mono 16-bit PCM output. Put a calibrated
`symthaea_inference.json` beside the voicebank's `config.yaml`; start from
`communication/singing/symthaea_inference.example.json`.

```bash
export SYMTHAEA_DIFFSINGER_ADAPTER="$PWD/communication/worker/openvpi_v251_adapter.py"
export SYMTHAEA_DIFFSINGER_MODEL_PATH=/absolute/path/to/consented-voicebank
export SYMTHAEA_DIFFSINGER_CHECKOUT=/srv/luminous-dynamics/.local/openvpi/diffsinger
export SYMTHAEA_DIFFSINGER_V251_COMMAND="$PWD/communication/worker/diffsinger_v251_infer.py"
export SYMTHAEA_OPENVPI_PYTHON=/srv/luminous-dynamics/.local/openvpi/runtime/bin/python
export SYMTHAEA_DEVICE=cuda

nix develop -c cargo run --bin symthaea-repl \
  --features "demo,audio,singing" -- \
  --voice --singing-backend diffsinger \
  --diffsinger-worker communication/worker/run_diffsinger_nixos.sh
```

If provisioning fails, the REPL logs the error and falls back to Kokoro. The
worker is offline, validates request identity, and caps output size/duration.

## Quality gate

Render a concealed comparison corpus:

```bash
nix develop -c cargo run --example muse_vocal_release_gate \
  --features "singing,voice-tts" -- \
  --output audio_output/muse_vocal_gate \
  --voice af_bella \
  --diffsinger-worker communication/worker/run_diffsinger_nixos.sh
```

This now renders 60 stratified cases spanning low/intimate, central/legato,
and high/luminous delivery. It includes melisma, attacks, sustained notes,
fricatives, affricates, nasals, diphthongs, consonant clusters, and phrase
releases. Every run receives fresh secure blinding salt. Pass `--blind-seed`
only to reproduce a diagnostic run.

It writes dry WAV stems, `blind_manifest.json`, a private salted backend key,
objective pitch/onset/voicing/duration/clipping/DC/RMS diagnostics, and a
ratings template. Copy each template row once per listener and provide a
stable pseudonymous `listener_id`. Fill ratings without viewing the backend
key, then rerun with `--ratings`. Release requires:

- median pitch error <= 25 cents;
- 95th percentile pitch error <= 60 cents;
- median onset error <= 30 ms;
- mean human naturalness >= 4/5;
- human lyric comprehension >= 95%;
- mean emotional fit, identity consistency, and artifact-free score >= 4/5;
- at least five listeners (configurable with `--minimum-listeners`);
- a complete listener-by-clip rating matrix.

Authorized human anchors can be included with `--reference-manifest`. The file
is a JSON array containing `case_id`, `audio`, `lyrics`, `category`, and
`challenge`. Reference files receive concealed IDs and participate in the
listening session, but their scores cannot inflate the synthetic release
thresholds.

ASR intelligibility remains available through `singing_intelligibility_gate`,
but is diagnostic rather than the final authority.

## Training a Symthaea identity

Do not train on scraped voices or pitch-shifted Kokoro output. Follow
`communication/singing/recording_plan.json`; capture dry mono PCM at 48 kHz,
preferably 24-bit, without pitch correction or dynamics processing. Create a
JSONL manifest using `communication/singing/example_record.json`, then run:

```bash
python communication/singing/prepare_dataset.py /dataset/root manifest.jsonl \
  --output build/authorized-training-index.jsonl

python communication/singing/validate_corpus_coverage.py \
  build/authorized-training-index.jsonl --profile pilot \
  --report build/pilot-coverage.json

python communication/singing/split_training_index.py \
  build/authorized-training-index.jsonl build/splits

python communication/singing/export_makediffsinger.py \
  build/splits/train.jsonl build/makediffsinger
```

The preparation gate verifies mono PCM audio, hashes, adult confirmation,
signed evidence, license identifiers, and explicit voice-identity/training
authorization. It fails the complete dataset on any invalid record. Coverage
must include all defined registers, dynamics, English phones, attacks,
breaths, vibrato, straight tone, portamento, staccato, legato, sustains, and
melisma. Validation is held out by complete recording session to prevent
near-duplicate leakage.

Use the exported raw layout with the locked MakeDiffSinger MFA forced-alignment
pipeline, then GAME for note/boundary assistance. Manually correct phonemes,
`AP` breaths, `SP` rests, slurs, and melisma before training. Copy
`diffsinger_training.template.yaml`, insert the verified split paths and
receipts, and only then enable training. Commercial release additionally
requires every record to authorize commercial use.

The ready-to-use producer checklist, detected 8 GiB training envelope, and
handoff boundary are in `communication/singing/PILOT_HANDOFF.md`.

## Iteration policy

Train the pilot before scheduling the production corpus. Evaluate it through
the same blind gate, classify failures by phoneme/register/technique, and use
those failures to select the next recording prompts. Do not grow the dataset
indiscriminately: corrected alignment and targeted coverage are generally more
valuable than additional mislabeled minutes.
