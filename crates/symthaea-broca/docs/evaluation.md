# Broca Evaluation

Broca quality is tracked with a small canonical JSONL suite plus larger
domain-specific eval files.

## Canonical Suite

`tests/fixtures/eval-canonical-v1.jsonl` contains 60 stable, hand-curated cases with:

- `category`: regression slice such as `intent`, `epistemic`, `code`, or `creative`
- `id`: stable case identifier
- `channels`: current 43-channel `ThoughtChannels` vector
- `target_text`: teacher-forced perplexity target
- `tags`: optional dashboard tags

Run the raw-vs-gated quality suite:

```bash
cargo run -p symthaea-broca --bin broca-eval -- \
  --checkpoint crates/symthaea-broca/data/broca-cfc-round7.best.bin \
  --canonical-eval crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl \
  --json-out /tmp/broca-quality.json
```

Add `--dump-generations /tmp/broca-generations.jsonl` to write one JSONL
record per evaluated canonical case with target text, raw output, gated output,
token IDs, repeated-token counts, coherence dynamics, hallucination flags, and
per-step decoder top-k logits/probabilities with entropy.
Use this before changing model size when quality metrics show low coherence.

The JSON report includes:

- raw generation metrics with `bypass_gating=true`
- gated generation metrics with the full gating stack enabled
- deltas for perplexity, English ratio, coherence, hallucination rate, and diversity
- absolute raw/gated values inside each delta object for CI dashboards
- target token overlap, a dependency-free lexical proxy for generated/target alignment
- moral refusal rate, most useful on the canonical `moral` category
- structured output validity for Rust, JSON, and action-shaped canonical cases
- per-category raw/gated breakdowns
- run metadata: backend, eval lane, checkpoint hash, git commit, feature set,
  training recipe, pair count, epochs, BPTT window, negative samples, learning
  rate, network LR scale, network layers, and neurons per layer when emitted by
  automation scripts

## Evaluation Lanes

Broca automation has two quality lanes:

- `fast`: teacher-forced canonical perplexity only. This is the default for
  training regression checks and every-checkpoint gating.
- `full`: generation-heavy canonical quality, including coherence, diversity,
  target overlap, refusal rate, and hallucination probes. Use this for
  promotion candidates and benchmark snapshots.

Select a lane with `BROCA_GATE_EVAL_LANE=fast|full` or
`BROCA_SMOKE_EVAL_LANE=fast|full`.

## Training Smoke

Before a longer training run, use the fast smoke script to verify the local
training path, checkpoint save, and checkpoint reload:

```bash
scripts/broca_train_smoke.sh
```

The script runs through `nix develop` by default and writes artifacts under
`/tmp/symthaea-broca-smoke`. Useful knobs:

- `BROCA_SMOKE_PAIRS=16` changes the generated training subset size
- `BROCA_SMOKE_EPOCHS=2` changes the short training duration
- `BROCA_SMOKE_BASELINE=1` also saves and reloads an epoch-0 baseline checkpoint
- `BROCA_SMOKE_CANONICAL=1` runs the slower canonical raw-vs-gated suite
- `BROCA_SMOKE_EVAL_LANE=fast|full` selects teacher-forced or generation-heavy eval
- `BROCA_SMOKE_BACKEND=auto|gpu|cpu` selects GPU when available by default
- `BROCA_SMOKE_TARGET_DIR=/tmp/symthaea-broca-gpu-target` keeps GPU builds out
  of the shared workspace target directory
- `BROCA_SMOKE_USE_NIX=0` skips `nix develop` when already inside the right shell

Smoke checkpoints use `broca-train --no-save-adam`, which keeps model weights
but omits optimizer resume state. Use normal training checkpoints when you need
to resume with Adam momentum.

## Train And Gate

Use the Tier 0 train-and-gate wrapper when a checkpoint should be treated as a
candidate artifact:

```bash
scripts/broca_train_and_gate.sh
```

It trains, writes a checkpoint, runs canonical eval, and exits non-zero if any
configured gate fails. Thresholds are controlled through environment variables,
for example:

```bash
BROCA_GATE_MIN_MORAL_REFUSAL_RATE=0.50 \
BROCA_GATE_MAX_COHERENCE_REGRESSION=0.10 \
BROCA_GATE_MIN_STRUCTURED_OUTPUT_VALIDITY_RATE=0.50 \
scripts/broca_train_and_gate.sh
```

`BROCA_GATE_EVAL_LANE=fast` is the default and writes `quality-fast.json`.
Set `BROCA_GATE_EVAL_LANE=full` for promotion runs that should write
`quality-full.json` with generation-heavy metrics.
`BROCA_GATE_TEACHER_FORCED_ONLY=1` is rejected with the full lane to avoid
mislabeled reports.
Set `BROCA_GATE_REPORT_ONLY=1` for calibration runs that should write the
quality JSON without failing the process on current thresholds.
GPU runs default `BROCA_GATE_TARGET_DIR` to `/tmp/symthaea-broca-gpu-target` so
training does not wait behind unrelated workspace builds.

Training is controlled by `BROCA_GATE_RECIPE`:

- `smoke`: the default, a tiny correctness and wiring check.
- `baseline-v1-small`: the first useful GPU baseline. It modestly increases
  examples, BPTT length, negative samples, and CfC capacity.
- `baseline-v1-binding`: a targeted experiment for flat-logit thought binding
  collapse. It keeps small-model capacity but increases negative samples,
  BPTT length, network LR scale, and enables light coherence alignment,
  contrastive, label-smoothing, and thought-to-logit auxiliary binding losses.
- `baseline-v1-medium`: a promotion-candidate recipe for full canonical eval
  and benchmark snapshots.
- `custom`: starts from smoke defaults but expects explicit overrides.

Every recipe can be overridden with:

- `BROCA_GATE_PAIRS`
- `BROCA_GATE_EPOCHS`
- `BROCA_GATE_EVAL_LIMIT`
- `BROCA_GATE_MAX_GEN_TOKENS`
- `BROCA_GATE_BPTT_WINDOW`
- `BROCA_GATE_NEGATIVE_SAMPLES`
- `BROCA_GATE_LR`
- `BROCA_GATE_NETWORK_LR_SCALE`
- `BROCA_GATE_NETWORK_LAYERS`
- `BROCA_GATE_NEURONS_PER_LAYER`
- `BROCA_GATE_COHERENCE_ALIGNMENT`
- `BROCA_GATE_ALIGNMENT_START`
- `BROCA_GATE_CONTRASTIVE`
- `BROCA_GATE_CONTRASTIVE_MARGIN`
- `BROCA_GATE_SCHEDULED_SAMPLING`
- `BROCA_GATE_LABEL_SMOOTHING`
- `BROCA_GATE_THOUGHT_LOGIT_AUX`
- `BROCA_GATE_MERGE_BIAS`

Prefer `baseline-v1-small` before increasing model size further. Only increase
neurons or layers when train and validation curves show under-capacity rather
than data or decoding limits. Widen neurons first, add layers second, and avoid
changing HDC/channel dimensions until baseline reports are stable because that
breaks checkpoint comparability and may require schema migrations.
If generation dumps show near-uniform logits with identical greedy outputs
across intents, run `baseline-v1-binding` before `baseline-v1-medium`.

`BROCA_GATE_BACKEND=auto` is the default. On machines where `nvidia-smi` works,
the script compiles with `--features gpu` and runs through `nix develop .#broca-gpu`;
otherwise it falls back to the portable CPU build. Use `BROCA_GATE_BACKEND=cpu`
or `BROCA_GATE_BACKEND=gpu` to force either path. GPU mode sets
`CUDA_COMPUTE_CAP=75` by default for the RTX 2070-class training host; override with
`BROCA_GATE_CUDA_COMPUTE_CAP` when running on different NVIDIA hardware.

Broca v0.1 treats structured readout as the grounded product path. Direct CfC/HDC
token generation and Mamba translation remain measurable decoder paths, but they
should not be promoted as the source of truth until their drift, hallucination,
and collapse gates pass. Enable decoder comparison with:

```bash
BROCA_GATE_DECODER_AB=1 \
BROCA_GATE_DECODER_AB_DECODERS=structured,direct \
BROCA_GATE_DECODER_AB_FAIL_ON_GATE=1 \
BROCA_GATE_DECODER_AB_MIN_STRUCTURED_VALIDITY=0.85 \
BROCA_GATE_DECODER_AB_MIN_STRUCTURED_TRANSLATION_VALIDITY=0.75 \
BROCA_GATE_DECODER_AB_MIN_STRUCTURED_TRANSLATION_GROUNDING_RATE=1.0 \
BROCA_GATE_DECODER_AB_MAX_STRUCTURED_TRANSLATION_DRIFT=0.25 \
scripts/broca_train_and_gate.sh
```

The structured translator is deterministic. Its report includes the role/filler
grounding surface, prose text, grounding preservation, hallucination markers,
translation validity, and a simple semantic-drift proxy. Use Mamba as an
optional humanizer outside that grounded path, not as the cognitive truth layer.

Add threshold flags to turn the canonical suite into a CI gate:

```bash
cargo run -p symthaea-broca --bin broca-eval -- \
  --checkpoint crates/symthaea-broca/data/broca-cfc-round7.best.bin \
  --canonical-eval crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl \
  --max-gated-perplexity 5000 \
  --min-gated-coherence 0.05 \
  --min-moral-refusal-rate 0.50 \
  --min-structured-output-validity-rate 0.50 \
  --max-coherence-regression 0.10 \
  --max-target-overlap-regression 0.10
```

Use `--allow-checkpoint-recovery` only for known legacy artifacts that predate
checkpoint compatibility metadata.

Use `--eval PATH` for the older human-readable held-out dataset report, or add
`--json-out PATH` to emit machine-readable `EvalResult` JSON.

## Compatibility

Current Broca uses 43 channels, or 47 with the `therapeutic` feature. Legacy
20-channel and 24-channel datasets are still accepted by `TrainingPair` and are
padded through `ThoughtChannels::default()`, but new canonical regression data
should use the full 43-channel schema.
