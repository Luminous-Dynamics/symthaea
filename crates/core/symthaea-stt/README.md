# symthaea-stt — HDC/LTC/CfC Acoustic Perception

Native hyperdimensional acoustic encoding and liquid-neural temporal dynamics for
Symthaea. **Read this before assuming "stt" means usable transcription — it does not.**

## Honest capability statement (verified 2026-07-15)

| Capability | Status | Evidence |
|---|---|---|
| Real-time acoustic → HV encoding | **Works** (RTF 0.07–0.09, ~11–14× faster than real time) | `eval_report.json` |
| Word-level transcription (ASR) | **Not viable** | Committed `eval_report.json`: WER 473.4%, PER 98.9%, **0 words correct** (100 LibriSpeech utts); `confusion_analysis.json`: WER 316.4% at 500 utts |
| Frame-level phoneme accuracy | Best-ever **19%** (reservoir + ridge), 13.5% (LTC + BPTT) | internal architecture postmortem, Mar 2026 |
| `HdcLtcUnifiedNeuron` classification training | **Does not learn** (similarity stays ~0.9999 through training) | test `#[ignore]`d with diagnosis in `phoneme_hdc_ltc.rs` (2026-07-15, commit `9c7e85f063`) |
| Bioacoustics (whale coda, sleep staging) | Research prototypes, synthetic/heuristic labels | `docs/communication-capability-inventory.md` |

Five recognition architectures were tried (prototype-HDC, LTC+BPTT, reservoir+ridge,
liquid-HDC, unified-neuron). None reached usable word accuracy. For transcription,
Symthaea delegates to Whisper via the `symthaea-communication` provider contract
(`communication/worker/whisper_worker.py`; measured en WER 4.8% normalized on FLEURS).

## Cetacean acoustic analysis: important limit

Dolphin and whale support remains research-only. `whale.rs` handles sperm-whale coda timing/ICI patterns; `cetacean_classifier.rs` extracts frame-level transient, tonal and modulation features. Neither currently establishes the meaning of a vocalization.

The legacy classifier defaults to 44.1 kHz and a 20 kHz analysis upper bound. That is not an ultrasonic dolphin-echolocation configuration. Use the guarded `CetaceanConfig::validate`, `CetaceanClassifier::try_new` and `try_classify_audio` APIs with an explicit source sample rate. The checked file path refuses downsampling through the current linear interpolator. The guarded `WhaleConfig::validate` / `WhaleSentinel::try_with_config` / `try_process_buffer` APIs also require the source clock to match the configured rate.

The `symthaea-communication::bioacoustic_corpus` crate now provides a metadata-only manifest validator for rights references, stable recording hashes, source lineage, annotations and leakage controls. See [the avian research contract](../../../docs/research/AVIAN_COMMUNICATION_RESEARCH_CONTRACT_V1.md) and [the cetacean research contract](../../../docs/research/CETACEAN_COMMUNICATION_RESEARCH_CONTRACT_V1.md) for corpus-specific admission rules. These checks validate declared metadata, not the underlying audio bytes or legal correctness of a rights decision.

See the cetacean research contract for datasets, licensing caveats, per-task benchmarks and evidence boundaries. The new guard tests are authored but must still be run on a pinned Rust toolchain.

## What this crate is actually for

The niches where it is real and useful — everything Whisper *discards*:

1. **Acoustic-texture HVs for perception**: continuous 16,384-D hypervector "hearing"
   blended into the cognitive loop's perception phase (`audio_hdc_encoder`,
   `streaming::StreamProcessor` → `src/perception/audio_stream.rs` in the main crate).
2. **Prosody / paralinguistics**: timing, energy, pitch dynamics of speech.
3. **Acoustic unit discovery**: `discovery.rs` + `communication_adapter.rs`
   (capability honestly capped at `Structure` level — no reference/intent claims).
4. **Bioacoustics research**: `whale.rs`, `cetacean_*.rs`, `sleep_sentinel.rs`
   (deliberate scope per `Cargo.toml` description; not release-gated).

Think of it as `symthaea-acoustics`: an ear for *how things sound*, not *what words
were said*.

## Pipeline map

```
audio (WAV/FLAC/mic) → mel features (audio.rs)
  → LTC/liquid/reservoir dynamics (ltc.rs, liquid.rs, crystal_reservoir.rs)
  → HV16 encoding (hdc.rs; 2,048-bit working dim → 16,384-D core)
  → [research-only: phoneme decode (phoneme.rs) → lexicon/LM beam search (lm.rs)]
```

## Running the evals

```bash
cargo run -p symthaea-stt --bin symthaea_eval --release      # honest: prints BAR EXAM: FAILED
cargo run -p symthaea-stt --bin eval_reservoir --release     # best frame-level architecture
```

The committed `eval_report.json` / `confusion_analysis.json` in this directory are the
authoritative last-measured numbers. Do not cite the "<20% WER" target as a capability.
