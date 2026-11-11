# SYNTH-VOCAL-001 — synthetic vocal acoustic plant

This standalone Rust tool generates a small, deterministic, permission-free acoustic fixture suite for Muse/Symthaea.

It is a **synthetic signal-processing test plant**, not a model of a particular person and not a claim of human vocal naturalness.

## What it exercises

- Explicit synthetic formant-frame parameters: F0, F1–F3, bandwidths, energy, vibrato and duration.
- The existing symthaea_vocal_tract::speech::vocoder::synthesize legacy source-filter renderer.
- Stable segment identities and bounded pitch/formant/energy edit proposals.
- Independent F0 measurement from PCM using normalized autocorrelation and parabolic lag interpolation.
- Exact sample-level protection outside a declared segment-plus-seam dependency cone.
- Float WAV output, parameter and PCM SHA-256 digests, and a machine-readable run manifest.

The renderer is intentionally identified as legacy_formant_source_filter. The vocal-tract source itself distinguishes this lightweight legacy formant path from the recovered Series 23 physical articulatory renderer. No physical-renderer capability or naturalness claim is made here.

## Run

From the repository root:

~~~bash
cargo run --manifest-path tools/synthetic-vocal-plant/Cargo.toml -- /tmp/synthetic-vocal-plant
~~~

The command writes:

- baseline.wav
- no_op.wav
- pitch_up_100c.wav
- wrong_direction_pitch_down_100c.wav
- formant_scale_110pct.wav
- collateral_corruption.wav
- manifest.json
- manifest.sha256

All WAVs are mono 24 kHz float32. The manifest records the renderer backend, sample/frame rates, crossfade guard, exact segment parameters, edit target, expected response, independent F0 result, PCM digest, and locality measurement.

SYNTH_VOCAL_PLANT_SOURCE_REVISION may be set by a reproducible build pipeline to bind the manifest to a source commit. If it is absent, the manifest explicitly says unknown-at-build; it does not fabricate a revision.

## Synthetic cases

Five generated vowel-like proxy segments form a short musical phrase. They are abstract control fixtures, not IPA annotations, a pronunciation lexicon, or a South African language ground-truth set.

| Case | Purpose | Expected result |
|---|---|---|
| baseline | Frozen source | Reference PCM |
| no-op | Negative control | Byte-identical PCM |
| pitch +100 cents | Positive control | Target F0 rises; protected PCM remains exact |
| pitch −100 cents | Wrong-direction control | Target F0 falls; must not satisfy a requested rise |
| formant scale 1.10 | Orthogonal-control probe | Formants change while declared F0 stays fixed |
| collateral corruption | Red-team control | A second, protected segment changes and locality fails |

The manifest also records a rejected unsupported-control proposal. Silence, invalid windows and unsupported parameters are treated as indeterminate/rejected inputs; none may be silently replaced by a guessed control.

Each segment is rendered independently with deterministic renderer state. Adjacent blocks are joined using a fixed 5 ms overlap. The target dependency cone includes the block and its seam overlap. Exact preservation is assessed outside that declared cone, not by pretending that crossfade samples are unaffected.

## Independence and limits

The ground-truth control parameters come from the fixture manifest. The F0 estimator receives only PCM; it does not read the input frame F0. The sample-locality check compares raw float sample bit patterns. These two measurements answer different questions:

- Did the target region's measured periodicity move?
- Did any sample outside the allowed edit/dependency cone change?

A passing result establishes only deterministic behavior of this generated fixture and the named legacy backend. It does **not** establish:
- naturalness or perceived singing quality;
- human intelligibility, native-language correctness or phonological validity;
- preservation of a human singer's timbre or identity;
- real-performer robustness;
- performance on MMAE, LyricEditBench or another external benchmark;
- operation of the Series 23 physical articulatory backend.

The next integration step is to export the measurements using the exact metric IDs, units, calibration receipts and provenance fields expected by VCRB-001, VIA-001, VCCR-001, VMEA-001 and VELB-001. It must not create a duplicate evaluator or allow the fixture generator to self-certify the analyzer.
