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


## Independent acoustic observations — SYNTH-OBS-001

The CLI also writes a second, independently versioned observation bundle:

- `observations.jsonl`: per-case/per-segment F0, F1–F3, relative energy and voicing-coverage measurements.
- `calibration.json`: deterministic no-op repeatability report from a separate calibration parameter family.
- `observer-manifest.json` and `observer-manifest.sha256`: analyzer identities/versions, extraction configuration digest, PCM/observation/calibration artifact digests, synthetic-only flags and split identities.

Two separate PCM-only F0 paths are retained:

1. `synthetic-vocal-plant.autocorrelation-f0` — this tool's normalized-autocorrelation implementation.
2. `symthaea_vocal_tract::formant_extraction::extract_formant_frames` — the existing vocal-tract crate's LPC/autocorrelation extraction path, which also supplies F1–F3, relative energy and voicing.

The observers are not averaged. Their cents disagreement is retained as a diagnostic, with a 50-cent alert threshold. If one or both observers cannot produce a valid F0, the report records that explicitly. An absent F0 or formant value is `INDETERMINATE`, not zero and not a pass.

All metric observations bind a stable metric ID, unit, analyzer ID/version, exact PCM digest, sample window and eligible frame count. The VCRB adapter exports only observations marked `Observed`; it does not duplicate VCRB's control-response evaluation logic.

Calibration uses a separately named and parameterized synthetic family. The report records exact PCM repeatability and per-metric maximum absolute repeat deltas. This deterministic calibration is a measurement-pipeline check; it is not statistical calibration for stochastic neural renderers.


## Measured response profile — SYNTH-CAL-001

The CLI also fits a backend-specific local response model from actual generated PCM interventions and writes:

- `response-profile.json`: measured metric × control sensitivity coefficients, conservative coefficient uncertainty, held-out prediction checks, split identities and explicit promotability.
- `response-calibration-pairs.json`: every baseline/intervention pair, control ID/unit/delta, target event, PCM digests and per-metric measured deltas.
- `vcca-response-profile.json`: a narrow field-compatible adapter for the existing VCCA-001 profile inputs (control bounds, metric residuals and sensitivity cells).
- SHA-256 sidecars for the profile and calibration-pair artifacts.

The initial control dimensions are pitch shift in cents, formant scaling delta, and energy scaling delta. Each is probed in both positive and negative directions at multiple magnitudes, across two distinct synthetic calibration parameter families.

For each metric/control cell, the compiler fits a through-origin least-squares slope. It derives coefficient uncertainty from the worst observed calibration residual divided by the corresponding absolute control input. Per-metric residual uncertainty comes from the independent no-op repeatability report. These are conservative local engineering bounds for this deterministic backend, not confidence intervals.

A separate confirmatory phrase is then used for one unseen probe per control. Every measured metric delta is compared with the fitted prediction interval. The profile is marked promotable only when all required cells exist, the no-op and split identities qualify, and every held-out prediction is covered.

The fitted matrix may still be **not promotable**. That is a valid result: it exposes model inadequacy instead of weakening a tolerance or forcing a correction. The profile is local to the exact legacy renderer, sample rate and analyzer versions; it must not be reused for another backend or extrapolated outside its measured trust radii.

This module prepares VCCA's inputs but does not duplicate its candidate-ranking algorithm. Every resulting VCCA action remains PROPOSAL_ONLY and requires capability admission plus fresh post-render measurement.


The compiler separately checks whether each advertised control produced a positive response in its designated target metric. An ignored or wrong-direction control is marked unsupported in the VCCA adapter, with zero-only action bounds, and prevents profile promotion. A complete-looking matrix cannot make an ineffective renderer control appear supported.
