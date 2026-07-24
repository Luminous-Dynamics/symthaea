# Symthaea Computational Vocal Physiology

## Stable contracts

```text
HDC/CfC phonetic and expressive intention
  -> GestureFrame (symthaea.gesture-frame.v1)
  -> GesturePlanner
  -> IdentityAnatomy + IdentityPhysiology
  -> PhysicalTractFrame (symthaea.physical-tract-frame.v1)
  -> interchangeable acoustic solver
```

`GestureFrame` is the only cacheable motor interface. Its named, normalized
coordinates contain jaw, tongue, lips, velum, pharynx, larynx, glottis,
respiratory effort, typed place and typed manner. It contains no centimetres,
square centimetres, delay counts, tube sections or renderer parameters.

`IdentityAnatomy` owns persistent oral/nasal geometry and branch positions.
`IdentityPhysiology` owns slower tissue, pressure, fold-mass, hydration,
fatigue, jitter and shimmer biases. `GesturePlanner` applies velocity limits,
carry-over and 35 ms anticipatory coarticulation before producing a
`PhysicalTractFrame` in physical units.

The physical realization preserves narrow stop apertures while bounding every
adjacent oral-area ratio to 12:1. Closures therefore occupy a short tissue
region instead of appearing as a one-section 400–500:1 impedance cliff.

## Renderer contract

Three deterministic solvers coexist deliberately:

1. `Kl24BaselineV1` is the frozen regression oracle. Its exact x86_64/Rust
   1.96 fixture hashes and portable signal tolerances are recorded in
   `communication/singing/kl24_baseline_v1.fixture.json`. It explicitly reports
   `velum_supported: false`. Output-affecting changes require a new Vn type.
2. `BranchedWaveguideV2` is the production candidate. It uses 48 oral and 36
   nasal sections, identity-relative internal propagation rate, a passive
   three-port oral/nasal junction, wall losses, lip/nostril radiation,
   rate-invariant LF-family source filtering, pressure/constriction turbulence,
   typed place filtering and closure-release bursts. Every continuous source
   control is smoothed in physical time at the internal acoustic rate; burst
   attacks are finite rather than single-sample gates. Oral and nasal log-area
   motion also uses rate-invariant 12/18 ms time constants, and a moving
   turbulence jet is distributed continuously across adjacent tube sections.
3. `TransmissionLineReference` doubles oral/nasal resolution to 96/72 and is
   an offline error reference.

All solvers preserve state. V2 emits raw glottal flow, its derivative, the
anti-aliased periodic source, aspiration, turbulence, oral, nasal and final
stems. Reflection warnings are distinct from the numerical
stability bound: positive areas permit passive near-unit closure reflection
without clamping at the warning threshold.

V2 and the reference use a shared-kernel, block-aware Blackman-windowed sinc
converter. It sees the neighbouring motor frames, uses half-open sample timing,
and rejects energy above the output Nyquist frequency. This replaces the old
seven-tap filter and its 1 ms boundary crossfade; render continuity must now
come from the physical source and tract states rather than a masking edit.

## Evidence design

`vocal_physiology_gallery` creates three controlled experiments:

- renderer isolation: same gesture, anatomy and physiology; different solver;
- identity isolation: same gesture, physiology and solver; different anatomy;
- planner isolation: same target, anatomy and solver; different gesture rates.

It writes raw dry WAVs, -20 dBFS RMS listening copies, intention-relative
`gestures.v1.json`, identity-relative `PhysicalTractFrame` data, ablation stems,
raw reflection events/episodes and a compact isolation report. The expensive
HDC/CfC stage can be bypassed with `--reuse-gestures` without caching rendered
geometry.

The first v2 evidence run achieves zero contextual clicks for the three
anatomies. Identity correlations are materially separated, unlike the old
shared-area gallery. Raw v2/reference correlation was about 0.94 before the
calibrated manifold and is about 0.53 afterward. Both solvers still pass their
independent pitch, stability, cleanliness, alias and nasal gates; the lower
phase-sensitive correlation is retained as evidence that formant/transfer-
function comparison must replace raw waveform correlation as the decisive
cross-resolution metric.

## Perceptual calibration campaign

Renderer cleanliness is frozen as a regression milestone. Naturalness work now
begins with static excitation and vowels rather than complete songs.

`LfGlottalSource` is independently renderable and exposes seven coherent
regimes: modal, breathy, pressed, head, falsetto, choral and belt. The source
report measures H1-H2, H1-H3, harmonic spectral tilt, normalized amplitude
quotient, a periodicity proxy and aspiration-to-periodic balance. A separate
sanity gate rejects degenerate excitation but does not choose an aesthetic
winner.

`VowelAnchorDecoder` replaces broad Gaussian deformation for the five cardinal
vowel calibration cells. It maps identity-relative, anatomically bounded anchor
area functions into `PhysicalTractFrame`; rounded vowels also extend effective
tract length to represent lip protrusion. The static impulse suite measures
target-region F1-F4 and bandwidths. Velvet anchor-v1 currently measures:

| Vowel | F1 | F2 | F3 | F4 | Mean relative error |
|---|---:|---:|---:|---:|---:|
| /a/ | 800 | 1240 | 2620 | 3700 | 3.5% |
| /e/ | 410 | 2010 | 2680 | 3780 | 8.7% |
| /i/ | 250 | 2190 | 3070 | 3740 | 7.2% |
| /o/ | 510 | 990 | 2420 | 3690 | 6.2% |
| /u/ | 330 | 870 | 2300 | 3640 | 9.1% |

All five pass their committed F1-F4 target windows. This is an acoustic gate,
not a claim of human naturalness; promotion still requires pitch/loudness
matched, blinded listening against a licensed human reference.

The production `GesturePlanner` now decodes open-tract gestures through a
continuous log-area interpolation of these five anchors. Procedural Gaussian
deformation remains available behind `use_calibrated_vowel_manifold = false`
only for controlled ablation; categorical consonant closures keep their typed
physical path.

`vowel_calibration_gallery` exports source-only stems, impulse responses,
twenty sustained-vowel cells (five vowels at 110/165/220/330 Hz), and the
source-by-tract 2x2 `/a/` ablation. It also includes a conventional formant
synthesizer control and explicitly leaves the real-human cell empty when no
locally licensed recording is available.

Raw physical renders may sit between -45 and -8 dBFS without failing objective
synthesis quality. Loudness is evaluated on the separately normalized listening
copy; it is not conflated with pitch, stability or render cleanliness.

## Passivity and diagnostics

The domain tests verify:

- positive and bounded physical geometry;
- anatomy-dependent geometry under identical gesture input;
- velum-dependent coupling area;
- larynx-dependent effective propagation length;
- closure preservation with bounded neighbouring impedance ratios;
- audible nasal-branch output;
- finite deterministic synthesis;
- source-free energy decay for fixed tracts;
- source-free decay across deterministic randomized oral/nasal shapes and
  velopharyngeal openings;
- reflection sample-hit grouping into closure-aware episodes.
- rejection of out-of-band energy during 4x downsampling;
- continuous-tone preservation across motor-frame boundaries;
- smoothing of abrupt motor targets before acoustic excitation;
- physiology-conditioned voiced, turbulent and nasal acoustic metrics.

Energy measured while geometry moves is reported separately: articulator motion
can exchange energy with the acoustic field and is not the static passivity
test.

## Remaining production work

V2 is an implemented, testable production candidate—not a completed human
vocal model. Remaining additions are subglottal coupling, sinus side branches,
measured viscothermal/radiation calibration, a bounded-memory streaming form of
the current band-limited converter, learned gesture priors, place-conditioned
consonant spectral gates, formant/antiformant tracking and blinded
MUSHRA/intelligibility studies.

The residual-detail model remains disabled until these deterministic solvers
establish a reproducible perceptual residual gap.
