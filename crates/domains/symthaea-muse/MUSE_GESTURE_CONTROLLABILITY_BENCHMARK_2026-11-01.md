# Muse gesture controllability benchmark — 2026-11-01

Status: **design and first executable receipt contract**. The accompanying gesture_controllability module verifies paired feature-measurement receipts. It does not yet launch renderers or grant capability.

## Research question

Does a renderer actually consume a declared instrument gesture, produce its intended effect, and preserve unrelated behavior under a controlled intervention?

A note-on event with a velocity and patch name is insufficient to answer this. A gesture may be syntactically accepted but ignored, quantized, mapped to the wrong note, applied to the wrong model, or approximated without disclosure.

The experiment is a paired counterfactual:

identical score + identical source/model + identical renderer + identical analyzer + one gesture change

The two renders must bind the same immutable score, producer-assigned note-identity map, source/asset manifest, renderer ID/build, and feature-extractor ID/build. Any mismatch makes the pair incomparable.

## Data contract

GestureCounterfactualReceiptV1 carries:

- score SHA-256 and score-note identity-map SHA-256;
- source/model/asset manifest digest;
- exact renderer ID and build digest;
- exact feature extractor ID and build digest;
- stable event ID, instrument ID, source model ID, target note/part and rational beat position;
- baseline and intervention values for one typed gesture parameter;
- both rendered output SHA-256 values;
- baseline and intervention feature measurements with stable feature IDs and units;
- predeclared expected feature effects and side-effect budgets;
- a domain-separated receipt digest.

The receipt digest is a tamper-evident content commitment, **not** a signature, independent measurement, source-rights proof or qualification attestation. An independent verifier must establish the provenance of the render and analysis artifacts. Capability promotion remains with the pinned signed exact-cell verifier in #9586.

## Experimental protocol

### A. Freeze the comparison

Before rendering, record the exact score digest, note-identity-map digest, source model and asset manifest, renderer build, feature-extractor build, target event, gesture parameter, baseline value, intervention value and all held-constant controls.

Reject a pair if a source asset, renderer build, feature extractor, score, target, unrelated control or instrument configuration differs between baseline and intervention.

### B. Calibrate repeatability

For the baseline condition, run repeated renders under the pinned environment. Measure the distribution of every proposed feature and document nondeterminism, numerical noise, device/driver variability and sample-rate conversion.

Choose the minimum expected effect and side-effect budgets **before** looking at the intervention result. The expected minimum should exceed baseline repeatability noise and reflect a domain-specific meaningful change. Use an empirical interval or bootstrap when distributional assumptions are not justified; do not assume a Gaussian three-sigma rule by default.

If baseline repeatability is unstable or not measured, the experiment is diagnostic only and cannot promote a Native mapping.

### C. Run one-parameter interventions

Change one gesture parameter at one event only. The parameter values must be type-correct, unit-correct and within the declared model domain. Repeat at low, middle, high and boundary values for the selected model/profile, and over independently seeded phrases where the renderer is stochastic.

Examples of falsifiable hypotheses:

- violin bow force changes a predeclared energy/spectral feature without exceeding the onset/timing side-effect budget;
- flute breath pressure changes an energy/noise feature while preserving the intended pitch trajectory;
- oud pluck position changes the expected spectral balance while leaving note identity and onset timing intact;
- vibraphone damper/pedal state changes decay/tail behavior;
- trombone slide position produces continuous pitch movement within its predeclared pitch tolerance;
- accordion bellows pressure changes dynamic response without cross-note leakage;
- ChoirAah versus ChoirOoh changes the declared vowel/formant feature;
- SawLead glide changes the pitch trajectory, not merely the output gain.

These are test hypotheses. They are not claims that the current renderer supports these gestures.

### D. Verify intended and unintended effects

For each expected feature, compute delta as intervention_feature minus baseline_feature.

Require the predeclared direction and minimum absolute delta. For each side-effect budget, require the absolute delta to remain within its predeclared maximum. Measurements must use identical feature definitions, units, analyzer build, windowing and normalization.

A renderer that produces different audio but fails the intended feature effect is not Native for that gesture. A renderer that produces the expected feature but violates a protected side-effect budget fails the current profile. A renderer that ignores the gesture fails the test.

### E. Add negative controls

Every suite must contain:

- no-op gesture change (same value);
- unsupported parameter;
- wrong instrument or source-model identity;
- wrong note/part ID;
- wrong score digest or note-map digest;
- stale renderer or analyzer build;
- invalid units and out-of-domain values;
- missing/corrupt source assets;
- a renderer mapping that accepts the request but intentionally ignores the parameter;
- an unrelated note used as a leakage sentinel;
- identical output bytes paired with deliberately altered measurement metadata.

The verifier must reject mismatched identity/build inputs, ignored controls, cross-note leakage and fabricated measurement changes.

## Benchmark dashboard

Keep these metrics separate by instrument, source model, gesture, renderer/build and held-out source. Do not collapse them into a single "realism" score.

| Metric | Definition | Initial target |
|---|---|---:|
| Decision receipt coverage | Requested gesture events with exactly one explicit Native/Approximation/Unsupported/Unknown/InvalidEvidence decision | 100% |
| Native controllability pass rate | Native mappings passing their preregistered expected-effect and side-effect tests | 100% of advertised Native cells |
| Ignored-gesture false acceptance | Renderer reports Native but paired intervention fails its intended effect | 0 |
| Side-effect breach rate | Advertised Native mappings exceeding a declared side-effect budget | 0 |
| Cross-note leakage | A gesture changes an unrelated note's protected feature beyond its declared budget | 0 |
| Identity/build mismatch acceptance | Stale score, note map, source, renderer or analyzer accepted as the intended comparison | 0 |
| Unknown/unsupported silent promotion | Unknown or Unsupported cells emitted as Native | 0 |
| Render integrity failures | NaN/Inf, invalid format, unexplained silence, malformed tail or unintended clipping in a successful render | 0 |
| Reproducibility | Same-input/same-build repeat-render identity or predeclared feature tolerances | Measured for 100% of qualification runs |
| OOD generalization | Per-instrument native-control success on held-out source models, assets, tunings and renderer builds | Report per stratum; no pooled-only score |
| Runtime | p50/p95 render latency, CPU, peak memory and polyphony | Establish baseline, then improve without weakening fidelity gates |

Report numerator, denominator, exclusions and confidence intervals for every rate. Zero observed failures is not proof of zero population risk; retain the challenge-set size and interval.

## Industry comparison and fair targets

### Sample engines: Kontakt and Spitfire

Kontakt documents group activation by keyswitch/controller and sequential or random round-robin cycling. Spitfire Chamber Strings advertises 244 articulations, multiple dynamic layers and round robins across a recorded 16-player section. This is the baseline for articulation switching, velocity-layer selection, repetition variety and release behavior.

Melothaea should not compare raw patch/articulation counts to a commercial library. Compare effective, tested capability cells: exact sample zone/technique selected, actual audio rendered, intended articulation response, round-robin repeat behavior, release-lane integrity and explicit fallback receipts. Target 100% receipt coverage and zero false Native claims before attempting library-depth parity.

### Physical modeling: Pianoteq

Pianoteq documents continuous timbral variation with velocity, sympathetic resonance, repedaling and progressive pedals, microtuning, note-per-note physical parameter editing and instrument morphing/layering. These establish a much richer piano benchmark than note and velocity alone.

A credible Melothaea piano parity suite must independently test velocity-to-spectrum response, pedal depth/transition, sympathetic resonance, repeated-note interaction, note-off and tuning. Do not claim piano-model parity from matching a static partial spectrum.

### Expressive modeling: SWAM

Audio Modeling explicitly states that note-on velocity alone is insufficient for SWAM expression because expression changes continuously through time. This supports a gesture-stream benchmark, not the claim that Muse already has comparable modeled instruments.

Compare continuous controller resolution, latency/jitter, expression envelope tracking, gesture-to-feature controllability and held-out phrase performance. Make all comparisons at equivalent controller and audio rates.

### Interoperability: MPE and MIDI 2.0

MIDI 2.0 extends rather than replaces MIDI 1.0. MIDI-CI allows capable bidirectional devices to discover and negotiate supported Profiles, Property Exchange and Process Inquiry. MPE provides an existing per-note expression pathway over MIDI 1.0 channel allocation.

Target explicit adapters:
- retain richer internal gesture values and score timing;
- expose each renderer/device's actual profile and parameter capabilities;
- record quantization, channel allocation, controller collision and lost semantics on export;
- never mark a down-converted gesture as fully preserved without a measured round-trip result.

### Recognition/generalization: MIREX

MIREX instrument recognition is a clip-level multi-label classification benchmark, not an instrument-rendering benchmark. In the 2026 results, the SSR system reports macro-F1 0.801 on public OpenMIC-2018 and 0.488 on the hidden MedleyDB set. The hidden set has fewer clips, ambiguous stem-derived labels and a different mixture population, so this is an OOD warning rather than a causal estimate of dataset shift alone.

Melothaea should hold out source recordings, instrument models, microphone positions, tunings, sample-library revisions and renderer builds. Report per-class and per-stratum results. Instrument recognition can be a diagnostic for emitted timbre, never the authority that a requested instrument/technique was actually realized.

## Test lanes

1. **Fast deterministic CI:** synthetic feature fixtures, schema validation, digest mutation, exact-target checks, no-op and ignored-controller negative controls.
2. **Offline source-backed renderer suite:** pinned local sample banks/procedural builds, real content hashes, render receipts, pitch/velocity/articulation boundaries and repeated-note trials.
3. **Physical instrument/measurement suite:** calibrated excitation/sensor profiles, as-built instrument identity, environmental conditions, raw capture digests and uncertainty. A simulation or procedural render is not a physical measurement.
4. **Blinded listening:** randomized, level-matched comparison with declared references, participant/task scope and uncertainty. Acoustic metrics cannot grant artistic preference.

## Initial pilot order

Use distinct excitation regimes rather than choosing instruments only because they are easy to implement:

1. SawLead — known electronic parameter controls as an adapter/measurement sanity check.
2. Piano — velocity, pedal, decay and repeated-note controls.
3. Violin — bow/continuous expression semantics.
4. Flute — breath/tonguing.
5. Plucked strings (Oud or AcousticGuitar) — pluck position/force and pitch movement.
6. Vibraphone — mallet, roll and damper behavior.
7. Trombone — continuous slide.
8. Accordion — bellows and register controls.
9. ChoirAah/ChoirOoh — separate vowel identities and formant response.

Passing one instrument does not qualify another member of its family. Passing a single gesture does not qualify all parameter values or all renderers.

## Sources

- [Native Instruments Kontakt manual — group start conditions, keyswitches and round robins](https://docs.native-instruments.com/ni-tech-manuals/kontakt-manual/en/classic-view)
- [Spitfire Chamber Strings — articulation, dynamic-layer and round-robin coverage](https://www.spitfireaudio.com/products/spitfire-chamber-strings)
- [Modartt Pianoteq manual — physical model, velocity, resonance, pedals and note edit](https://www.modartt.com/user_manual?product=pianoteq)
- [Audio Modeling — SWAM note-on velocity versus continuous Expression](https://kb.audiomodeling.com/support/solutions/articles/206000050973-is-it-possible-to-control-swam-instruments-using-only-note-on-velocity-)
- [MIDI 2.0 and MIDI-CI](https://midi.org/midi-2-0), [MIDI-CI specification](https://midi.org/midi-ci-specification)
- [MIREX 2026 Audio Instrument Recognition results](https://music-ir.org/mirex/wiki/2026%3AAudio_Instrument_Recognition_Results)

## Nonclaims and release gate

This benchmark does not prove that a renderer is perceptually superior, that any physical instrument has been measured, or that the current Muse renderer implements a listed gesture. A receipt hash does not make self-reported measurements trustworthy. A capability becomes promotable only after independent source/build/measurement verification and the signed exact-cell decision in #9586. Exact-head compile/tests and challenge-set evidence are required before removing draft status.
