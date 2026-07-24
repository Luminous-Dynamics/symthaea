# Cognitive-to-Physical Voice Integration Campaign

Status: preregistered engineering and promotion campaign for Series 23.

## Purpose

This campaign evaluates the first live path in which cognitive state and a
canonical ARPAbet phoneme drive a normalized motor gesture, identity-specific
anatomy and physiology, and the passive physical renderer without caching or
reconstructing absolute tract areas in the cognitive controller.

The authoritative path under test is:

`VoiceCognitiveState + phoneme + prosody -> CfC latent -> GestureProjection -> GestureFrame -> GesturePlanner -> PhysicalTractFrame -> BranchedWaveguideV2`

`FormantFrame` remains an acoustic observer for comparison. It is not accepted
as the motor contract and cannot satisfy physical-path coverage.

## Frozen implementation claims

Series 23 claims only the following implementation changes:

1. Every continuous `GestureFrame` articulator participates in coarticulation.
2. Canonical English diphthongs move between two vowel targets over time.
3. Phonation responds to F0, energy, phoneme voicing, and source class.
4. Explicit silence suppresses energy, adduction, and respiratory effort.
5. Scheduler output includes the configured final carryover tail.
6. Liquids retain approximant constriction instead of collapsing to open vowels.
7. Live pipeline routing falls back to canonical ARPAbet manner and voicing.
8. The live pipeline exposes normalized and physical motor frames directly.
9. The physical gesture head can be bootstrapped deterministically from the
   canonical phoneme inventory without absolute-area targets.
10. Gesture fitting rejects non-finite input atomically.
11. Stream diagnostics include the final delayed block and regrouped reflection
    episodes.
12. Bounded physiology coordinates fail closed on non-finite values.
13. Target-independent articulatory trajectory metrics are available on both
    gestures and completed physical speech renders.

These claims do not imply naturalness, human preference, speaker identity, or
scientific validation of every anatomical parameter.

## Frozen identities and clocks

Run every required lane for:

- Velvet
- Luminous
- Silk

Use all of the following motor clocks:

- 160 Hz
- 197 Hz
- 200 Hz
- 240 Hz

Use output sample rates 16 kHz, 24 kHz, and 48 kHz. Renderer configuration must
validate before a trial begins. Low-rate radiation cutoffs must remain below
Nyquist.

## Canonical phoneme coverage

Every symbol in `CANONICAL_ARPABET_SYMBOLS` must be exercised by the bootstrap
head and by at least one live motor trial. The campaign must separately cover:

- monophthongs;
- `EY`, `AY`, `AW`, `OW`, and `OY` diphthongs;
- voiced and unvoiced stops;
- nasals at labial, alveolar, and velar places;
- fricatives;
- `CH` and `JH` affricates;
- liquids `L` and `R`;
- glides;
- explicit silence.

Unknown symbols must be rejected before projection or physical realization.

## Lane A: continuous-coordinate causality

For each continuous motor coordinate, perturb only that coordinate around a
validated neutral gesture and require:

- a finite realized physical frame;
- a nonzero change in an anatomically relevant physical quantity;
- no change in unrelated categorical manner or place fields;
- no absolute-area state stored in the gesture or learned head.

The coordinates are jaw aperture, tongue-body height and frontness, tongue-tip
constriction and location, lip aperture and protrusion, velum opening,
pharyngeal constriction, larynx height, glottal adduction, vocal-fold tension,
and respiratory effort.

## Lane B: coarticulation and diphthongs

Use minimal pairs and three-phoneme contexts to measure anticipation and
carryover. Required cases include:

- `AA -> IY`;
- `UW -> IY`;
- `S -> T -> R`;
- `M -> AY -> N`;
- `AH -> SIL -> N`;
- every canonical diphthong between consonantal contexts.

Require:

- tongue-tip location, larynx height, and vocal-fold tension to move when their
  neighboring targets differ;
- diphthong start and end targets to be measurably distinct;
- no discontinuity at the diphthong midpoint;
- the configured final carryover duration to be present;
- no anticipatory voicing or respiratory leakage during explicit silence.

## Lane C: gesture-head bootstrap

For each genesis seed, bootstrap the physical gesture head twice with identical
configuration and require bit-identical fitted coefficients.

Use at least 20 independent seeds. For every seed:

- all canonical symbols must be present exactly once;
- every latent and target must be finite;
- fitting must either commit a complete finite head or leave the prior head
  unchanged;
- front/back vowel ordering and rounded/unrounded lip ordering must be correct;
- stops, nasals, fricatives, affricates, liquids, and glides must retain their
  canonical categorical realization.

Also inject NaN and infinity into latent inputs and regularization. Every
injection must fail without modifying any learned coefficient.

## Lane D: live cognitive-to-physical integration

Drive the pipeline with controlled sweeps over valence, arousal, confidence,
Phi, and expected free energy while holding phoneme and identity fixed. Then
hold cognitive state fixed while varying phoneme and identity.

Require:

- all emitted acoustic, gesture, and physical frames to be finite and valid;
- phoneme place and manner to follow canonical metadata when no manual maps are
  registered;
- prosodic F0 and energy to reach the physical glottal and pressure state;
- anatomy changes to affect physical geometry without altering the normalized
  gesture contract;
- no physical frame to be obtained through the legacy formant-vocoder backend.

## Lane E: target-independent trajectory quality

For every utterance, calculate `ArticulatoryQualityMetrics`. Promotion requires:

- at least 40 evaluated motor frames;
- zero non-finite measurements;
- maximum continuous-coordinate slew no greater than 80 normalized units/s;
- maximum F0 slew no greater than 10 octaves/s;
- maximum energy slew no greater than 80 normalized units/s;
- explicit silence coverage;
- zero silence-leakage frames.

The campaign must publish raw maxima, not only pass/fail values.

## Lane F: acoustic and renderer regression

Render every physical trajectory through the production waveguide. Require:

- finite PCM and every debug stem;
- exact expected sample count within the renderer clock contract;
- final diagnostics captured after stream flush;
- no regression in the existing transition-pitch, click, reflection,
  passivity, nasal-coupling, and source-alias gates;
- observation renderer PCM parity where that lane is enabled.

Series 23 does not preregister a naturalness score. Listening results may be
reported separately but cannot replace the objective gates above.

## Required negative controls

The evidence bundle must include exercised rejection for:

- unsupported phoneme;
- NaN in every bounded gesture wrapper family;
- NaN latent during fit;
- infinite regularization;
- invalid identity anatomy or physiology;
- invalid physical constriction location;
- silence with nonzero adduction or respiratory effort;
- insufficient quality-metric coverage;
- a render captured before final stream flush.

## Evidence bundle

For each trial, retain:

- genesis seed digest;
- identity and renderer configuration;
- phoneme sequence and motor clock;
- cognitive and prosody inputs;
- normalized gesture frames;
- physical tract frames;
- PCM and diagnostic stems where storage permits;
- articulatory quality metrics and gates;
- renderer diagnostics and existing evidence reports;
- exact source revision and dependency lock.

The published summary must distinguish `pass`, `fail`, and `not_exercised`.
Missing identity, phoneme, silence, clock, or negative-control coverage is
`not_exercised`, never an implicit pass.

## Promotion rule

Promotion requires every mandatory lane to pass for all three identities, all
four motor clocks, all three output rates, and the complete canonical phoneme
inventory. Any non-finite state, silent leakage, unknown-phoneme acceptance,
legacy-backend substitution, incomplete bootstrap, or missing negative-control
lane blocks promotion.
