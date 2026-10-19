# Melothaea Style-Conditioned Render Profiles — Benchmark v1

**Status: experimental profile proposal.** The preset path now carries the
style's declared grammar performance dialect through native rendering, FluidSynth,
MIDI downloads, performed-note inspection, and keeper bundles. All eight
grammar-level dialects have distinct renderer counterparts. Profile values remain
hypotheses: no new perceptual validation is claimed, and this does not expand the
instrument registry or qualify any bank.

## The paradigm shift

Treat style identity as a **render contract**, not a single instrument label:
symbolic grammar + ensemble + articulation/timing dialect + room + mastering.
The layers remain independently switchable in controlled comparisons. We should
not reward "novelty" as a proxy for quality, nor let mastering cover for weak
composition or incorrect instruments.

The first seam is a typed performance contract. Style.grammar_profile() keeps
grammar family/phrase/harmony architecture but specializes the performance dialect
by style, so a tango and a bossa do not inherit identical timing merely because
their specifications happen to share texture flags. The selected dialect then
controls timing/expression, a directional room hypothesis in both native and
FluidSynth renderers, and broad mastering balance. The reverb engines are not
acoustically equivalent, so the invariant is shared *directional intent*, not
identical room response. MIDI downloads, performed notes, ListenPerformanceBundle,
and keeper MIDI use that same selection. User-authored specs still have no
declared style identity and are explicitly left on the versioned heuristic fallback.

Every mastering profile retains the same integrated-loudness target, limiter
ceiling, and section-leveling behavior; only three-band balance targets and EQ
correction limits vary. ClassicalRubato remains the historical default.
Calling render_midi_to_wav without an explicit profile remains the default path
for neutral previews and imported-work audition.

## Profile hypotheses (not measured truths)

| Performance dialect | Low / mid / high target | Max EQ correction | Max high-band boost | Rationale to test |
|---|---:|---:|---:|---|
| ClassicalRubato | 0.30 / 0.45 / 0.25 | 6.0 dB | 1.5 dB | Compatibility reference |
| DanceLocked | 0.32 / 0.45 / 0.23 | 4.5 dB | 1.0 dB | Slightly more low-band weight; restrained top-end compensation |
| ProcessExact | 0.28 / 0.43 / 0.29 | 3.5 dB | 1.0 dB | Less aggressive global correction for sustained/process material |
| FolkLift | 0.30 / 0.44 / 0.26 | 4.5 dB | 1.25 dB | Slightly open upper band without increasing loudness |
| JazzLaidBack | 0.29 / 0.47 / 0.24 | 4.0 dB | 0.9 dB | Mid-focused balance and restrained upper-band boosts |
| DroneElastic | 0.29 / 0.43 / 0.28 | 3.5 dB | 1.0 dB | Less EQ intervention in unfolding drone/modal material |
| DramaticTiming | 0.31 / 0.44 / 0.25 | 4.5 dB | 1.2 dB | Conservative broad balance for dramatic arcs |
| ChoralBlend | 0.28 / 0.47 / 0.25 | 3.5 dB | 0.75 dB | Protect a unified midrange; keep treble boosts modest |

The ratios are the mastering algorithm's target band proportions, not spectral
measurements of a genre and not an assertion about every culture or recording.
The preset mapping is now explicit in Style::grammar_profile; dialect_for_spec
remains the fallback for custom specs only. RendererRecipe schema v4 records the
declared grammar-level dialect when one exists. The next architecture step is to
let a user-authored CompositionSpec carry an explicit performance profile too,
with named, versioned rendering controls rather than adding more heuristic flags.

## Required evidence

### Deterministic contract tests

- Every profile keeps the same LUFS target and limiter ceiling.
- Every target ratio is finite, between zero and one, and the three targets sum
  to one within floating-point tolerance.
- Every correction cap is finite and positive.
- ClassicalRubato is field-by-field compatible with the existing default.
- The eight grammar dialects map one-to-one to renderer dialects; test that
  each timing scale is finite and bounded, and each mastering profile shares
  loudness/limiter targets.
- The native renderer and FluidSynth both receive the same directional room
  hypothesis and broad mastering profile. Assert dry/groove versus wet/drone
  ordering separately per backend; do not assert acoustic equivalence between
  their different reverb algorithms.
- MIDI download, notes endpoint, ListenPerformanceBundle, analyst evidence, and
  keeper MIDI use the candidate's same declared dialect; custom specs explicitly
  take the heuristic fallback.
- The renderer recipe records the dialect in schema v4 whenever it is declared.

### Audio sanity gate

Run the paired renderer probe on fixed seeds and a qualified asset environment:

    nix shell nixpkgs#fluidsynth nixpkgs#soundfont-fluid --command bash -c '
      export SYMTHAEA_FLUIDSYNTH=$(which fluidsynth)
      export SYMTHAEA_SOUNDFONT=$(ls /nix/store/*/share/soundfonts/FluidR3_GM2-2.sf2 | head -1)
      cargo run --release -p symthaea-muse --features theory --example renderer_ab_fair
    '

Record the source revision, FluidSynth version, SoundFont SHA-256, VCSL state,
per-piece LUFS / peak / crest / band balance, failed renders, and exact profile.
A missing renderer, missing asset, or empty cohort is a failed run, not a null
result. Objective spectral movement shows the profile is applied; it does not
prove musical quality.

### Perceptual gate

Use fixed symbolic scores and a preregistered, randomized blind A/B or ABX:
neutral profile versus dialect profile, level matched, identical MIDI, identical
SoundFont, same sample rate and mastering chain. Keep the composition and
instrument ensemble constant for the renderer-profile ablation. For a separate
style-identity trial, allow the score, ensemble and dialect to vary, and balance
listener familiarity. Never combine the two into one score.

Targets for the first pilot (proposed, not existing results):

- **Style identity:** at least 70% correct pairwise identification for each
  preselected contrasting pair, with a 95% interval reported and a minimum of
  30 independent ratings per pair for the *pilot*; publish misses as well as hits.
  Treat 30 as a pilot floor, not as a powered confirmatory sample size. Preregister
  a power/sample-size plan before a confirmatory study, and cluster uncertainty by
  listener if the same listener contributes multiple ratings.
- **Profile benefit:** the profile must improve its preregistered listener-rated
  fit or preference over the neutral profile without reducing audio-quality
  ratings. If not, revert or revise the profile; objective spectral movement is
  not sufficient.
- **Level fairness:** both arms stay within 1.0 LU of the intended integrated
  loudness target and below the configured peak ceiling, or the comparison is
  invalid. (The current WAV peak check is sample peak; a true-peak meter is
  still needed before calling this a true-peak guarantee.)
- **Instrument integrity:** qualify selected sample banks separately from names,
  file presence, and MIDI program labels. Use pitch/velocity grid renders,
  positive/negative gesture tests, non-silence checks, and blinded human review.
  Do not count unavailable or untested notes as successful proficiency.

## Dataset and metric boundary

MTG-Jamendo provides a large multi-label reference for instrument, genre, and
mood/theme tagging, with more than 55,000 tracks and predefined research splits.
It can help test whether audio descriptors or a separate classifier recognize
broad labels, subject to checking each source track's license and split protocol.
It is not ground truth for whether a newly generated style is musically correct,
nor does a genre tag validate a particular instrument's range or articulation.

For generated-audio metrics, use a portfolio rather than one scalar: spectral
band balance, integrated loudness and peak, onset/duration consistency,
instrument/pitch/velocity coverage, style-tag similarity where appropriate,
and blind listener judgments. Distributional or text/audio scores can be useful
diagnostics but cannot substitute for human assessments of quality, relevance,
or stylistic credibility. A 2026 survey likewise concludes that no single measure
is sufficient and distinguishes symbolic evaluation (tonality, rhythm, structure,
diversity, controllability) from audio evaluation (timbre, transients, perceptual
quality). We should preserve that separation and add a control-fidelity test:
changing a declared dialect should alter only its intended rendering channels
while leaving the score and declared grammar unchanged.

## References

- MTG-Jamendo dataset and published tag splits:
  https://github.com/MTG/mtg-jamendo-dataset
- Grötschla et al., *Benchmarking Music Generation Models and Metrics via Human
  Preference Studies* (2025): 6,000 generated songs, 15,000 pairwise comparisons,
  2,500 participants; evaluates alignment between objective metrics and human
  preference: https://arxiv.org/abs/2506.19085
- *From Aesthetics to Human Preferences* (IEEE MLSP 2025) discusses inconsistent
  relationships between distributional metrics and human preference:
  https://doi.org/10.1109/MLSP62443.2025.11204254
- *Recent advances in music generation: methods, evaluation, and challenges*
  (Artificial Intelligence Review, 2026) distinguishes symbolic and audio
  evaluation and recommends multiple complementary measures:
  https://doi.org/10.1007/s10462-026-11582-x
- MusicGen model card summarizing FAD, KL, CLAP, and human evaluations of audio
  quality and text relevance on held-out evaluation sets:
  https://github.com/aime-labs/MusicGen/blob/main/MODEL_CARD.md
- Current instrument qualification work:
  https://github.com/Luminous-Dynamics/symthaea/pull/7943
- Instrument provenance consistency verifier:
  https://github.com/Luminous-Dynamics/symthaea/pull/8019
