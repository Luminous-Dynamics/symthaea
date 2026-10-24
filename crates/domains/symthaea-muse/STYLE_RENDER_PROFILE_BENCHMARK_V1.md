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

## Machine-readable renderer evidence receipt

The current `renderer_ab_fair` probe emits a versioned JSON receipt and hashes
the artifacts that its outputs depend on. The receipt is a diagnostic/replay
record, not a trust root. SHA-256 ties a claim to bytes, but a caller-provided
Git revision does not prove those bytes were built from that revision; likewise,
a declared VCSL manifest digest is not proof of which files were loaded until
that manifest is independently checked against the asset root. Receipts must
remain `incomplete` if a planned cell fails, metrics are non-finite, a source
revision is absent/ill-formed, FluidSynth version probing fails, or VCSL is active
without a syntactically valid manifest digest.

The emitted profile fingerprint covers grammar/renderer dialect, mastering
targets/caps, and FluidSynth room controls. Each row also reports whether the
profile's controls differ from defaults, profile-minus-default signal deltas,
and exact PCM/WAV hash equality for the no-op control. It deliberately does not
fold asset identity into the profile hash: tool, SoundFont, source revision,
input MIDI, and rendered output hashes remain separate fields so a difference
can be localized instead of being hidden in a single composite digest.

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
- The same-MIDI 2×2 example computes the room effect at default mastering, the
  mastering effect at default room, and their interaction. Synthetic tests inject
  known factor effects and an interaction, then verify the recovered values;
  a separate case verifies that purely additive effects produce zero interaction.
- Malformed WAV input fails closed: sample decoding errors are not dropped,
  partial-channel frames are rejected, and non-finite float samples must not flow
  into loudness or spectral measurements. Unit tests cover NaN and truncated PCM
  fixtures; they are authored but remain unexecuted until a Rust toolchain runs them.

### Audio sanity gate

Run the paired renderer probe on fixed seeds and a qualified asset environment:

    export SYMTHAEA_GIT_REVISION="$(git rev-parse HEAD)"
    nix shell nixpkgs#fluidsynth nixpkgs#soundfont-fluid --command bash -c '
      export SYMTHAEA_FLUIDSYNTH=$(which fluidsynth)
      export SYMTHAEA_SOUNDFONT=$(ls /nix/store/*/share/soundfonts/FluidR3_GM2-2.sf2 | head -1)
      cargo run --release -p symthaea-muse --features theory --example renderer_ab_fair
    '

The probe now renders **five arms per style** and writes
`audio_output/renderer-ab-fair-{synth|vcsl}/renderer-ab-fair-manifest.json`
alongside the WAV/MIDI cohort: native with the dialect profile, plus a 2×2
FluidSynth design with (1) default room/default mastering, (2) dialect room/
default mastering, (3) default room/dialect mastering, and (4) dialect room/
dialect mastering. All four FluidSynth cells use the same exact MIDI file,
SoundFont, binary, and requested sample rate. This separates the room effect at
default mastering, the mastering effect at default room, and their interaction
at the signal-measurement level. The native-versus-FluidSynth comparison remains
a separate backend diagnostic. Neither factorial signal effects nor cross-backend
differences establish a perceptual benefit.

The JSON receipt records the caller-supplied source revision, executable
SHA-256 and byte length, FluidSynth executable hash and version output, SoundFont
SHA-256 and byte length, per-cell MIDI SHA-256, canonical PCM hashes for all five
arms, four FluidSynth WAV hashes, profile fingerprints, exact mastering values,
room settings, signal measurements, factorial contrasts and per-cell failure
codes. Native/FluidSynth PCM hashes use interleaved little-endian f32 stereo
samples; FluidSynth WAV hashes include the file container. The four FluidSynth
variants are retained for a later randomized study; the probe does not itself
create or score a blinded listening pack.

Set `SYMTHAEA_GIT_REVISION` to the exact checkout SHA before the run. A complete
status requires all eight cohort rows, a valid 40- or 64-character hexadecimal
revision, successful FluidSynth version output, and valid source provenance
metadata. If VCSL is enabled, also set
`SYMTHAEA_VCSL_MANIFEST_SHA256` to the digest of the independently reviewed
complete VCSL asset manifest used for that checkout. **The probe records and
format-checks this VCSL digest; it does not recompute the manifest or prove that
the supplied digest corresponds to the configured root.** This receipt is not a
signed build attestation. Verify the source revision, manifest-to-root binding,
and build provenance independently.

Partial cohorts and non-finite measurements are now marked as failed cells;
the program writes an `incomplete` receipt and exits nonzero rather than
presenting a partial run as a successful comparison. Missing FluidSynth or
SoundFont aborts preflight before a cohort can run. A complete receipt proves
identity/replay metadata was captured, not that either renderer sounds better.
Objective spectral movement shows the profile is applied; it does not prove
musical quality.

### Perceptual gate

Use a preregistered, randomized **2×2 factorial listening design** on the four
FluidSynth variants from the same cell: default/default, dialect room only,
dialect mastering only, and combined dialect room/mastering. Level match for
presentation and counterbalance order; record any presentation gain separately
from the original audio receipt. All variants must share the exact MIDI SHA-256,
SoundFont SHA-256, renderer binary SHA-256 and requested sample rate. Estimate the
room effect, mastering effect, and their interaction separately; the primary
preference endpoint may remain default versus combined. Keep composition and
instrument ensemble fixed. Any human-rated effect is a separate result from the
objective signal contrasts. For a separate
style-identity trial, allow the score, ensemble and dialect to vary, and balance
listener familiarity. Randomize presentation order and hide renderer/model
identity until debrief; if AI-versus-human labelling is of interest, test that in
a separate randomized factor instead of confounding it with sound. Recent work
finds disclosure labels can affect attitudes to music, so provenance should not
quietly become the style-recognition treatment. Never combine the profile
ablation and full style-identity trial into one score.

Targets for the first pilot (proposed, not existing results):

- **Style identity:** at least 70% correct pairwise identification for each
  preselected contrasting pair, with a 95% interval reported and a minimum of
  30 independent ratings per pair for the *pilot*; publish misses as well as hits.
  Treat 30 as a pilot floor, not as a powered confirmatory sample size. Preregister
  a power/sample-size plan before a confirmatory study, and cluster uncertainty by
  listener if the same listener contributes multiple ratings.
- **Profile benefit:** the combined profile must improve preregistered listener-
  rated style-fit or preference over default/default without reducing overall
  quality. Use the room-only/mastering-only variants to identify which change
  drives an observed benefit and whether the combined effect is non-additive.
  If not, revert or revise the profile; objective spectral movement is not
  sufficient. Replicate across multiple seeds/styles and listeners before
  promoting exploratory findings to confirmatory claims.
- **Profile-control fidelity:** each successful cell records one MIDI digest
  reused by all four FluidSynth arms; the tool and SoundFont hashes, sample rate,
  and controls are explicitly reported. Require 8/8 complete style cells and a
  complete receipt before aggregation. For every cell, verify identical MIDI
  hashes and expected sample rates across the factorial arms. As a positive
  control, the ClassicalRubato/default profile declares no room or mastering
  changes: compare its default-vs-combined PCM/WAV hashes. Exact equality is
  the target if the backend is deterministic; any divergence must be investigated
  as possible DSP randomness, hidden defaults, or a parity defect before the
  control is treated as passed. The probe reports hash equality but does not yet
  fail solely on mismatch. The current cohort has one seed/style and is a smoke/
  diagnostic matrix, not a statistically useful population estimate; add multiple
  preregistered seeds per style before drawing robust style-general conclusions.
- **Level fairness:** both arms stay within 1.0 LU of the intended integrated
  loudness target and below the configured peak ceiling, or the comparison is
  invalid. (The current WAV peak check is sample peak; a true-peak meter is
  still needed before calling this a true-peak guarantee.)
- **Instrument integrity:** qualify selected sample banks separately from names,
  file presence, and MIDI program labels. Use pitch/velocity grid renders,
  positive/negative gesture tests, non-silence checks, and blinded human review.
  Do not count unavailable or untested notes as successful proficiency.

## Next paradigm: compositional-control contract tests

A style preset should become a **testable set of counterfactual promises**, not a
bag of correlated style labels. The factorial profile probe is the first step:
it estimates room-only and mastering-only changes on fixed MIDI, then exposes
their interaction. The next benchmark should generalize that design to declared
control axes—dialect, room, mastering, ensemble, articulation, timing, dynamics,
and structure—without letting one metric stand in for all of them.

For each control, freeze the input and perturb only that control. Record (a) the
intended target change, (b) non-target attributes that should remain invariant,
(c) pairwise interaction effects, and (d) failure/unsupported combinations.
Then run selected combined-control cases to measure whether the system composes
requests or silently drops / entangles them. Keep renderer-level tests on the
same symbolic input separate from generation-level tests where changing a style
is expected to change the score. A style change is not a clean room-control
ablation.

Proposed starting gates, explicitly engineering targets rather than established
field standards:

- **Contract completeness:** 100% of shipped controls have a versioned definition
  of the intended effect, allowed range, invariants, and explicit unsupported
  states.
- **Execution fidelity:** 100% of requested render-profile settings are recorded
  in the receipt and the executed settings match them; zero silent fallback.
- **Counterfactual isolation:** in a same-MIDI single-control sweep, the intended
  channel changes by a preregistered detectable amount in 8/8 dialect cases,
  while declared invariant channels remain inside preregistered tolerance bands.
  Establish practical tolerances from repeated-render noise first; do not invent
  a universal threshold for every audio metric.
- **Composition:** all preregistered supported pairwise control combinations
  complete, with interaction terms and failed combinations reported rather than
  averaged away.
- **Human validation:** pre-register one primary style-fit / preference endpoint
  and one non-inferiority endpoint for overall quality. Use listener-clustered
  uncertainty and randomized presentation; metric gains alone do not accept a
  profile.

A closely related external frontier is compositional instruction evaluation:
CMI-RewardBench (ICML 2026) evaluates music reward models on musicality,
text–music alignment, and compositional-instruction alignment, with both
pseudo-labelled and human-annotated preference resources. It is a useful external
reference for instruction-compliance evaluation, but it is not ground truth for
our symbolic music grammar or for instrument proficiency. See
https://proceedings.mlr.press/v306/ma26am.html and the official evaluation
toolkit at https://github.com/Haiwen-Xia/CMI-RewardBench.

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
- Chia et al. (December 2025), *Do listeners devalue AI-generated pop music?*
  experimentally studies AI-labelled versus human-labelled music and motivates
  treating provenance disclosure as a separate randomized factor:
  https://doi.org/10.1016/j.chbah.2025.100217
- MusicGen model card summarizing FAD, KL, CLAP, and human evaluations of audio
  quality and text relevance on held-out evaluation sets:
  https://github.com/aime-labs/MusicGen/blob/main/MODEL_CARD.md
- Current instrument qualification work:
  https://github.com/Luminous-Dynamics/symthaea/pull/7943
- Instrument provenance consistency verifier:
  https://github.com/Luminous-Dynamics/symthaea/pull/8019
