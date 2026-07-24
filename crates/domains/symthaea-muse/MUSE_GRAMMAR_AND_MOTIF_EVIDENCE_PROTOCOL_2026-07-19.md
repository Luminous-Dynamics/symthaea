# Muse grammar and motif evidence protocol

**Protocol version:** 1.0  
**Frozen:** 2026-07-19  
**Scope:** four flagship grammar families and Motif Foundry v2

This protocol separates two claims:

1. **Ecological identity:** the whole output of a grammar is recognizable.
2. **Structural identity:** the grammar remains recognizable after obvious
   performance and production cues are reduced.

It also separates short vocabulary recognition from complete-form grammar.

| Pack | Duration | Primary claim |
|---|---:|---|
| Identity excerpt | 30 seconds, 1.5-second fade | Immediate family recognition |
| Complete form | 90–180 seconds, no truncation/fade | Development, return, trajectory, closure |

## Families and labels

- `PeriodSentence`
- `GrooveCycle`
- `ProcessAdditive`
- `ModalArcInformed`

`ModalArcInformed` is intentionally culturally qualified. This study does not
establish authenticity for a South Asian tradition.

## Arms

The ecological arm retains natural tempo, density, register, dynamics, silence,
and performance dialect. Both arms use the same piano path and remove drums.

The structural-challenge arm additionally uses a melody-only projection, shared
72/96/120 BPM strata, octave-normalized median register, a common dynamic range,
final -27 dBFS waveform RMS, and the `ProcessExact` performance dialect. Onset density is measured and reported
but is not forcibly equalized: thinning notes would alter the grammar under test.
Residual density differences must therefore be included as covariates and stated
as a limitation.

## Splits and selection

- Development seeds: `11, 28, 45, 62`
- Validation seeds (12 per family): `79, 96, 113, 130, 147, 164, 181,
  198, 215, 232, 249, 266`
- Locked holdout: no seeds embedded in source; generation exits without an
  externally committed seed-file hash and a future explicit unlock workflow.

Every pre-registered seed is included. The generated `rejection_ledger.json`
records inclusion, any mechanical rejection, source bar count, duration, note
count, onset density, and median register. No aesthetic rejection is permitted.

For complete forms, the generator evaluates the fixed candidate bar counts
`4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64` and selects the result closest to 120 seconds
within the pre-registered 90–180 second interval. This is a duration rule, not a
quality filter. The initial grid began at 8 bars; it was amended before accepting
or listening to any complete-form pack because GrooveCycle's 8-bar realization
measured 183.27 seconds. Adding 4 and 6 bars fixes protocol feasibility without
selecting on musical quality.

Controlled validation uses shared tempo strata at 72, 96, and 120 BPM. The
same seed is assigned the same stratum in every grammar. This replaces a single
tempo target while preserving nuisance-variable overlap.

Controlled audio is cropped and faded first, then the final participant-facing
waveform is normalized to -27 dBFS RMS. The conservative target avoids limiting
high-crest sparse material. Normalizing before the crop is prohibited because
the fade and excerpt silence would reintroduce family-level loudness differences.

## Within-family multiplicity gate

Every generated pack writes `clone_warnings.json`. Same-family pairs are flagged
when any pre-registered warning threshold is crossed:

- normalized onset-trajectory Pearson correlation ≥ 0.95;
- time-binned chroma cosine similarity ≥ 0.95;
- cadence-position similarity ≥ 0.98;
- normalized climax-position difference ≤ 0.02.

Warnings do not silently remove a pre-registered seed. Development packs expose
them for engine revision; validation reports them as multiplicity failures or
limitations. `artifacts.json` additionally records literal and normalized score
hashes so exact symbolic reuse cannot hide behind different performances.

## Causal minimal grammar pairs

The `minimal-pairs` arm supplies the same seed-indexed literal motif bank to all
four grammar owners, disables style-specific hook grafting, and applies the
controlled rendering policy. The grammar engines still own phrase/cycle logic,
harmonic syntax, development, and obligations. This jointly tests grammar
differentiation and motif context transfer without pasting a motif over an
already-composed score.

The first minimal-pair development render exposed the intended bottleneck:
ProcessAdditive and ModalArc still had effectively fixed temporal envelopes.
The engines now expose four seed-selected but grammar-valid internal profiles:
additive arch, terraced, peak-hold, and accelerating-release trajectories; and
modal expansive-opening, balanced, early-pulse, and extended-intensification
stage allocations. GrooveCycle retains its five section roles while varying
their whole-cycle proportions. These are registered multiplicity dimensions,
not arbitrary post-render perturbations.

The second causal development render provides a mechanical sanity check. At the
onset-trajectory warning threshold, ModalArc fell from 6 flagged pairs to 1 and
ProcessAdditive from 6 to 1. GrooveCycle remained at 1; PeriodSentence retained
2 warnings. This is evidence that the new profiles reduce template collapse,
not evidence of listener-perceived individuality. The remaining pairs stay in
the pack and require review.

Each seed is an explicit four-treatment block with stable `block_id`,
`premise_id`, and `motif_id`. `paired_trials.json` exposes only blinded filenames
and block metadata. `paired_study.html` randomizes blocks and treatments per
participant and requires every grammar exactly once per block. It records
classification, confidence, coherence, interest, beauty, memorability, replay
desire, the most structurally similar pair, best motif preservation, the most
convincing treatment, and a constrained musical-evidence checklist.

Two paired arms share the same symbolic controls. `minimal-pairs` uses the common
neutral `ProcessExact` performance dialect; `minimal-pairs-natural` retains each
family's native performance dialect. Their difference estimates how much
classification and preference come from performance rather than composition.

## Human responses

`study.html` randomizes presentation per participant and records:

- family classification;
- confidence from 0–100;
- free-text evidence heard;
- distinctiveness from 1–7;
- musical quality from 1–7;
- desire to replay from 1–7;
- coherence and perceived similarity to an earlier piece;
- intentional development, meaningful repetition, earned climax, recognizable
  returns, conclusive ending, and sustained attention;
- trained or untrained cohort.

Primary analysis is the family confusion matrix. Report permutation-test
significance and participant/bootstrap confidence intervals. Distinctiveness,
quality, and replay desire remain separate outcomes. Repeat-exposure learning and
same/different pair trials are secondary studies and must not be silently pooled
with the primary classification endpoint.

## Motif Foundry v2 evidence

The lifecycle axes are orthogonal:

- mechanical: unchecked, valid, invalid;
- listening: untested, pilot, recognition pass, recall pass;
- originality: unchecked, low concern, review required, cleared;
- cultural: not applicable, required, in progress, approved;
- promotion: candidate, curated, foundational, retired.

Procedural generation can set only `mechanical = valid`. A family is eligible for
foundational promotion only after recall pass, originality clearance, and either
non-applicable or approved cultural review.

Semantic roles are contextual bindings, not identity fields. Compound materials
reference constituent family IDs rather than copying their data. Material kinds
include melodic, rhythmic, harmonic, bass, contrapuntal, textural, performance,
process identity, modal identity, and compound.

Each recognition study should include transformations at increasing declared
distance and difficult lures: same contour/different rhythm, same rhythm/different
intervals, shared opening/different continuation, same process/different seed,
same pitch set/different hierarchy, and same durations/shifted metric phase. The
result is stored as a validated psychometric identity curve.

## Artifact policy

Each candidate evidence directory must eventually contain:

```text
candidate.json
canonical.mid
canonical.opus
transformations/
lures/
symbolic_report.json
similarity_neighbors.json
provenance.json
study_manifest.json
lineage.json
artifact_hashes.json
```

HDC remains derived and rebuildable. Compare symbolic distance, HDC similarity,
and human recognition probability before granting HDC any retrieval authority.

Every motif candidate carries three separate human tests: transformation
identity, lure rejection, and cross-grammar context transfer. Transfer evidence
records the source and target grammar and the hypothesized identity carriers
(interval, rhythm, accent, contour, anchor, rule, metric phase, or gesture).

## Hash-addressed grammar artifacts

Each pack writes `artifacts.json` with a stable clip ID, WAV SHA-256, score
SHA-256, normalized-score SHA-256, recipe SHA-256, seed, exact rendered and
symbolic durations, grammar profile, performance dialect, source bars, and
automatic-versus-curated inclusion. The global manifest records the Git commit,
FluidSynth version, and soundfont SHA-256. `answer_key_by_sha256.json` maps audio
hashes to labels. Complete-form `clip_duration_secs` is JSON `null`, never `0.0`.

Participant-facing directories contain no answer or structural labels. `sealed/`
contains answer keys, artifact metadata, clone warnings, nuisance baselines, and
`structural_truth_by_sha256.json`. Symbolic truth includes the grammar plan and
obligations, phrase starts, cadences, climax, recurrence intervals, detected
literal/transposed motif occurrences, density arc, pitch-class trajectory, and
declared development operations.

Before listening, a sealed leave-one-out nearest-centroid baseline predicts
families using only tempo, density, register, velocity statistics, final RMS,
and low-activity proportion. Overall and pairwise accuracies are frozen. Human
grammar evidence is interpreted relative to this shallow-cue baseline,
especially for GrooveCycle versus PeriodSentence.

## Commands

```bash
# Development structural challenge, 30-second identity excerpts
cargo run -p symthaea-muse --features studio --bin muse152_listening_pack -- \
  study development structural identity audio_output/muse152_structural_challenge_dev

# Validation ecological complete forms
cargo run -p symthaea-muse --features studio --bin muse152_listening_pack -- \
  study validation ecological complete audio_output/muse152_ecological_complete_validation

# Causal same-motif grammar matrix
cargo run -p symthaea-muse --features studio --bin muse152_listening_pack -- \
  study development minimal-pairs identity audio_output/muse152_minimal_pairs_dev

# Same controlled composition with family-native performance dialects
cargo run -p symthaea-muse --features studio --bin muse152_listening_pack -- \
  study development minimal-pairs-natural identity audio_output/muse152_minimal_pairs_natural_dev
```

The holdout command intentionally refuses to run. Human evidence, not artifact
generation, determines whether a family or motif passes.
