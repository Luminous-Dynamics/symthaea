# Muse factorial composer adapter

This example executes the frozen case manifest against the pinned
symthaea-music-theory symbolic composer and emits observations in the strict
CSV format consumed by the standalone factorial analyzer.

It intentionally lives beside the composer it measures. It is not a new
production API and does not modify the composition engine's behavior.

## Execution sequence

Generate the exact case manifest:

    cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin emit-cases -- discovery > /tmp/muse-discovery-cases.csv

Run the composer adapter from the repository root:

    cargo run --manifest-path crates/domains/symthaea-music-theory/Cargo.toml --example muse_factorial_adapter -- discovery /tmp/muse-discovery-cases.csv /tmp/muse-observations.csv "$(git rev-parse HEAD)"

The cohort selector is discovery, confirmation, or both. The adapter validates
the manifest against the exact frozen style, seed, and 3^4 cell set before
composing anything. The final argument is a required 40-hex source commit SHA.
The adapter verifies that it equals the checked-out HEAD, refuses a dirty
worktree, checks 13 pinned source/lockfile blob digests, and verifies that the
entire music-theory source tree is unchanged from the preregistered composer
reference. The SHA is retained in every result row.

The full-tree diff requires the pinned reference commit to exist locally. A
shallow checkout may contain all current files but omit the historical commit
object, causing the adapter to fail before composition. Fetch the reference
branch history before running:

    if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
      git fetch --unshallow origin muse/style-genre-profile-v1
    else
      git fetch origin muse/style-genre-profile-v1
    fi
    git cat-file -e 'bd227d416878a4373eb87a53e182e7067a2411a5^{commit}'

The adapter also checks the 13 pinned file blobs, so the fetch does not replace
the exact source checks. Put generated manifests and reports outside the source
checkout (for example under /tmp) so they do not contaminate the next run.

Analyze the resulting observations. The metric registry must be explicit:

    METRICS='score_note_count,score_total_beats,score_total_seconds,score_tempo_bpm,score_notes_per_beat,score_mean_pitch_midi,score_pitch_span_semitones,score_pitch_stddev_midi,score_mean_velocity,score_velocity_stddev,score_mean_section_intensity,score_mean_note_duration_beats,score_assigned_part_fraction,score_universal_fatal_count,score_style_fatal_count,score_warning_count,score_universal_valid,score_full_validation_valid,score_pitch_class_diversity,score_pitch_class_entropy_bits,score_melody_mean_interval_semitones,score_melody_step_fraction,score_melody_leap_fraction,score_onset_ioi_stddev_beats,score_offbeat_onset_fraction,score_voice_role_entropy_bits,score_duration_variety_count,score_section_intensity_stddev,identity_score_meter_beats,score_distinct_midi_pitches,identity_score_key_tonic_pc,score_key_major_indicator,score_key_minor_indicator,score_key_modal_indicator,score_emphasis_normal_count,score_emphasis_climax_count,score_emphasis_cadential_count,score_emphasis_phrase_start_count,score_form_section_count,score_plan_requested_bars,score_plan_realized_bars,score_fingerprint_form_distance,score_fingerprint_harmony_distance,score_fingerprint_orchestration_distance,score_fingerprint_rhythm_distance,score_fingerprint_contour_distance,score_fingerprint_tempo_meter_distance,score_fingerprint_pitch_class_distance,score_fingerprint_interval_distance,score_fingerprint_chord_quality_distance,score_fingerprint_cadence_distance,score_fingerprint_motif_distance,identity_exact_fingerprint_high32,identity_exact_fingerprint_low32'
    cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin analyze-results -- discovery /tmp/muse-observations.csv "$METRICS" coverage.csv effects.csv summary.csv axis-invariance.csv

The root Cargo workspace currently needs the separate unmanifested-member
exclusion repair before this example can be compiled through Cargo. Do not
interpret that workspace blocker as a composer or adapter failure.

## What is measured

The adapter materializes the registered factors exactly:

- valence: -1, 0, +1
- arousal: 0, 0.5, 1
- energy: 0, 0.5, 1
- requested bars: 8, 16, 24
- tonic: C
- seed: the cohort's registered replication seed

Every composition routes through the style's canonical
compose_with_grammar_plan entry point and its frozen grammar profile/spec.

For each symbolic score, it serializes the score with serde_json, computes a
SHA-256 digest over those exact bytes, and emits 54 scalar/identity metrics:

1. note count
2. actual score length in beats
3. score duration in seconds
4. realized tempo in BPM
5. notes per beat
6. mean MIDI pitch
7. pitch span in semitones
8. pitch standard deviation
9. mean velocity
10. velocity standard deviation
11. mean section intensity
12. mean note duration in beats
13. fraction of notes with explicit PartId
14. fatal universal-invariant count
15. fatal style-specific judgment count
16. warning count
17. universal structural validity
18. full default-validator validity
19. pitch-class diversity (distinct classes present)
20. pitch-class entropy in bits
21. mean absolute melodic interval in semitones
22. melody step fraction (intervals at most two semitones)
23. melody leap fraction (intervals greater than seven semitones)
24. onset inter-onset-interval standard deviation in beats
25. offbeat-onset fraction relative to the quarter-note beat grid
26. voice-role entropy in bits
27. distinct note-duration count
28. section-intensity standard deviation
29. meter in beats per bar (identity-only channel)
30. distinct absolute MIDI-pitch count
31. tonic pitch-class code (identity-only channel)
32–34. major, minor and modal one-hot indicators
35–38. note counts for Normal, Climax, Cadential and PhraseStart emphasis
39. explicit form-section count (not applicable when no Form object is emitted)
40–41. plan-requested and plan-realized bars (only for CallResponse/JazzChorus plans that expose both)
42–52. eleven structural-fingerprint layer distances from the same style/seed's centre condition: form, harmony, orchestration, rhythm, contour, tempo/meter, pitch class, interval, chord quality, cadence and motif
53–54. exact fingerprint high/low 32-bit words (identity-only channels), preserving the 64-bit identity without lossy f64 conversion

The centre factorial condition (valence 0, arousal 0.5, energy 0.5,
bars 16) is executed first for each style/seed. It becomes that block's
within-style reference. Each other condition reports eleven L2 layer distances
from that reference. This adds no additional composer calls: the reference
condition is already part of the 81-case factorial block. Its own layer
distances are zero by construction.

The pitch, melody, rhythm and role metrics are structural descriptors. They are
not proxies for beauty, cultural authenticity, complexity or quality. In
particular, offbeat fraction is defined against the score's quarter-note beat
coordinate and must be interpreted alongside meter and style.

Metric names beginning with identity_ are emitted and coverage-audited but are
not submitted to the scalar contrast estimator. This applies to meter, tonic
pitch-class identifiers and exact fingerprint words; the numeric encoding is
not a meaningful interval scale. Tonality uses major/minor/modal 0/1 indicators
instead.

The two validity metrics are deliberately separate. The universal metric only
counts the validator's universal malformed-score invariants. The full validator
also includes style-dependent judgments such as strong-beat consonance, parallel
perfect motion, melodic leap, and final tonic arrival; those are not universal
rules for every grammar or musical culture.

Part identity is measured only from explicit PartId values. The adapter never
reconstructs part identity from register, voice role, channel, or timing. A
measured fraction of zero is reported as zero coverage, not as proof that all
notes belong to one part.

## Outcome and evidence semantics

- A finite scalar with an exact score SHA-256 is emitted as observed.
- A metric that cannot be calculated from a valid score is emitted as
  not_applicable with no scalar. It is not imputed as zero.
- A spec error, serialization error, composer error, or caught composer panic
  emits failed rows without a score digest.
- Every row records the source commit, case-manifest SHA-256, composer package
  version, grammar family, declared supported axes, and validator counts.
- Output rows follow canonical manifest order and fixed metric order.

The adapter's CSV is an execution receipt, not an authenticity certificate.
It hashes the serialized symbolic score and binds its source commit and input
manifest digest as note fields. It does not render audio, hash rendered audio,
measure listener preference, certify legal rights, or establish that the
composer's metric definitions are perceptually valid.

Any not_applicable or failed value suppresses effect estimates for that
style/seed/metric block in analyze-results. A block with 81 explicit
not_applicable rows is retained as not_applicable and does not fail the run;
mixed not_applicable/observed, failed, or missing rows remain incomplete and
cause a non-zero analyzer exit after writing all reports. The axis-invariance
report independently compares exact score digests over 27 matched slices per
factor and marks declared unsupported axes as negative controls. A command exit
of zero from the adapter only means that no composer execution failed; it does
not mean every metric block is eligible for effect estimation.

## Qualification boundary

The example and its tests are authored but have not been compiled or executed
in the current environment. The repository root workspace repair must land
before Cargo can load the composer package. A green standalone analyzer test
does not qualify the composer adapter, and symbolic score measurements do not
qualify rendered audio behavior.
