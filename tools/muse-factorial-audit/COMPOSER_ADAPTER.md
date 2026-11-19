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
composing anything. The final argument is a required 40-hex source commit SHA. The adapter verifies
that it equals the checked-out HEAD, refuses a dirty worktree, checks 13 pinned
source/lockfile blob digests, and verifies that the entire music-theory source
tree is unchanged from the preregistered composer reference. The pinned base
commit must exist in local Git history. The SHA is retained in every result
row. Put generated manifests and reports outside the source checkout (for
example under /tmp) so they do not contaminate the next run.

Analyze the resulting observations. The metric registry must be explicit:

    cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin analyze-results -- discovery /tmp/muse-observations.csv score_note_count,score_total_beats,score_total_seconds,score_tempo_bpm,score_notes_per_beat,score_mean_pitch_midi,score_pitch_span_semitones,score_pitch_stddev_midi,score_mean_velocity,score_velocity_stddev,score_mean_section_intensity,score_mean_note_duration_beats,score_assigned_part_fraction,score_universal_fatal_count,score_style_fatal_count,score_warning_count,score_universal_valid,score_full_validation_valid,score_pitch_class_diversity,score_pitch_class_entropy_bits,score_melody_mean_interval_semitones,score_melody_step_fraction,score_melody_leap_fraction,score_onset_ioi_stddev_beats,score_offbeat_onset_fraction,score_voice_role_entropy_bits,score_duration_variety_count,score_section_intensity_stddev coverage.csv effects.csv summary.csv

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
SHA-256 digest over those exact bytes, and emits 52 scalar/identity metrics:

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
29. meter in beats per bar
30. distinct absolute MIDI-pitch count
31. tonic pitch-class code (0–11)
32. tonality code (0=major, 1=minor, 2=modal; exact modal name remains in the receipt note)
33–36. note counts for Normal, Climax, Cadential and PhraseStart emphasis
37. explicit form-section count (not applicable when no Form object is emitted)
38–39. plan-requested and plan-realized bars (only for CallResponse/JazzChorus plans that expose both)
40–50. eleven structural-fingerprint layer distances from the same style/seed's centre condition: form, harmony, orchestration, rhythm, contour, tempo/meter, pitch class, interval, chord quality, cadence and motif
51–52. exact fingerprint high/low 32-bit words, preserving the 64-bit identity without lossy f64 conversion

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
style/seed/metric block in analyze-results. The analyzer still writes complete
coverage reports and returns non-zero for incomplete blocks. A command exit of
zero from the adapter only means that no composer execution failed; it does not
mean every metric block is eligible for estimation.

## Qualification boundary

The example and its tests are authored but have not been compiled or executed
in the current environment. The repository root workspace repair must land
before Cargo can load the composer package. A green standalone analyzer test
does not qualify the composer adapter, and symbolic score measurements do not
qualify rendered audio behavior.
