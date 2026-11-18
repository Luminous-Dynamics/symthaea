# Muse factorial composer adapter

This example executes the frozen case manifest against the pinned
symthaea-music-theory symbolic composer and emits observations in the strict
CSV format consumed by the standalone factorial analyzer.

It intentionally lives beside the composer it measures. It is not a new
production API and does not modify the composition engine's behavior.

## Execution sequence

Generate the exact case manifest:

    cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin emit-cases -- discovery > discovery.csv

Run the composer adapter from the repository root:

    cargo run --manifest-path crates/domains/symthaea-music-theory/Cargo.toml --example muse_factorial_adapter -- discovery discovery.csv observations.csv "$(git rev-parse HEAD)"

The cohort selector is discovery, confirmation, or both. The adapter validates
the manifest against the exact frozen style, seed, and 3^4 cell set before
composing anything. The final argument is a required 40-hex source commit SHA
and is retained in every result row.

Analyze the resulting observations. The metric registry must be explicit:

    cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin analyze-results -- discovery observations.csv score_note_count,score_total_beats,score_total_seconds,score_tempo_bpm,score_notes_per_beat,score_mean_pitch_midi,score_pitch_span_semitones,score_pitch_stddev_midi,score_mean_velocity,score_velocity_stddev,score_mean_section_intensity,score_mean_note_duration_beats,score_assigned_part_fraction,score_universal_fatal_count,score_style_fatal_count,score_warning_count,score_universal_valid,score_full_validation_valid coverage.csv effects.csv summary.csv

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
SHA-256 digest over those exact bytes, and emits 18 scalar metrics:

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
