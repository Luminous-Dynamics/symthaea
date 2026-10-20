// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First executable Melothaea style-interaction probe.
//!
//! This is a symbolic contract/diagnostic benchmark, not an audio-quality or
//! cultural-authenticity test. It uses the production music-theory composition
//! path and compares a fixed Classical structural owner under two independent
//! experimental recipe interventions:
//!   A: use only the Shuffle accompaniment cell;
//!   B: pin the pitch language to Dorian.
//!
//! Run from the workspace root with:
//!   SYMTHAEA_GIT_REVISION="$(git rev-parse HEAD)" SYMTHAEA_RUSTC_VERSION="$(rustc --version)" \
//!     cargo run -p symthaea-muse --features theory --example style_interaction_probe
//!
//! The caller-supplied revision is recorded as a claim only; this program does
//! not independently attest which revision built the executable.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process::ExitCode;

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_music_theory::accompaniment::Accompaniment;
use symthaea_music_theory::scale::Mode;
use symthaea_music_theory::score::{Score, VoiceRole};
use symthaea_music_theory::{
    MusicalIntent, PitchClass, Style, compose_with_grammar_plan,
};

const SCHEMA_VERSION: u32 = 1;
const SEEDS: [u64; 8] = [17, 53, 89, 144, 233, 377, 610, 987];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Intervention {
    ShuffleCell,
    DorianMode,
}

impl Intervention {
    fn id(self) -> &'static str {
        match self {
            Self::ShuffleCell => "shuffle_accompaniment",
            Self::DorianMode => "dorian_mode",
        }
    }

    fn apply(self, spec: &mut symthaea_music_theory::spec::CompositionSpec) {
        match self {
            Self::ShuffleCell => {
                // A new experimental recipe, not a claim that the Classical
                // profile has become an authentic blues or any other tradition.
                spec.accompaniment_pool = vec![Accompaniment::Shuffle];
            }
            Self::DorianMode => {
                spec.mode = Some(Mode::Dorian);
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ScoreMetrics {
    note_count: usize,
    melody_note_count: usize,
    harmony_note_count: usize,
    bass_note_count: usize,
    distinct_pitch_classes: usize,
    distinct_onsets: usize,
    half-beat_offgrid_note_count: usize,
    tempo_bpm: f32,
    meter_beats_per_bar: u8,
    total_beats: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ConditionReceipt {
    condition: String,
    interventions: Vec<String>,
    spec_sha256: String,
    score_sha256: String,
    metrics: ScoreMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct InteractionDiagnostics {
    note_count_factorial_delta: i64,
    distinct_pitch_class_factorial_delta: i64,
    half_beat_offgrid_factorial_delta: i64,
    reversed_application_score_equal: bool,
    reversed_application_spec_equal: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SeedReceipt {
    seed: u64,
    /// Full symbolic input, not merely a seed label.
    intent: MusicalIntent,
    conditions: Vec<ConditionReceipt>,
    interaction: InteractionDiagnostics,
}

#[derive(Debug, Serialize)]
struct BenchmarkReceipt {
    schema_version: u32,
    benchmark_id: &'static str,
    source_revision_claim: String,
    /// Caller-supplied toolchain label; not independent compiler attestation.
    rustc_version_claim: String,
    executable_sha256: String,
    executable_byte_count: u64,
    operating_system: &'static str,
    target_architecture: &'static str,
    evidence_boundary: &'static str,
    seed_set: Vec<u64>,
    pair_definition: &'static str,
    condition_run_count: usize,
    seed_receipts: Vec<SeedReceipt>,
}

fn sha256_serialized<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("serializable benchmark value");
    format!("{:x}", Sha256::digest(bytes))
}

fn intent(seed: u64) -> MusicalIntent {
    MusicalIntent {
        valence: 0.2,
        arousal: 0.52,
        energy: 0.6,
        bars: 4,
        seed,
        tonic: PitchClass::C,
    }
}

fn recipe(
    interventions: &[Intervention],
) -> symthaea_music_theory::spec::CompositionSpec {
    let mut spec = Style::Classical.spec();
    for intervention in interventions {
        intervention.apply(&mut spec);
    }
    spec
}

fn compose(
    seed: u64,
    interventions: &[Intervention],
) -> (symthaea_music_theory::spec::CompositionSpec, Score) {
    let spec = recipe(interventions);
    let realized =
        compose_with_grammar_plan(Style::Classical.grammar_profile(), &intent(seed), &spec);
    (spec, realized.score)
}

fn validate_score(score: &Score) -> Result<(), String> {
    if score.notes.is_empty() {
        return Err("empty_score".into());
    }
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 {
        return Err("invalid_tempo".into());
    }
    if !score.total_beats.beats().is_finite() || score.total_beats.beats() <= 0.0 {
        return Err("invalid_total_duration".into());
    }
    for (index, note) in score.notes.iter().enumerate() {
        if !note.onset.beats().is_finite() || note.onset.beats() < 0.0 {
            return Err(format!("invalid_onset_at_note_{index}"));
        }
        if !note.duration.beats().is_finite() || note.duration.beats() <= 0.0 {
            return Err(format!("invalid_duration_at_note_{index}"));
        }
        if !note.velocity.is_finite() || !(0.0..=1.0).contains(&note.velocity) {
            return Err(format!("invalid_velocity_at_note_{index}"));
        }
        if !note.section_intensity.is_finite() {
            return Err(format!("invalid_section_intensity_at_note_{index}"));
        }
    }
    Ok(())
}

fn metrics(score: &Score) -> ScoreMetrics {
    let pitch_classes = score
        .notes
        .iter()
        .map(|note| note.pitch.pitch_class().value())
        .collect::<BTreeSet<_>>();
    let onsets = score
        .notes
        .iter()
        .map(|note| (note.onset.num(), note.onset.den()))
        .collect::<BTreeSet<_>>();
    let half_beat_offgrid_note_count = score
        .notes
        .iter()
        .filter(|note| {
            let phase = note.onset.beats().rem_euclid(1.0);
            (phase - 0.5).abs() < 1e-8
        })
        .count();

    ScoreMetrics {
        note_count: score.notes.len(),
        melody_note_count: score.voice(VoiceRole::Melody).len(),
        harmony_note_count: score.voice(VoiceRole::Harmony).len(),
        bass_note_count: score.voice(VoiceRole::Bass).len(),
        distinct_pitch_classes: pitch_classes.len(),
        distinct_onsets: onsets.len(),
        half_beat_offgrid_note_count,
        tempo_bpm: score.tempo_bpm,
        meter_beats_per_bar: score.meter,
        total_beats: score.total_beats.beats(),
    }
}

fn condition_receipt(
    seed: u64,
    condition: &str,
    interventions: &[Intervention],
) -> Result<(ConditionReceipt, Score, symthaea_music_theory::spec::CompositionSpec), String> {
    let (spec, score) = compose(seed, interventions);
    validate_score(&score).map_err(|reason| format!("{condition}, seed {seed}: {reason}"))?;
    let receipt = ConditionReceipt {
        condition: condition.into(),
        interventions: interventions.iter().map(|item| item.id().to_owned()).collect(),
        spec_sha256: sha256_serialized(&spec),
        score_sha256: sha256_serialized(&score),
        metrics: metrics(&score),
    };
    Ok((receipt, score, spec))
}

fn run_seed(seed: u64) -> Result<SeedReceipt, String> {
    let conditions: [(&str, &[Intervention]); 5] = [
        ("baseline", &[]),
        ("shuffle_only", &[Intervention::ShuffleCell]),
        ("dorian_only", &[Intervention::DorianMode]),
        (
            "combined_shuffle_then_dorian",
            &[Intervention::ShuffleCell, Intervention::DorianMode],
        ),
        (
            "combined_dorian_then_shuffle",
            &[Intervention::DorianMode, Intervention::ShuffleCell],
        ),
    ];

    let mut receipts = Vec::with_capacity(conditions.len());
    let mut scores = Vec::with_capacity(conditions.len());
    let mut specs = Vec::with_capacity(conditions.len());

    for (name, interventions) in conditions {
        let (receipt, score, spec) = condition_receipt(seed, name, interventions)?;
        receipts.push(receipt);
        scores.push(score);
        specs.push(spec);
    }

    let base = &receipts[0].metrics;
    let a = &receipts[1].metrics;
    let b = &receipts[2].metrics;
    let ab = &receipts[3].metrics;
    let reversed_equal = scores[3] == scores[4];
    let reversed_spec_equal = specs[3] == specs[4];

    Ok(SeedReceipt {
        seed,
        intent: intent(seed),
        conditions: receipts,
        interaction: InteractionDiagnostics {
            note_count_factorial_delta: ab.note_count as i64
                - a.note_count as i64
                - b.note_count as i64
                + base.note_count as i64,
            distinct_pitch_class_factorial_delta: ab.distinct_pitch_classes as i64
                - a.distinct_pitch_classes as i64
                - b.distinct_pitch_classes as i64
                + base.distinct_pitch_classes as i64,
            half_beat_offgrid_factorial_delta: ab.half_beat_offgrid_note_count as i64
                - a.half_beat_offgrid_note_count as i64
                - b.half_beat_offgrid_note_count as i64
                + base.half_beat_offgrid_note_count as i64,
            reversed_application_score_equal: reversed_equal,
            reversed_application_spec_equal: reversed_spec_equal,
        },
    })
}

fn valid_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn executable_digest() -> Result<(String, u64), String> {
    let path = env::current_exe().map_err(|error| format!("current_exe: {error}"))?;
    let bytes = fs::read(&path).map_err(|error| format!("read executable {}: {error}", path.display()))?;
    let size = u64::try_from(bytes.len()).map_err(|error| format!("executable size: {error}"))?;
    Ok((format!("{:x}", Sha256::digest(&bytes)), size))
}

fn main() -> ExitCode {
    let revision = env::var("SYMTHAEA_GIT_REVISION").unwrap_or_default();
    if !valid_sha1(&revision) {
        eprintln!("FAIL: set SYMTHAEA_GIT_REVISION to the exact 40-hex checkout HEAD; this is a caller claim, not build attestation.");
        return ExitCode::FAILURE;
    }

    let rustc_version = env::var("SYMTHAEA_RUSTC_VERSION").unwrap_or_default();
    if !rustc_version.starts_with("rustc ") {
        eprintln!("FAIL: set SYMTHAEA_RUSTC_VERSION to the exact `rustc --version` output; this is a caller claim, not independent attestation.");
        return ExitCode::FAILURE;
    }
    let (executable_sha256, executable_byte_count) = match executable_digest() {
        Ok(identity) => identity,
        Err(error) => {
            eprintln!("FAIL: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut seed_receipts = Vec::with_capacity(SEEDS.len());
    for seed in SEEDS {
        match run_seed(seed) {
            Ok(receipt) => seed_receipts.push(receipt),
            Err(error) => {
                eprintln!("FAIL: {error}");
                return ExitCode::FAILURE;
            }
        }
    }

    let order_metamorphic_pass = seed_receipts.iter().all(|item| {
        item.interaction.reversed_application_score_equal
            && item.interaction.reversed_application_spec_equal
    });
    let receipt = BenchmarkReceipt {
        schema_version: SCHEMA_VERSION,
        benchmark_id: "MUSE-STYLE-INTERACTION-001",
        source_revision_claim: revision,
        rustc_version_claim: rustc_version,
        executable_sha256,
        executable_byte_count,
        operating_system: std::env::consts::OS,
        target_architecture: std::env::consts::ARCH,
        evidence_boundary: "symbolic diagnostics only; not audio quality, style authenticity, or human preference",
        seed_set: SEEDS.to_vec(),
        pair_definition: "Classical structural owner with independent Shuffle-accompaniment and Dorian-mode interventions",
        condition_run_count: seed_receipts.iter().map(|item| item.conditions.len()).sum(),
        seed_receipts,
    };

    match serde_json::to_string_pretty(&receipt) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("FAIL: could not serialize report: {error}");
            return ExitCode::FAILURE;
        }
    }

    if !order_metamorphic_pass {
        eprintln!(
            "FAIL: independent shuffle/mode interventions changed output when application order was reversed."
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intervention_a_changes_only_the_declared_accompaniment_field() {
        let baseline = Style::Classical.spec();
        let mut expected = baseline.clone();
        expected.accompaniment_pool = vec![Accompaniment::Shuffle];
        assert_eq!(recipe(&[Intervention::ShuffleCell]), expected);
    }

    #[test]
    fn intervention_b_changes_only_the_declared_mode_field() {
        let baseline = Style::Classical.spec();
        let mut expected = baseline.clone();
        expected.mode = Some(Mode::Dorian);
        assert_eq!(recipe(&[Intervention::DorianMode]), expected);
    }

    #[test]
    fn independent_interventions_commute_at_recipe_level() {
        assert_eq!(
            recipe(&[Intervention::ShuffleCell, Intervention::DorianMode]),
            recipe(&[Intervention::DorianMode, Intervention::ShuffleCell])
        );
    }

    #[test]
    fn the_production_composer_replays_exactly_for_same_seed_and_recipe() {
        for seed in SEEDS {
            let (_, first) = compose(seed, &[Intervention::ShuffleCell, Intervention::DorianMode]);
            let (_, replay) = compose(seed, &[Intervention::ShuffleCell, Intervention::DorianMode]);
            assert_eq!(first, replay, "seed {seed}");
        }
    }

    #[test]
    fn every_frozen_condition_has_a_mechanically_valid_score() {
        for seed in SEEDS {
            for axes in [
                &[][..],
                &[Intervention::ShuffleCell][..],
                &[Intervention::DorianMode][..],
                &[Intervention::ShuffleCell, Intervention::DorianMode][..],
                &[Intervention::DorianMode, Intervention::ShuffleCell][..],
            ] {
                let (_, score) = compose(seed, axes);
                validate_score(&score).unwrap_or_else(|reason| {
                    panic!("seed {seed}, interventions {axes:?}: {reason}")
                });
            }
        }
    }

    #[test]
    fn the_two_interventions_have_an_observable_symbolic_effect_in_the_sweep() {
        let mut shuffle_effect_seen = false;
        let mut mode_effect_seen = false;
        for seed in SEEDS {
            let (_, baseline) = compose(seed, &[]);
            let (_, shuffle) = compose(seed, &[Intervention::ShuffleCell]);
            let (_, dorian) = compose(seed, &[Intervention::DorianMode]);
            shuffle_effect_seen |= baseline != shuffle;
            mode_effect_seen |= baseline != dorian;
        }
        assert!(shuffle_effect_seen, "shuffle had no symbolic effect on any frozen seed");
        assert!(mode_effect_seen, "Dorian mode had no symbolic effect on any frozen seed");
    }
}
