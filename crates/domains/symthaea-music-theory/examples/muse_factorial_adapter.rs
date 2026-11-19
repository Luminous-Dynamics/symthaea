//! Execute the preregistered Muse factorial manifest against the pinned
//! symbolic composer and emit analyzer-compatible measurement rows.
//!
//! This is an experiment adapter, not a production composition API. Run it
//! from the symthaea-music-theory package after the root workspace is healthy.
//!
//! Usage:
//!   cargo run -p symthaea-music-theory --example muse_factorial_adapter --
//!     discovery cases.csv observations.csv <40-character-source-commit>

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::process::{Command, ExitCode};

use serde_json::to_vec;
use sha2::{Digest, Sha256};
use symthaea_music_theory::score_validation::is_universal_invariant;
use symthaea_music_theory::{
    Emphasis, GrammarPlanEvidence, MusicalIntent, PitchClass, Score, ScoreValidationConfig,
    Style, Tonality, ValidationSeverity, VoiceRole, compose_with_grammar_plan, validate_score,
};

const PINNED_RUSTC_RELEASE: &str = "1.96.0";

const HEADER: &str =
    "case_id,cohort,style,seed,valence_code,arousal_code,energy_code,bars_code";

const METRICS: [&str; 54] = [
    "score_note_count",
    "score_total_beats",
    "score_total_seconds",
    "score_tempo_bpm",
    "score_notes_per_beat",
    "score_mean_pitch_midi",
    "score_pitch_span_semitones",
    "score_pitch_stddev_midi",
    "score_mean_velocity",
    "score_velocity_stddev",
    "score_mean_section_intensity",
    "score_mean_note_duration_beats",
    "score_assigned_part_fraction",
    "score_universal_fatal_count",
    "score_style_fatal_count",
    "score_warning_count",
    "score_universal_valid",
    "score_full_validation_valid",
    "score_pitch_class_diversity",
    "score_pitch_class_entropy_bits",
    "score_melody_mean_interval_semitones",
    "score_melody_step_fraction",
    "score_melody_leap_fraction",
    "score_onset_ioi_stddev_beats",
    "score_offbeat_onset_fraction",
    "score_voice_role_entropy_bits",
    "score_duration_variety_count",
    "score_section_intensity_stddev",
    "identity_score_meter_beats",
    "score_distinct_midi_pitches",
    "identity_score_key_tonic_pc",
    "score_key_major_indicator",
    "score_key_minor_indicator",
    "score_key_modal_indicator",
    "score_emphasis_normal_count",
    "score_emphasis_climax_count",
    "score_emphasis_cadential_count",
    "score_emphasis_phrase_start_count",
    "score_form_section_count",
    "score_plan_requested_bars",
    "score_plan_realized_bars",
    "score_fingerprint_form_distance",
    "score_fingerprint_harmony_distance",
    "score_fingerprint_orchestration_distance",
    "score_fingerprint_rhythm_distance",
    "score_fingerprint_contour_distance",
    "score_fingerprint_tempo_meter_distance",
    "score_fingerprint_pitch_class_distance",
    "score_fingerprint_interval_distance",
    "score_fingerprint_chord_quality_distance",
    "score_fingerprint_cadence_distance",
    "score_fingerprint_motif_distance",
    "identity_exact_fingerprint_high32",
    "identity_exact_fingerprint_low32",
];

const CONFIRMATION_STYLES: [&str; 6] = [
    "Classical",
    "Fugue",
    "Passacaglia",
    "Blues",
    "AfroCuban",
    "HindustaniInspired",
];

const DISCOVERY_SEEDS: [u64; 4] = [90_210, 130_363, 222_731, 918_273];

const PINNED_COMPOSER_SOURCE_COMMIT: &str =
    "bd227d416878a4373eb87a53e182e7067a2411a5";

const PINNED_COMPOSER_BLOBS: [(&str, &str); 15] = [
    (
        "crates/domains/symthaea-music-theory/src/style.rs",
        "81192045a724823b47107702185b814cb9921cbf",
    ),
    (
        "crates/domains/symthaea-music-theory/src/grammar.rs",
        "a44ec71c774cc749dcdbded96ec2ced5500ccd7c",
    ),
    (
        "crates/domains/symthaea-music-theory/src/composer.rs",
        "e9a7bf7de176a8f8091163c6809bb4e8855b3a5a",
    ),
    (
        "crates/domains/symthaea-music-theory/src/spec.rs",
        "2ea178ab065ab7582ab6d3619818b88a3da5787e",
    ),
    (
        "crates/domains/symthaea-music-theory/src/fingerprint.rs",
        "97c4f127a1b460f424cb4201d4290c89899932ae",
    ),
    (
        "crates/domains/symthaea-music-theory/src/score.rs",
        "7e42614f0bd79bb5d769d4bea7d289ebf6c503c8",
    ),
    (
        "crates/domains/symthaea-music-theory/examples/style_control_response_curves.rs",
        "bf51ab11df3c1ca2c4ca28b894e8597eaef490e3",
    ),
    ("Cargo.lock", "53f6c6da5ff9c253682fa331264897472f2fa2c3"),
    (
        "crates/domains/symthaea-music-theory/src/lib.rs",
        "ecfc73d187418cac7168794ea20e07dd70fc6cf4",
    ),
    (
        "crates/domains/symthaea-music-theory/src/score_validation.rs",
        "2f139e2c1f67e67603b2391149007cf1385f4c36",
    ),
    (
        "crates/domains/symthaea-music-theory/src/motif.rs",
        "c69edee65b92374c7517b453edf9a34121214a71",
    ),
    (
        "crates/domains/symthaea-music-theory/src/rhythm.rs",
        "444f70eec498a79e77c5cf041cde45458eda2bc6",
    ),
    (
        "crates/domains/symthaea-music-theory/Cargo.toml",
        "2b2ce7caed933be6837ca50f2f34ddd254de224a",
    ),
    ("Cargo.toml", "9a6eafdce0b03ab08aef24dee6ed384cab2a63f3"),
    ("rust-toolchain.toml", "4f0430eac96d545bcfaa0df23ce475faf4aee96a"),
];

const CONFIRMATION_SEEDS: [u64; 12] = [
    1_000_003,
    1_000_033,
    1_000_037,
    1_000_039,
    1_000_081,
    1_000_099,
    1_000_117,
    1_000_121,
    1_000_133,
    1_000_151,
    1_000_159,
    1_000_171,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Cohort {
    Discovery,
    Confirmation,
}

impl Cohort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Discovery => "DISCOVERY",
            Self::Confirmation => "CONFIRMATION",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cell {
    valence: u8,
    arousal: u8,
    energy: u8,
    bars: u8,
}

impl Cell {
    fn all() -> Vec<Self> {
        let mut cells = Vec::with_capacity(81);
        for valence in 0..3 {
            for arousal in 0..3 {
                for energy in 0..3 {
                    for bars in 0..3 {
                        cells.push(Self {
                            valence,
                            arousal,
                            energy,
                            bars,
                        });
                    }
                }
            }
        }
        cells
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CaseRow {
    case_id: String,
    cohort: Cohort,
    style: Style,
    style_name: String,
    seed: u64,
    cell: Cell,
}

#[derive(Debug, Clone)]
struct Measurement {
    case_id: String,
    metric: &'static str,
    status: &'static str,
    value: Option<f64>,
    artifact_sha256: String,
    note: String,
}

#[derive(Debug, Clone)]
struct ReferenceArtifact {
    structural_fingerprint: [f64; symthaea_music_theory::fingerprint::STRUCT_DIMS],
    exact_fingerprint: u64,
}

#[derive(Debug)]
struct ComposedArtifact {
    metrics: Vec<(&'static str, Option<f64>)>,
    artifact_sha256: String,
    note: String,
    reference: Option<ReferenceArtifact>,
}

fn style_name(style: Style) -> String {
    format!("{style:?}")
}

fn case_id(cohort: Cohort, style: &str, seed: u64, cell: Cell) -> String {
    format!(
        "{}__{}__s{}__v{}a{}e{}b{}",
        cohort.as_str(),
        style,
        seed,
        cell.valence,
        cell.arousal,
        cell.energy,
        cell.bars
    )
}

fn style_from_name(name: &str) -> Result<Style, String> {
    Style::ALL
        .into_iter()
        .find(|style| style_name(*style) == name)
        .ok_or_else(|| format!("style {name:?} is not in Style::ALL"))
}

fn parse_cohort(value: &str) -> Result<Vec<Cohort>, String> {
    match value {
        "discovery" => Ok(vec![Cohort::Discovery]),
        "confirmation" => Ok(vec![Cohort::Confirmation]),
        "both" => Ok(vec![Cohort::Discovery, Cohort::Confirmation]),
        other => Err(format!(
            "unknown cohort selector {other:?}; expected discovery, confirmation, or both"
        )),
    }
}

fn expected_cases(cohorts: &[Cohort]) -> Vec<CaseRow> {
    let mut result = Vec::new();
    for cohort in cohorts {
        let seeds: &[u64] = match cohort {
            Cohort::Discovery => &DISCOVERY_SEEDS,
            Cohort::Confirmation => &CONFIRMATION_SEEDS,
        };
        for style in Style::ALL {
            let name = style_name(style);
            if *cohort == Cohort::Confirmation
                && !CONFIRMATION_STYLES.contains(&name.as_str())
            {
                continue;
            }
            for seed in seeds {
                for cell in Cell::all() {
                    result.push(CaseRow {
                        case_id: case_id(*cohort, &name, *seed, cell),
                        cohort: *cohort,
                        style,
                        style_name: name.clone(),
                        seed: *seed,
                        cell,
                    });
                }
            }
        }
    }
    result
}

fn parse_manifest(input: &str, expected: &[CaseRow]) -> Result<Vec<CaseRow>, String> {
    let mut lines = input.lines();
    let header = lines
        .next()
        .ok_or_else(|| "case manifest is empty".to_string())?;
    if header.trim_start_matches('\u{feff}') != HEADER {
        return Err(format!("unexpected case manifest header {header:?}"));
    }

    let expected_by_id: HashMap<&str, &CaseRow> = expected
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect();
    if expected_by_id.len() != expected.len() {
        return Err("internal expected case manifest contains duplicate IDs".to_string());
    }

    let mut seen = HashSet::with_capacity(expected.len());
    for (offset, raw_line) in lines.enumerate() {
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            return Err(format!("blank manifest row at line {}", offset + 2));
        }
        let columns: Vec<&str> = line.split(',').collect();
        if columns.len() != 8 {
            return Err(format!(
                "manifest row {} has {} columns; expected 8",
                offset + 2,
                columns.len()
            ));
        }

        let cohort = match columns[1] {
            "DISCOVERY" => Cohort::Discovery,
            "CONFIRMATION" => Cohort::Confirmation,
            other => return Err(format!("unknown cohort {other:?} at row {}", offset + 2)),
        };
        let style = style_from_name(columns[2])?;
        let seed = columns[3]
            .parse::<u64>()
            .map_err(|error| format!("invalid seed at row {}: {error}", offset + 2))?;
        let parse_code = |column: &str, name: &str| -> Result<u8, String> {
            let value = column
                .parse::<u8>()
                .map_err(|error| format!("invalid {name} at row {}: {error}", offset + 2))?;
            if value > 2 {
                return Err(format!(
                    "{name} is outside 0..=2 at row {}",
                    offset + 2
                ));
            }
            Ok(value)
        };
        let cell = Cell {
            valence: parse_code(columns[4], "valence_code")?,
            arousal: parse_code(columns[5], "arousal_code")?,
            energy: parse_code(columns[6], "energy_code")?,
            bars: parse_code(columns[7], "bars_code")?,
        };
        let actual = CaseRow {
            case_id: columns[0].to_string(),
            cohort,
            style,
            style_name: columns[2].to_string(),
            seed,
            cell,
        };

        let expected_case = expected_by_id
            .get(actual.case_id.as_str())
            .ok_or_else(|| {
                format!(
                    "unexpected or mutated case ID {:?} at row {}",
                    actual.case_id,
                    offset + 2
                )
            })?;
        if &actual != *expected_case {
            return Err(format!(
                "case fields disagree with the frozen manifest for {:?} at row {}",
                actual.case_id,
                offset + 2
            ));
        }
        if !seen.insert(actual.case_id.clone()) {
            return Err(format!("duplicate case ID {:?} at row {}", actual.case_id, offset + 2));
        }
    }

    if seen.len() != expected.len() {
        return Err(format!(
            "manifest is incomplete: expected {} unique cases, found {}",
            expected.len(),
            seen.len()
        ));
    }
    Ok(expected.to_vec())
}

fn centre_cell() -> Cell {
    Cell {
        valence: 1,
        arousal: 1,
        energy: 1,
        bars: 1,
    }
}

fn centre_first_cases(cases: &[CaseRow]) -> Result<Vec<CaseRow>, String> {
    if cases.is_empty() || cases.len() % 81 != 0 {
        return Err(format!(
            "case list must contain complete 81-case blocks, found {} rows",
            cases.len()
        ));
    }
    let mut ordered = Vec::with_capacity(cases.len());
    for (block_index, block) in cases.chunks_exact(81).enumerate() {
        let Some(centre_index) = block.iter().position(|case| case.cell == centre_cell()) else {
            return Err(format!("block {block_index} has no centre reference condition"));
        };
        ordered.push(block[centre_index].clone());
        ordered.extend(
            block
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != centre_index)
                .map(|(_, case)| case.clone()),
        );
    }
    Ok(ordered)
}

fn factor_value(code: u8, levels: [f32; 3]) -> f32 {
    levels[code as usize]
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn population_stddev(values: &[f64]) -> Option<f64> {
    let average = mean(values)?;
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt())
}

fn categorical_entropy_bits(counts: &[usize], total: usize) -> Option<f64> {
    if total == 0 {
        return None;
    }
    Some(
        counts
            .iter()
            .filter(|count| **count > 0)
            .map(|count| {
                let probability = *count as f64 / total as f64;
                -probability * probability.log2()
            })
            .sum(),
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn compose_case(
    case: &CaseRow,
    source_commit: &str,
    manifest_sha256: &str,
    rustc_version: &str,
    reference: Option<&ReferenceArtifact>,
) -> Result<ComposedArtifact, String> {
    let spec = case.style.spec();
    spec.validate()
        .map_err(|errors| format!("style spec failed validation: {}", errors.join("; ")))?;

    let intent = MusicalIntent {
        valence: factor_value(case.cell.valence, [-1.0, 0.0, 1.0]),
        arousal: factor_value(case.cell.arousal, [0.0, 0.5, 1.0]),
        energy: factor_value(case.cell.energy, [0.0, 0.5, 1.0]),
        bars: [8, 16, 24][case.cell.bars as usize],
        seed: case.seed,
        tonic: PitchClass::C,
    };
    let is_centre = case.cell == centre_cell();
    if is_centre != reference.is_none() {
        return Err("centre-reference execution ordering contract was violated".to_string());
    }

    let realization =
        compose_with_grammar_plan(case.style.grammar_profile(), &intent, &spec);
    let score: Score = realization.score;
    let validation = validate_score(&score, &ScoreValidationConfig::default());

    let artifact_bytes = to_vec(&score)
        .map_err(|error| format!("cannot serialize symbolic score: {error}"))?;
    let artifact_sha256 = sha256_hex(&artifact_bytes);
    let current_fingerprint =
        symthaea_music_theory::fingerprint::structural_fingerprint(&score, &realization.form);
    let current_exact_fingerprint =
        symthaea_music_theory::fingerprint::exact_fingerprint(&score);
    let layer_distances = match reference {
        Some(reference) => symthaea_music_theory::fingerprint::layer_dists(
            &current_fingerprint,
            &reference.structural_fingerprint,
        ),
        None => [0.0; symthaea_music_theory::fingerprint::LAYERS.len()],
    };

    let note_count = score.notes.len();
    let pitches: Vec<f64> = score
        .notes
        .iter()
        .map(|note| f64::from(note.pitch.midi()))
        .collect();
    let velocities: Vec<f64> = score
        .notes
        .iter()
        .map(|note| f64::from(note.velocity))
        .collect();
    let intensities: Vec<f64> = score
        .notes
        .iter()
        .map(|note| f64::from(note.section_intensity))
        .collect();
    let durations: Vec<f64> = score
        .notes
        .iter()
        .map(|note| note.duration.beats())
        .collect();

    let pitch_span = if pitches.is_empty() {
        None
    } else {
        let min = pitches.iter().copied().fold(f64::INFINITY, f64::min);
        let max = pitches
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Some(max - min)
    };

    let universal_fatal_count = validation
        .issues
        .iter()
        .filter(|issue| {
            issue.severity == ValidationSeverity::Fatal && is_universal_invariant(issue.rule)
        })
        .count();
    let all_fatal_count = validation
        .issues
        .iter()
        .filter(|issue| issue.severity == ValidationSeverity::Fatal)
        .count();
    let warning_count = validation
        .issues
        .iter()
        .filter(|issue| issue.severity == ValidationSeverity::Warning)
        .count();

    let total_beats = score.total_beats.beats();
    let notes_per_beat = if total_beats.is_finite() && total_beats > 0.0 {
        Some(note_count as f64 / total_beats)
    } else {
        None
    };
    let assigned_part_fraction = if note_count == 0 {
        None
    } else {
        Some(
            score
                .notes
                .iter()
                .filter(|note| note.part.is_assigned())
                .count() as f64
                / note_count as f64,
        )
    };

    let mut pitch_class_counts = [0usize; 12];
    let mut distinct_midi_pitches = HashSet::new();
    for note in &score.notes {
        pitch_class_counts[(note.pitch.midi() % 12) as usize] += 1;
        distinct_midi_pitches.insert(note.pitch.midi());
    }
    let pitch_class_diversity = pitch_class_counts.iter().filter(|count| **count > 0).count();
    let pitch_class_entropy = categorical_entropy_bits(&pitch_class_counts, note_count);

    let mut melody_notes: Vec<_> = score
        .notes
        .iter()
        .filter(|note| note.role == VoiceRole::Melody)
        .collect();
    melody_notes.sort_by(|left, right| left.onset.beats().total_cmp(&right.onset.beats()));
    let melody_intervals: Vec<f64> = melody_notes
        .windows(2)
        .map(|pair| {
            (f64::from(pair[1].pitch.midi()) - f64::from(pair[0].pitch.midi())).abs()
        })
        .collect();
    let melody_mean_interval = mean(&melody_intervals);
    let melody_step_fraction = if melody_intervals.is_empty() {
        None
    } else {
        Some(
            melody_intervals.iter().filter(|interval| **interval <= 2.0).count() as f64
                / melody_intervals.len() as f64,
        )
    };
    let melody_leap_fraction = if melody_intervals.is_empty() {
        None
    } else {
        Some(
            melody_intervals.iter().filter(|interval| **interval > 7.0).count() as f64
                / melody_intervals.len() as f64,
        )
    };

    let mut onsets: Vec<f64> = score.notes.iter().map(|note| note.onset.beats()).collect();
    onsets.sort_by(f64::total_cmp);
    let onset_intervals: Vec<f64> = onsets.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let onset_ioi_stddev = population_stddev(&onset_intervals);
    let offbeat_onset_fraction = if onsets.is_empty() {
        None
    } else {
        Some(
            onsets
                .iter()
                .filter(|onset| (*onset - onset.round()).abs() > 1.0e-9)
                .count() as f64
                / onsets.len() as f64,
        )
    };

    let mut role_counts = [0usize; 4];
    let mut emphasis_counts = [0usize; 4];
    for note in &score.notes {
        let role_index = match note.role {
            VoiceRole::Melody => 0,
            VoiceRole::Harmony => 1,
            VoiceRole::Bass => 2,
            VoiceRole::CounterMelody => 3,
        };
        role_counts[role_index] += 1;

        let emphasis_index = match note.emphasis {
            Emphasis::Normal => 0,
            Emphasis::Climax => 1,
            Emphasis::Cadential => 2,
            Emphasis::PhraseStart => 3,
        };
        emphasis_counts[emphasis_index] += 1;
    }
    let role_entropy = categorical_entropy_bits(&role_counts, note_count);

    let duration_variety_count = if durations.is_empty() {
        None
    } else {
        let distinct: HashSet<u64> = durations.iter().map(|duration| duration.to_bits()).collect();
        Some(distinct.len() as f64)
    };
    let intensity_stddev = population_stddev(&intensities);
    let is_major = matches!(score.key.tonality, Tonality::Major);
    let is_minor = matches!(score.key.tonality, Tonality::Minor);
    let is_modal = matches!(score.key.tonality, Tonality::Modal(_));

    let (plan_kind, plan_requested_bars, plan_realized_bars) = match &realization.plan {
        GrammarPlanEvidence::CallResponse(plan) => (
            "call_response",
            Some(plan.requested_bars as f64),
            Some(plan.realized_bars as f64),
        ),
        GrammarPlanEvidence::JazzChorus(plan) => (
            "jazz_chorus",
            Some(plan.requested_bars as f64),
            Some(plan.realized_bars as f64),
        ),
        GrammarPlanEvidence::PeriodSentence(_) => ("period_sentence", None, None),
        GrammarPlanEvidence::Contrapuntal(_) => ("contrapuntal", None, None),
        GrammarPlanEvidence::GrooveCycle(_) => ("groove_cycle", None, None),
        GrammarPlanEvidence::AdditiveProcess(_) => ("additive_process", None, None),
        GrammarPlanEvidence::ModalArc(_) => ("modal_arc", None, None),
        GrammarPlanEvidence::Compatibility { .. } => ("compatibility", None, None),
    };
    let form_section_count = realization
        .form
        .as_ref()
        .map(|form| form.sections.len() as f64);

    let exact_high = ((current_exact_fingerprint >> 32) as u32) as f64;
    let exact_low = (current_exact_fingerprint as u32) as f64;

    let metrics = vec![
        ("score_note_count", Some(note_count as f64)),
        ("score_total_beats", Some(total_beats)),
        ("score_total_seconds", Some(score.seconds())),
        ("score_tempo_bpm", Some(f64::from(score.tempo_bpm))),
        ("score_notes_per_beat", notes_per_beat),
        ("score_mean_pitch_midi", mean(&pitches)),
        ("score_pitch_span_semitones", pitch_span),
        ("score_pitch_stddev_midi", population_stddev(&pitches)),
        ("score_mean_velocity", mean(&velocities)),
        ("score_velocity_stddev", population_stddev(&velocities)),
        ("score_mean_section_intensity", mean(&intensities)),
        ("score_mean_note_duration_beats", mean(&durations)),
        ("score_assigned_part_fraction", assigned_part_fraction),
        ("score_universal_fatal_count", Some(universal_fatal_count as f64)),
        (
            "score_style_fatal_count",
            Some(all_fatal_count.saturating_sub(universal_fatal_count) as f64),
        ),
        ("score_warning_count", Some(warning_count as f64)),
        (
            "score_universal_valid",
            Some(if universal_fatal_count == 0 { 1.0 } else { 0.0 }),
        ),
        (
            "score_full_validation_valid",
            Some(if validation.valid { 1.0 } else { 0.0 }),
        ),
        ("score_pitch_class_diversity", Some(pitch_class_diversity as f64)),
        ("score_pitch_class_entropy_bits", pitch_class_entropy),
        ("score_melody_mean_interval_semitones", melody_mean_interval),
        ("score_melody_step_fraction", melody_step_fraction),
        ("score_melody_leap_fraction", melody_leap_fraction),
        ("score_onset_ioi_stddev_beats", onset_ioi_stddev),
        ("score_offbeat_onset_fraction", offbeat_onset_fraction),
        ("score_voice_role_entropy_bits", role_entropy),
        ("score_duration_variety_count", duration_variety_count),
        ("score_section_intensity_stddev", intensity_stddev),
        ("identity_score_meter_beats", Some(f64::from(score.meter))),
        ("score_distinct_midi_pitches", Some(distinct_midi_pitches.len() as f64)),
        ("identity_score_key_tonic_pc", Some(f64::from(score.key.tonic.value()))),
        ("score_key_major_indicator", Some(if is_major { 1.0 } else { 0.0 })),
        ("score_key_minor_indicator", Some(if is_minor { 1.0 } else { 0.0 })),
        ("score_key_modal_indicator", Some(if is_modal { 1.0 } else { 0.0 })),
        ("score_emphasis_normal_count", Some(emphasis_counts[0] as f64)),
        ("score_emphasis_climax_count", Some(emphasis_counts[1] as f64)),
        ("score_emphasis_cadential_count", Some(emphasis_counts[2] as f64)),
        ("score_emphasis_phrase_start_count", Some(emphasis_counts[3] as f64)),
        ("score_form_section_count", form_section_count),
        ("score_plan_requested_bars", plan_requested_bars),
        ("score_plan_realized_bars", plan_realized_bars),
        ("score_fingerprint_form_distance", Some(layer_distances[0])),
        ("score_fingerprint_harmony_distance", Some(layer_distances[1])),
        ("score_fingerprint_orchestration_distance", Some(layer_distances[2])),
        ("score_fingerprint_rhythm_distance", Some(layer_distances[3])),
        ("score_fingerprint_contour_distance", Some(layer_distances[4])),
        ("score_fingerprint_tempo_meter_distance", Some(layer_distances[5])),
        ("score_fingerprint_pitch_class_distance", Some(layer_distances[6])),
        ("score_fingerprint_interval_distance", Some(layer_distances[7])),
        ("score_fingerprint_chord_quality_distance", Some(layer_distances[8])),
        ("score_fingerprint_cadence_distance", Some(layer_distances[9])),
        ("score_fingerprint_motif_distance", Some(layer_distances[10])),
        ("identity_exact_fingerprint_high32", Some(exact_high)),
        ("identity_exact_fingerprint_low32", Some(exact_low)),
    ];

    let note = format!(
        "source_commit={source_commit};composer_reference_commit={};manifest_sha256={manifest_sha256};rustc_version={rustc_version};engine_version={};grammar_family={:?};support_axes={};tonality={:?};plan_kind={plan_kind};universal_fatal={universal_fatal_count};style_fatal={};warnings={warning_count};validation_valid={};exact_fingerprint={current_exact_fingerprint:016x};reference_exact_fingerprint={:016x}",
        PINNED_COMPOSER_SOURCE_COMMIT,
        symthaea_music_theory::MUSIC_THEORY_ENGINE_VERSION,
        case.style.grammar_family(),
        case.style.supported_intent_axes().join("|"),
        score.key.tonality,
        all_fatal_count.saturating_sub(universal_fatal_count),
        validation.valid,
        reference
            .map(|reference| reference.exact_fingerprint)
            .unwrap_or(current_exact_fingerprint)
    );

    Ok(ComposedArtifact {
        metrics,
        artifact_sha256,
        note,
        reference: is_centre.then_some(ReferenceArtifact {
            structural_fingerprint: current_fingerprint,
            exact_fingerprint: current_exact_fingerprint,
        }),
    })
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_row(fields: &[String]) -> String {
    let mut line = fields
        .iter()
        .map(|field| csv_escape(field))
        .collect::<Vec<_>>()
        .join(",");
    line.push('\n');
    line
}

fn failed_measurements(
    case_id: &str,
    error: &str,
    note: &str,
    rustc_version: &str,
) -> Vec<Measurement> {
    METRICS
        .iter()
        .map(|metric| Measurement {
            case_id: case_id.to_string(),
            metric: *metric,
            status: "failed",
            value: None,
            artifact_sha256: String::new(),
            note: format!("{note};rustc_version={rustc_version};composer_failure={error}"),
        })
        .collect()
}

fn measurements_for_case(
    case: &CaseRow,
    source_commit: &str,
    manifest_sha256: &str,
    rustc_version: &str,
    reference: Option<&ReferenceArtifact>,
) -> (Vec<Measurement>, Option<ReferenceArtifact>) {
    let composed = catch_unwind(AssertUnwindSafe(|| {
        compose_case(case, source_commit, manifest_sha256, rustc_version, reference)
    }));

    match composed {
        Ok(Ok(artifact)) => {
            let ComposedArtifact {
                metrics,
                artifact_sha256,
                note,
                reference,
            } = artifact;
            let measurements = metrics
                .into_iter()
                .map(|(metric, value)| match value {
                    Some(value) if value.is_finite() => Measurement {
                        case_id: case.case_id.clone(),
                        metric,
                        status: "observed",
                        value: Some(value),
                        artifact_sha256: artifact_sha256.clone(),
                        note: note.clone(),
                    },
                    Some(_) => Measurement {
                        case_id: case.case_id.clone(),
                        metric,
                        status: "failed",
                        value: None,
                        artifact_sha256: artifact_sha256.clone(),
                        note: format!("{note};metric_value_non_finite"),
                    },
                    None => Measurement {
                        case_id: case.case_id.clone(),
                        metric,
                        status: "not_applicable",
                        value: None,
                        artifact_sha256: artifact_sha256.clone(),
                        note: note.clone(),
                    },
                })
                .collect();
            (measurements, reference)
        }
        Ok(Err(error)) => (
            failed_measurements(
                &case.case_id,
                &error,
                &format!("source_commit={source_commit};manifest_sha256={manifest_sha256}"),
                rustc_version,
            ),
            None,
        ),
        Err(_) => (
            failed_measurements(
                &case.case_id,
                "composer_panicked",
                &format!("source_commit={source_commit};manifest_sha256={manifest_sha256}"),
                rustc_version,
            ),
            None,
        ),
    }
}

fn is_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn git_output(args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .map_err(|error| format!("cannot execute git {:?}: {error}", args))?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed with status {}",
            args,
            output.status
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("git {:?} output is not UTF-8: {error}", args))
}

fn rustc_version() -> Result<String, String> {
    let output = Command::new("rustc")
        .arg("--version")
        .output()
        .map_err(|error| format!("cannot execute rustc --version: {error}"))?;
    if !output.status.success() {
        return Err(format!("rustc --version failed with status {}", output.status));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("rustc --version output is not UTF-8: {error}"))
}

fn is_pinned_rustc_version(version: &str) -> bool {
    version == "rustc 1.96.0" || version.starts_with("rustc 1.96.0 (")
}

fn verify_rustc_version() -> Result<String, String> {
    let version = rustc_version()?;
    if !is_pinned_rustc_version(&version) {
        return Err(format!(
            "factorial measurements require rustc {}; found {version}",
            PINNED_RUSTC_RELEASE
        ));
    }
    Ok(version)
}

fn verify_source_revision(source_commit: &str) -> Result<(), String> {
    let head = git_output(&["rev-parse", "HEAD"])?;
    if head != source_commit {
        return Err(format!(
            "provided source commit {source_commit} does not match checked-out HEAD {head}"
        ));
    }
    let dirty = git_output(&["status", "--porcelain", "--untracked-files=all"])?;
    if !dirty.is_empty() {
        return Err(format!(
            "source worktree is not clean; refusing to bind measurements to commit {head}"
        ));
    }

    let pinned_commit_ref = format!("{PINNED_COMPOSER_SOURCE_COMMIT}^{{commit}}");
    git_output(&["cat-file", "-e", &pinned_commit_ref]).map_err(|_| {
        format!(
            "pinned composer reference commit {} is not present locally; fetch the full history of origin/muse/style-genre-profile-v1 before running the adapter",
            PINNED_COMPOSER_SOURCE_COMMIT
        )
    })?;

    for (path, expected_blob) in PINNED_COMPOSER_BLOBS {
        let revision = format!("HEAD:{path}");
        let actual_blob = git_output(&["rev-parse", &revision])?;
        if actual_blob != expected_blob {
            return Err(format!(
                "pinned composer source mismatch for {path}: expected blob {expected_blob}, got {actual_blob}"
            ));
        }
    }

    // Pin the complete composer source subtree, not just the hand-selected
    // interface files above. This prevents an unreviewed transitive module
    // change (e.g. fugue/process/modal realization) from riding along.
    git_output(&[
        "diff",
        "--quiet",
        PINNED_COMPOSER_SOURCE_COMMIT,
        "HEAD",
        "--",
        "crates/domains/symthaea-music-theory/src",
    ])
    .map_err(|_| {
        "symbolic music-theory source tree differs from the frozen composer source commit"
            .to_string()
    })?;
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: muse_factorial_adapter <discovery|confirmation|both> <case-manifest.csv> <observations.csv> <source-commit-sha>"
                .to_string(),
        );
    }

    let cohorts = parse_cohort(&args[0])?;
    let manifest_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    if manifest_path == output_path {
        return Err("manifest and output paths must be distinct".to_string());
    }
    if !is_commit_sha(&args[3]) {
        return Err("source commit must be exactly 40 hexadecimal characters".to_string());
    }
    verify_source_revision(&args[3])?;
    let rustc_version = verify_rustc_version()?;

    let manifest_bytes = fs::read(manifest_path)
        .map_err(|error| format!("cannot read {}: {error}", manifest_path.display()))?;
    let manifest_text = std::str::from_utf8(&manifest_bytes)
        .map_err(|error| format!("case manifest is not UTF-8: {error}"))?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let expected = expected_cases(&cohorts);
    let cases = parse_manifest(manifest_text, &expected)?;

    let file = fs::File::create(output_path)
        .map_err(|error| format!("cannot create {}: {error}", output_path.display()))?;
    let mut output = BufWriter::new(file);
    writeln!(
        output,
        "case_id,metric,status,value,artifact_sha256,note"
    )
    .map_err(|error| format!("cannot write observation header: {error}"))?;

    let execution_cases = centre_first_cases(&cases)?;
    let mut reference_by_block: HashMap<(Cohort, Style, u64), ReferenceArtifact> =
        HashMap::new();
    let mut failed_cases = 0usize;
    let mut not_applicable_rows = 0usize;
    let mut observed_rows = 0usize;
    for (index, case) in execution_cases.iter().enumerate() {
        let key = (case.cohort, case.style, case.seed);
        let is_centre = case.cell == centre_cell();
        let reference = if is_centre {
            None
        } else {
            reference_by_block.get(&key)
        };
        let (measurements, reference_artifact) = if is_centre || reference.is_some() {
            measurements_for_case(case, &args[3], &manifest_sha256, &rustc_version, reference)
        } else {
            (
                failed_measurements(
                    &case.case_id,
                    "centre_reference_unavailable",
                    &format!("source_commit={};manifest_sha256={manifest_sha256}", args[3]),
                    &rustc_version,
                ),
                None,
            )
        };
        if let Some(reference_artifact) = reference_artifact {
            if reference_by_block.insert(key, reference_artifact).is_some() {
                return Err(format!(
                    "duplicate centre reference produced for {}/{}/seed {}",
                    case.cohort.as_str(),
                    case.style_name,
                    case.seed
                ));
            }
        }
        let case_failed = measurements.iter().any(|row| row.status == "failed");
        if case_failed {
            failed_cases += 1;
        }
        for measurement in measurements {
            match measurement.status {
                "observed" => observed_rows += 1,
                "not_applicable" => not_applicable_rows += 1,
                "failed" => {}
                _ => return Err("internal adapter status is invalid".to_string()),
            }
            let fields = vec![
                measurement.case_id,
                measurement.metric.to_string(),
                measurement.status.to_string(),
                measurement.value.map(|value| value.to_string()).unwrap_or_default(),
                measurement.artifact_sha256,
                measurement.note,
            ];
            output
                .write_all(csv_row(&fields).as_bytes())
                .map_err(|error| format!("cannot write observation row: {error}"))?;
        }
        if (index + 1) % 256 == 0 {
            eprintln!(
                "composed {} / {} cases ({} failed cases)",
                index + 1,
                execution_cases.len(),
                failed_cases
            );
        }
    }
    output
        .flush()
        .map_err(|error| format!("cannot flush observations: {error}"))?;

    eprintln!(
        "case_count={}, metric_count={}, observed_rows={}, not_applicable_rows={}, failed_cases={}, source_commit={}, manifest_sha256={}, rustc_version={}",
        cases.len(),
        METRICS.len(),
        observed_rows,
        not_applicable_rows,
        failed_cases,
        args[3],
        manifest_sha256,
        rustc_version
    );

    if failed_cases > 0 {
        return Err(format!(
            "composer adapter produced {} failed cases; inspect the CSV and run analyze-results to preserve the incomplete denominator",
            failed_cases
        ));
    }
    if not_applicable_rows > 0 {
        eprintln!(
            "adapter completed with {} not-applicable metric rows; the result analyzer will retain them and suppress incomplete-block estimates",
            not_applicable_rows
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse-factorial-adapter: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_discovery_and_confirmation_manifests_have_frozen_sizes() {
        assert_eq!(expected_cases(&[Cohort::Discovery]).len(), 9_396);
        assert_eq!(expected_cases(&[Cohort::Confirmation]).len(), 5_832);
        assert_eq!(
            expected_cases(&[Cohort::Discovery, Cohort::Confirmation]).len(),
            15_228
        );
    }

    #[test]
    fn generated_case_manifest_round_trips_and_rejects_missing_rows() {
        let expected = expected_cases(&[Cohort::Confirmation]);
        let mut csv = String::from(HEADER);
        csv.push('\n');
        for case in &expected {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                case.case_id,
                case.cohort.as_str(),
                case.style_name,
                case.seed,
                case.cell.valence,
                case.cell.arousal,
                case.cell.energy,
                case.cell.bars
            ));
        }
        assert_eq!(parse_manifest(&csv, &expected).unwrap(), expected);

        let truncated = csv.lines().take(100).collect::<Vec<_>>().join("\n");
        assert!(parse_manifest(&truncated, &expected).is_err());
    }

    #[test]
    fn centre_reference_is_first_in_every_style_seed_block() {
        let cases = expected_cases(&[Cohort::Confirmation]);
        let ordered = centre_first_cases(&cases).unwrap();
        assert_eq!(ordered.len(), cases.len());
        for block in ordered.chunks_exact(81) {
            assert_eq!(block[0].cell, centre_cell());
            assert_eq!(
                block.iter().filter(|case| case.cell == centre_cell()).count(),
                1
            );
        }
    }

    #[test]
    fn metric_registry_is_unique_and_matches_the_contract() {
        assert_eq!(METRICS.len(), 54);
        let unique: HashSet<&str> = METRICS.iter().copied().collect();
        assert_eq!(unique.len(), METRICS.len());
    }

    #[test]
    fn pinned_composer_manifest_has_fifteen_valid_blob_pins() {
        assert_eq!(PINNED_COMPOSER_BLOBS.len(), 15);
        assert!(is_commit_sha(PINNED_COMPOSER_SOURCE_COMMIT));
        for (path, blob) in PINNED_COMPOSER_BLOBS {
            assert!(!path.is_empty());
            assert!(is_commit_sha(blob));
        }
    }

    #[test]
    fn rustc_release_gate_accepts_only_the_pinned_compiler() {
        assert!(is_pinned_rustc_version("rustc 1.96.0"));
        assert!(is_pinned_rustc_version("rustc 1.96.0 (abc123 2026-01-01)"));
        assert!(!is_pinned_rustc_version("rustc 1.96.1 (abc123 2026-01-01)"));
        assert!(!is_pinned_rustc_version("rustc 1.96.00 (abc123 2026-01-01)"));
    }

    #[test]
    fn source_commit_gate_rejects_short_or_non_hex_hashes() {
        assert!(is_commit_sha("15cf388016a578b524602e0e72e3c2a6b3f35a2b"));
        assert!(!is_commit_sha("15cf388016a578b524602e0e72e3c2a6b3f35a2"));
        assert!(!is_commit_sha("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
    }

    #[test]
    fn csv_writer_escapes_notes() {
        assert_eq!(
            csv_escape("field, with \"quotes\"\nand linebreak"),
            "\"field, with \"\"quotes\"\"\nand linebreak\""
        );
    }

    #[test]
    fn contrast_input_factors_map_to_registered_levels() {
        assert_eq!(factor_value(0, [-1.0, 0.0, 1.0]), -1.0);
        assert_eq!(factor_value(1, [-1.0, 0.0, 1.0]), 0.0);
        assert_eq!(factor_value(2, [-1.0, 0.0, 1.0]), 1.0);
        assert_eq!([8, 16, 24][2], 24);
    }
}
