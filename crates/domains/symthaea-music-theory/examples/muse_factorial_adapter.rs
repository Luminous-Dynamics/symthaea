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
use std::process::ExitCode;

use serde_json::to_vec;
use sha2::{Digest, Sha256};
use symthaea_music_theory::score_validation::is_universal_invariant;
use symthaea_music_theory::{
    MusicalIntent, PitchClass, Score, ScoreValidationConfig, Style, ValidationSeverity,
    compose_with_grammar_plan, validate_score,
};

const HEADER: &str =
    "case_id,cohort,style,seed,valence_code,arousal_code,energy_code,bars_code";

const METRICS: [&str; 18] = [
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

#[derive(Debug)]
struct ComposedArtifact {
    metrics: Vec<(&'static str, Option<f64>)>,
    artifact_sha256: String,
    note: String,
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

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn compose_case(
    case: &CaseRow,
    source_commit: &str,
    manifest_sha256: &str,
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

    let realization =
        compose_with_grammar_plan(case.style.grammar_profile(), &intent, &spec);
    let score: Score = realization.score;
    let validation = validate_score(&score, &ScoreValidationConfig::default());

    let artifact_bytes = to_vec(&score)
        .map_err(|error| format!("cannot serialize symbolic score: {error}"))?;
    let artifact_sha256 = sha256_hex(&artifact_bytes);

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
    ];

    let note = format!(
        "source_commit={source_commit};manifest_sha256={manifest_sha256};engine_version={};grammar_family={:?};support_axes={};universal_fatal={universal_fatal_count};style_fatal={};warnings={warning_count};validation_valid={}",
        symthaea_music_theory::MUSIC_THEORY_ENGINE_VERSION,
        case.style.grammar_family(),
        case.style.supported_intent_axes().join("|"),
        all_fatal_count.saturating_sub(universal_fatal_count),
        validation.valid
    );

    Ok(ComposedArtifact {
        metrics,
        artifact_sha256,
        note,
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

fn failed_measurements(case_id: &str, error: &str, note: &str) -> Vec<Measurement> {
    METRICS
        .iter()
        .map(|metric| Measurement {
            case_id: case_id.to_string(),
            metric: *metric,
            status: "failed",
            value: None,
            artifact_sha256: String::new(),
            note: format!("{note};composer_failure={error}"),
        })
        .collect()
}

fn measurements_for_case(
    case: &CaseRow,
    source_commit: &str,
    manifest_sha256: &str,
) -> Vec<Measurement> {
    let composed = catch_unwind(AssertUnwindSafe(|| {
        compose_case(case, source_commit, manifest_sha256)
    }));

    match composed {
        Ok(Ok(artifact)) => artifact
            .metrics
            .into_iter()
            .map(|(metric, value)| match value {
                Some(value) if value.is_finite() => Measurement {
                    case_id: case.case_id.clone(),
                    metric,
                    status: "observed",
                    value: Some(value),
                    artifact_sha256: artifact.artifact_sha256.clone(),
                    note: artifact.note.clone(),
                },
                Some(_) => Measurement {
                    case_id: case.case_id.clone(),
                    metric,
                    status: "failed",
                    value: None,
                    artifact_sha256: artifact.artifact_sha256.clone(),
                    note: format!("{};metric_value_non_finite", artifact.note),
                },
                None => Measurement {
                    case_id: case.case_id.clone(),
                    metric,
                    status: "not_applicable",
                    value: None,
                    artifact_sha256: artifact.artifact_sha256.clone(),
                    note: artifact.note.clone(),
                },
            })
            .collect(),
        Ok(Err(error)) => failed_measurements(
            &case.case_id,
            &error,
            &format!(
                "source_commit={source_commit};manifest_sha256={manifest_sha256}"
            ),
        ),
        Err(_) => failed_measurements(
            &case.case_id,
            "composer_panicked",
            &format!(
                "source_commit={source_commit};manifest_sha256={manifest_sha256}"
            ),
        ),
    }
}

fn is_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

    let mut failed_cases = 0usize;
    let mut not_applicable_rows = 0usize;
    let mut observed_rows = 0usize;
    for (index, case) in cases.iter().enumerate() {
        let measurements =
            measurements_for_case(case, &args[3], &manifest_sha256);
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
                cases.len(),
                failed_cases
            );
        }
    }
    output
        .flush()
        .map_err(|error| format!("cannot flush observations: {error}"))?;

    eprintln!(
        "case_count={}, metric_count={}, observed_rows={}, not_applicable_rows={}, failed_cases={}, source_commit={}, manifest_sha256={}",
        cases.len(),
        METRICS.len(),
        observed_rows,
        not_applicable_rows,
        failed_cases,
        args[3],
        manifest_sha256
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
