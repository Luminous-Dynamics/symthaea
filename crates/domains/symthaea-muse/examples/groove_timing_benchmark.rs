// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Role-aware drum timing profile for the local Groove MIDI Dataset (GMD).
//!
//! Example:
//! cargo run --release -p symthaea-muse --features theory --example groove_timing_benchmark -- \
//!   --dataset-root /path/to/groove --split test --styles 24 --seeds 2
//!
//! Reads GMD MIDI and info.csv only; it does not need or redistribute the
//! dataset audio. It compares descriptive timing/velocity profiles, not human
//! quality. Style labels and tasks are not perfectly matched between GMD and
//! Melothaea, so the receipt explicitly forbids a general "better drummer" claim.

use midly::{MidiMessage, Smf, Timing, TrackEventKind};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Component, Path, PathBuf},
    process::ExitCode,
};

use symthaea_muse::{MusicalState, midi_export::export_performance_midi};
use symthaea_music_theory::{MusicalIntent, Style, compose_styled};

#[derive(Debug)]
struct Arguments {
    dataset_root: PathBuf,
    split: String,
    file_limit: Option<usize>,
    generated_style_limit: usize,
    seeds_per_style: u64,
    output: Option<PathBuf>,
}

#[derive(Debug)]
struct DatasetRow {
    drummer: String,
    session: String,
    performance_id: String,
    style: String,
    bpm: f64,
    beat_type: String,
    time_signature: String,
    midi_filename: String,
    split: String,
}

#[derive(Debug, Clone, Copy)]
struct DrumHit {
    onset_beats: f64,
    pitch: u8,
    velocity: u8,
}

#[derive(Debug, Serialize)]
struct SourceFileReceipt {
    relative_path: String,
    sha256: Option<String>,
    drummer: String,
    session: String,
    performance_id: String,
    style: String,
    bpm: f64,
    beat_type: String,
    time_signature: String,
    split: String,
    hit_count: usize,
    status: &'static str,
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct GeneratedFileReceipt {
    style: String,
    seed: u64,
    bpm: f64,
    sha256: String,
    hit_count: usize,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct RoleSummary {
    files_with_hits: usize,
    hit_count: usize,
    mean_signed_sixteenth_residual_ms: Option<f64>,
    median_absolute_sixteenth_residual_ms: Option<f64>,
    p95_absolute_sixteenth_residual_ms: Option<f64>,
    median_absolute_triplet_residual_ms: Option<f64>,
    mean_velocity_0_to_127: Option<f64>,
    velocity_standard_deviation: Option<f64>,
    mean_file_lag1_residual_correlation: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CorpusSummary {
    files_examined: usize,
    files_with_drum_hits: usize,
    files_without_drum_hits: usize,
    files_excluded: usize,
    total_drum_hits: usize,
    roles: BTreeMap<String, RoleSummary>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReceipt {
    schema: &'static str,
    result_scope: &'static str,
    dataset: &'static str,
    selected_split: String,
    metadata_sha256: String,
    source_rows_selected: usize,
    generated_styles_requested: usize,
    generated_seeds_per_style: u64,
    human_reference: CorpusSummary,
    melothaea_generated: CorpusSummary,
    source_files: Vec<SourceFileReceipt>,
    generated_files: Vec<GeneratedFileReceipt>,
    exclusions: Vec<String>,
    scope_boundary: &'static str,
}

#[derive(Debug, Default)]
struct RoleAccumulator {
    signed_sixteenth_residual_ms: Vec<f64>,
    absolute_triplet_residual_ms: Vec<f64>,
    velocities: Vec<f64>,
    file_lag1_correlations: Vec<f64>,
    files_with_hits: usize,
}

#[derive(Debug, Default)]
struct CorpusAccumulator {
    files_examined: usize,
    files_with_drum_hits: usize,
    files_without_drum_hits: usize,
    files_excluded: usize,
    total_drum_hits: usize,
    roles: BTreeMap<String, RoleAccumulator>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("groove_timing_benchmark: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_arguments()? else {
        print_usage();
        return Ok(());
    };

    let dataset_root = locate_dataset_root(&args.dataset_root)?;
    let metadata_path = dataset_root.join("info.csv");
    ensure_regular_non_symlink(&metadata_path)?;
    let metadata_bytes = fs::read(&metadata_path)?;
    let metadata_sha256 = sha256_hex(&metadata_bytes);
    let metadata_text = std::str::from_utf8(&metadata_bytes)?;
    let rows = selected_dataset_rows(metadata_text, &args.split, args.file_limit)?;
    if rows.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("info.csv has no rows for split {:?}", args.split),
        )
        .into());
    }

    let mut human = CorpusAccumulator::default();
    let mut source_files = Vec::with_capacity(rows.len());
    let mut exclusions = Vec::new();

    for row in rows {
        let resolved = match resolve_dataset_file(&dataset_root, &row.midi_filename) {
            Ok(path) => path,
            Err(reason) => {
                human.files_examined += 1;
                human.files_excluded += 1;
                exclusions.push(format!("{}: {reason}", row.midi_filename));
                source_files.push(source_receipt(&row, None, 0, "excluded", Some(reason)));
                continue;
            }
        };

        let bytes = match fs::read(&resolved) {
            Ok(bytes) => bytes,
            Err(error) => {
                human.files_examined += 1;
                human.files_excluded += 1;
                let reason = format!("cannot read MIDI: {error}");
                exclusions.push(format!("{}: {reason}", row.midi_filename));
                source_files.push(source_receipt(&row, None, 0, "excluded", Some(reason)));
                continue;
            }
        };
        let digest = sha256_hex(&bytes);
        match parse_drum_hits(&bytes, row.bpm) {
            Ok(hits) => {
                human.record_file(&hits, row.bpm);
                let hit_count = hits.len();
                let status = if hits.is_empty() { "no_drum_hits" } else { "measured" };
                source_files.push(source_receipt(
                    &row,
                    Some(digest),
                    hit_count,
                    status,
                    None,
                ));
            }
            Err(reason) => {
                human.files_examined += 1;
                human.files_excluded += 1;
                exclusions.push(format!("{}: {reason}", row.midi_filename));
                source_files.push(source_receipt(
                    &row,
                    Some(digest),
                    0,
                    "excluded",
                    Some(reason),
                ));
            }
        }
    }

    if human.files_with_drum_hits == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no valid drum events were parsed from the selected GMD split",
        )
        .into());
    }

    let temp_dir = create_temp_dir()?;
    let generated_result = generate_melothaea_corpus(
        &args,
        &temp_dir,
    );
    let _cleanup_result = fs::remove_dir_all(&temp_dir);
    let (generated, generated_files) = generated_result?;

    let receipt = BenchmarkReceipt {
        schema: "melothaea-groove-timing-benchmark-v1",
        result_scope: "descriptive_profile_only_not_quality_qualified",
        dataset: "Groove MIDI Dataset; metadata and MIDI only; local user-supplied copy",
        selected_split: args.split,
        metadata_sha256,
        source_rows_selected: source_files.len(),
        generated_styles_requested: args.generated_style_limit,
        generated_seeds_per_style: args.seeds_per_style,
        human_reference: human.summarize(),
        melothaea_generated: generated.summarize(),
        source_files,
        generated_files,
        exclusions,
        scope_boundary: "profile statistics only; no style-equated or score-matched superiority claim; no human preference evidence; generated and reference corpora differ in prompts, structure and style taxonomy",
    };

    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(output) = args.output {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        file.write_all(json.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    println!("{json}");
    Ok(())
}

fn parse_arguments() -> Result<Option<Arguments>, Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let mut dataset_root = PathBuf::from("/opt/datasets/groove");
    let mut split = "test".to_owned();
    let mut file_limit = None;
    let mut generated_style_limit = 24_usize;
    let mut seeds_per_style = 2_u64;
    let mut output = None;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--help" | "-h" => return Ok(None),
            "--dataset-root" => {
                dataset_root = PathBuf::from(arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--dataset-root requires a path")
                })?);
            }
            "--split" => {
                split = arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--split requires train, validation or test")
                })?;
                if !["train", "validation", "test"].contains(&split.as_str()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--split must be train, validation, or test",
                    )
                    .into());
                }
            }
            "--limit" => {
                let value = arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--limit requires a positive integer")
                })?;
                let parsed = value.parse::<usize>()?;
                if parsed == 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "--limit must be positive").into());
                }
                file_limit = Some(parsed);
            }
            "--styles" => {
                let value = arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--styles requires a positive integer")
                })?;
                generated_style_limit = value.parse::<usize>()?;
            }
            "--seeds" => {
                let value = arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--seeds requires a positive integer")
                })?;
                seeds_per_style = value.parse::<u64>()?;
            }
            "--output" => {
                output = Some(PathBuf::from(arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--output requires a path")
                })?));
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument {other:?}"),
                )
                .into());
            }
        }
    }

    if generated_style_limit == 0 || generated_style_limit > Style::ALL.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("--styles must be within 1..={}", Style::ALL.len()),
        )
        .into());
    }
    if seeds_per_style == 0 || seeds_per_style > 1000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--seeds must be within 1..=1000",
        )
        .into());
    }

    Ok(Some(Arguments {
        dataset_root,
        split,
        file_limit,
        generated_style_limit,
        seeds_per_style,
        output,
    }))
}

fn print_usage() {
    eprintln!(
        "Usage: groove_timing_benchmark [--dataset-root PATH] [--split train|validation|test] \
         [--limit N] [--styles N] [--seeds N] [--output NEW_FILE.json]"
    );
}

fn locate_dataset_root(requested: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let root = requested.canonicalize()?;
    if !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dataset root must be a directory",
        )
        .into());
    }
    let direct = root.join("info.csv");
    if direct.is_file() {
        return Ok(root);
    }
    let nested = root.join("groove");
    if nested.join("info.csv").is_file() {
        return Ok(nested.canonicalize()?);
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "could not find info.csv in dataset root or its groove/ child",
    )
    .into())
}

fn ensure_regular_non_symlink(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected a regular non-symlink file",
        )
        .into());
    }
    Ok(())
}

fn selected_dataset_rows(
    csv_text: &str,
    split: &str,
    limit: Option<usize>,
) -> Result<Vec<DatasetRow>, Box<dyn Error>> {
    let mut lines = csv_text.lines();
    let header = parse_csv_record(lines.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "info.csv is empty")
    })?)
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "info.csv header is malformed"))?;
    let columns: BTreeMap<String, usize> = header
        .iter()
        .enumerate()
        .map(|(index, name)| (name.trim().to_owned(), index))
        .collect();
    let index = |name: &str| {
        columns.get(name).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("info.csv is missing required column {name:?}"),
            )
        })
    };
    let (
        drummer_i,
        session_i,
        id_i,
        style_i,
        bpm_i,
        beat_type_i,
        meter_i,
        midi_i,
        split_i,
    ) = (
        index("drummer")?,
        index("session")?,
        index("id")?,
        index("style")?,
        index("bpm")?,
        index("beat_type")?,
        index("time_signature")?,
        index("midi_filename")?,
        index("split")?,
    );

    let mut rows = Vec::new();
    for (row_offset, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_record(line).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed CSV record at line {}", row_offset + 2),
            )
        })?;
        if fields.len() != header.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "CSV column count mismatch at line {}: expected {}, got {}",
                    row_offset + 2,
                    header.len(),
                    fields.len()
                ),
            )
            .into());
        }
        if fields[split_i] != split {
            continue;
        }
        let bpm = fields[bpm_i].parse::<f64>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid BPM at CSV line {}: {error}", row_offset + 2),
            )
        })?;
        if !bpm.is_finite() || bpm <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("BPM must be positive and finite at CSV line {}", row_offset + 2),
            )
            .into());
        }
        rows.push(DatasetRow {
            drummer: fields[drummer_i].clone(),
            session: fields[session_i].clone(),
            performance_id: fields[id_i].clone(),
            style: fields[style_i].clone(),
            bpm,
            beat_type: fields[beat_type_i].clone(),
            time_signature: fields[meter_i].clone(),
            midi_filename: fields[midi_i].clone(),
            split: fields[split_i].clone(),
        });
        if limit.is_some_and(|maximum| rows.len() >= maximum) {
            break;
        }
    }
    Ok(rows)
}

/// Parse one-line CSV records including escaped quotes and quoted commas.
fn parse_csv_record(line: &str) -> Option<Vec<String>> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted => {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            }
            '"' if field.trim().is_empty() => quoted = true,
            ',' if !quoted => {
                fields.push(field.trim().to_owned());
                field.clear();
            }
            _ => field.push(character),
        }
    }
    if quoted {
        return None;
    }
    fields.push(field.trim().to_owned());
    Some(fields)
}

fn resolve_dataset_file(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let candidate = Path::new(relative);
    if relative.is_empty()
        || relative.contains('\\')
        || candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe relative MIDI path {relative:?}"));
    }

    let mut current = root.to_path_buf();
    for component in candidate.components() {
        let Component::Normal(name) = component else {
            return Err(format!("unsafe MIDI path component in {relative:?}"));
        };
        current.push(name);
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| format!("cannot access MIDI path {relative:?}: {error}"))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("MIDI path traverses a symlink: {relative:?}"));
        }
    }
    let metadata = fs::symlink_metadata(&current)
        .map_err(|error| format!("cannot inspect MIDI file {relative:?}: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err(format!("MIDI path is not a regular file: {relative:?}"));
    }
    Ok(current)
}

fn parse_drum_hits(bytes: &[u8], bpm: f64) -> Result<Vec<DrumHit>, String> {
    if !bpm.is_finite() || bpm <= 0.0 {
        return Err("BPM must be finite and positive".to_owned());
    }
    let smf = Smf::parse(bytes).map_err(|error| format!("invalid MIDI: {error}"))?;
    let ppq = match smf.header.timing {
        Timing::Metrical(value) => f64::from(value.as_int()),
        _ => return Err("SMPTE-based MIDI timing is unsupported".to_owned()),
    };
    if !ppq.is_finite() || ppq < 1.0 {
        return Err("MIDI PPQ must be a positive integer".to_owned());
    }

    let mut hits = Vec::new();
    for track in &smf.tracks {
        let mut tick = 0_u64;
        for event in track {
            tick = tick
                .checked_add(u64::from(event.delta.as_int()))
                .ok_or_else(|| "MIDI absolute tick overflow".to_owned())?;
            let TrackEventKind::Midi { channel, message } = &event.kind else {
                continue;
            };
            if channel.as_int() != 9 {
                continue;
            }
            if let MidiMessage::NoteOn { key, vel } = message {
                let velocity = vel.as_int();
                if velocity > 0 {
                    hits.push(DrumHit {
                        onset_beats: tick as f64 / ppq,
                        pitch: key.as_int(),
                        velocity,
                    });
                }
            }
        }
    }
    Ok(hits)
}

impl CorpusAccumulator {
    fn record_excluded_file(&mut self) {
        self.files_examined += 1;
        self.files_excluded += 1;
    }

    fn record_file(&mut self, hits: &[DrumHit], bpm: f64) {
        self.files_examined += 1;
        if hits.is_empty() {
            self.files_without_hits += 1;
            return;
        }
        self.files_with_drum_hits += 1;
        self.total_drum_hits += hits.len();
        let milliseconds_per_beat = 60_000.0 / bpm;
        let mut residuals_by_role: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for hit in hits {
            let role = drum_role(hit.pitch).to_owned();
            let nearest_sixteenth = (hit.onset_beats * 4.0).round() / 4.0;
            let signed_residual_beats = hit.onset_beats - nearest_sixteenth;
            let nearest_triplet = (hit.onset_beats * 3.0).round() / 3.0;
            let absolute_triplet_residual_ms =
                (hit.onset_beats - nearest_triplet).abs() * milliseconds_per_beat;
            let accumulator = self.roles.entry(role.clone()).or_default();
            accumulator
                .signed_sixteenth_residual_ms
                .push(signed_residual_beats * milliseconds_per_beat);
            accumulator
                .absolute_triplet_residual_ms
                .push(absolute_triplet_residual_ms);
            accumulator.velocities.push(f64::from(hit.velocity));
            residuals_by_role
                .entry(role)
                .or_default()
                .push(signed_residual_beats);
        }

        for (role, residuals) in residuals_by_role {
            let accumulator = self.roles.entry(role).or_default();
            accumulator.files_with_hits += 1;
            if let Some(correlation) = lag1_autocorrelation(&residuals) {
                accumulator.file_lag1_correlations.push(correlation);
            }
        }
    }

    fn summarize(&self) -> CorpusSummary {
        let roles = self
            .roles
            .iter()
            .map(|(role, accumulator)| {
                (
                    role.clone(),
                    RoleSummary {
                        files_with_hits: accumulator.files_with_hits,
                        hit_count: accumulator.signed_sixteenth_residual_ms.len(),
                        mean_signed_sixteenth_residual_ms: mean(
                            &accumulator.signed_sixteenth_residual_ms,
                        ),
                        median_absolute_sixteenth_residual_ms: quantile_abs(
                            &accumulator.signed_sixteenth_residual_ms,
                            0.50,
                        ),
                        p95_absolute_sixteenth_residual_ms: quantile_abs(
                            &accumulator.signed_sixteenth_residual_ms,
                            0.95,
                        ),
                        median_absolute_triplet_residual_ms: median(
                            &accumulator.absolute_triplet_residual_ms,
                        ),
                        mean_velocity_0_to_127: mean(&accumulator.velocities),
                        velocity_standard_deviation: standard_deviation(&accumulator.velocities),
                        mean_file_lag1_residual_correlation: mean(
                            &accumulator.file_lag1_correlations,
                        ),
                    },
                )
            })
            .collect();
        CorpusSummary {
            files_examined: self.files_examined,
            files_with_drum_hits: self.files_with_drum_hits,
            files_without_drum_hits: self.files_without_hits,
            files_excluded: self.files_excluded,
            total_drum_hits: self.total_drum_hits,
            roles,
        }
    }
}

fn drum_role(pitch: u8) -> &'static str {
    match pitch {
        35 | 36 => "kick",
        37 | 38 | 40 => "snare",
        22 | 26 | 42 | 44 | 46 => "hi_hat",
        41 | 43 | 45 | 47 | 48 | 50 => "tom",
        49 | 51 | 52 | 53 | 55 | 57 | 59 => "cymbal",
        _ => "other",
    }
}

fn lag1_autocorrelation(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let left = &values[..values.len() - 1];
    let right = &values[1..];
    let left_mean = mean(left)?;
    let right_mean = mean(right)?;
    let covariance = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - left_mean) * (right - right_mean))
        .sum::<f64>()
        / left.len() as f64;
    let left_variance = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>()
        / left.len() as f64;
    let right_variance = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>()
        / right.len() as f64;
    let denominator = (left_variance * right_variance).sqrt();
    (denominator > f64::EPSILON)
        .then_some((covariance / denominator).clamp(-1.0, 1.0))
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn standard_deviation(values: &[f64]) -> Option<f64> {
    let average = mean(values)?;
    Some(
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt(),
    )
}

fn median(values: &[f64]) -> Option<f64> {
    quantile(values, 0.50)
}

fn quantile_abs(values: &[f64], quantile: f64) -> Option<f64> {
    let absolute: Vec<f64> = values.iter().map(|value| value.abs()).collect();
    quantile_value(&absolute, quantile)
}

fn quantile(values: &[f64], quantile: f64) -> Option<f64> {
    quantile_value(values, quantile)
}

fn quantile_value(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((quantile.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted.get(rank).copied()
}

fn source_receipt(
    row: &DatasetRow,
    sha256: Option<String>,
    hit_count: usize,
    status: &'static str,
    reason: Option<String>,
) -> SourceFileReceipt {
    SourceFileReceipt {
        relative_path: row.midi_filename.clone(),
        sha256,
        drummer: row.drummer.clone(),
        session: row.session.clone(),
        performance_id: row.performance_id.clone(),
        style: row.style.clone(),
        bpm: row.bpm,
        beat_type: row.beat_type.clone(),
        time_signature: row.time_signature.clone(),
        split: row.split.clone(),
        hit_count,
        status,
        reason,
    }
}

fn generate_melothaea_corpus(
    args: &Arguments,
    temp_dir: &Path,
) -> Result<(CorpusAccumulator, Vec<GeneratedFileReceipt>), Box<dyn Error>> {
    let mut corpus = CorpusAccumulator::default();
    let mut receipts = Vec::new();
    let state = MusicalState::default();

    for (style_index, style) in Style::ALL.into_iter().enumerate() {
        if style_index >= args.generated_style_limit {
            break;
        }
        let spec = style.spec();
        for seed in 0..args.seeds_per_style {
            let intent = MusicalIntent {
                seed,
                ..Default::default()
            };
            let score = compose_styled(&intent, style);
            let bpm = score.tempo_bpm as f64;
            if !bpm.is_finite() || bpm <= 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("generated score has invalid BPM for style {style:?}"),
                )
                .into());
            }
            let filename = format!("style_{style_index:03}_seed_{seed:04}.mid");
            let midi_path = temp_dir.join(&filename);
            export_performance_midi(&score, &spec, seed, &state, &midi_path)
                .map_err(|error| io::Error::other(format!("MIDI export failed: {error}")))?;
            let bytes = fs::read(&midi_path)?;
            let digest = sha256_hex(&bytes);
            let hits = parse_drum_hits(&bytes, bpm).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("generated MIDI {filename} is invalid: {error}"),
                )
            })?;
            corpus.record_file(&hits, bpm);
            receipts.push(GeneratedFileReceipt {
                style: format!("{style:?}"),
                seed,
                bpm,
                sha256: digest,
                hit_count: hits.len(),
                status: if hits.is_empty() { "no_drum_hits" } else { "measured" },
            });
        }
    }
    Ok((corpus, receipts))
}

fn create_temp_dir() -> Result<PathBuf, Box<dyn Error>> {
    let base = env::temp_dir();
    let process_id = std::process::id();
    for attempt in 0..100_u32 {
        let path = base.join(format!("melothaea-gmd-benchmark-{process_id}-{attempt}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not reserve a unique temporary benchmark directory",
    )
    .into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_parser_preserves_quoted_commas_and_escaped_quotes() {
        let fields = parse_csv_record(r#"a,"b,c","quoted ""word"""#).unwrap();
        assert_eq!(fields, vec!["a", "b,c", "quoted \"word\""]);
    }

    #[test]
    fn percussion_roles_match_gmd_pitch_mapping() {
        assert_eq!(drum_role(36), "kick");
        assert_eq!(drum_role(38), "snare");
        assert_eq!(drum_role(42), "hi_hat");
        assert_eq!(drum_role(45), "tom");
        assert_eq!(drum_role(49), "cymbal");
        assert_eq!(drum_role(88), "other");
    }

    #[test]
    fn metrics_handle_empty_and_constant_sequences() {
        assert_eq!(median(&[]), None);
        assert_eq!(mean(&[]), None);
        assert_eq!(lag1_autocorrelation(&[0.0, 0.0, 0.0]), None);
        assert_eq!(lag1_autocorrelation(&[0.0, 1.0]), None);
        assert!(lag1_autocorrelation(&[-1.0, 0.0, 1.0, 2.0]).is_some());
    }

    #[test]
    fn csv_parser_rejects_unclosed_quoted_field() {
        assert!(parse_csv_record(r#"a,"unfinished"#).is_none());
    }
}
