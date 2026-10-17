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

const MAX_GMD_INFO_BYTES: u64 = 32 * 1024 * 1024;
const MAX_GMD_MIDI_BYTES: u64 = 16 * 1024 * 1024;

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
    median_absolute_sixteenth_grid_residual_beats: Option<f64>,
    p95_absolute_sixteenth_grid_residual_beats: Option<f64>,
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
struct PerformerMacroSummary {
    performers_with_role: usize,
    mean_performer_median_abs_sixteenth_grid_residual_beats: Option<f64>,
    median_performer_median_abs_sixteenth_grid_residual_beats: Option<f64>,
    mean_performer_p95_abs_sixteenth_grid_residual_beats: Option<f64>,
    mean_performer_velocity_mean_0_to_127: Option<f64>,
    mean_performer_velocity_standard_deviation: Option<f64>,
    mean_performer_lag1_residual_correlation: Option<f64>,
}

#[derive(Debug, Default)]
struct PerformerMacroAccumulator {
    timing_medians_beats: Vec<f64>,
    timing_p95_beats: Vec<f64>,
    velocity_means: Vec<f64>,
    velocity_standard_deviations: Vec<f64>,
    lag1_correlations: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReceipt {
    schema: &'static str,
    result_scope: &'static str,
    quality_qualified: bool,
    dataset: &'static str,
    generated_method: &'static str,
    selected_split: String,
    metadata_sha256: String,
    source_rows_selected: usize,
    performer_count: usize,
    performer_session_count: usize,
    generated_styles_requested: usize,
    generated_seeds_per_style: u64,
    human_reference: CorpusSummary,
    human_performer_macro: BTreeMap<String, PerformerMacroSummary>,
    human_by_drummer: BTreeMap<String, CorpusSummary>,
    human_by_drummer_session: BTreeMap<String, CorpusSummary>,
    melothaea_generated: CorpusSummary,
    source_files: Vec<SourceFileReceipt>,
    generated_files: Vec<GeneratedFileReceipt>,
    exclusions: Vec<String>,
    scope_boundary: &'static str,
}

#[derive(Debug, Default)]
struct RoleAccumulator {
    signed_sixteenth_residual_ms: Vec<f64>,
    signed_sixteenth_grid_residual_beats: Vec<f64>,
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
    let metadata_bytes = read_regular_file_limited(&metadata_path, MAX_GMD_INFO_BYTES)?;
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
    let mut human_by_drummer: BTreeMap<String, CorpusAccumulator> = BTreeMap::new();
    let mut human_by_drummer_session: BTreeMap<String, CorpusAccumulator> = BTreeMap::new();
    let mut source_files = Vec::with_capacity(rows.len());
    let mut exclusions = Vec::new();

    for row in rows {
        let resolved = match resolve_dataset_file(&dataset_root, &row.midi_filename) {
            Ok(path) => path,
            Err(reason) => {
                record_excluded_grouped(
                    &mut human,
                    &mut human_by_drummer,
                    &mut human_by_drummer_session,
                    &row,
                );
                exclusions.push(format!("{}: {reason}", row.midi_filename));
                source_files.push(source_receipt(&row, None, 0, "excluded", Some(reason)));
                continue;
            }
        };

        let bytes = match read_regular_file_limited(&resolved, MAX_GMD_MIDI_BYTES) {
            Ok(bytes) => bytes,
            Err(error) => {
                record_excluded_grouped(
                    &mut human,
                    &mut human_by_drummer,
                    &mut human_by_drummer_session,
                    &row,
                );
                let reason = format!("cannot read MIDI: {error}");
                exclusions.push(format!("{}: {reason}", row.midi_filename));
                source_files.push(source_receipt(&row, None, 0, "excluded", Some(reason)));
                continue;
            }
        };
        let digest = sha256_hex(&bytes);
        match parse_drum_hits(&bytes, row.bpm) {
            Ok(hits) => {
                record_observed_grouped(
                    &mut human,
                    &mut human_by_drummer,
                    &mut human_by_drummer_session,
                    &row,
                    &hits,
                );
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
                record_excluded_grouped(
                    &mut human,
                    &mut human_by_drummer,
                    &mut human_by_drummer_session,
                    &row,
                );
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

    let performer_count = human_by_drummer.len();
    let performer_session_count = human_by_drummer_session.len();
    let human_performer_macro = performer_macro_summary(&human_by_drummer);
    let receipt = BenchmarkReceipt {
        schema: "melothaea-groove-timing-benchmark-v1",
        result_scope: "descriptive_profile_only_not_quality_qualified",
        quality_qualified: false,
        dataset: "Groove MIDI Dataset; metadata and MIDI only; local user-supplied copy",
        generated_method: concat!(
            "compose_styled + export_performance_midi; existing production realization path; ",
            "new phrase curve is not applied",
        ),
        selected_split: args.split,
        metadata_sha256,
        source_rows_selected: source_files.len(),
        performer_count,
        performer_session_count,
        generated_styles_requested: args.generated_style_limit,
        generated_seeds_per_style: args.seeds_per_style,
        human_reference: human.summarize(),
        human_performer_macro,
        human_by_drummer: human_by_drummer
            .into_iter()
            .map(|(drummer, corpus)| (drummer, corpus.summarize()))
            .collect(),
        human_by_drummer_session: human_by_drummer_session
            .into_iter()
            .map(|(session, corpus)| (session, corpus.summarize()))
            .collect(),
        melothaea_generated: generated.summarize(),
        source_files,
        generated_files,
        exclusions,
        scope_boundary: concat!(
            "profile statistics only; no style-equated or score-matched superiority claim; ",
            "no human preference evidence; generated and reference corpora differ in prompts, ",
            "structure and style taxonomy",
        ),
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
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--split requires train, validation or test",
                    )
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
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--limit requires a positive integer",
                    )
                })?;
                let parsed = value.parse::<usize>()?;
                if parsed == 0 {
                    return Err(
                        io::Error::new(io::ErrorKind::InvalidInput, "--limit must be positive")
                            .into(),
                    );
                }
                file_limit = Some(parsed);
            }
            "--styles" => {
                let value = arguments.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--styles requires a positive integer",
                    )
                })?;
                generated_style_limit = value.parse::<usize>()?;
            }
            "--seeds" => {
                let value = arguments.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--seeds requires a positive integer",
                    )
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
        let metadata = fs::symlink_metadata(&nested)?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "nested groove directory must be a regular non-symlink directory",
            )
            .into());
        }
        return Ok(nested.canonicalize()?);
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "could not find info.csv in dataset root or its groove/ child",
    )
    .into())
}

fn read_regular_file_limited(
    path: &Path,
    max_bytes: u64,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected a regular non-symlink file",
        )
        .into());
    }
    if metadata.len() > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("file exceeds {max_bytes} byte read limit"),
        )
        .into());
    }
    Ok(fs::read(path)?)
}

fn selected_dataset_rows(
    csv_text: &str,
    split: &str,
    limit: Option<usize>,
) -> Result<Vec<DatasetRow>, Box<dyn Error>> {
    let mut lines = csv_text.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "info.csv is empty"))?;
    let header_line = header_line.trim_start_matches('\u{feff}');
    let header = parse_csv_record(header_line)
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
        if fields[split_i].as_str() != split {
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
    let smf = Smf::parse(bytes).map_err(|error| format!("invalid MIDI: {error}"))?;
    extract_drum_hits(&smf, bpm)
}

fn extract_drum_hits(smf: &Smf<'_>, bpm: f64) -> Result<Vec<DrumHit>, String> {
    if !bpm.is_finite() || bpm <= 0.0 {
        return Err("BPM must be finite and positive".to_owned());
    }
    let ppq = match &smf.header.timing {
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
    hits.sort_by(|left, right| {
        left.onset_beats
            .total_cmp(&right.onset_beats)
            .then_with(|| left.pitch.cmp(&right.pitch))
            .then_with(|| left.velocity.cmp(&right.velocity))
    });
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
                .signed_sixteenth_grid_residual_beats
                .push(signed_residual_beats);
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
                        median_absolute_sixteenth_grid_residual_beats: quantile_abs(
                            &accumulator.signed_sixteenth_grid_residual_beats,
                            0.50,
                        ),
                        p95_absolute_sixteenth_grid_residual_beats: quantile_abs(
                            &accumulator.signed_sixteenth_grid_residual_beats,
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

fn drummer_session_key(row: &DatasetRow) -> String {
    format!("drummer={};session={}", row.drummer, row.session)
}

fn record_excluded_grouped(
    all: &mut CorpusAccumulator,
    by_drummer: &mut BTreeMap<String, CorpusAccumulator>,
    by_drummer_session: &mut BTreeMap<String, CorpusAccumulator>,
    row: &DatasetRow,
) {
    all.record_excluded_file();
    by_drummer
        .entry(row.drummer.clone())
        .or_default()
        .record_excluded_file();
    by_drummer_session
        .entry(drummer_session_key(row))
        .or_default()
        .record_excluded_file();
}

fn record_observed_grouped(
    all: &mut CorpusAccumulator,
    by_drummer: &mut BTreeMap<String, CorpusAccumulator>,
    by_drummer_session: &mut BTreeMap<String, CorpusAccumulator>,
    row: &DatasetRow,
    hits: &[DrumHit],
) {
    all.record_file(hits, row.bpm);
    by_drummer
        .entry(row.drummer.clone())
        .or_default()
        .record_file(hits, row.bpm);
    by_drummer_session
        .entry(drummer_session_key(row))
        .or_default()
        .record_file(hits, row.bpm);
}

/// Macro-average performer summaries so one high-volume drummer cannot
/// dominate the descriptive human profile. These are descriptive summaries,
/// not confidence intervals or population estimates.
fn performer_macro_summary(
    by_drummer: &BTreeMap<String, CorpusSummary>,
) -> BTreeMap<String, PerformerMacroSummary> {
    let mut roles: BTreeMap<String, PerformerMacroAccumulator> = BTreeMap::new();
    for corpus in by_drummer.values() {
        for (role, summary) in &corpus.roles {
            let accumulator = roles.entry(role.clone()).or_default();
            if let Some(value) = summary.median_absolute_sixteenth_grid_residual_beats {
                accumulator.timing_medians_beats.push(value);
            }
            if let Some(value) = summary.p95_absolute_sixteenth_grid_residual_beats {
                accumulator.timing_p95_beats.push(value);
            }
            if let Some(value) = summary.mean_velocity_0_to_127 {
                accumulator.velocity_means.push(value);
            }
            if let Some(value) = summary.velocity_standard_deviation {
                accumulator.velocity_standard_deviations.push(value);
            }
            if let Some(value) = summary.mean_file_lag1_residual_correlation {
                accumulator.lag1_correlations.push(value);
            }
        }
    }

    roles
        .into_iter()
        .map(|(role, accumulator)| {
            let performers_with_role = accumulator.timing_medians_beats.len();
            (
                role,
                PerformerMacroSummary {
                    performers_with_role,
                    mean_performer_median_abs_sixteenth_grid_residual_beats: mean(
                        &accumulator.timing_medians_beats,
                    ),
                    median_performer_median_abs_sixteenth_grid_residual_beats: median(
                        &accumulator.timing_medians_beats,
                    ),
                    mean_performer_p95_abs_sixteenth_grid_residual_beats: mean(
                        &accumulator.timing_p95_beats,
                    ),
                    mean_performer_velocity_mean_0_to_127: mean(&accumulator.velocity_means),
                    mean_performer_velocity_standard_deviation: mean(
                        &accumulator.velocity_standard_deviations,
                    ),
                    mean_performer_lag1_residual_correlation: mean(
                        &accumulator.lag1_correlations,
                    ),
                },
            )
        })
        .collect()
}

fn drum_role(pitch: u8) -> &'static str {
    match pitch {
        35 | 36 => "kick",
        37 | 38 | 40 => "snare",
        22 | 26 | 42 | 44 | 46 => "hi_hat",
        41 | 43 | 45 | 47 | 48 | 50 | 58 => "tom",
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
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some((sorted[middle - 1] + sorted[middle]) / 2.0)
    } else {
        sorted.get(middle).copied()
    }
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
    fn drum_parser_selects_positive_note_ons_on_percussion_channel_only() {
        use midly::{
            Format, Header, TrackEvent,
            num::{u4, u7, u15, u28},
        };

        let event = |delta, channel, key, velocity| TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(channel),
                message: MidiMessage::NoteOn {
                    key: u7::new(key),
                    vel: u7::new(velocity),
                },
            },
        };
        let smf = Smf {
            header: Header {
                format: Format::SingleTrack,
                timing: Timing::Metrical(u15::new(480)),
            },
            tracks: vec![vec![
                event(0, 9, 36, 100),
                event(120, 0, 38, 90),
                event(120, 9, 42, 0),
                event(0, 9, 42, 80),
            ]],
        };
        let hits = extract_drum_hits(&smf, 120.0).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].pitch, 36);
        assert_eq!(hits[0].onset_beats, 0.0);
        assert_eq!(hits[1].pitch, 42);
        assert_eq!(hits[1].onset_beats, 0.5);
        assert_eq!(hits[1].velocity, 80);
    }

    #[test]
    fn metrics_handle_empty_and_constant_sequences() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[1.0, 3.0]), Some(2.0));
        assert_eq!(median(&[1.0, 2.0, 3.0]), Some(2.0));
        assert_eq!(mean(&[]), None);
        assert_eq!(lag1_autocorrelation(&[0.0, 0.0, 0.0]), None);
        assert_eq!(lag1_autocorrelation(&[0.0, 1.0]), None);
        assert!(lag1_autocorrelation(&[-1.0, 0.0, 1.0, 2.0]).is_some());
    }

    fn test_corpus_with_kick_profile(
        hit_count: usize,
        median_residual_beats: f64,
        p95_residual_beats: f64,
        mean_velocity: f64,
    ) -> CorpusSummary {
        let role = RoleSummary {
            files_with_hits: 1,
            hit_count,
            mean_signed_sixteenth_residual_ms: Some(0.0),
            median_absolute_sixteenth_residual_ms: Some(median_residual_beats * 100.0),
            p95_absolute_sixteenth_residual_ms: Some(p95_residual_beats * 100.0),
            median_absolute_sixteenth_grid_residual_beats: Some(median_residual_beats),
            p95_absolute_sixteenth_grid_residual_beats: Some(p95_residual_beats),
            median_absolute_triplet_residual_ms: Some(1.0),
            mean_velocity_0_to_127: Some(mean_velocity),
            velocity_standard_deviation: Some(10.0),
            mean_file_lag1_residual_correlation: Some(0.1),
        };
        CorpusSummary {
            files_examined: 1,
            files_with_drum_hits: 1,
            files_without_drum_hits: 0,
            files_excluded: 0,
            total_drum_hits: hit_count,
            roles: BTreeMap::from([("kick".to_owned(), role)]),
        }
    }

    #[test]
    fn performer_macro_weights_drummer_summaries_equally_not_by_hit_count() {
        let by_drummer = BTreeMap::from([
            (
                "drummer-a".to_owned(),
                test_corpus_with_kick_profile(10_000, 0.01, 0.12, 10.0),
            ),
            (
                "drummer-b".to_owned(),
                test_corpus_with_kick_profile(10, 0.09, 0.22, 100.0),
            ),
        ]);
        let summary = performer_macro_summary(&by_drummer);
        let kick = &summary["kick"];
        assert_eq!(kick.performers_with_role, 2);
        assert!(
            (kick
                .mean_performer_median_abs_sixteenth_grid_residual_beats
                .unwrap()
                - 0.05)
                .abs()
                < 1e-12
        );
        assert!(
            (kick
                .median_performer_median_abs_sixteenth_grid_residual_beats
                .unwrap()
                - 0.05)
                .abs()
                < 1e-12
        );
        assert!(
            (kick.mean_performer_velocity_mean_0_to_127.unwrap() - 55.0).abs() < 1e-12
        );
    }

    #[test]
    fn dataset_selection_filters_split_and_preserves_group_metadata() {
        let csv = concat!(
            "drummer,session,id,style,bpm,beat_type,time_signature,",
            "midi_filename,audio_filename,duration,split\n",
            "1,2,100,rock,120,beat,4/4,drummer1/session2/a.mid,a.wav,12.0,test\n",
            "2,3,101,jazz,90,half,4/4,drummer2/session3/b.mid,b.wav,10.0,train\n",
            "1,2,102,funk,110,beat,4/4,drummer1/session2/c.mid,c.wav,11.0,test\n",
        );
        let selected = selected_dataset_rows(csv, "test", Some(1)).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].drummer, "1");
        assert_eq!(selected[0].session, "2");
        assert_eq!(selected[0].performance_id, "100");
        assert_eq!(selected[0].midi_filename, "drummer1/session2/a.mid");
        assert_eq!(selected[0].split, "test");
    }

    #[test]
    fn dataset_selection_accepts_utf8_bom_and_rejects_missing_columns() {
        let csv = concat!(
            "\u{feff}drummer,session,id,style,bpm,beat_type,time_signature,",
            "midi_filename,audio_filename,duration,split\n",
            "1,2,100,rock,120,beat,4/4,drummer1/session2/a.mid,a.wav,12.0,test\n",
        );
        assert_eq!(selected_dataset_rows(csv, "test", None).unwrap().len(), 1);
        assert!(selected_dataset_rows("drummer,session\n1,2\n", "test", None).is_err());
    }

    #[test]
    fn csv_parser_rejects_unclosed_quoted_field() {
        assert!(parse_csv_record(r#"a,"unfinished"#).is_none());
    }
}
