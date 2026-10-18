// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auditable instrument capability matrix for Melothaea.
//!
//! This tool summarizes *observed source inventory* separately from *proven
//! render capability*. A filename-derived MIDI range is never promoted to a
//! playable range, and inventory-only output cannot mark an instrument as
//! qualified.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use symthaea_muse::vcsl::{
    AssetRootEvidenceV1, InstrumentCapabilityV1, InstrumentCapabilityInventoryV1,
    SampleDirectoryEvidenceV1, capability_inventory_with_provenance_options,
};

#[derive(Debug)]
struct Options {
    vcsl_dir: PathBuf,
    vsco2_dir: PathBuf,
    soundfont: Option<PathBuf>,
    hash_indexed_samples: bool,
    hash_root_wav_contents: bool,
}

#[derive(Debug, Serialize)]
struct QualificationMatrixV1 {
    schema_version: &'static str,
    benchmark: &'static str,
    qualification_result: &'static str,
    evidence_boundary: &'static str,
    source_revision: Option<String>,
    vcsl_root: AssetRootEvidenceV1,
    vsco2_root: AssetRootEvidenceV1,
    indexed_sample_content_hashing_enabled: bool,
    root_content_manifest_hashing_enabled: bool,
    instruments: Vec<InstrumentQualificationRowV1>,
    next_evidence_gates: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct InstrumentQualificationRowV1 {
    instrument: String,
    gm_program_0_based: u8,
    declared_procedural_renderer_path: String,
    source_mapping_status: &'static str,
    sustained_samples: Option<SampleBankQualificationV1>,
    short_articulation_samples: Option<SampleBankQualificationV1>,
    additional_style_count: usize,
    additional_styles_with_non_silent_decoded_samples: usize,
    /// Detailed style-bank identity so render receipts can be cross-checked
    /// against the exact selected source bank.
    additional_sample_styles: Vec<SampleStyleQualificationV1>,
    soundfont_bank0_program_matches: Option<Vec<String>>,
    observed_filename_pitch_bounds_midi: Option<(u8, u8)>,
    playable_range_status: &'static str,
    gesture_coverage_status: &'static str,
    planned_gesture_tests: Vec<PlannedGestureV1>,
    overall_qualification_status: &'static str,
}

#[derive(Debug, Serialize)]
struct SampleBankQualificationV1 {
    source_library: String,
    relative_directory: String,
    bank_status: &'static str,
    directory_present: bool,
    indexed_sample_file_count: usize,
    decoded_sample_file_count: usize,
    failed_sample_decode_count: usize,
    silent_sample_file_count: usize,
    observed_filename_pitch_bounds_midi: Option<(u8, u8)>,
    filename_pitch_gaps_midi_within_observed_range: Vec<u8>,
    indexed_pitch_count: usize,
    distinct_filename_dynamic_strength_count: usize,
    indexed_sample_content_manifest_sha256: Option<String>,
}

#[derive(Debug, Serialize)]
struct SampleStyleQualificationV1 {
    style_name: &'static str,
    style_kind: &'static str,
    evidence: SampleBankQualificationV1,
}

#[derive(Debug, Serialize)]
struct PlannedGestureV1 {
    gesture: &'static str,
    status: &'static str,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse_instrument_qualification: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let inventory = capability_inventory_with_provenance_options(
        &options.vcsl_dir,
        &options.vsco2_dir,
        options.soundfont.as_deref(),
        None,
        options.hash_indexed_samples,
        options.hash_root_wav_contents,
    )
    .map_err(|error| format!("could not inspect sample inventory: {error}"))?;

    let rows = inventory
        .instruments
        .iter()
        .map(qualification_row)
        .collect::<Vec<_>>();
    let report = QualificationMatrixV1 {
        schema_version: "melothaea.instrument-capability-matrix.v3",
        benchmark: "MEL-INST-CAP-001",
        qualification_result: "not_qualified_by_source_inventory_alone",
        evidence_boundary: "source inventory, filename pitch coverage, WAV decode and silence audit only; optional root content manifests hash all WAVs found by the root walker, including files outside the pitch indexer; no render-grid, gesture, or human-preference result is implied",
        source_revision: inventory.source_revision.clone(),
        vcsl_root: inventory.vcsl_root,
        vsco2_root: inventory.vsco2_root,
        indexed_sample_content_hashing_enabled: inventory.indexed_sample_content_hashing_enabled,
        root_content_manifest_hashing_enabled: inventory.root_content_manifest_hashing_enabled,
        instruments: rows,
        next_evidence_gates: vec![
            "instrument-specific render sweep with per-note renderer/source receipts",
            "pitch accuracy against independent reference tones where fundamental pitch is meaningful",
            "velocity and articulation behavior against instrument-family expectations",
            "positive and negative controls for every advertised gesture",
            "blind level-matched instrument-identity and musical-quality listening study",
        ],
    };

    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)
        .map_err(|error| format!("could not serialize qualification matrix: {error}"))?;
    println!();
    Ok(())
}

fn qualification_row(instrument: &InstrumentCapabilityV1) -> InstrumentQualificationRowV1 {
    let sustained_samples = instrument.sustained_samples.as_ref().map(sample_bank_row);
    let short_articulation_samples = instrument
        .short_articulation_samples
        .as_ref()
        .map(sample_bank_row);

    let additional_styles_with_non_silent_decoded_samples = instrument
        .additional_sample_styles
        .iter()
        .filter(|style| style.evidence.decoded_sample_file_count > style.evidence.silent_sample_file_count)
        .count();

    let observed_filename_pitch_bounds_midi = sustained_samples
        .as_ref()
        .and_then(|bank| bank.observed_filename_pitch_bounds_midi);
    let source_mapping_status = match sustained_samples.as_ref() {
        None => "no_declared_sustained_sample_mapping",
        Some(bank) => bank.bank_status,
    };

    InstrumentQualificationRowV1 {
        instrument: instrument.instrument.clone(),
        gm_program_0_based: instrument.gm_program_0_based,
        declared_procedural_renderer_path: instrument.procedural_renderer_path.to_string(),
        source_mapping_status,
        sustained_samples,
        short_articulation_samples,
        additional_style_count: instrument.additional_sample_styles.len(),
        additional_styles_with_non_silent_decoded_samples,
        additional_sample_styles: instrument
            .additional_sample_styles
            .iter()
            .map(|style| SampleStyleQualificationV1 {
                style_name: style.style_name,
                style_kind: style.style_kind,
                evidence: sample_bank_row(&style.evidence),
            })
            .collect(),
        soundfont_bank0_program_matches: instrument.soundfont_bank0_program_matches.clone(),
        observed_filename_pitch_bounds_midi,
        playable_range_status: "not_established_by_filename_inventory",
        gesture_coverage_status: "not_measured_by_source_inventory",
        planned_gesture_tests: planned_gestures(&instrument.instrument),
        overall_qualification_status: "not_qualified",
    }
}

fn sample_bank_row(evidence: &SampleDirectoryEvidenceV1) -> SampleBankQualificationV1 {
    let observed_filename_pitch_bounds_midi = match (evidence.min_midi, evidence.max_midi) {
        (Some(minimum), Some(maximum)) => Some((minimum, maximum)),
        _ => None,
    };
    SampleBankQualificationV1 {
        source_library: evidence.source_library.to_string(),
        relative_directory: evidence.relative_directory.clone(),
        bank_status: sample_bank_status(
            evidence.directory_present,
            evidence.indexed_sample_count,
            evidence.decoded_sample_file_count,
            evidence.failed_sample_decode_count,
            evidence.silent_sample_file_count,
        ),
        directory_present: evidence.directory_present,
        indexed_sample_file_count: evidence.indexed_sample_count,
        decoded_sample_file_count: evidence.decoded_sample_file_count,
        failed_sample_decode_count: evidence.failed_sample_decode_count,
        silent_sample_file_count: evidence.silent_sample_file_count,
        observed_filename_pitch_bounds_midi,
        filename_pitch_gaps_midi_within_observed_range: evidence
            .filename_pitch_gaps_midi_within_observed_range
            .clone(),
        indexed_pitch_count: evidence.distinct_pitch_count,
        distinct_filename_dynamic_strength_count: evidence.distinct_dynamic_strength_count,
        indexed_sample_content_manifest_sha256: evidence
            .indexed_sample_content_manifest_sha256
            .clone(),
    }
}

fn sample_bank_status(
    directory_present: bool,
    indexed_samples: usize,
    decoded_samples: usize,
    failed_decodes: usize,
    silent_samples: usize,
) -> &'static str {
    if !directory_present {
        "mapped_directory_missing"
    } else if indexed_samples == 0 {
        "directory_present_no_indexed_pitch_samples"
    } else if decoded_samples == 0 {
        "no_indexed_samples_decoded"
    } else if silent_samples >= decoded_samples {
        "no_non_silent_decoded_samples"
    } else if failed_decodes > 0 || decoded_samples < indexed_samples || silent_samples > 0 {
        "partial_sample_audio_evidence"
    } else {
        "indexed_samples_decoded_non_silent"
    }
}

fn planned_gestures(instrument: &str) -> Vec<PlannedGestureV1> {
    let gestures: &[&str] = match instrument {
        "Recorder" => &["tongued_attack", "slur", "breath_phrasing", "register_transition"],
        "Ocarina" => &["breath_dynamics", "phrase_breath", "pitch_transition", "register_transition"],
        "Harmonica" => &["blow_draw", "pitch_bend", "overblow_overdraw", "chord_double_stop"],
        "DanTranh" => &["pluck_dynamic", "tremolo", "pitch_bend", "tuning_scale"],
        _ => &[],
    };
    gestures
        .iter()
        .map(|gesture| PlannedGestureV1 {
            gesture: *gesture,
            status: "planned_unvalidated",
        })
        .collect()
}

fn parse_args() -> Result<Options, String> {
    let data_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples");
    let default_vcsl = std::env::var_os("SYMTHAEA_VCSL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_root.join("vcsl"));
    let default_vsco2 = std::env::var_os("SYMTHAEA_VSCO2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_root.join("vsco2-ce"));
    let mut options = Options {
        vcsl_dir: default_vcsl,
        vsco2_dir: default_vsco2,
        soundfont: None,
        hash_indexed_samples: false,
        hash_root_wav_contents: false,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--vcsl-dir" => {
                options.vcsl_dir = PathBuf::from(
                    args.next().ok_or_else(|| "--vcsl-dir requires a path".to_string())?,
                );
            }
            "--vsco2-dir" => {
                options.vsco2_dir = PathBuf::from(
                    args.next().ok_or_else(|| "--vsco2-dir requires a path".to_string())?,
                );
            }
            "--soundfont" => {
                options.soundfont = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--soundfont requires a path".to_string())?,
                ));
            }
            "--hash-indexed-samples" => options.hash_indexed_samples = true,
            "--hash-root-wav-contents" => options.hash_root_wav_contents = true,
            _ => return Err(format!("unknown argument: {arg} (use --help)")),
        }
    }
    Ok(options)
}

fn print_help() {
    println!(
        "MEL-INST-CAP-001: auditable instrument capability matrix\n\
         Usage: cargo run -p symthaea-muse --bin muse_instrument_qualification -- [OPTIONS]\n\
         Options:\n\
           --vcsl-dir PATH         VCSL sample root (default: SYMTHAEA_VCSL_DIR or data/samples/vcsl)\n\
           --vsco2-dir PATH        VSCO2-CE sample root (default: SYMTHAEA_VSCO2_DIR or data/samples/vsco2-ce)\n\
           --soundfont PATH        Optional SoundFont header inventory (no FluidSynth rendering)\n\
           --hash-indexed-samples  Hash all indexed WAV contents (can be I/O intensive)\n\
           --hash-root-wav-contents Hash all WAV bytes under each complete root (very I/O intensive)\n\
           -h, --help              Show this help\n\
         Writes JSON to stdout. Source inventory does not certify playable range or instrument proficiency.\n\
         Indexed-bank and root-wide content hashing are independently opt-in; root hashing includes WAVs rejected by the pitch indexer."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_bank_status_never_promotes_inventory_to_playability() {
        assert_eq!(
            sample_bank_status(false, 0, 0, 0, 0),
            "mapped_directory_missing"
        );
        assert_eq!(
            sample_bank_status(true, 0, 0, 0, 0),
            "directory_present_no_indexed_pitch_samples"
        );
        assert_eq!(
            sample_bank_status(true, 5, 0, 5, 0),
            "no_indexed_samples_decoded"
        );
        assert_eq!(
            sample_bank_status(true, 5, 5, 0, 5),
            "no_non_silent_decoded_samples"
        );
        assert_eq!(
            sample_bank_status(true, 5, 5, 1, 0),
            "partial_sample_audio_evidence"
        );
        assert_eq!(
            sample_bank_status(true, 6, 5, 0, 0),
            "partial_sample_audio_evidence"
        );
        assert_eq!(
            sample_bank_status(true, 5, 5, 0, 0),
            "indexed_samples_decoded_non_silent"
        );
    }

    #[test]
    fn requested_new_instruments_have_explicit_unvalidated_gesture_gaps() {
        for (name, expected) in [
            ("Recorder", 4),
            ("Ocarina", 4),
            ("Harmonica", 4),
            ("DanTranh", 4),
        ] {
            let gestures = planned_gestures(name);
            assert_eq!(gestures.len(), expected);
            assert!(gestures.iter().all(|gesture| gesture.status == "planned_unvalidated"));
        }
        assert!(planned_gestures("Piano").is_empty());
    }
}
