// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auditable instrument capability matrix for Melothaea.
//!
//! This tool summarizes *observed source inventory* separately from *proven
//! render capability*. A filename-derived MIDI range is never promoted to a
//! playable range, and inventory-only output cannot mark an instrument as
//! qualified.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
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
    source_terms_evidence: Vec<AssetSourceTermsEvidenceV1>,
    instruments: Vec<InstrumentQualificationRowV1>,
    next_evidence_gates: Vec<&'static str>,
}

/// License/readme bytes and upstream checkout identity are independent from WAV
/// integrity: a valid audio digest is not evidence of licensing or source lineage.
#[derive(Debug, Serialize)]
struct AssetSourceTermsEvidenceV1 {
    source_library: &'static str,
    upstream_repository_url: &'static str,
    declared_license_spdx: &'static str,
    license_declaration_boundary: &'static str,
    license_document_relative_path: &'static str,
    license_document_sha256: Option<String>,
    license_document_status: &'static str,
    readme_document_relative_path: &'static str,
    readme_document_sha256: Option<String>,
    readme_document_status: &'static str,
    upstream_readme_request: &'static str,
    upstream_revision: Option<String>,
    upstream_origin_matches_expected: bool,
    source_checkout_clean: Option<bool>,
    source_revision_status: &'static str,
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
    let source_terms_evidence = vec![
        inspect_asset_source_terms(
            &options.vcsl_dir,
            "vcsl",
            "https://github.com/sgossner/VCSL",
            "sgossner/VCSL",
            "README.md",
            "No attribution required according to the upstream README; preserve its license/readme bytes for review.",
        ),
        inspect_asset_source_terms(
            &options.vsco2_dir,
            "vsco2_ce",
            "https://github.com/sgossner/VSCO-2-CE",
            "sgossner/VSCO-2-CE",
            "Readme.txt",
            "Upstream README asks for courtesy/author credit; keep this request distinct from the declared SPDX license and review the source terms rather than silently discarding the request.",
        ),
    ];
    let report = QualificationMatrixV1 {
        schema_version: "melothaea.instrument-capability-matrix.v4",
        benchmark: "MEL-INST-CAP-001",
        qualification_result: "not_qualified_by_source_inventory_alone",
        evidence_boundary: "source inventory, filename pitch coverage, WAV decode and silence audit only; optional root content manifests hash all WAVs found by the root walker, including files outside the pitch indexer; no render-grid, gesture, or human-preference result is implied",
        source_revision: inventory.source_revision.clone(),
        vcsl_root: inventory.vcsl_root,
        vsco2_root: inventory.vsco2_root,
        indexed_sample_content_hashing_enabled: inventory.indexed_sample_content_hashing_enabled,
        root_content_manifest_hashing_enabled: inventory.root_content_manifest_hashing_enabled,
        source_terms_evidence,
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

fn inspect_asset_source_terms(
    root: &Path,
    source_library: &'static str,
    upstream_repository_url: &'static str,
    expected_remote: &str,
    readme_document_relative_path: &'static str,
    upstream_readme_request: &'static str,
) -> AssetSourceTermsEvidenceV1 {
    let (license_document_sha256, license_document_status) =
        hash_local_source_document(root, "LICENSE");
    let (readme_document_sha256, readme_document_status) =
        hash_local_source_document(root, readme_document_relative_path);
    let (upstream_revision, upstream_origin_matches_expected, source_checkout_clean, source_revision_status) =
        inspect_upstream_checkout(root, expected_remote);

    AssetSourceTermsEvidenceV1 {
        source_library,
        upstream_repository_url,
        declared_license_spdx: "CC0-1.0",
        license_declaration_boundary: "Expected SPDX-style license identifier from reviewed upstream-repository policy; not automatically inferred from local bytes, not a legal opinion, and never a substitute for inspecting the hashed license/readme files.",
        license_document_relative_path: "LICENSE",
        license_document_sha256,
        license_document_status,
        readme_document_relative_path,
        readme_document_sha256,
        readme_document_status,
        upstream_readme_request,
        upstream_revision,
        upstream_origin_matches_expected,
        source_checkout_clean,
        source_revision_status,
    }
}

fn hash_local_source_document(
    root: &Path,
    relative_path: &str,
) -> (Option<String>, &'static str) {
    let canonical_root = match std::fs::canonicalize(root) {
        Ok(path) if path.is_dir() => path,
        _ => return (None, "root_unavailable"),
    };
    let requested = root.join(relative_path);
    let canonical_file = match std::fs::canonicalize(&requested) {
        Ok(path) if path.starts_with(&canonical_root) && path.is_file() => path,
        Ok(_) => return (None, "unsafe_or_not_regular_file"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return (None, "missing"),
        Err(_) => return (None, "read_error"),
    };
    match sha256_file(&canonical_file) {
        Ok(digest) => (Some(digest), "hashed"),
        Err(_) => (None, "read_error"),
    }
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn inspect_upstream_checkout(
    root: &Path,
    expected_remote: &str,
) -> (Option<String>, bool, Option<bool>, &'static str) {
    let canonical_root = match std::fs::canonicalize(root) {
        Ok(path) if path.is_dir() => path,
        _ => return (None, false, None, "root_unavailable"),
    };
    let git = |arguments: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(arguments)
            .output()
    };

    let top_level_output = match git(&["rev-parse", "--show-toplevel"]) {
        Ok(output) if output.status.success() => output,
        Ok(_) => return (None, false, None, "not_a_git_checkout"),
        Err(_) => return (None, false, None, "git_unavailable"),
    };
    let top_level_text = String::from_utf8_lossy(&top_level_output.stdout).trim().to_owned();
    let top_level = match std::fs::canonicalize(&top_level_text) {
        Ok(path) => path,
        Err(_) => return (None, false, None, "git_root_unreadable"),
    };
    if top_level != canonical_root {
        return (None, false, None, "asset_root_is_not_repository_root");
    }

    let upstream_revision = match git(&["rev-parse", "HEAD"]) {
        Ok(output) if output.status.success() => {
            let revision = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if valid_git_revision(&revision) {
                Some(revision)
            } else {
                return (None, false, None, "invalid_git_revision");
            }
        }
        _ => return (None, false, None, "git_revision_unavailable"),
    };

    let remote = match git(&["remote", "get-url", "origin"]) {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
        }
        _ => None,
    };
    let upstream_origin_matches_expected = remote
        .as_deref()
        .and_then(normalize_github_remote)
        .is_some_and(|remote| remote == expected_remote.to_ascii_lowercase());

    let clean = match git(&["status", "--porcelain", "--untracked-files=all"]) {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).trim().is_empty())
        }
        _ => None,
    };
    let status = match (upstream_origin_matches_expected, clean) {
        (true, Some(true)) => "clean_upstream_checkout",
        (false, _) => "upstream_origin_mismatch",
        (true, Some(false)) => "dirty_worktree",
        (true, None) => "git_status_unavailable",
    };
    (
        upstream_revision,
        upstream_origin_matches_expected,
        clean,
        status,
    )
}

fn normalize_github_remote(remote: &str) -> Option<String> {
    let remote = remote.trim().trim_end_matches('/');
    let path = remote
        .strip_prefix("https://github.com/")
        .or_else(|| remote.strip_prefix("http://github.com/"))
        .or_else(|| remote.strip_prefix("ssh://git@github.com/"))
        .or_else(|| remote.strip_prefix("git@github.com:"))?;
    let path = path.split(['?', '#']).next().unwrap_or(path).trim_end_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);
    if path.is_empty() || path.contains(':') || path.contains('@') || path.contains("..") {
        return None;
    }
    Some(path.to_ascii_lowercase())
}

fn valid_git_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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
         Indexed-bank and root-wide content hashing are independently opt-in; root hashing includes WAVs rejected by the pitch indexer.\n\
         Source terms evidence hashes each root's LICENSE/readme and separately checks Git root, expected origin, revision, and worktree state. Missing docs or non-upstream/non-clean roots remain explicit; digest evidence is not legal advice."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_remote_normalization_accepts_expected_github_forms() {
        assert_eq!(
            normalize_github_remote("https://github.com/sgossner/VCSL.git"),
            Some("sgossner/vcsl".into())
        );
        assert_eq!(
            normalize_github_remote("git@github.com:sgossner/VSCO-2-CE.git"),
            Some("sgossner/vsco-2-ce".into())
        );
        assert_eq!(
            normalize_github_remote("ssh://git@github.com/sgossner/VCSL"),
            Some("sgossner/vcsl".into())
        );
    }

    #[test]
    fn upstream_remote_normalization_rejects_unknown_hosts_and_paths() {
        assert_eq!(normalize_github_remote("https://example.com/sgossner/VCSL"), None);
        assert_eq!(normalize_github_remote("https://github.com/sgossner/../VCSL"), None);
        assert_eq!(normalize_github_remote("git@github.com:sgossner/VCSL:other.git"), None);
    }

    #[test]
    fn source_document_hashing_reports_missing_files_explicitly() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "muse-source-terms-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("temporary source root should be created");
        let (digest, status) = hash_local_source_document(&root, "LICENSE");
        assert_eq!(digest, None);
        assert_eq!(status, "missing");

        std::fs::write(root.join("LICENSE"), b"CC0 fixture document")
            .expect("license fixture should be written");
        let (digest, status) = hash_local_source_document(&root, "LICENSE");
        assert_eq!(status, "hashed");
        assert_eq!(
            digest,
            Some(sha256_file(&root.join("LICENSE")).expect("fixture should hash"))
        );
        std::fs::remove_dir_all(&root).expect("temporary source root should be removed");
    }

    #[test]
    fn source_license_and_readme_requests_are_separate_fields() {
        let source = inspect_asset_source_terms(
            Path::new("/path/that/does/not/exist"),
            "vsco2_ce",
            "https://github.com/sgossner/VSCO-2-CE",
            "sgossner/VSCO-2-CE",
            "Readme.txt",
            "Upstream README asks for courtesy/author credit; keep this request distinct from the declared SPDX license and review the source terms rather than silently discarding the request.",
        );
        assert_eq!(source.declared_license_spdx, "CC0-1.0");
        assert!(source.upstream_readme_request.contains("courtesy/author credit"));
        assert_eq!(source.license_document_status, "root_unavailable");
        assert_ne!(source.upstream_readme_request, source.license_declaration_boundary);
    }

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
