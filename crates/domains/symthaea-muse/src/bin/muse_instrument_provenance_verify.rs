// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed cross-check for Melothaea instrument-probe and source-inventory reports.
//!
//! This verifies provenance consistency, not musical quality or instrument qualification.
//! Input reports are treated as untrusted data. A successful result is a consistency check,
//! not a signature or an authenticated build attestation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug)]
struct Options {
    probe_path: PathBuf,
    inventory_path: PathBuf,
    renderer_binary_path: PathBuf,
    vcsl_root: PathBuf,
    vsco2_root: PathBuf,
}

#[derive(Debug)]
struct RootObservation {
    canonical_path: Option<PathBuf>,
    content_manifest_sha256: Option<String>,
    wav_file_count: Option<usize>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct VerificationReport {
    schema_version: &'static str,
    benchmark: &'static str,
    verdict: &'static str,
    probe_report_sha256: String,
    capability_report_sha256: String,
    source_revision: Option<String>,
    source_revision_match: bool,
    renderer_binary_sha256: Option<String>,
    renderer_binary_file_sha256: String,
    renderer_binary_file_size_bytes: u64,
    renderer_binary_file_name_matches_report: bool,
    renderer_binary_identity_verified: bool,
    vcsl_root_content_manifest_sha256: Option<String>,
    vsco2_root_content_manifest_sha256: Option<String>,
    root_content_manifests_verified: usize,
    selected_samples_total: usize,
    selected_samples_with_matching_actual_file_digest: usize,
    selected_samples_with_valid_file_digest: usize,
    selected_samples_with_valid_bank_digest: usize,
    selected_samples_with_matching_inventory_bank_digest: usize,
    cases_total: usize,
    error_count: usize,
    incomplete_count: usize,
    findings: Vec<Finding>,
    evidence_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct Finding {
    severity: &'static str,
    code: &'static str,
    detail: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("muse_instrument_provenance_verify: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let options = parse_args()?;
    // Hash bytes through the same reader used by the JSON parser, so the
    // reported input digest binds the exact document that was actually parsed.
    let (renderer_binary_file_sha256, renderer_binary_file_size_bytes) =
        file_sha256(&options.renderer_binary_path)
            .map_err(|error| format!("cannot hash supplied renderer binary: {error}"))?;
    let renderer_binary_file_name = options
        .renderer_binary_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| "renderer binary path has no filename".to_string())?;
    let (probe, probe_report_sha256) = read_json_with_sha256(&options.probe_path)
        .map_err(|error| format!("cannot read probe report: {error}"))?;
    let (inventory, capability_report_sha256) =
        read_json_with_sha256(&options.inventory_path)
            .map_err(|error| format!("cannot read capability report: {error}"))?;
    let vcsl_root_observation = observe_asset_root(&options.vcsl_root);
    let vsco2_root_observation = observe_asset_root(&options.vsco2_root);

    let report = verify_reports(
        &probe,
        &inventory,
        probe_report_sha256,
        capability_report_sha256,
        renderer_binary_file_sha256,
        renderer_binary_file_size_bytes,
        &renderer_binary_file_name,
        &vcsl_root_observation,
        &vsco2_root_observation,
    );
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, &report)
        .map_err(|error| format!("cannot serialize verification report: {error}"))?;
    writeln!(lock).map_err(|error| format!("cannot flush verification report: {error}"))?;

    if report.verdict == "consistent" {
        Ok(ExitCode::SUCCESS)
    } else {
        // Machine-readable findings are already emitted. Non-zero means this evidence
        // set must not be used as a complete provenance gate.
        Ok(ExitCode::from(1))
    }
}

fn parse_args() -> Result<Options, String> {
    let mut probe_path = None;
    let mut inventory_path = None;
    let mut renderer_binary_path = None;
    let mut vcsl_root = None;
    let mut vsco2_root = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--probe" => {
                probe_path = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--probe requires a JSON file path".to_string())?,
                ));
            }
            "--inventory" => {
                inventory_path = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--inventory requires a JSON file path".to_string())?,
                ));
            }
            "--renderer-binary" => {
                renderer_binary_path = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--renderer-binary requires a file path".to_string())?,
                ));
            }
            "--vcsl-root" => {
                vcsl_root = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--vcsl-root requires an asset-root path".to_string())?,
                ));
            }
            "--vsco2-root" => {
                vsco2_root = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--vsco2-root requires an asset-root path".to_string())?,
                ));
            }
            _ => return Err(format!("unknown argument: {arg} (use --help)")),
        }
    }

    Ok(Options {
        probe_path: probe_path.ok_or_else(|| "--probe is required".to_string())?,
        inventory_path: inventory_path.ok_or_else(|| "--inventory is required".to_string())?,
        renderer_binary_path: renderer_binary_path
            .ok_or_else(|| "--renderer-binary is required".to_string())?,
        vcsl_root: vcsl_root.ok_or_else(|| "--vcsl-root is required".to_string())?,
        vsco2_root: vsco2_root.ok_or_else(|| "--vsco2-root is required".to_string())?,
    })
}

fn print_help() {
    println!(
        "MEL-PROV-VERIFY-001: fail-closed provenance consistency check\n\
         Usage: cargo run -p symthaea-muse --bin muse_instrument_provenance_verify -- --probe REPORT.json --inventory MATRIX.json --renderer-binary /path/to/muse_instrument_probe --vcsl-root /assets/VCSL --vsco2-root /assets/VSCO2-CE\n\
         Required evidence: supplied executable bytes, independently rehashed root WAV manifests and selected WAV files, matching source revision,\n\
         indexed-bank manifests, and selected-sample bank digests matching the inventory mapping.\n\
         JSON findings are written to stdout. Exit 0 means consistent; exit 1 means incomplete or inconsistent;\n\
         exit 2 means invocation or input reports could not be processed. This is not instrument qualification."
    );
}

fn verify_reports(
    probe: &Value,
    inventory: &Value,
    probe_report_sha256: String,
    capability_report_sha256: String,
    renderer_binary_file_sha256: String,
    renderer_binary_file_size_bytes: u64,
    renderer_binary_file_name: &str,
    vcsl_root_observation: &RootObservation,
    vsco2_root_observation: &RootObservation,
) -> VerificationReport {
    let mut findings = Vec::new();
    let mut root_content_manifests_verified = 0_usize;
    let mut selected_samples_total = 0_usize;
    let mut selected_samples_with_valid_file_digest = 0_usize;
    let mut selected_samples_with_valid_bank_digest = 0_usize;
    let mut selected_samples_with_matching_inventory_bank_digest = 0_usize;

    if string_field(probe, "schema_version") != Some("melothaea.instrument-probe.v4") {
        push_error(
            &mut findings,
            "probe_schema_unsupported",
            "probe report must use melothaea.instrument-probe.v4".into(),
        );
    }
    if string_field(inventory, "schema_version")
        != Some("melothaea.instrument-capability-matrix.v3")
    {
        push_error(
            &mut findings,
            "inventory_schema_unsupported",
            "capability report must use melothaea.instrument-capability-matrix.v3".into(),
        );
    }

    let probe_revision = string_field(probe, "source_revision");
    let inventory_revision = string_field(inventory, "source_revision");
    let source_revision = probe_revision.map(str::to_owned);
    let revision_status = string_field(probe, "source_revision_status");
    let source_revision_match = match (probe_revision, inventory_revision) {
        (Some(left), Some(right)) if !valid_git_revision(left) || !valid_git_revision(right) => {
            push_incomplete(
                &mut findings,
                "source_revision_invalid",
                "source revision must be a 40- or 64-character hexadecimal Git object ID".into(),
            );
            false
        }
        (Some(left), Some(right)) if left != right => {
            push_error(
                &mut findings,
                "source_revision_mismatch",
                format!("probe source revision {left} differs from inventory revision {right}"),
            );
            false
        }
        (Some(_), Some(_))
            if !matches!(
                revision_status,
                Some("runtime_symthaea_source_rev" | "runtime_github_sha" | "compile_time_embedded")
            ) =>
        {
            push_incomplete(
                &mut findings,
                "source_revision_status_untrusted",
                "source revision is present but its provenance status is missing or unavailable".into(),
            );
            false
        }
        (Some(_), Some(_)) => true,
        _ => {
            push_incomplete(
                &mut findings,
                "source_revision_unavailable",
                "both reports must contain the same explicit 40- or 64-hex Git revision".into(),
            );
            false
        }
    };

    let renderer_binary_sha256 = string_field(probe, "renderer_binary_sha256").map(str::to_owned);
    let renderer_binary_file_name_matches_report =
        string_field(probe, "renderer_binary_name") == Some(renderer_binary_file_name);
    let renderer_binary_identity_verified = match (
        string_field(probe, "renderer_binary_identity_status"),
        renderer_binary_sha256.as_deref(),
        string_field(probe, "renderer_binary_name"),
        probe.get("renderer_binary_size_bytes").and_then(Value::as_u64),
    ) {
        (Some("hashed"), Some(digest), Some(name), Some(size))
            if valid_sha256(digest) && !name.trim().is_empty() && size > 0 =>
        {
            if digest != renderer_binary_file_sha256.as_str() {
                push_error(
                    &mut findings,
                    "renderer_binary_digest_mismatch",
                    "supplied executable bytes do not match renderer_binary_sha256 in the probe report".into(),
                );
                false
            } else if size != renderer_binary_file_size_bytes {
                push_error(
                    &mut findings,
                    "renderer_binary_size_mismatch",
                    "supplied executable byte length does not match the probe report".into(),
                );
                false
            } else if !renderer_binary_file_name_matches_report {
                push_error(
                    &mut findings,
                    "renderer_binary_name_mismatch",
                    "supplied executable basename does not match the probe report".into(),
                );
                false
            } else {
                true
            }
        }
        (Some("hashed"), Some(digest), _, _) if !valid_sha256(digest) => {
            push_error(
                &mut findings,
                "renderer_digest_invalid",
                "renderer claims hashed identity but its SHA-256 is malformed".into(),
            );
            false
        }
        _ => {
            push_incomplete(
                &mut findings,
                "renderer_identity_incomplete",
                "renderer binary must have status=hashed, a valid SHA-256, a basename, and nonzero byte length".into(),
            );
            false
        }
    };

    if bool_field(inventory, "root_content_manifest_hashing_enabled") != Some(true) {
        push_incomplete(
            &mut findings,
            "root_manifest_hashing_not_enabled",
            "inventory must be produced with --hash-root-wav-contents".into(),
        );
    }
    for (field, label) in [("vcsl_root", "VCSL"), ("vsco2_root", "VSCO2-CE")] {
        let Some(root) = inventory.get(field) else {
            push_incomplete(
                &mut findings,
                "root_evidence_missing",
                format!("{label} root evidence is missing"),
            );
            continue;
        };
        match root.get("root_present").and_then(Value::as_bool) {
            Some(true) => {
                let status = string_field(root, "content_manifest_status");
                let digest = string_field(root, "content_manifest_sha256");
                if status == Some("hashed") && digest.is_some_and(valid_sha256) {
                    root_content_manifests_verified += 1;
                } else if status == Some("hashed") && digest.is_some_and(|value| !valid_sha256(value)) {
                    push_error(
                        &mut findings,
                        "root_manifest_digest_invalid",
                        format!("{label} root claims hashed content but digest is malformed"),
                    );
                } else {
                    push_incomplete(
                        &mut findings,
                        "root_manifest_incomplete",
                        format!("{label} root is present but has no complete content manifest"),
                    );
                }
            }
            Some(false)
                if string_field(root, "content_manifest_status") == Some("root_missing")
                    && root.get("content_manifest_sha256").is_none_or(Value::is_null) =>
            {
                push_incomplete(
                    &mut findings,
                    "asset_root_missing",
                    format!("{label} asset root is absent; no source bytes can be attested for it"),
                );
            }
            Some(false) => push_error(
                &mut findings,
                "root_presence_status_conflict",
                format!("{label} root is marked absent but does not carry an explicit root_missing status"),
            ),
            None => push_incomplete(
                &mut findings,
                "root_presence_unknown",
                format!("{label} root evidence has no root_present boolean"),
            ),
        }
    }

    if bool_field(inventory, "indexed_sample_content_hashing_enabled") != Some(true) {
        push_incomplete(
            &mut findings,
            "inventory_bank_hashing_not_enabled",
            "inventory must be produced with --hash-indexed-samples".into(),
        );
    }
    let probe_bank_hashing = bool_field(probe, "source_bank_manifest_hashing_enabled");
    if probe_bank_hashing != Some(true) {
        push_incomplete(
            &mut findings,
            "probe_bank_hashing_not_enabled",
            "probe must run with SYMTHAEA_MUSE_HASH_SOURCE_BANK_MANIFEST=1 before startup".into(),
        );
    }

    let cases: &[Value] = match probe.get("cases").and_then(Value::as_array) {
        Some(cases) => cases.as_slice(),
        None => {
            push_error(
                &mut findings,
                "probe_cases_missing",
                "probe report has no cases array".into(),
            );
            &[]
        }
    };
    let cases_total = cases.len();
    if cases.is_empty() {
        push_incomplete(
            &mut findings,
            "probe_has_no_cases",
            "an empty probe cannot establish per-case provenance consistency".into(),
        );
    }
    if probe.get("cases_total").and_then(Value::as_u64) != Some(cases_total as u64) {
        push_error(
            &mut findings,
            "case_count_mismatch",
            "cases_total does not match the serialized cases array length".into(),
        );
    }

    for (index, case) in cases.iter().enumerate() {
        let case_label = format!("case {index}");
        match string_field(case, "output_sha256") {
            Some(digest) if valid_sha256(digest) => {}
            Some(_) => push_error(
                &mut findings,
                "output_digest_invalid",
                format!("{case_label} has a malformed output_sha256"),
            ),
            None => push_incomplete(
                &mut findings,
                "output_digest_missing",
                format!("{case_label} has no output_sha256"),
            ),
        }
        match bool_field(case, "receipt_matches_request") {
            Some(true) => {}
            Some(false) => push_error(
                &mut findings,
                "render_receipt_mismatch",
                format!("{case_label} receipt does not match the requested note controls"),
            ),
            None => push_incomplete(
                &mut findings,
                "render_receipt_missing",
                format!("{case_label} lacks receipt_matches_request"),
            ),
        }
        match bool_field(case, "sample_evidence_repeats_match") {
            Some(true) => {}
            Some(false) => push_error(
                &mut findings,
                "sample_evidence_repeat_mismatch",
                format!("{case_label} selected-sample evidence differs across repeats"),
            ),
            None => push_incomplete(
                &mut findings,
                "sample_repeat_evidence_missing",
                format!("{case_label} lacks sample_evidence_repeats_match"),
            ),
        }

        let Some(selected_sample_value) = case.get("selected_sample") else {
            push_incomplete(
                &mut findings,
                "selected_sample_field_missing",
                format!("{case_label} has no selected_sample field"),
            );
            continue;
        };
        if selected_sample_value.is_null() {
            continue;
        }
        selected_samples_total += 1;
        let Some(sample) = selected_sample_value.as_object().map(|_| selected_sample_value) else {
            push_error(
                &mut findings,
                "selected_sample_invalid",
                format!("{case_label} selected_sample is neither null nor an object"),
            );
            continue;
        };

        match string_field(sample, "source_file_sha256") {
            Some(digest) if valid_sha256(digest) => selected_samples_with_valid_file_digest += 1,
            Some(_) => push_error(
                &mut findings,
                "source_file_digest_invalid",
                format!("{case_label} selected source has a malformed byte digest"),
            ),
            None => push_incomplete(
                &mut findings,
                "source_file_digest_missing",
                format!("{case_label} selected source has no exact-file digest"),
            ),
        }
        let source_library = string_field(sample, "source_library");
        let relative_path = string_field(sample, "source_relative_path");
        if source_library.is_none_or(|value| value.is_empty())
            || relative_path.is_none_or(|value| value.is_empty())
        {
            push_incomplete(
                &mut findings,
                "source_location_incomplete",
                format!("{case_label} selected source lacks its library or root-relative path"),
            );
        }

        if probe_bank_hashing == Some(true) {
            let source_bank_digest = string_field(sample, "source_bank_content_manifest_sha256");
            let bank_status = string_field(sample, "source_bank_manifest_status");
            let bank_digest_valid = match (bank_status, source_bank_digest) {
                (Some("hashed"), Some(digest)) if valid_sha256(digest) => {
                    selected_samples_with_valid_bank_digest += 1;
                    true
                }
                (Some("hashed"), Some(_)) => {
                    push_error(
                        &mut findings,
                        "source_bank_digest_invalid",
                        format!("{case_label} claims a hashed source bank but digest is malformed"),
                    );
                    false
                }
                (Some("hash_failed"), _) => {
                    push_incomplete(
                        &mut findings,
                        "source_bank_hash_failed",
                        format!("{case_label} selected sample's bank manifest hash failed"),
                    );
                    false
                }
                _ => {
                    push_incomplete(
                        &mut findings,
                        "source_bank_digest_missing",
                        format!("{case_label} selected sample has no successful bank manifest digest"),
                    );
                    false
                }
            };

            if bank_digest_valid {
                let case_instrument = string_field(case, "instrument");
                let matching_banks = matching_inventory_banks(
                    inventory,
                    case_instrument,
                    source_library,
                    relative_path,
                );
                match matching_banks.as_slice() {
                    [bank] => match string_field(bank, "indexed_sample_content_manifest_sha256") {
                        Some(inventory_digest) if valid_sha256(inventory_digest) => {
                            if source_bank_digest == Some(inventory_digest) {
                                selected_samples_with_matching_inventory_bank_digest += 1;
                            } else {
                                push_error(
                                    &mut findings,
                                    "source_bank_manifest_mismatch",
                                    format!(
                                        "{case_label} selected-bank digest differs from the inventory bank for the same instrument/library/path"
                                    ),
                                );
                            }
                        }
                        Some(_) => push_error(
                            &mut findings,
                            "inventory_bank_digest_invalid",
                            format!("{case_label} matching inventory bank has a malformed digest"),
                        ),
                        None => push_incomplete(
                            &mut findings,
                            "inventory_bank_digest_missing",
                            format!("{case_label} matching inventory bank was not content-hashed"),
                        ),
                    },
                    [] => push_incomplete(
                        &mut findings,
                        "inventory_bank_not_found",
                        format!(
                            "{case_label} source bank cannot be matched to a declared sustained, short-articulation, or additional-style inventory entry"
                        ),
                    ),
                    _ => push_error(
                        &mut findings,
                        "inventory_bank_ambiguous",
                        format!("{case_label} source bank matches more than one inventory entry"),
                    ),
                }
            }
        }
    }

    let error_count = findings.iter().filter(|item| item.severity == "error").count();
    let incomplete_count = findings.iter().filter(|item| item.severity == "incomplete").count();
    let verdict = if error_count > 0 {
        "inconsistent"
    } else if incomplete_count > 0 {
        "incomplete"
    } else {
        "consistent"
    };

    VerificationReport {
        schema_version: "melothaea.provenance-verification.v1",
        benchmark: "MEL-PROV-VERIFY-001",
        verdict,
        probe_report_sha256,
        capability_report_sha256,
        source_revision,
        source_revision_match,
        renderer_binary_sha256,
        renderer_binary_file_sha256,
        renderer_binary_file_size_bytes,
        renderer_binary_file_name_matches_report,
        renderer_binary_identity_verified,
        root_content_manifests_verified,
        selected_samples_total,
        selected_samples_with_valid_file_digest,
        selected_samples_with_valid_bank_digest,
        selected_samples_with_matching_inventory_bank_digest,
        cases_total,
        error_count,
        incomplete_count,
        findings,
        evidence_boundary: "provenance consistency only; this does not qualify instrument fidelity, validate licensing, authenticate the source revision, attest the build environment, or prove musical quality",
    }
}

fn matching_inventory_banks<'a>(
    inventory: &'a Value,
    instrument: Option<&str>,
    source_library: Option<&str>,
    source_relative_path: Option<&str>,
) -> Vec<&'a Value> {
    let (Some(instrument), Some(source_library), Some(source_path)) =
        (instrument, source_library, source_relative_path)
    else {
        return Vec::new();
    };
    let Some(rows) = inventory.get("instruments").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut banks = Vec::new();
    for row in rows.iter().filter(|row| string_field(row, "instrument") == Some(instrument)) {
        for field in ["sustained_samples", "short_articulation_samples"] {
            if let Some(bank) = row.get(field).filter(|value| bank_matches(value, source_library, source_path)) {
                banks.push(bank);
            }
        }
        if let Some(styles) = row.get("additional_sample_styles").and_then(Value::as_array) {
            for style in styles {
                if let Some(bank) = style
                    .get("evidence")
                    .filter(|value| bank_matches(value, source_library, source_path))
                {
                    banks.push(bank);
                }
            }
        }
    }
    banks
}

fn bank_matches(bank: &Value, source_library: &str, source_relative_path: &str) -> bool {
    let Some(library) = string_field(bank, "source_library") else {
        return false;
    };
    let Some(directory) = string_field(bank, "relative_directory") else {
        return false;
    };
    if library != source_library || directory.trim_matches('/').is_empty() {
        return false;
    }
    let prefix = format!("{}/", directory.trim_matches('/'));
    source_relative_path.starts_with(&prefix)
}

fn string_field<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn bool_field(value: &Value, field: &str) -> Option<bool> {
    value.get(field).and_then(Value::as_bool)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_git_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn push_error(findings: &mut Vec<Finding>, code: &'static str, detail: String) {
    findings.push(Finding { severity: "error", code, detail });
}

fn push_incomplete(findings: &mut Vec<Finding>, code: &'static str, detail: String) {
    findings.push(Finding { severity: "incomplete", code, detail });
}

struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..count]);
        Ok(count)
    }
}

fn read_json_with_sha256(path: &Path) -> Result<(Value, String), String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut reader = HashingReader {
        inner: file,
        hasher: Sha256::new(),
    };
    let value = {
        let mut deserializer = serde_json::Deserializer::from_reader(&mut reader);
        let value = Value::deserialize(&mut deserializer)
            .map_err(|error| format!("invalid JSON: {error}"))?;
        deserializer
            .end()
            .map_err(|error| format!("trailing data after JSON: {error}"))?;
        value
    };
    let digest = format!("{:x}", reader.hasher.finalize());
    Ok((value, digest))
}

fn file_sha256(path: &Path) -> io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let expected_size = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes_read = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes_read = bytes_read
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "file length overflow while hashing"))?;
        hasher.update(&buffer[..count]);
    }
    if bytes_read != expected_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file size changed while hashing",
        ));
    }
    Ok((format!("{:x}", hasher.finalize()), bytes_read))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixtures() -> (Value, Value) {
        let revision = "a".repeat(40);
        let bank_digest = "b".repeat(64);
        let root_digest = "c".repeat(64);
        let probe = json!({
            "schema_version": "melothaea.instrument-probe.v4",
            "source_revision": revision,
            "source_revision_status": "runtime_symthaea_source_rev",
            "renderer_binary_name": "muse_instrument_probe",
            "renderer_binary_sha256": "d".repeat(64),
            "renderer_binary_size_bytes": 4096,
            "renderer_binary_identity_status": "hashed",
            "source_bank_manifest_hashing_enabled": true,
            "cases_total": 1,
            "cases": [{
                "instrument": "Piano",
                "output_sha256": "e".repeat(64),
                "receipt_matches_request": true,
                "sample_evidence_repeats_match": true,
                "selected_sample": {
                    "source_library": "vcsl",
                    "source_relative_path": "Piano/sample_C4.wav",
                    "source_file_sha256": "f".repeat(64),
                    "source_bank_content_manifest_sha256": bank_digest,
                    "source_bank_manifest_status": "hashed"
                }
            }]
        });
        let bank = json!({
            "source_library": "vcsl",
            "relative_directory": "Piano",
            "directory_present": true,
            "indexed_sample_content_manifest_sha256": bank_digest
        });
        let inventory = json!({
            "schema_version": "melothaea.instrument-capability-matrix.v3",
            "source_revision": revision,
            "indexed_sample_content_hashing_enabled": true,
            "root_content_manifest_hashing_enabled": true,
            "vcsl_root": {
                "root_present": true,
                "content_manifest_status": "hashed",
                "content_manifest_sha256": root_digest
            },
            "vsco2_root": {
                "root_present": true,
                "content_manifest_status": "hashed",
                "content_manifest_sha256": root_digest
            },
            "instruments": [{
                "instrument": "Piano",
                "sustained_samples": bank,
                "short_articulation_samples": null,
                "additional_sample_styles": []
            }]
        });
        (probe, inventory)
    }

    fn test_report(probe: &Value, inventory: &Value) -> VerificationReport {
        verify_reports(
            probe,
            inventory,
            "0".repeat(64),
            "1".repeat(64),
            "d".repeat(64),
            4096,
            "muse_instrument_probe",
        )
    }

    #[test]
    fn verifies_matching_revision_executable_roots_and_source_bank_digest() {
        let (probe, inventory) = fixtures();
        let report = test_report(&probe, &inventory);
        assert_eq!(report.verdict, "consistent");
        assert_eq!(report.root_content_manifests_verified, 2);
        assert_eq!(report.selected_samples_total, 1);
        assert_eq!(report.selected_samples_with_valid_file_digest, 1);
        assert_eq!(report.selected_samples_with_valid_bank_digest, 1);
        assert_eq!(report.selected_samples_with_matching_inventory_bank_digest, 1);
        assert_eq!(report.error_count, 0);
        assert_eq!(report.incomplete_count, 0);
    }

    #[test]
    fn mismatched_supplied_binary_is_rejected_even_when_report_sha_is_well_formed() {
        let (mut probe, inventory) = fixtures();
        probe["renderer_binary_sha256"] = serde_json::json!("9".repeat(64));
        let report = test_report(&probe, &inventory);
        assert_eq!(report.verdict, "inconsistent");
        assert!(!report.renderer_binary_identity_verified);
        assert!(report.findings.iter().any(|finding| finding.code == "renderer_binary_digest_mismatch"));
    }

    #[test]
    fn same_schema_with_a_conflicting_bank_digest_is_rejected() {
        let (probe, mut inventory) = fixtures();
        inventory["instruments"][0]["sustained_samples"]
            ["indexed_sample_content_manifest_sha256"] = json!("9".repeat(64));
        let report = test_report(&probe, &inventory);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "source_bank_manifest_mismatch"));
    }

    #[test]
    fn different_source_revisions_are_rejected() {
        let (probe, mut inventory) = fixtures();
        inventory["source_revision"] = json!("1".repeat(40));
        let report = test_report(&probe, &inventory);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "source_revision_mismatch"));
    }

    #[test]
    fn unavailable_source_revision_status_is_incomplete_not_mislabeled_as_mismatch() {
        let (mut probe, inventory) = fixtures();
        probe["source_revision_status"] = json!("unavailable");
        let report = test_report(&probe, &inventory);
        assert_eq!(report.verdict, "incomplete");
        assert!(report.findings.iter().any(|finding| finding.code == "source_revision_status_untrusted"));
        assert!(!report.findings.iter().any(|finding| finding.code == "source_revision_mismatch"));
    }

    #[test]
    fn missing_root_manifest_is_incomplete_not_a_pass() {
        let (probe, mut inventory) = fixtures();
        inventory["vcsl_root"]["content_manifest_status"] = json!("not_requested");
        inventory["vcsl_root"]["content_manifest_sha256"] = Value::Null;
        let report = test_report(&probe, &inventory);
        assert_eq!(report.verdict, "incomplete");
        assert!(report.findings.iter().any(|finding| finding.code == "root_manifest_incomplete"));
    }

    #[test]
    fn report_digest_covers_the_exact_json_bytes_that_were_parsed() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "muse-provenance-report-{}-{nonce}.json",
            std::process::id()
        ));
        let bytes = br#"{ "schema_version": "fixture", "value": 7 }"#;
        std::fs::write(&path, bytes).expect("fixture should be written");
        let (value, digest) = read_json_with_sha256(&path).expect("fixture should parse and hash");
        assert_eq!(value["value"], 7);
        assert_eq!(digest, format!("{:x}", Sha256::digest(bytes)));
        std::fs::remove_file(&path).expect("fixture should be removed");
    }

    #[test]
    fn malformed_sha256_is_rejected_by_the_digest_validator() {
        assert!(valid_sha256(&"a".repeat(64)));
        assert!(!valid_sha256(&"g".repeat(64)));
        assert!(!valid_sha256(&"a".repeat(63)));
    }

    #[test]
    fn style_bank_receipts_can_be_matched_to_the_inventory() {
        let (probe, mut inventory) = fixtures();
        inventory["instruments"][0]["sustained_samples"] = Value::Null;
        inventory["instruments"][0]["additional_sample_styles"] = json!([{
            "style_name": "normal",
            "style_kind": "sustained",
            "evidence": {
                "source_library": "vcsl",
                "relative_directory": "Piano",
                "indexed_sample_content_manifest_sha256": "b".repeat(64)
            }
        }]);
        let report = test_report(&probe, &inventory);
        assert_eq!(report.verdict, "consistent");
        assert_eq!(report.selected_samples_with_matching_inventory_bank_digest, 1);
    }
}
