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
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

#[derive(Debug)]
struct Options {
    probe_path: PathBuf,
    inventory_path: PathBuf,
    renderer_binary_path: PathBuf,
    vcsl_root: PathBuf,
    vsco2_root: PathBuf,
}

#[derive(Debug, Default)]
struct SourceTermsAudit {
    roots_verified: usize,
    license_documents_verified: usize,
    readme_documents_verified: usize,
    clean_upstream_checkouts_verified: usize,
}

#[derive(Debug, Clone, Copy)]
struct SourceTermsPolicy {
    source_library: &'static str,
    upstream_repository_url: &'static str,
    expected_remote: &'static str,
    declared_license_spdx: &'static str,
    readme_relative_path: &'static str,
    upstream_readme_request: &'static str,
}

#[derive(Debug)]
struct GitCheckoutObservation {
    upstream_revision: Option<String>,
    origin_matches_expected: bool,
    clean: Option<bool>,
    status: &'static str,
}

#[derive(Debug)]
struct RootObservation {
    canonical_path: Option<PathBuf>,
    content_manifest_sha256: Option<String>,
    wav_file_count: Option<usize>,
    wav_relative_paths: Option<HashSet<String>>,
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
    source_terms_roots_verified: usize,
    source_license_documents_verified: usize,
    source_readme_documents_verified: usize,
    clean_upstream_checkouts_verified: usize,
    selected_samples_total: usize,
    selected_samples_with_matching_actual_file_digest: usize,
    selected_samples_with_valid_file_digest: usize,
    selected_samples_with_valid_bank_digest: usize,
    selected_samples_with_matching_inventory_bank_digest: usize,
    selected_samples_with_matching_actual_bank_digest: usize,
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
    let mut selected_samples_with_matching_actual_file_digest = 0_usize;
    let mut selected_samples_with_valid_file_digest = 0_usize;
    let mut selected_samples_with_valid_bank_digest = 0_usize;
    let mut selected_samples_with_matching_inventory_bank_digest = 0_usize;
    let mut selected_samples_with_matching_actual_bank_digest = 0_usize;
    let mut actual_bank_manifest_cache: HashMap<(String, String), Result<String, String>> =
        HashMap::new();

    if string_field(probe, "schema_version") != Some("melothaea.instrument-probe.v4") {
        push_error(
            &mut findings,
            "probe_schema_unsupported",
            "probe report must use melothaea.instrument-probe.v4".into(),
        );
    }
    if string_field(inventory, "schema_version")
        != Some("melothaea.instrument-capability-matrix.v4")
    {
        push_error(
            &mut findings,
            "inventory_schema_unsupported",
            "capability report must use melothaea.instrument-capability-matrix.v4".into(),
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
    for (field, label, observed) in [
        ("vcsl_root", "VCSL", vcsl_root_observation),
        ("vsco2_root", "VSCO2-CE", vsco2_root_observation),
    ] {
        if verify_root_evidence(inventory.get(field), label, observed, &mut findings) {
            root_content_manifests_verified += 1;
        }
    }

    let source_terms_audit = verify_source_terms_evidence(
        inventory,
        vcsl_root_observation,
        vsco2_root_observation,
        &mut findings,
    );

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
        } else if verify_selected_sample_file(
            sample,
            source_library.unwrap(),
            relative_path.unwrap(),
            vcsl_root_observation,
            vsco2_root_observation,
            &case_label,
            &mut findings,
        ) {
            selected_samples_with_matching_actual_file_digest += 1;
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
                    [bank] => {
                        let inventory_digest =
                            string_field(bank, "indexed_sample_content_manifest_sha256");
                        match inventory_digest {
                            Some(digest) if valid_sha256(digest) => {
                                if source_bank_digest == Some(digest) {
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

                                let bank_directory = string_field(bank, "relative_directory");
                                if let Some(bank_directory) = bank_directory {
                                    let cache_key = (
                                        source_library.unwrap_or_default().to_owned(),
                                        bank_directory.to_owned(),
                                    );
                                    let observation = actual_bank_manifest_cache
                                        .entry(cache_key)
                                        .or_insert_with(|| {
                                            indexed_bank_content_manifest_sha256(
                                                match source_library.unwrap_or_default() {
                                                    "vcsl" => vcsl_root_observation,
                                                    "vsco2_ce" => vsco2_root_observation,
                                                    _ => return Err("unknown sample library".into()),
                                                },
                                                bank_directory,
                                            )
                                        });
                                    match observation {
                                        Ok(actual_digest) => {
                                            let sample_digest_matches =
                                                source_bank_digest == Some(actual_digest.as_str());
                                            let inventory_digest_matches =
                                                digest == actual_digest.as_str();
                                            if !sample_digest_matches {
                                                push_error(
                                                    &mut findings,
                                                    "selected_bank_actual_digest_mismatch",
                                                    format!(
                                                        "{case_label} selected bank manifest does not match independently hashed bank bytes"
                                                    ),
                                                );
                                            }
                                            if !inventory_digest_matches {
                                                push_error(
                                                    &mut findings,
                                                    "inventory_bank_actual_digest_mismatch",
                                                    format!(
                                                        "{case_label} inventory bank digest does not match independently hashed bank bytes"
                                                    ),
                                                );
                                            }
                                            if sample_digest_matches && inventory_digest_matches {
                                                selected_samples_with_matching_actual_bank_digest += 1;
                                            }
                                        }
                                        Err(error) => push_incomplete(
                                            &mut findings,
                                            "indexed_bank_recompute_failed",
                                            format!(
                                                "{case_label} selected bank manifest could not be recomputed from disk: {error}"
                                            ),
                                        ),
                                    }
                                } else {
                                    push_incomplete(
                                        &mut findings,
                                        "inventory_bank_path_missing",
                                        format!("{case_label} matching inventory bank has no relative_directory"),
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
                        }
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
        vcsl_root_content_manifest_sha256: vcsl_root_observation.content_manifest_sha256.clone(),
        vsco2_root_content_manifest_sha256: vsco2_root_observation.content_manifest_sha256.clone(),
        root_content_manifests_verified,
        source_terms_roots_verified: source_terms_audit.roots_verified,
        source_license_documents_verified: source_terms_audit.license_documents_verified,
        source_readme_documents_verified: source_terms_audit.readme_documents_verified,
        clean_upstream_checkouts_verified: source_terms_audit.clean_upstream_checkouts_verified,
        selected_samples_total,
        selected_samples_with_valid_file_digest,
        selected_samples_with_valid_bank_digest,
        selected_samples_with_matching_inventory_bank_digest,
        selected_samples_with_matching_actual_bank_digest,
        selected_samples_with_matching_actual_file_digest,
        cases_total,
        error_count,
        incomplete_count,
        findings,
        evidence_boundary: "provenance consistency only; this does not qualify instrument fidelity, validate licensing, authenticate the source revision, attest the build environment, or prove musical quality",
    }
}

fn verify_root_evidence(
    expected: Option<&Value>,
    label: &str,
    observed: &RootObservation,
    findings: &mut Vec<Finding>,
) -> bool {
    let Some(expected) = expected else {
        push_incomplete(
            findings,
            "root_evidence_missing",
            format!("{label} root evidence is missing from the inventory"),
        );
        return false;
    };

    let expected_present = expected.get("root_present").and_then(Value::as_bool);
    let Some(actual_root) = observed.canonical_path.as_ref() else {
        push_incomplete(
            findings,
            "asset_root_unreadable",
            format!(
                "{label} root could not be independently inspected: {}",
                observed.error.as_deref().unwrap_or("unknown root access error")
            ),
        );
        return false;
    };
    if expected_present == Some(false) {
        push_error(
            findings,
            "root_presence_mismatch",
            format!(
                "{label} inventory claims this root is absent, but supplied root {} exists",
                actual_root.display()
            ),
        );
        return false;
    }
    if expected_present != Some(true) {
        push_incomplete(
            findings,
            "root_presence_unknown",
            format!("{label} inventory has no valid root_present=true claim"),
        );
        return false;
    }

    let Some(actual_digest) = observed.content_manifest_sha256.as_deref() else {
        push_incomplete(
            findings,
            "root_content_manifest_unavailable",
            format!(
                "{label} supplied root could not be fully hashed: {}",
                observed.error.as_deref().unwrap_or("unknown manifest error")
            ),
        );
        return false;
    };
    if string_field(expected, "content_manifest_status") != Some("hashed") {
        push_incomplete(
            findings,
            "root_manifest_incomplete",
            format!("{label} inventory does not claim a completed root content manifest"),
        );
        return false;
    }
    let Some(expected_digest) = string_field(expected, "content_manifest_sha256") else {
        push_incomplete(
            findings,
            "root_manifest_digest_missing",
            format!("{label} inventory has no root content digest"),
        );
        return false;
    };
    if !valid_sha256(expected_digest) {
        push_error(
            findings,
            "root_manifest_digest_invalid",
            format!("{label} inventory has a malformed root content digest"),
        );
        return false;
    }
    if expected_digest != actual_digest {
        push_error(
            findings,
            "root_manifest_content_mismatch",
            format!("{label} supplied root bytes do not match the inventory root content manifest"),
        );
        return false;
    }
    match (
        expected.get("wav_file_count").and_then(Value::as_u64),
        observed.wav_file_count,
    ) {
        (Some(expected_count), Some(actual_count)) if expected_count == actual_count as u64 => {}
        (Some(expected_count), Some(actual_count)) => {
            push_error(
                findings,
                "root_wav_file_count_mismatch",
                format!(
                    "{label} inventory declares {expected_count} WAV files but supplied root contains {actual_count}"
                ),
            );
            return false;
        }
        _ => {
            push_incomplete(
                findings,
                "root_wav_file_count_missing",
                format!("{label} WAV file count is unavailable or malformed"),
            );
            return false;
        }
    }
    true
}

fn source_terms_policy(source_library: &str) -> Option<SourceTermsPolicy> {
    match source_library {
        "vcsl" => Some(SourceTermsPolicy {
            source_library: "vcsl",
            upstream_repository_url: "https://github.com/sgossner/VCSL",
            expected_remote: "sgossner/vcsl",
            declared_license_spdx: "CC0-1.0",
            readme_relative_path: "README.md",
            upstream_readme_request: "No attribution required according to the upstream README; preserve its license/readme bytes for review.",
        }),
        "vsco2_ce" => Some(SourceTermsPolicy {
            source_library: "vsco2_ce",
            upstream_repository_url: "https://github.com/sgossner/VSCO-2-CE",
            expected_remote: "sgossner/vsco-2-ce",
            declared_license_spdx: "CC0-1.0",
            readme_relative_path: "Readme.txt",
            upstream_readme_request: "Upstream README asks for courtesy/author credit; keep this request distinct from the declared SPDX license and review the source terms rather than silently discarding the request.",
        }),
        _ => None,
    }
}

fn verify_source_terms_evidence(
    inventory: &Value,
    vcsl_root: &RootObservation,
    vsco2_root: &RootObservation,
    findings: &mut Vec<Finding>,
) -> SourceTermsAudit {
    let mut audit = SourceTermsAudit::default();
    let Some(entries) = inventory.get("source_terms_evidence").and_then(Value::as_array) else {
        push_incomplete(
            findings,
            "source_terms_evidence_missing",
            "capability matrix v4 must contain source_terms_evidence".into(),
        );
        return audit;
    };
    if entries.len() != 2 {
        push_incomplete(
            findings,
            "source_terms_row_count_incomplete",
            format!("expected exactly two source-terms rows, found {}", entries.len()),
        );
    }

    for (source_library, label, root) in [
        ("vcsl", "VCSL", vcsl_root),
        ("vsco2_ce", "VSCO-2-CE", vsco2_root),
    ] {
        let matching: Vec<&Value> = entries
            .iter()
            .filter(|row| string_field(row, "source_library") == Some(source_library))
            .collect();
        match matching.as_slice() {
            [entry] => {
                let policy = source_terms_policy(source_library).unwrap();
                let declarations_match =
                    source_terms_declarations_match(entry, policy, label, findings);
                let license_matches = source_terms_document_matches(
                    entry,
                    "license_document_relative_path",
                    "license_document_sha256",
                    "license_document_status",
                    "LICENSE",
                    root,
                    label,
                    findings,
                );
                let readme_matches = source_terms_document_matches(
                    entry,
                    "readme_document_relative_path",
                    "readme_document_sha256",
                    "readme_document_status",
                    policy.readme_relative_path,
                    root,
                    label,
                    findings,
                );
                let checkout_matches =
                    source_terms_checkout_matches(entry, policy, label, root, findings);
                if license_matches {
                    audit.license_documents_verified += 1;
                }
                if readme_matches {
                    audit.readme_documents_verified += 1;
                }
                if checkout_matches {
                    audit.clean_upstream_checkouts_verified += 1;
                }
                if declarations_match && license_matches && readme_matches && checkout_matches {
                    audit.roots_verified += 1;
                }
            }
            [] => push_incomplete(
                findings,
                "source_terms_row_missing",
                format!("{label} source-terms evidence row is missing"),
            ),
            _ => push_error(
                findings,
                "source_terms_row_duplicate",
                format!("{label} has multiple source-terms rows"),
            ),
        }
    }

    for entry in entries {
        let source_library = string_field(entry, "source_library").unwrap_or("");
        if source_terms_policy(source_library).is_none() {
            push_error(
                findings,
                "source_terms_library_unknown",
                format!("source-terms row has an unknown source library {source_library:?}"),
            );
        }
    }
    audit
}

fn source_terms_declarations_match(
    entry: &Value,
    policy: SourceTermsPolicy,
    label: &str,
    findings: &mut Vec<Finding>,
) -> bool {
    let declarations_match =
        string_field(entry, "upstream_repository_url") == Some(policy.upstream_repository_url)
            && string_field(entry, "declared_license_spdx") == Some(policy.declared_license_spdx)
            && string_field(entry, "license_document_relative_path") == Some("LICENSE")
            && string_field(entry, "readme_document_relative_path") == Some(policy.readme_relative_path)
            && string_field(entry, "upstream_readme_request") == Some(policy.upstream_readme_request)
            && string_field(entry, "license_declaration_boundary")
                .is_some_and(|value| value.contains("not a legal opinion"));
    if !declarations_match {
        push_error(
            findings,
            "source_terms_declaration_mismatch",
            format!("{label} source-terms declaration differs from the verifier's explicit policy"),
        );
    }
    declarations_match
}

fn source_terms_document_matches(
    entry: &Value,
    path_field: &str,
    digest_field: &str,
    status_field: &str,
    expected_relative_path: &str,
    root: &RootObservation,
    label: &str,
    findings: &mut Vec<Finding>,
) -> bool {
    if string_field(entry, path_field) != Some(expected_relative_path) {
        push_error(
            findings,
            "source_terms_document_path_mismatch",
            format!("{label} {path_field} differs from the required relative path"),
        );
        return false;
    }
    if string_field(entry, status_field) != Some("hashed") {
        push_incomplete(
            findings,
            "source_terms_document_not_hashed",
            format!("{label} {expected_relative_path} is not marked hashed"),
        );
        return false;
    }
    let Some(expected_digest) = string_field(entry, digest_field) else {
        push_incomplete(
            findings,
            "source_terms_document_digest_missing",
            format!("{label} {expected_relative_path} has no content digest"),
        );
        return false;
    };
    if !valid_sha256(expected_digest) {
        push_error(
            findings,
            "source_terms_document_digest_invalid",
            format!("{label} {expected_relative_path} has a malformed SHA-256"),
        );
        return false;
    }
    let Some(root_path) = root.canonical_path.as_ref() else {
        push_incomplete(
            findings,
            "source_terms_root_unavailable",
            format!("{label} source root is unavailable for document verification"),
        );
        return false;
    };
    match hash_relative_source_document(root_path, expected_relative_path) {
        Ok(actual_digest) if actual_digest == expected_digest => true,
        Ok(_) => {
            push_error(
                findings,
                "source_terms_document_digest_mismatch",
                format!("{label} {expected_relative_path} bytes do not match the capability-matrix digest"),
            );
            false
        }
        Err(error) => {
            push_incomplete(
                findings,
                "source_terms_document_unavailable",
                format!("{label} {expected_relative_path} cannot be independently read: {error}"),
            );
            false
        }
    }
}

fn source_terms_checkout_matches(
    entry: &Value,
    policy: SourceTermsPolicy,
    label: &str,
    root: &RootObservation,
    findings: &mut Vec<Finding>,
) -> bool {
    let Some(root_path) = root.canonical_path.as_ref() else {
        push_incomplete(
            findings,
            "source_checkout_root_unavailable",
            format!("{label} source root is unavailable for Git lineage verification"),
        );
        return false;
    };
    let observed = inspect_upstream_checkout(root_path, policy.expected_remote);
    if string_field(entry, "source_revision_status") != Some(observed.status) {
        // A claimed clean checkout that no longer has the recorded state is an
        // inconsistency; other non-clean states remain incomplete evidence.
        if string_field(entry, "source_revision_status") == Some("clean_upstream_checkout") {
            push_error(
                findings,
                "source_checkout_status_mismatch",
                format!("{label} source checkout no longer matches its reported clean upstream status"),
            );
        } else {
            push_incomplete(
                findings,
                "source_checkout_not_clean_upstream",
                format!("{label} checkout status is {}", observed.status),
            );
        }
        return false;
    }
    if bool_field(entry, "upstream_origin_matches_expected") != Some(observed.origin_matches_expected) {
        push_error(
            findings,
            "source_checkout_origin_status_mismatch",
            format!("{label} reported origin-match status differs from the actual Git remote"),
        );
        return false;
    }
    if observed.origin_matches_expected != true {
        push_incomplete(
            findings,
            "source_checkout_origin_unverified",
            format!("{label} Git origin does not match the expected upstream repository"),
        );
        return false;
    }
    if bool_field(entry, "source_checkout_clean") != observed.clean {
        push_error(
            findings,
            "source_checkout_clean_status_mismatch",
            format!("{label} reported clean-worktree status differs from actual Git status"),
        );
        return false;
    }
    if observed.clean != Some(true) {
        push_incomplete(
            findings,
            "source_checkout_dirty_or_unknown",
            format!("{label} upstream checkout is dirty or its status could not be read"),
        );
        return false;
    }
    let Some(reported_revision) = string_field(entry, "upstream_revision") else {
        push_incomplete(
            findings,
            "source_checkout_revision_missing",
            format!("{label} has no upstream commit revision"),
        );
        return false;
    };
    if !valid_git_revision(reported_revision) {
        push_error(
            findings,
            "source_checkout_revision_invalid",
            format!("{label} upstream revision is not a valid Git object ID"),
        );
        return false;
    }
    if observed.upstream_revision.as_deref() != Some(reported_revision) {
        push_error(
            findings,
            "source_checkout_revision_mismatch",
            format!("{label} upstream commit differs from the revision recorded in the matrix"),
        );
        return false;
    }
    true
}

fn hash_relative_source_document(root: &Path, relative_path: &str) -> Result<String, String> {
    let relative = Path::new(relative_path);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative_path.contains('\\')
        || relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err("document path is not a normalized relative path".into());
    }
    let joined = root.join(relative);
    let canonical = std::fs::canonicalize(&joined).map_err(|error| error.to_string())?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err("document path escaped its source root or is not a regular file".into());
    }
    file_sha256(&canonical)
        .map(|(digest, _)| digest)
        .map_err(|error| error.to_string())
}

fn inspect_upstream_checkout(root: &Path, expected_remote: &str) -> GitCheckoutObservation {
    let canonical_root = match std::fs::canonicalize(root) {
        Ok(path) if path.is_dir() => path,
        _ => {
            return GitCheckoutObservation {
                upstream_revision: None,
                origin_matches_expected: false,
                clean: None,
                status: "root_unavailable",
            };
        }
    };
    let git = |arguments: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(arguments)
            .output()
    };
    let top_level = match git(&["rev-parse", "--show-toplevel"]) {
        Ok(output) if output.status.success() => {
            let output = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            match std::fs::canonicalize(output) {
                Ok(path) if path == canonical_root => path,
                Ok(_) => {
                    return GitCheckoutObservation {
                        upstream_revision: None,
                        origin_matches_expected: false,
                        clean: None,
                        status: "asset_root_is_not_repository_root",
                    };
                }
                Err(_) => {
                    return GitCheckoutObservation {
                        upstream_revision: None,
                        origin_matches_expected: false,
                        clean: None,
                        status: "git_root_unreadable",
                    };
                }
            }
        }
        Ok(_) => {
            return GitCheckoutObservation {
                upstream_revision: None,
                origin_matches_expected: false,
                clean: None,
                status: "not_a_git_checkout",
            };
        }
        Err(_) => {
            return GitCheckoutObservation {
                upstream_revision: None,
                origin_matches_expected: false,
                clean: None,
                status: "git_unavailable",
            };
        }
    };
    let _ = top_level;

    let upstream_revision = match git(&["rev-parse", "HEAD"]) {
        Ok(output) if output.status.success() => {
            let revision = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if valid_git_revision(&revision) {
                Some(revision)
            } else {
                return GitCheckoutObservation {
                    upstream_revision: None,
                    origin_matches_expected: false,
                    clean: None,
                    status: "invalid_git_revision",
                };
            }
        }
        _ => {
            return GitCheckoutObservation {
                upstream_revision: None,
                origin_matches_expected: false,
                clean: None,
                status: "git_revision_unavailable",
            };
        }
    };

    let remote = match git(&["remote", "get-url", "origin"]) {
        Ok(output) if output.status.success() => Some(String::from_utf8_lossy(&output.stdout).trim().to_owned()),
        _ => None,
    };
    let origin_matches_expected = remote
        .as_deref()
        .and_then(normalize_github_remote)
        .is_some_and(|remote| remote == expected_remote.to_ascii_lowercase());
    let clean = match git(&["status", "--porcelain", "--untracked-files=all"]) {
        Ok(output) if output.status.success() => Some(String::from_utf8_lossy(&output.stdout).trim().is_empty()),
        _ => None,
    };
    let status = match (origin_matches_expected, clean) {
        (true, Some(true)) => "clean_upstream_checkout",
        (false, _) => "upstream_origin_mismatch",
        (true, Some(false)) => "dirty_worktree",
        (true, None) => "git_status_unavailable",
    };
    GitCheckoutObservation {
        upstream_revision,
        origin_matches_expected,
        clean,
        status,
    }
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

fn verify_selected_sample_file(
    sample: &Value,
    source_library: &str,
    relative_path: &str,
    vcsl_root: &RootObservation,
    vsco2_root: &RootObservation,
    case_label: &str,
    findings: &mut Vec<Finding>,
) -> bool {
    let root = match source_library {
        "vcsl" => vcsl_root,
        "vsco2_ce" => vsco2_root,
        other => {
            push_error(
                findings,
                "source_library_unknown",
                format!("{case_label} uses unrecognized source library {other:?}"),
            );
            return false;
        }
    };
    let Some(root_path) = root.canonical_path.as_ref() else {
        push_incomplete(
            findings,
            "selected_sample_root_unavailable",
            format!(
                "{case_label} source root is unavailable: {}",
                root.error.as_deref().unwrap_or("unknown root access error")
            ),
        );
        return false;
    };

    let relative = Path::new(relative_path);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative_path.contains('\\')
        || relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        push_error(
            findings,
            "selected_sample_path_unsafe",
            format!("{case_label} selected source path is not a safe normalized relative path"),
        );
        return false;
    }
    if relative.extension().and_then(|extension| extension.to_str()) != Some("wav") {
        push_error(
            findings,
            "selected_sample_not_wav",
            format!("{case_label} selected source does not use the exact lowercase .wav extension"),
        );
        return false;
    }
    if !root
        .wav_relative_paths
        .as_ref()
        .is_some_and(|paths| paths.contains(relative_path))
    {
        push_error(
            findings,
            "selected_sample_not_in_root_manifest",
            format!("{case_label} selected source was not discovered by the supplied root manifest walker"),
        );
        return false;
    }
    let joined = root_path.join(relative);
    let resolved = match std::fs::canonicalize(&joined) {
        Ok(path) => path,
        Err(error) => {
            push_error(
                findings,
                "selected_sample_file_unavailable",
                format!("{case_label} selected source file cannot be opened under the supplied root: {error}"),
            );
            return false;
        }
    };
    if !resolved.starts_with(root_path) || !resolved.is_file() {
        push_error(
            findings,
            "selected_sample_path_escape",
            format!("{case_label} selected source resolves outside its supplied root or is not a regular file"),
        );
        return false;
    }
    let actual_digest = match file_sha256(&resolved) {
        Ok((digest, _)) => digest,
        Err(error) => {
            push_error(
                findings,
                "selected_sample_file_hash_failed",
                format!("{case_label} selected source bytes could not be hashed: {error}"),
            );
            return false;
        }
    };
    let Some(expected_digest) = string_field(sample, "source_file_sha256") else {
        push_incomplete(
            findings,
            "selected_sample_digest_missing",
            format!("{case_label} selected source has no claimed exact-file SHA-256"),
        );
        return false;
    };
    if !valid_sha256(expected_digest) {
        push_error(
            findings,
            "source_file_digest_invalid",
            format!("{case_label} selected source file digest is malformed"),
        );
        return false;
    }
    if expected_digest != actual_digest {
        push_error(
            findings,
            "selected_sample_file_digest_mismatch",
            format!("{case_label} selected source WAV bytes do not match source_file_sha256"),
        );
        return false;
    }
    true
}

fn observe_asset_root(path: &Path) -> RootObservation {
    let canonical_path = match std::fs::canonicalize(path) {
        Ok(path) if path.is_dir() => path,
        Ok(_) => {
            return RootObservation {
                canonical_path: None,
                content_manifest_sha256: None,
                wav_file_count: None,
                wav_relative_paths: None,
                error: Some("supplied asset root is not a directory".into()),
            };
        }
        Err(error) => {
            return RootObservation {
                canonical_path: None,
                content_manifest_sha256: None,
                wav_file_count: None,
                wav_relative_paths: None,
                error: Some(error.to_string()),
            };
        }
    };

    let mut entries = Vec::new();
    if let Err(error) = collect_wav_manifest(&canonical_path, &canonical_path, &mut entries) {
        return RootObservation {
            canonical_path: Some(canonical_path),
            content_manifest_sha256: None,
            wav_file_count: None,
            wav_relative_paths: None,
            error: Some(error.to_string()),
        };
    }
    match asset_root_content_manifest_sha256(&canonical_path, &entries) {
        Ok(digest) => RootObservation {
            canonical_path: Some(canonical_path),
            content_manifest_sha256: Some(digest),
            wav_file_count: Some(entries.len()),
            wav_relative_paths: Some(entries.iter().map(|(path, _)| path.clone()).collect()),
            error: None,
        },
        Err(error) => RootObservation {
            canonical_path: Some(canonical_path),
            content_manifest_sha256: None,
            wav_file_count: Some(entries.len()),
            wav_relative_paths: Some(entries.iter().map(|(path, _)| path.clone()).collect()),
            error: Some(error.to_string()),
        },
    }
}

/// Match the capability-inventory root walker exactly: sorted entries within each
/// directory, depth-first traversal, skipping .git/.svn and non-regular WAV links.
fn collect_wav_manifest(
    root: &Path,
    current: &Path,
    output: &mut Vec<(String, u64)>,
) -> io::Result<()> {
    let mut entries = std::fs::read_dir(current)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_unstable_by_key(|entry| entry.file_name());
    for entry in entries {
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        if file_type.is_dir() {
            if name == ".git" || name == ".svn" {
                continue;
            }
            collect_wav_manifest(root, &entry.path(), output)?;
        } else if file_type.is_file()
            && entry.path().extension().and_then(|extension| extension.to_str()) == Some("wav")
        {
            let path = entry.path();
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let relative_path = relative
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            output.push((relative_path, entry.metadata()?.len()));
        }
    }
    Ok(())
}

fn asset_root_content_manifest_sha256(
    root: &Path,
    entries: &[(String, u64)],
) -> io::Result<String> {
    let mut manifest = Sha256::new();
    manifest.update(b"muse-asset-root-content-manifest-v1\0");
    let mut buffer = [0_u8; 64 * 1024];
    for (relative_path, expected_size) in entries {
        manifest.update((relative_path.len() as u64).to_le_bytes());
        manifest.update(relative_path.as_bytes());
        manifest.update(expected_size.to_le_bytes());
        let path = root.join(relative_path);
        let mut file = File::open(&path)?;
        if file.metadata()?.len() != *expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("asset size changed before hashing: {relative_path}"),
            ));
        }
        let mut file_hasher = Sha256::new();
        let mut bytes_read = 0_u64;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            bytes_read = bytes_read
                .checked_add(count as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "asset length overflow"))?;
            file_hasher.update(&buffer[..count]);
        }
        if bytes_read != *expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("asset size changed while hashing: {relative_path}"),
            ));
        }
        manifest.update(&file_hasher.finalize()[..]);
    }
    Ok(format!("{:x}", manifest.finalize()))
}

fn indexed_bank_content_manifest_sha256(
    root: &RootObservation,
    relative_directory: &str,
) -> Result<String, String> {
    let root_path = root
        .canonical_path
        .as_ref()
        .ok_or_else(|| root.error.clone().unwrap_or_else(|| "asset root is unavailable".into()))?;
    let relative = Path::new(relative_directory);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative_directory.contains('\\')
        || relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err("inventory bank path is not a safe normalized relative directory".into());
    }
    let joined = root_path.join(relative);
    let canonical_directory =
        std::fs::canonicalize(&joined).map_err(|error| error.to_string())?;
    if !canonical_directory.starts_with(root_path) || !canonical_directory.is_dir() {
        return Err("inventory bank directory resolves outside the supplied root".into());
    }

    let entries = std::fs::read_dir(&canonical_directory)
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let mut indexed_paths = Vec::new();
    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("wav") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| "indexed WAV has a non-UTF-8 file stem".to_string())?;
        let contains_midi_token = stem.split('_').any(|token| parse_note_token(token).is_some());
        if contains_midi_token {
            indexed_paths.push(path);
        }
    }
    // InstrumentBank::index path-sorts accepted samples so the content manifest
    // is independent of the filesystem's read_dir order.
    indexed_paths.sort_unstable();
    if indexed_paths.is_empty() {
        return Err("bank contains no WAV filenames accepted by the instrument pitch indexer".into());
    }

    let mut manifest = Sha256::new();
    manifest.update(b"muse-indexed-sample-content-manifest-v1\0");
    let mut buffer = [0_u8; 64 * 1024];
    for path in indexed_paths {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let name_bytes = name.as_bytes();
        manifest.update((name_bytes.len() as u64).to_be_bytes());

        let mut file = File::open(&path).map_err(|error| error.to_string())?;
        let expected_size = file.metadata().map_err(|error| error.to_string())?.len();
        manifest.update(expected_size.to_be_bytes());
        let mut read_size = 0_u64;
        let mut file_hasher = Sha256::new();
        loop {
            let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            read_size = read_size
                .checked_add(count as u64)
                .ok_or_else(|| "indexed WAV length overflow while hashing".to_string())?;
            file_hasher.update(&buffer[..count]);
        }
        if read_size != expected_size {
            return Err(format!("indexed WAV changed size while hashing: {name}"));
        }
        manifest.update(&file_hasher.finalize()[..]);
    }
    Ok(format!("{:x}", manifest.finalize()))
}

fn parse_note_token(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    let letter = *bytes.first()?;
    if !(b'A'..=b'G').contains(&letter) {
        return None;
    }
    let mut semitones: i32 = match letter {
        b'C' => 0,
        b'D' => 2,
        b'E' => 4,
        b'F' => 5,
        b'G' => 7,
        b'A' => 9,
        b'B' => 11,
        _ => return None,
    };
    let mut index = 1;
    match bytes.get(index) {
        Some(b'#') => {
            semitones += 1;
            index += 1;
        }
        Some(b'b') => {
            semitones -= 1;
            index += 1;
        }
        _ => {}
    }
    let octave: i32 = token.get(index..)?.parse().ok()?;
    let midi = (octave + 1) * 12 + semitones;
    u8::try_from(midi).ok().filter(|&note| note <= 127)
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

    struct TestAssets {
        base: PathBuf,
        vcsl_root: PathBuf,
        vsco2_root: PathBuf,
        sample_wav: PathBuf,
    }

    impl Drop for TestAssets {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    fn fixtures() -> (Value, Value, TestAssets) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "muse-provenance-assets-{}-{nonce}",
            std::process::id()
        ));
        let vcsl_root = base.join("vcsl");
        let vsco2_root = base.join("vsco2_ce");
        let sample_wav = vcsl_root.join("Piano").join("sample_C4.wav");
        std::fs::create_dir_all(sample_wav.parent().expect("sample parent"))
            .expect("VCSL fixture should be created");
        std::fs::create_dir_all(&vsco2_root).expect("VSCO2 fixture should be created");
        std::fs::write(&sample_wav, b"sample-WAV-content-for-provenance-test")
            .expect("source sample fixture should be written");
        std::fs::write(vcsl_root.join("LICENSE"), b"CC0 1.0 Universal test license\n")
            .expect("VCSL license fixture should be written");
        std::fs::write(
            vcsl_root.join("README.md"),
            b"VCSL fixture README: CC0; no attribution required.\n",
        )
        .expect("VCSL README fixture should be written");
        std::fs::write(vsco2_root.join("LICENSE"), b"CC0 1.0 Universal test license\n")
            .expect("VSCO-2-CE license fixture should be written");
        std::fs::write(
            vsco2_root.join("Readme.txt"),
            b"VSCO-2-CE fixture README: use allowed; credit requested.\n",
        )
        .expect("VSCO-2-CE README fixture should be written");
        initialize_git_fixture(&vcsl_root, "https://github.com/sgossner/VCSL.git");
        initialize_git_fixture(&vsco2_root, "https://github.com/sgossner/VSCO-2-CE.git");

        let source_file_sha256 = file_sha256(&sample_wav)
            .expect("source sample fixture should hash")
            .0;
        let vcsl_observation = observe_asset_root(&vcsl_root);
        let vsco2_observation = observe_asset_root(&vsco2_root);
        let vcsl_digest = vcsl_observation
            .content_manifest_sha256
            .clone()
            .expect("VCSL root should hash");
        let vsco2_digest = vsco2_observation
            .content_manifest_sha256
            .clone()
            .expect("VSCO2 root should hash");
        let revision = "a".repeat(40);
        let bank_digest = indexed_bank_content_manifest_sha256(&vcsl_observation, "Piano")
            .expect("indexed fixture bank should hash");
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
                    "source_file_sha256": source_file_sha256,
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
            "schema_version": "melothaea.instrument-capability-matrix.v4",
            "source_revision": revision,
            "source_terms_evidence": [
                fixture_source_terms(&vcsl_root, "vcsl"),
                fixture_source_terms(&vsco2_root, "vsco2_ce")
            ],
            "indexed_sample_content_hashing_enabled": true,
            "root_content_manifest_hashing_enabled": true,
            "vcsl_root": {
                "root_present": true,
                "wav_file_count": vcsl_observation.wav_file_count.unwrap(),
                "content_manifest_status": "hashed",
                "content_manifest_sha256": vcsl_digest
            },
            "vsco2_root": {
                "root_present": true,
                "wav_file_count": vsco2_observation.wav_file_count.unwrap(),
                "content_manifest_status": "hashed",
                "content_manifest_sha256": vsco2_digest
            },
            "instruments": [{
                "instrument": "Piano",
                "sustained_samples": bank,
                "short_articulation_samples": null,
                "additional_sample_styles": []
            }]
        });
        (
            probe,
            inventory,
            TestAssets { base, vcsl_root, vsco2_root, sample_wav },
        )
    }

    fn initialize_git_fixture(root: &Path, origin: &str) {
        let run = |arguments: &[&str]| {
            let output = Command::new("git")
                .arg("-C")
                .arg(root)
                .args(arguments)
                .output()
                .expect("git must be available for upstream provenance fixtures");
            assert!(
                output.status.success(),
                "git {:?} failed: {}",
                arguments,
                String::from_utf8_lossy(&output.stderr)
            );
        };
        run(&["init", "-q"]);
        run(&["config", "user.name", "Melothaea verifier fixture"]);
        run(&["config", "user.email", "muse-provenance-fixture@example.invalid"]);
        run(&["remote", "add", "origin", origin]);
        run(&["add", "--all"]);
        run(&["commit", "-q", "-m", "fixture source terms"]);
    }

    fn fixture_source_terms(root: &Path, source_library: &str) -> Value {
        let policy = source_terms_policy(source_library).expect("known fixture source");
        let canonical = std::fs::canonicalize(root).expect("source root should canonicalize");
        let license_digest =
            hash_relative_source_document(&canonical, "LICENSE").expect("license should hash");
        let readme_digest = hash_relative_source_document(&canonical, policy.readme_relative_path)
            .expect("README should hash");
        let checkout = inspect_upstream_checkout(&canonical, policy.expected_remote);
        json!({
            "source_library": policy.source_library,
            "upstream_repository_url": policy.upstream_repository_url,
            "declared_license_spdx": policy.declared_license_spdx,
            "license_declaration_boundary": "Declared from the upstream repository license metadata; this field is not a legal opinion and does not replace inspection of the hashed license/readme bytes.",
            "license_document_relative_path": "LICENSE",
            "license_document_sha256": license_digest,
            "license_document_status": "hashed",
            "readme_document_relative_path": policy.readme_relative_path,
            "readme_document_sha256": readme_digest,
            "readme_document_status": "hashed",
            "upstream_readme_request": policy.upstream_readme_request,
            "upstream_revision": checkout.upstream_revision,
            "upstream_origin_matches_expected": checkout.origin_matches_expected,
            "source_checkout_clean": checkout.clean,
            "source_revision_status": checkout.status
        })
    }

    fn test_report(
        probe: &Value,
        inventory: &Value,
        assets: &TestAssets,
    ) -> VerificationReport {
        let vcsl_observation = observe_asset_root(&assets.vcsl_root);
        let vsco2_observation = observe_asset_root(&assets.vsco2_root);
        verify_reports(
            probe,
            inventory,
            "0".repeat(64),
            "1".repeat(64),
            "d".repeat(64),
            4096,
            "muse_instrument_probe",
            &vcsl_observation,
            &vsco2_observation,
        )
    }

    #[test]
    fn verifies_matching_revision_executable_roots_and_source_bank_digest() {
        let (probe, inventory, assets) = fixtures();
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "consistent");
        assert_eq!(report.root_content_manifests_verified, 2);
        assert_eq!(report.source_terms_roots_verified, 2);
        assert_eq!(report.source_license_documents_verified, 2);
        assert_eq!(report.source_readme_documents_verified, 2);
        assert_eq!(report.clean_upstream_checkouts_verified, 2);
        assert_eq!(report.selected_samples_total, 1);
        assert_eq!(report.selected_samples_with_matching_actual_file_digest, 1);
        assert_eq!(report.selected_samples_with_valid_file_digest, 1);
        assert_eq!(report.selected_samples_with_valid_bank_digest, 1);
        assert_eq!(report.selected_samples_with_matching_inventory_bank_digest, 1);
        assert_eq!(report.selected_samples_with_matching_actual_bank_digest, 1);
        assert_eq!(report.error_count, 0);
        assert_eq!(report.incomplete_count, 0);
    }

    #[test]
    fn modified_license_document_is_rejected_even_when_wav_assets_are_unchanged() {
        let (probe, inventory, assets) = fixtures();
        std::fs::write(assets.vcsl_root.join("LICENSE"), b"different license bytes\\n")
            .expect("modified license should be written");
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "source_terms_document_digest_mismatch"));
        assert!(report.findings.iter().any(|finding| finding.code == "source_checkout_status_mismatch"));
    }

    #[test]
    fn changed_upstream_origin_is_not_accepted_as_clean_source_lineage() {
        let (probe, inventory, assets) = fixtures();
        let output = Command::new("git")
            .arg("-C")
            .arg(&assets.vcsl_root)
            .args(["remote", "set-url", "origin", "https://example.com/sgossner/VCSL.git"])
            .output()
            .expect("git must be available");
        assert!(output.status.success());
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "source_checkout_status_mismatch"));
        assert!(report.findings.iter().any(|finding| finding.code == "source_checkout_origin_status_mismatch"));
    }

    #[test]
    fn mismatched_supplied_binary_is_rejected_even_when_report_sha_is_well_formed() {
        let (mut probe, inventory, assets) = fixtures();
        probe["renderer_binary_sha256"] = serde_json::json!("9".repeat(64));
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(!report.renderer_binary_identity_verified);
        assert!(report.findings.iter().any(|finding| finding.code == "renderer_binary_digest_mismatch"));
    }

    #[test]
    fn same_schema_with_a_conflicting_bank_digest_is_rejected() {
        let (probe, mut inventory, assets) = fixtures();
        inventory["instruments"][0]["sustained_samples"]
            ["indexed_sample_content_manifest_sha256"] = json!("9".repeat(64));
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "source_bank_manifest_mismatch"));
    }

    #[test]
    fn same_forged_bank_digest_in_both_reports_is_rejected_against_disk() {
        let (mut probe, mut inventory, assets) = fixtures();
        let forged_digest = json!("9".repeat(64));
        probe["cases"][0]["selected_sample"]["source_bank_content_manifest_sha256"] =
            forged_digest.clone();
        inventory["instruments"][0]["sustained_samples"]
            ["indexed_sample_content_manifest_sha256"] = forged_digest;
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "selected_bank_actual_digest_mismatch"));
        assert!(report.findings.iter().any(|finding| finding.code == "inventory_bank_actual_digest_mismatch"));
    }

    #[test]
    fn different_source_revisions_are_rejected() {
        let (probe, mut inventory, assets) = fixtures();
        inventory["source_revision"] = json!("1".repeat(40));
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "source_revision_mismatch"));
    }

    #[test]
    fn unavailable_source_revision_status_is_incomplete_not_mislabeled_as_mismatch() {
        let (mut probe, inventory, assets) = fixtures();
        probe["source_revision_status"] = json!("unavailable");
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "incomplete");
        assert!(report.findings.iter().any(|finding| finding.code == "source_revision_status_untrusted"));
        assert!(!report.findings.iter().any(|finding| finding.code == "source_revision_mismatch"));
    }

    #[test]
    fn missing_root_manifest_is_incomplete_not_a_pass() {
        let (probe, mut inventory, assets) = fixtures();
        inventory["vcsl_root"]["content_manifest_status"] = json!("not_requested");
        inventory["vcsl_root"]["content_manifest_sha256"] = Value::Null;
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "incomplete");
        assert!(report.findings.iter().any(|finding| finding.code == "root_manifest_incomplete"));
    }

    #[test]
    fn tampered_source_bytes_are_rejected_by_root_and_selected_file_digests() {
        let (probe, inventory, assets) = fixtures();
        std::fs::write(&assets.sample_wav, b"tampered-WAV-content-for-provenance-test")
            .expect("tampered source should be written");
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "root_manifest_content_mismatch"));
        assert!(report.findings.iter().any(|finding| finding.code == "selected_sample_file_digest_mismatch"));
        assert!(report.findings.iter().any(|finding| finding.code == "selected_bank_actual_digest_mismatch"));
        assert!(report.findings.iter().any(|finding| finding.code == "inventory_bank_actual_digest_mismatch"));
    }

    #[test]
    fn path_traversal_in_a_selected_sample_receipt_is_rejected() {
        let (mut probe, inventory, assets) = fixtures();
        probe["cases"][0]["selected_sample"]["source_relative_path"] = json!("../outside.wav");
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "inconsistent");
        assert!(report.findings.iter().any(|finding| finding.code == "selected_sample_path_unsafe"));
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
    fn indexed_bank_manifest_matches_the_source_binary_encoding() {
        let (probe, _inventory, assets) = fixtures();
        let observed = observe_asset_root(&assets.vcsl_root);
        let digest = indexed_bank_content_manifest_sha256(&observed, "Piano")
            .expect("fixture bank should hash");
        assert_eq!(
            digest,
            "1dca576377e3860838fcd8a9b39e50081431bbde531d62d914097fa468dc3c8d"
        );
        assert_eq!(
            probe["cases"][0]["selected_sample"]["source_bank_content_manifest_sha256"],
            digest
        );
    }

    #[test]
    fn malformed_sha256_is_rejected_by_the_digest_validator() {
        assert!(valid_sha256(&"a".repeat(64)));
        assert!(!valid_sha256(&"g".repeat(64)));
        assert!(!valid_sha256(&"a".repeat(63)));
    }

    #[test]
    fn style_bank_receipts_can_be_matched_to_the_inventory() {
        let (probe, mut inventory, assets) = fixtures();
        let bank_digest =
            probe["cases"][0]["selected_sample"]["source_bank_content_manifest_sha256"].clone();
        inventory["instruments"][0]["sustained_samples"] = Value::Null;
        inventory["instruments"][0]["additional_sample_styles"] = json!([{
            "style_name": "normal",
            "style_kind": "sustained",
            "evidence": {
                "source_library": "vcsl",
                "relative_directory": "Piano",
                "indexed_sample_content_manifest_sha256": bank_digest
            }
        }]);
        let report = test_report(&probe, &inventory, &assets);
        assert_eq!(report.verdict, "consistent");
        assert_eq!(report.selected_samples_with_matching_inventory_bank_digest, 1);
    }
}
