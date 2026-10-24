// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Read-only artifact integrity checks for exported MADE benchmark runs.
//!
//! This verifier checks consistency across exported trajectories, episode/CSV/aggregate
//! metrics, a declared source/task/config manifest, file digests, and a per-attempt oracle
//! ledger. A self-declared manifest or ledger is not external attestation of source identity,
//! legal rights, billed costs, scientific validity, or experimental confirmation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const ABS_TOLERANCE: f64 = 1.0e-10;
const REL_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, Serialize)]
pub struct AuditFinding {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub status: String,
    pub run_dir: String,
    pub trajectory_files: usize,
    pub episode_rows: usize,
    pub findings: Vec<AuditFinding>,
    pub checks_completed: Vec<String>,
    /// "not_provided", "manifest_consistent", or "manifest_invalid".
    pub provenance_manifest_status: String,
    pub scope_ceiling: Vec<String>,
}

impl AuditReport {
    fn new(run_dir: &Path) -> Self {
        Self {
            status: "fail".to_owned(),
            run_dir: run_dir.display().to_string(),
            trajectory_files: 0,
            episode_rows: 0,
            findings: Vec::new(),
            checks_completed: Vec::new(),
            provenance_manifest_status: "not_provided".to_owned(),
            scope_ceiling: vec![
                "An artifact-integrity pass is not a scientific-performance pass.".to_owned(),
                "Source commit, data rights and true upstream dataset identity are not externally attested; declarations and local hashes are checked only.".to_owned(),
                "Oracle calls, caching, retries and costs reconcile to a self-declared ledger, not source-system logs or billing evidence.".to_owned(),
                "No physical experiment or independent replication is verified.".to_owned(),
            ],
        }
    }

    fn fail(&mut self, code: impl Into<String>, detail: impl Into<String>) {
        self.findings.push(AuditFinding {
            code: code.into(),
            detail: detail.into(),
        });
    }

    fn checked(&mut self, check: impl Into<String>) {
        self.checks_completed.push(check.into());
    }
}

/// Audits a standard exported MADE run directory.
pub fn audit_run(run_dir: &Path) -> AuditReport {
    let mut report = AuditReport::new(run_dir);
    let config_path = run_dir.join(".hydra").join("config.yaml");
    match fs::read_to_string(&config_path) {
        Ok(config) if !config.trim().is_empty() => {
            report.checked("resolved_config_present_and_nonempty");
        }
        Ok(_) => report.fail(
            "empty_config",
            format!("resolved config is empty: {}", config_path.display()),
        ),
        Err(error) => report.fail(
            "missing_or_unreadable_config",
            format!("{}: {error}", config_path.display()),
        ),
    }

    let episodes_path = run_dir.join("summary").join("episodes.json");
    let aggregate_path = run_dir.join("summary").join("summary.json");
    let csv_path = run_dir.join("summary").join("episodes.csv");
    let episodes_value = read_json(&episodes_path, &mut report);
    let aggregate_value = read_json(&aggregate_path, &mut report);
    let episode_rows = episodes_value.as_ref().and_then(Value::as_array).cloned();
    let aggregate = aggregate_value.as_ref().and_then(Value::as_object);

    if episodes_value.is_some() && episode_rows.is_none() {
        report.fail(
            "episodes_json_not_array",
            format!("{} must contain a JSON array", episodes_path.display()),
        );
    }
    if aggregate_value.is_some() && aggregate.is_none() {
        report.fail(
            "summary_json_not_object",
            format!("{} must contain a JSON object", aggregate_path.display()),
        );
    }

    let mut trajectory_files = match discover_trajectories(&run_dir.join("trajectories")) {
        Ok(files) => files,
        Err(error) => {
            report.fail("trajectory_directory_unreadable", error);
            Vec::new()
        }
    };
    trajectory_files.sort_by_key(|(index, _)| *index);
    report.trajectory_files = trajectory_files.len();
    report.episode_rows = episode_rows.as_ref().map_or(0, Vec::len);
    if episode_rows.as_ref().is_some_and(|rows| rows.is_empty()) {
        report.fail("no_episode_rows", "an audit run must contain at least one completed episode");
    }

    if let Some(rows) = episode_rows.as_ref() {
        if let Some(aggregate) = aggregate {
            match aggregate.get("episodes").and_then(Value::as_u64) {
                Some(count) if count as usize == rows.len() => {
                    report.checked("aggregate_episode_count_matches_rows");
                }
                Some(count) => report.fail(
                    "aggregate_episode_count_mismatch",
                    format!(
                        "summary.json declares {count} episodes, episodes.json has {}",
                        rows.len()
                    ),
                ),
                None => report.fail(
                    "aggregate_episode_count_missing",
                    "summary.json must contain a non-negative integer field named episodes",
                ),
            }
        }

        let trajectory_findings_before = report.findings.len();
        if trajectory_files.len() == rows.len() {
            report.checked("trajectory_count_matches_episode_rows");
        } else {
            report.fail(
                "trajectory_count_mismatch",
                format!(
                    "found {} episode trajectory files but {} rows in episodes.json",
                    trajectory_files.len(),
                    rows.len()
                ),
            );
        }

        for (expected_index, (actual_index, _)) in trajectory_files.iter().enumerate() {
            if expected_index != *actual_index {
                report.fail(
                    "trajectory_sequence_gap",
                    format!(
                        "expected episode index {expected_index} but found {actual_index}; indices must start at zero and be contiguous"
                    ),
                );
                break;
            }
        }

        for (episode_index, ((file_index, path), row)) in
            trajectory_files.iter().zip(rows.iter()).enumerate()
        {
            if *file_index != episode_index {
                continue;
            }
            let Some(trajectory_doc) = read_json(path, &mut report) else {
                continue;
            };
            let Some(metrics) = trajectory_doc.get("metrics").and_then(Value::as_object) else {
                report.fail(
                    "trajectory_metrics_missing",
                    format!("{} has no metrics object", path.display()),
                );
                continue;
            };

            if !metrics
                .get("episode_id")
                .and_then(Value::as_u64)
                .is_some_and(|id| id == episode_index as u64)
            {
                report.fail(
                    "trajectory_episode_id_mismatch",
                    format!(
                        "{} metrics.episode_id does not equal {episode_index}",
                        path.display()
                    ),
                );
            }
            if !values_equal(&Value::Object(metrics.clone()), row) {
                report.fail(
                    "episode_metrics_mismatch",
                    format!(
                        "{} metrics object differs from its corresponding summary/episodes.json row",
                        path.display()
                    ),
                );
            }

            let Some(raw_trajectory) = trajectory_doc.get("trajectory").and_then(Value::as_array) else {
                report.fail(
                    "raw_trajectory_missing",
                    format!("{} has no trajectory array", path.display()),
                );
                continue;
            };
            match metrics.get("final/queries_used").and_then(Value::as_u64) {
                Some(query_count) if query_count as usize == raw_trajectory.len() => {}
                Some(query_count) => report.fail(
                    "query_count_mismatch",
                    format!(
                        "{} declares {query_count} queries but contains {} trajectory observations",
                        path.display(),
                        raw_trajectory.len()
                    ),
                ),
                None => report.fail(
                    "query_count_missing",
                    format!(
                        "{} has no non-negative integer final/queries_used metric",
                        path.display()
                    ),
                ),
            }

            if trajectory_doc
                .get("final_env_state")
                .and_then(Value::as_object)
                .and_then(|state| state.get("phase_diagram_all_entries"))
                .and_then(Value::as_array)
                .is_none()
            {
                report.fail(
                    "final_phase_diagram_missing",
                    format!(
                        "{} lacks final_env_state.phase_diagram_all_entries",
                        path.display()
                    ),
                );
            }
            if trajectory_doc.get("phase_diagram_gt").and_then(Value::as_array).is_none() {
                report.fail(
                    "ground_truth_phase_diagram_missing",
                    format!("{} lacks phase_diagram_gt array", path.display()),
                );
            }
        }

        if !rows.is_empty()
            && trajectory_files.len() == rows.len()
            && report.findings.len() == trajectory_findings_before
        {
            report.checked("trajectory_and_per_episode_records_reconcile");
        }

        audit_csv(&csv_path, rows, &mut report);
        audit_aggregate(rows, aggregate_value.as_ref(), &mut report);
    }

    let proposal_requests = episode_rows.as_ref().and_then(sum_proposal_requests);
    verify_provenance_manifest(run_dir, proposal_requests, &mut report);

    report.status = if report.findings.is_empty() {
        "artifact_integrity_pass".to_owned()
    } else {
        "fail".to_owned()
    };
    report
}

const PROVENANCE_MANIFEST_NAME: &str = "provenance-manifest.json";
const PROVENANCE_SCHEMA_VERSION: &str = "made-run-provenance-v1";
const MAX_PROVENANCE_MANIFEST_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProvenanceManifest {
    schema_version: String,
    run_id: String,
    benchmark_id: String,
    upstream_repository: String,
    upstream_commit: String,
    policy_revision: String,
    task_contract_sha256: String,
    evaluation_manifest_sha256: String,
    source_manifest_sha256: String,
    resolved_config_sha256: String,
    oracle_call_ledger_path: String,
    oracle_call_ledger_sha256: String,
    #[serde(default)]
    shift_manifest_path: Option<String>,
    #[serde(default)]
    shift_manifest_sha256: Option<String>,
    data_sources: Vec<DataSourceDeclaration>,
    oracle_accounting: OracleAccountingDeclaration,
    artifacts: Vec<ArtifactDeclaration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DataSourceDeclaration {
    source_id: String,
    source_url: String,
    revision: String,
    license: String,
    rights_status: String,
    source_artifact_path: String,
    source_artifact_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleAccountingDeclaration {
    setup_requests: usize,
    proposal_requests: usize,
    retry_requests: usize,
    cache_hits: usize,
    local_rejections: usize,
    external_calls: usize,
    failed_external_calls: usize,
    total_cost: f64,
    cost_unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactDeclaration {
    path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleCallRecord {
    sequence: usize,
    stage: String,
    status: String,
    candidate_id: Option<String>,
    receipt_path: String,
    receipt_sha256: String,
    cost: f64,
    cache_hit: bool,
    external_invocation: bool,
}

const SHIFT_MANIFEST_SCHEMA_VERSION: &str = "nano-shift-manifest-v1";
const MAX_SHIFT_MANIFEST_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SHIFT_GROUPS: usize = 10_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShiftManifest {
    schema_version: String,
    benchmark_id: String,
    task_contract_sha256: String,
    split_policy: String,
    shift_groups: Vec<ShiftGroupDeclaration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShiftGroupDeclaration {
    group_id: String,
    /// Physical perturbation families present in this group; separate from mechanism axes.
    shift_families: Vec<String>,
    /// Statistical/physical mechanism axes can differ independently of physical families.
    shift_axes: Vec<String>,
    shift_kind: String,
    severity: String,
    split: String,
    physical_rationale: String,
    severity_definition: String,
    source_artifact_path: String,
    source_artifact_sha256: String,
    independence_keys: Vec<String>,
    sample_ids: Vec<String>,
}

const MAX_ORACLE_CALL_LEDGER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ORACLE_CALL_RECORDS: usize = 500_000;

fn sum_proposal_requests(rows: &[Value]) -> Option<usize> {
    rows.iter().try_fold(0usize, |total, row| {
        let count = row
            .as_object()?
            .get("final/queries_used")?
            .as_u64()?;
        total.checked_add(usize::try_from(count).ok()?)
    })
}

fn verify_provenance_manifest(
    run_dir: &Path,
    expected_proposal_requests: Option<usize>,
    report: &mut AuditReport,
) {
    let manifest_path = run_dir.join(PROVENANCE_MANIFEST_NAME);
    let manifest_metadata = match fs::symlink_metadata(&manifest_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            report.provenance_manifest_status = "not_provided".to_owned();
            return;
        }
        Err(error) => {
            report.provenance_manifest_status = "manifest_invalid".to_owned();
            report.fail(
                "provenance_manifest_unreadable",
                format!("{}: {error}", manifest_path.display()),
            );
            return;
        }
    };
    if manifest_metadata.file_type().is_symlink() || !manifest_metadata.is_file() {
        report.provenance_manifest_status = "manifest_invalid".to_owned();
        report.fail(
            "provenance_manifest_not_regular_file",
            "provenance-manifest.json must be a regular file, not a symlink",
        );
        return;
    }
    if manifest_metadata.len() > MAX_PROVENANCE_MANIFEST_BYTES {
        report.provenance_manifest_status = "manifest_invalid".to_owned();
        report.fail(
            "provenance_manifest_too_large",
            format!("provenance manifest exceeds {MAX_PROVENANCE_MANIFEST_BYTES} bytes"),
        );
        return;
    }

    let raw = match fs::read(&manifest_path) {
        Ok(raw) => raw,
        Err(error) => {
            report.provenance_manifest_status = "manifest_invalid".to_owned();
            report.fail(
                "provenance_manifest_unreadable",
                format!("{}: {error}", manifest_path.display()),
            );
            return;
        }
    };
    let manifest: ProvenanceManifest = match serde_json::from_slice(&raw) {
        Ok(manifest) => manifest,
        Err(error) => {
            report.provenance_manifest_status = "manifest_invalid".to_owned();
            report.fail(
                "provenance_manifest_invalid_json_or_schema",
                format!("{}: {error}", manifest_path.display()),
            );
            return;
        }
    };

    let findings_before = report.findings.len();
    if manifest.schema_version != PROVENANCE_SCHEMA_VERSION {
        report.fail(
            "provenance_schema_version_mismatch",
            format!("expected {PROVENANCE_SCHEMA_VERSION}, found {}", manifest.schema_version),
        );
    }
    for (field, value) in [
        ("run_id", manifest.run_id.as_str()),
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("upstream_repository", manifest.upstream_repository.as_str()),
        ("upstream_commit", manifest.upstream_commit.as_str()),
        ("policy_revision", manifest.policy_revision.as_str()),
    ] {
        if value.trim().is_empty() {
            report.fail("provenance_required_field_empty", format!("{field} must not be empty"));
        }
    }
    for (field, value) in [
        ("upstream_commit", manifest.upstream_commit.as_str()),
        ("policy_revision", manifest.policy_revision.as_str()),
    ] {
        if !is_lower_hex(value, 40) {
            report.fail(
                "provenance_git_revision_invalid",
                format!("{field} must be a 40-character lowercase hexadecimal Git revision"),
            );
        }
    }
    for (field, value) in [
        ("task_contract_sha256", manifest.task_contract_sha256.as_str()),
        ("evaluation_manifest_sha256", manifest.evaluation_manifest_sha256.as_str()),
        ("source_manifest_sha256", manifest.source_manifest_sha256.as_str()),
        ("resolved_config_sha256", manifest.resolved_config_sha256.as_str()),
        ("oracle_call_ledger_sha256", manifest.oracle_call_ledger_sha256.as_str()),
    ] {
        if !is_lower_hex(value, 64) {
            report.fail(
                "provenance_digest_invalid",
                format!("{field} must be a 64-character lowercase SHA-256 digest"),
            );
        }
    }
    if manifest.oracle_call_ledger_path.trim().is_empty() {
        report.fail(
            "oracle_call_ledger_path_empty",
            "oracle_call_ledger_path must not be empty",
        );
    }
    if !manifest.upstream_repository.starts_with("https://") {
        report.fail(
            "provenance_repository_url_invalid",
            "upstream_repository must be an HTTPS URL",
        );
    }
    if manifest.data_sources.is_empty() {
        report.fail("provenance_sources_missing", "at least one data/source declaration is required");
    }
    for (index, source) in manifest.data_sources.iter().enumerate() {
        for (field, value) in [
            ("source_id", source.source_id.as_str()),
            ("source_url", source.source_url.as_str()),
            ("revision", source.revision.as_str()),
            ("license", source.license.as_str()),
            ("rights_status", source.rights_status.as_str()),
            ("source_artifact_path", source.source_artifact_path.as_str()),
            ("source_artifact_sha256", source.source_artifact_sha256.as_str()),
        ] {
            if value.trim().is_empty() {
                report.fail(
                    "provenance_source_field_empty",
                    format!("data_sources[{index}].{field} must not be empty"),
                );
            }
        }
        if !source.source_url.starts_with("https://") {
            report.fail(
                "provenance_source_url_invalid",
                format!("data_sources[{index}].source_url must be an HTTPS URL"),
            );
        }
        if !is_lower_hex(&source.source_artifact_sha256, 64) {
            report.fail(
                "provenance_source_digest_invalid",
                format!("data_sources[{index}].source_artifact_sha256 must be a 64-character lowercase SHA-256 digest"),
            );
        }
        if !matches!(
            source.rights_status.as_str(),
            "verified" | "unknown" | "restricted" | "not_applicable"
        ) {
            report.fail(
                "provenance_rights_status_invalid",
                format!(
                    "data_sources[{index}].rights_status must be verified, unknown, restricted, or not_applicable"
                ),
            );
        }
    }

    let accounting = &manifest.oracle_accounting;
    if accounting.cost_unit.trim().is_empty()
        || !accounting.total_cost.is_finite()
        || accounting.total_cost < 0.0
    {
        report.fail(
            "oracle_accounting_invalid_cost",
            "cost_unit must be non-empty and total_cost must be finite and non-negative",
        );
    }
    let request_total = accounting
        .setup_requests
        .checked_add(accounting.proposal_requests)
        .and_then(|value| value.checked_add(accounting.retry_requests));
    let classified_total = accounting
        .external_calls
        .checked_add(accounting.cache_hits)
        .and_then(|value| value.checked_add(accounting.local_rejections));
    match (request_total, classified_total) {
        (Some(requests), Some(classified)) if requests == classified => {
            report.checked("oracle_accounting_requests_partitioned_into_external_cache_or_local_rejection");
        }
        (Some(requests), Some(classified)) => report.fail(
            "oracle_accounting_request_partition_mismatch",
            format!(
                "setup + proposal + retry requests totals {requests}, but external + cache-hit + local-rejection outcomes total {classified}"
            ),
        ),
        _ => report.fail(
            "oracle_accounting_count_overflow",
            "oracle request or classification counters overflow usize",
        ),
    }
    if accounting.local_rejections > accounting.proposal_requests {
        report.fail(
            "oracle_accounting_local_rejection_mismatch",
            "local_rejections cannot exceed proposal_requests",
        );
    }
    if accounting.failed_external_calls > accounting.external_calls {
        report.fail(
            "oracle_accounting_failed_call_mismatch",
            "failed_external_calls cannot exceed external_calls",
        );
    }
    match expected_proposal_requests {
        Some(expected) if accounting.proposal_requests == expected => {
            report.checked("provenance_proposal_requests_match_episode_queries");
        }
        Some(expected) => report.fail(
            "oracle_accounting_proposal_count_mismatch",
            format!(
                "manifest declares {} proposal requests but episode metrics sum to {expected}",
                accounting.proposal_requests
            ),
        ),
        None => report.fail(
            "oracle_accounting_proposal_count_unavailable",
            "cannot reconcile proposal requests because episode query counts are missing or overflow",
        ),
    }

    if manifest.artifacts.is_empty() {
        report.fail("provenance_artifacts_missing", "artifact list must not be empty");
    }
    if manifest.artifacts.len() > 10_000 {
        report.fail(
            "provenance_artifact_limit_exceeded",
            "artifact list exceeds the 10000-record safety limit",
        );
        report.provenance_manifest_status = "manifest_invalid".to_owned();
        return;
    }

    let mut by_path: BTreeMap<String, &ArtifactDeclaration> = BTreeMap::new();
    for artifact in &manifest.artifacts {
        if artifact.path == PROVENANCE_MANIFEST_NAME {
            report.fail(
                "provenance_manifest_self_reference",
                "provenance-manifest.json must not list itself as a hashed artifact",
            );
            continue;
        }
        if artifact.path.trim().is_empty() {
            report.fail("provenance_artifact_path_empty", "artifact paths must not be empty");
            continue;
        }
        if by_path.insert(artifact.path.clone(), artifact).is_some() {
            report.fail(
                "provenance_artifact_path_duplicate",
                format!("artifact path occurs more than once: {}", artifact.path),
            );
        }
        if !is_lower_hex(&artifact.sha256, 64) {
            report.fail(
                "provenance_artifact_digest_invalid",
                format!("{} has an invalid SHA-256 digest", artifact.path),
            );
            continue;
        }
        match hash_artifact(run_dir, &artifact.path) {
            Ok((digest, size_bytes))
                if digest == artifact.sha256 && size_bytes == artifact.size_bytes => {}
            Ok((digest, size_bytes)) => report.fail(
                "provenance_artifact_hash_mismatch",
                format!(
                    "{} declared sha256/bytes {}/{} but actual values are {}/{}",
                    artifact.path, artifact.sha256, artifact.size_bytes, digest, size_bytes
                ),
            ),
            Err(error) => report.fail(
                "provenance_artifact_invalid_path_or_unreadable",
                format!("{}: {error}", artifact.path),
            ),
        }
    }

    for (index, source) in manifest.data_sources.iter().enumerate() {
        match by_path.get(&source.source_artifact_path) {
            Some(artifact) if artifact.sha256 == source.source_artifact_sha256 => {}
            Some(_) => report.fail(
                "provenance_source_artifact_digest_mismatch",
                format!("data_sources[{index}] source artifact digest does not match its artifact record"),
            ),
            None => report.fail(
                "provenance_source_artifact_missing",
                format!("data_sources[{index}] source artifact {} is not listed in artifacts", source.source_artifact_path),
            ),
        }
    }

    let mut required_paths = vec![
        manifest.oracle_call_ledger_path.clone(),
        ".hydra/config.yaml".to_owned(),
        "source-manifest.json".to_owned(),
        "task-contract.json".to_owned(),
        "evaluation-manifest.json".to_owned(),
        "summary/episodes.json".to_owned(),
        "summary/episodes.csv".to_owned(),
        "summary/summary.json".to_owned(),
    ];
    let trajectory_dir = run_dir.join("trajectories");
    if let Ok(entries) = fs::read_dir(&trajectory_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path.file_name().and_then(|name| name.to_str()).is_some_and(|name| {
                    name.starts_with("episode_") && name.ends_with(".json")
                })
            {
                if let Ok(relative) = path.strip_prefix(run_dir) {
                    required_paths.push(relative.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    required_paths.sort();
    required_paths.dedup();
    for required in required_paths {
        if !by_path.contains_key(&required) {
            report.fail(
                "provenance_required_artifact_missing",
                format!("artifact manifest must include {required}"),
            );
        }
    }

    for (path, expected_digest) in [
        ("source-manifest.json", manifest.source_manifest_sha256.as_str()),
        ("task-contract.json", manifest.task_contract_sha256.as_str()),
        ("evaluation-manifest.json", manifest.evaluation_manifest_sha256.as_str()),
        (".hydra/config.yaml", manifest.resolved_config_sha256.as_str()),
        (manifest.oracle_call_ledger_path.as_str(), manifest.oracle_call_ledger_sha256.as_str()),
    ] {
        if let Some(artifact) = by_path.get(path) {
            if artifact.sha256 != expected_digest {
                report.fail(
                    "provenance_named_digest_mismatch",
                    format!("{path} artifact digest does not match the digest declared in the manifest"),
                );
            }
        }
    }

    verify_oracle_call_ledger(run_dir, &manifest, &by_path, expected_proposal_requests, report);
    verify_optional_shift_manifest(run_dir, &manifest, &by_path, report);

    if report.findings.len() == findings_before {
        report.provenance_manifest_status = "manifest_consistent".to_owned();
        report.checked("provenance_manifest_artifact_hashes_and_accounting_reconcile");
    } else {
        report.provenance_manifest_status = "manifest_invalid".to_owned();
    }
}

fn verify_optional_shift_manifest(
    run_dir: &Path,
    provenance: &ProvenanceManifest,
    artifacts: &BTreeMap<String, &ArtifactDeclaration>,
    report: &mut AuditReport,
) {
    let (path, expected_digest) = match (
        provenance.shift_manifest_path.as_deref(),
        provenance.shift_manifest_sha256.as_deref(),
    ) {
        (None, None) => return,
        (Some(path), Some(digest)) => (path, digest),
        _ => {
            report.fail(
                "shift_manifest_reference_incomplete",
                "shift_manifest_path and shift_manifest_sha256 must either both be present or both be absent",
            );
            return;
        }
    };
    let findings_before = report.findings.len();

    if !is_lower_hex(expected_digest, 64) {
        report.fail(
            "shift_manifest_digest_invalid",
            "shift_manifest_sha256 must be a 64-character lowercase SHA-256 digest",
        );
        return;
    }
    let Some(artifact) = artifacts.get(path) else {
        report.fail(
            "shift_manifest_artifact_missing",
            format!("shift manifest {path} is not listed in the provenance artifact manifest"),
        );
        return;
    };
    if artifact.sha256 != expected_digest {
        report.fail(
            "shift_manifest_digest_link_mismatch",
            "shift_manifest_sha256 does not match the artifact-list digest",
        );
        return;
    }
    let (_, size_bytes) = match hash_artifact(run_dir, path) {
        Ok(result) => result,
        Err(error) => {
            report.fail("shift_manifest_invalid_path", error);
            return;
        }
    };
    if size_bytes > MAX_SHIFT_MANIFEST_BYTES {
        report.fail(
            "shift_manifest_too_large",
            format!("shift manifest exceeds {MAX_SHIFT_MANIFEST_BYTES} bytes"),
        );
        return;
    }
    let raw = match fs::read(run_dir.join(path)) {
        Ok(raw) => raw,
        Err(error) => {
            report.fail(
                "shift_manifest_unreadable",
                format!("cannot read shift manifest: {error}"),
            );
            return;
        }
    };
    let shift: ShiftManifest = match serde_json::from_slice(&raw) {
        Ok(shift) => shift,
        Err(error) => {
            report.fail(
                "shift_manifest_invalid_schema",
                format!("shift manifest does not match the strict Rust contract: {error}"),
            );
            return;
        }
    };

    if shift.schema_version != SHIFT_MANIFEST_SCHEMA_VERSION {
        report.fail(
            "shift_manifest_schema_version_mismatch",
            format!(
                "expected {SHIFT_MANIFEST_SCHEMA_VERSION}, found {}",
                shift.schema_version
            ),
        );
    }
    if shift.benchmark_id != provenance.benchmark_id {
        report.fail(
            "shift_manifest_benchmark_mismatch",
            "shift manifest benchmark_id must match the provenance manifest",
        );
    }
    if shift.task_contract_sha256 != provenance.task_contract_sha256
        || !is_lower_hex(&shift.task_contract_sha256, 64)
    {
        report.fail(
            "shift_manifest_task_contract_mismatch",
            "shift manifest task_contract_sha256 must match the valid provenance task contract hash",
        );
    }
    if shift.split_policy != "independence_key_group_holdout_v1" {
        report.fail(
            "shift_manifest_split_policy_unsupported",
            "split_policy must be independence_key_group_holdout_v1",
        );
    }
    if shift.shift_groups.is_empty() {
        report.fail("shift_manifest_groups_missing", "shift_groups must not be empty");
        return;
    }
    if shift.shift_groups.len() > MAX_SHIFT_GROUPS {
        report.fail(
            "shift_manifest_group_limit_exceeded",
            format!("shift manifest exceeds {MAX_SHIFT_GROUPS} groups"),
        );
        return;
    }

    let valid_families = [
        "composition",
        "surface_coordination",
        "defects",
        "temperature_energy",
        "strain_pressure_phase",
        "size_interface_passivation_substrate",
    ];
    let valid_severity = ["baseline", "mild", "moderate", "severe"];
    let valid_kinds = ["in_distribution", "single_factor", "compounded"];
    let valid_axes = [
        "feature_support",
        "label_distribution",
        "graph_topology",
        "physical_regime",
        "chemical_environment",
        "measurement_process",
    ];
    let valid_splits = ["train", "calibration", "validation", "test_lockbox"];
    let mut group_ids = BTreeSet::new();
    let mut independence_splits: BTreeMap<String, String> = BTreeMap::new();
    let mut sample_splits: BTreeMap<String, String> = BTreeMap::new();

    for (index, group) in shift.shift_groups.iter().enumerate() {
        for (field, value) in [
            ("group_id", group.group_id.as_str()),
            ("shift_kind", group.shift_kind.as_str()),
            ("severity", group.severity.as_str()),
            ("split", group.split.as_str()),
            ("physical_rationale", group.physical_rationale.as_str()),
            ("severity_definition", group.severity_definition.as_str()),
            ("source_artifact_path", group.source_artifact_path.as_str()),
            ("source_artifact_sha256", group.source_artifact_sha256.as_str()),
        ] {
            if value.trim().is_empty() {
                report.fail(
                    "shift_manifest_group_field_empty",
                    format!("shift_groups[{index}].{field} must not be empty"),
                );
            }
        }
        if !group_ids.insert(group.group_id.clone()) {
            report.fail(
                "shift_manifest_duplicate_group_id",
                format!("group_id {:?} appears more than once", group.group_id),
            );
        }
        let mut physical_families_seen = BTreeSet::new();
        for family in &group.shift_families {
            if !valid_families.contains(&family.as_str()) {
                report.fail(
                    "shift_manifest_family_invalid",
                    format!("shift_groups[{index}] has unsupported physical shift family {family:?}"),
                );
            }
            if !physical_families_seen.insert(family) {
                report.fail(
                    "shift_manifest_family_duplicate",
                    format!("shift_groups[{index}] repeats physical shift family {family:?}"),
                );
            }
        }
        if !valid_kinds.contains(&group.shift_kind.as_str()) {
            report.fail(
                "shift_manifest_kind_invalid",
                format!("shift_groups[{index}] has unsupported shift_kind {:?}", group.shift_kind),
            );
        }
        if !valid_severity.contains(&group.severity.as_str()) {
            report.fail(
                "shift_manifest_severity_invalid",
                format!("shift_groups[{index}] has unsupported severity {:?}", group.severity),
            );
        }
        if (group.shift_kind == "in_distribution") != (group.severity == "baseline") {
            report.fail(
                "shift_manifest_kind_severity_mismatch",
                format!(
                    "shift_groups[{index}] must use baseline severity only for in_distribution data"
                ),
            );
        }
        if group.shift_kind == "in_distribution" && !group.shift_families.is_empty() {
            report.fail(
                "shift_manifest_baseline_family_mismatch",
                format!(
                    "shift_groups[{index}] cannot declare changed physical families for in_distribution data"
                ),
            );
        }
        let mut axes_seen = BTreeSet::new();
        for axis in &group.shift_axes {
            if !valid_axes.contains(&axis.as_str()) {
                report.fail(
                    "shift_manifest_axis_invalid",
                    format!("shift_groups[{index}] has unsupported shift axis {axis:?}"),
                );
            }
            if !axes_seen.insert(axis) {
                report.fail(
                    "shift_manifest_axis_duplicate",
                    format!("shift_groups[{index}] repeats shift axis {axis:?}"),
                );
            }
        }
        let axes_cardinality_valid = match group.shift_kind.as_str() {
            "in_distribution" => group.shift_axes.is_empty(),
            "single_factor" => group.shift_axes.len() == 1,
            "compounded" => group.shift_axes.len() >= 2,
            _ => false,
        };
        if !axes_cardinality_valid {
            report.fail(
                "shift_manifest_axis_kind_mismatch",
                format!(
                    "shift_groups[{index}] must have 0 axes for in_distribution, exactly 1 for single_factor, and at least 2 for compounded shifts"
                ),
            );
        }
        if !valid_splits.contains(&group.split.as_str()) {
            report.fail(
                "shift_manifest_split_invalid",
                format!("shift_groups[{index}] has unsupported split {:?}", group.split),
            );
        }
        if !is_lower_hex(&group.source_artifact_sha256, 64) {
            report.fail(
                "shift_manifest_source_digest_invalid",
                format!("shift_groups[{index}].source_artifact_sha256 is invalid"),
            );
        }
        match artifacts.get(&group.source_artifact_path) {
            Some(source) if source.sha256 == group.source_artifact_sha256 => {}
            Some(_) => report.fail(
                "shift_manifest_source_digest_mismatch",
                format!("shift_groups[{index}] source digest differs from the artifact list"),
            ),
            None => report.fail(
                "shift_manifest_source_artifact_missing",
                format!(
                    "shift_groups[{index}] source artifact {} is not listed",
                    group.source_artifact_path
                ),
            ),
        }
        if group.independence_keys.is_empty() {
            report.fail(
                "shift_manifest_independence_keys_missing",
                format!("shift_groups[{index}] must declare at least one independence key"),
            );
        }
        if group.sample_ids.is_empty() {
            report.fail(
                "shift_manifest_sample_ids_missing",
                format!("shift_groups[{index}] must declare at least one sample ID"),
            );
        }

        let mut local_keys = BTreeSet::new();
        for key in &group.independence_keys {
            if key.trim().is_empty() || !local_keys.insert(key.clone()) {
                report.fail(
                    "shift_manifest_independence_key_empty_or_duplicate",
                    format!("shift_groups[{index}] contains an empty or duplicate independence key"),
                );
                continue;
            }
            if let Some(previous_split) = independence_splits.get(key) {
                if previous_split != &group.split {
                    report.fail(
                        "shift_manifest_independence_leakage",
                        format!(
                            "independence key {key:?} occurs in both {previous_split:?} and {:?} splits",
                            group.split
                        ),
                    );
                }
            } else {
                independence_splits.insert(key.clone(), group.split.clone());
            }
        }
        let mut local_samples = BTreeSet::new();
        for sample_id in &group.sample_ids {
            if sample_id.trim().is_empty() || !local_samples.insert(sample_id.clone()) {
                report.fail(
                    "shift_manifest_sample_id_empty_or_duplicate",
                    format!("shift_groups[{index}] contains an empty or duplicate sample ID"),
                );
                continue;
            }
            if let Some(previous_split) = sample_splits.get(sample_id) {
                if previous_split != &group.split {
                    report.fail(
                        "shift_manifest_sample_leakage",
                        format!(
                            "sample ID {sample_id:?} occurs in both {previous_split:?} and {:?} splits",
                            group.split
                        ),
                    );
                }
            } else {
                sample_splits.insert(sample_id.clone(), group.split.clone());
            }
        }
    }

    if report.findings.len() == findings_before {
        report.checked("shift_manifest_provenance_and_group_split_integrity");
    }
}

fn verify_oracle_call_ledger(
    run_dir: &Path,
    manifest: &ProvenanceManifest,
    artifacts: &BTreeMap<String, &ArtifactDeclaration>,
    expected_proposal_requests: Option<usize>,
    report: &mut AuditReport,
) {
    let findings_before = report.findings.len();
    let Some(ledger_artifact) = artifacts.get(&manifest.oracle_call_ledger_path) else {
        report.fail(
            "oracle_call_ledger_missing",
            format!(
                "oracle call ledger {} is not included in the artifact manifest",
                manifest.oracle_call_ledger_path
            ),
        );
        return;
    };
    if ledger_artifact.sha256 != manifest.oracle_call_ledger_sha256 {
        report.fail(
            "oracle_call_ledger_digest_link_mismatch",
            "oracle_call_ledger_sha256 does not match the listed ledger artifact digest",
        );
        return;
    }

    let (actual_digest, size_bytes) = match hash_artifact(run_dir, &manifest.oracle_call_ledger_path) {
        Ok(result) => result,
        Err(error) => {
            report.fail("oracle_call_ledger_invalid_path", error);
            return;
        }
    };
    if size_bytes > MAX_ORACLE_CALL_LEDGER_BYTES {
        report.fail(
            "oracle_call_ledger_too_large",
            format!(
                "oracle call ledger exceeds the {MAX_ORACLE_CALL_LEDGER_BYTES}-byte limit"
            ),
        );
        return;
    }
    if actual_digest != manifest.oracle_call_ledger_sha256 {
        report.fail(
            "oracle_call_ledger_hash_mismatch",
            "oracle call ledger bytes do not match the declared SHA-256",
        );
        return;
    }

    let raw = match fs::read(run_dir.join(&manifest.oracle_call_ledger_path)) {
        Ok(raw) => raw,
        Err(error) => {
            report.fail(
                "oracle_call_ledger_unreadable",
                format!("cannot read oracle call ledger: {error}"),
            );
            return;
        }
    };
    let ledger_text = match std::str::from_utf8(&raw) {
        Ok(text) => text,
        Err(error) => {
            report.fail(
                "oracle_call_ledger_invalid_utf8",
                format!("oracle call ledger must be UTF-8 JSONL: {error}"),
            );
            return;
        }
    };
    let line_count = ledger_text.lines().count();
    if line_count > MAX_ORACLE_CALL_RECORDS {
        report.fail(
            "oracle_call_ledger_record_limit_exceeded",
            format!("oracle call ledger exceeds {MAX_ORACLE_CALL_RECORDS} records"),
        );
        return;
    }
    let lines: Vec<&str> = ledger_text.lines().collect();
    if lines.is_empty() {
        report.fail("oracle_call_ledger_empty", "oracle call ledger contains no records");
        return;
    }
    let mut setup_requests = 0usize;
    let mut proposal_requests = 0usize;
    let mut retry_requests = 0usize;
    let mut cache_hits = 0usize;
    let mut local_rejections = 0usize;
    let mut external_calls = 0usize;
    let mut failed_external_calls = 0usize;
    let mut total_cost = 0.0_f64;

    for (index, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            report.fail(
                "oracle_call_ledger_blank_line",
                format!("oracle call ledger line {} is blank", index + 1),
            );
            continue;
        }
        let call: OracleCallRecord = match serde_json::from_str(line) {
            Ok(call) => call,
            Err(error) => {
                report.fail(
                    "oracle_call_ledger_record_invalid",
                    format!("line {} is not a valid strict call record: {error}", index + 1),
                );
                continue;
            }
        };
        if call.sequence != index + 1 {
            report.fail(
                "oracle_call_ledger_sequence_mismatch",
                format!(
                    "line {} declares sequence {}; sequence numbers must be contiguous from 1",
                    index + 1,
                    call.sequence
                ),
            );
        }
        match call.stage.as_str() {
            "setup" => setup_requests += 1,
            "proposal" => {
                proposal_requests += 1;
                if !call
                    .candidate_id
                    .as_deref()
                    .is_some_and(|candidate_id| !candidate_id.trim().is_empty())
                {
                    report.fail(
                        "oracle_call_ledger_candidate_missing",
                        format!("proposal record on line {} requires candidate_id", index + 1),
                    );
                }
            }
            "retry" => retry_requests += 1,
            _ => report.fail(
                "oracle_call_ledger_stage_invalid",
                format!("line {} stage must be setup, proposal, or retry", index + 1),
            ),
        }
        if call.cache_hit {
            cache_hits += 1;
        }
        if call.external_invocation {
            external_calls += 1;
            if call.status != "evaluated" {
                failed_external_calls += 1;
            }
        }
        match call.status.as_str() {
            "evaluated" => {
                if call.external_invocation == call.cache_hit {
                    report.fail(
                        "oracle_call_ledger_invocation_cache_contradiction",
                        format!(
                            "evaluated line {} must be exactly one of cache_hit or external_invocation",
                            index + 1
                        ),
                    );
                }
            }
            "invalid_candidate" => {
                local_rejections += 1;
                if call.stage != "proposal" || call.cache_hit || call.external_invocation {
                    report.fail(
                        "oracle_call_ledger_local_rejection_mismatch",
                        format!(
                            "invalid_candidate line {} must be a proposal rejected locally with no cache hit and no external invocation",
                            index + 1
                        ),
                    );
                }
            }
            "oracle_failed" | "timeout" => {
                if !call.external_invocation || call.cache_hit {
                    report.fail(
                        "oracle_call_ledger_failed_call_mismatch",
                        format!(
                            "failed line {} must be an external invocation, not a cache hit",
                            index + 1
                        ),
                    );
                }
            }
            _ => report.fail(
                "oracle_call_ledger_status_invalid",
                format!("line {} has unsupported status {:?}", index + 1, call.status),
            ),
        }
        if !call.cost.is_finite() || call.cost < 0.0 || call.cost > 1.0e12 {
            report.fail(
                "oracle_call_ledger_cost_invalid",
                format!("line {} cost must be finite and between 0 and 1e12", index + 1),
            );
        } else {
            total_cost += call.cost;
        }
        if !is_lower_hex(&call.receipt_sha256, 64) {
            report.fail(
                "oracle_call_ledger_receipt_digest_invalid",
                format!("line {} receipt_sha256 is not a SHA-256 digest", index + 1),
            );
        }
        match artifacts.get(&call.receipt_path) {
            Some(artifact) if artifact.sha256 == call.receipt_sha256 => {}
            Some(_) => report.fail(
                "oracle_call_ledger_receipt_digest_mismatch",
                format!("line {} receipt digest differs from the artifact manifest", index + 1),
            ),
            None => report.fail(
                "oracle_call_ledger_receipt_missing",
                format!(
                    "line {} receipt {} is not included in the artifact manifest",
                    index + 1,
                    call.receipt_path
                ),
            ),
        }
    }

    let accounting = &manifest.oracle_accounting;
    for (counter, actual, expected) in [
        ("setup_requests", setup_requests, accounting.setup_requests),
        ("proposal_requests", proposal_requests, accounting.proposal_requests),
        ("retry_requests", retry_requests, accounting.retry_requests),
        ("cache_hits", cache_hits, accounting.cache_hits),
        ("local_rejections", local_rejections, accounting.local_rejections),
        ("external_calls", external_calls, accounting.external_calls),
        (
            "failed_external_calls",
            failed_external_calls,
            accounting.failed_external_calls,
        ),
    ] {
        if actual != expected {
            report.fail(
                "oracle_call_ledger_counter_mismatch",
                format!("ledger {counter}={actual}, manifest declares {expected}"),
            );
        }
    }
    if !floats_equal(total_cost, accounting.total_cost) {
        report.fail(
            "oracle_call_ledger_cost_mismatch",
            format!(
                "ledger total cost is {total_cost}, manifest declares {}",
                accounting.total_cost
            ),
        );
    }
    match expected_proposal_requests {
        Some(expected) if proposal_requests == expected => {}
        Some(expected) => report.fail(
            "oracle_call_ledger_proposal_count_mismatch",
            format!("ledger has {proposal_requests} proposal records but episode metrics sum to {expected}"),
        ),
        None => report.fail(
            "oracle_call_ledger_proposal_count_unavailable",
            "cannot reconcile proposal ledger records because episode query counts are unavailable",
        ),
    }

    if report.findings.len() == findings_before {
        report.checked("oracle_call_ledger_reconciles_manifest_and_receipts");
    }
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_artifact(run_dir: &Path, relative_path: &str) -> Result<(String, u64), String> {
    let relative = Path::new(relative_path);
    if relative.as_os_str().is_empty()
        || relative_path.contains('\\')
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("artifact path must be a non-empty relative path without dot or parent components".to_owned());
    }
    let root = run_dir
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize run directory: {error}"))?;
    let candidate = run_dir.join(relative);
    let resolved = candidate
        .canonicalize()
        .map_err(|error| format!("cannot resolve artifact: {error}"))?;
    if !resolved.starts_with(&root) {
        return Err("artifact resolves outside the run directory".to_owned());
    }
    let metadata = fs::metadata(&resolved).map_err(|error| format!("cannot stat artifact: {error}"))?;
    if !metadata.is_file() {
        return Err("artifact must be a regular file".to_owned());
    }

    let mut file = File::open(&resolved).map_err(|error| format!("cannot open artifact: {error}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot read artifact: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), metadata.len()))
}

fn read_json(path: &Path, report: &mut AuditReport) -> Option<Value> {
    match fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<Value>(&content) {
            Ok(value) => Some(value),
            Err(error) => {
                report.fail("invalid_json", format!("{}: {error}", path.display()));
                None
            }
        },
        Err(error) => {
            report.fail(
                "missing_or_unreadable_json",
                format!("{}: {error}", path.display()),
            );
            None
        }
    }
}

fn discover_trajectories(directory: &Path) -> Result<Vec<(usize, PathBuf)>, String> {
    let entries =
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", directory.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("episode_") || !name.ends_with(".json") {
            continue;
        }
        let digits = &name["episode_".len()..name.len() - ".json".len()];
        if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
            return Err(format!("malformed episode filename: {}", path.display()));
        }
        let index = digits
            .parse::<usize>()
            .map_err(|error| format!("episode index out of range in {}: {error}", path.display()))?;
        files.push((index, path));
    }
    Ok(files)
}

fn audit_csv(path: &Path, rows: &[Value], report: &mut AuditReport) {
    let mut reader = match csv::ReaderBuilder::new().has_headers(true).from_path(path) {
        Ok(reader) => reader,
        Err(error) => {
            report.fail("csv_unreadable", format!("{}: {error}", path.display()));
            return;
        }
    };
    let headers = match reader.headers() {
        Ok(headers) => headers.iter().map(str::to_owned).collect::<Vec<_>>(),
        Err(error) => {
            report.fail("csv_header_invalid", format!("{}: {error}", path.display()));
            return;
        }
    };

    let expected_headers = rows
        .iter()
        .filter_map(Value::as_object)
        .flat_map(|object| object.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    if headers != expected_headers {
        report.fail(
            "csv_header_mismatch",
            format!(
                "{} header differs from the sorted union of episode metric keys",
                path.display()
            ),
        );
        return;
    }

    let mut row_count = 0usize;
    for (row_index, csv_row) in reader.records().enumerate() {
        let csv_row = match csv_row {
            Ok(row) => row,
            Err(error) => {
                report.fail(
                    "csv_row_invalid",
                    format!("{} row {}: {error}", path.display(), row_index + 2),
                );
                continue;
            }
        };
        row_count += 1;
        let Some(json_row) = rows.get(row_index).and_then(Value::as_object) else {
            report.fail(
                "csv_has_extra_rows",
                format!(
                    "{} has a row without a matching episodes.json object",
                    path.display()
                ),
            );
            continue;
        };

        for (column, key) in headers.iter().enumerate() {
            let actual = csv_row.get(column).unwrap_or_default();
            let expected = json_row.get(key);
            let matches = match expected {
                None | Some(Value::Null) => actual.is_empty(),
                Some(Value::Number(number)) => number
                    .as_f64()
                    .zip(actual.parse::<f64>().ok())
                    .is_some_and(|(left, right)| floats_equal(left, right)),
                Some(Value::Bool(value)) => actual == if *value { "True" } else { "False" },
                Some(Value::String(value)) => actual == value,
                Some(_) => false,
            };
            if !matches {
                report.fail(
                    "csv_cell_mismatch",
                    format!(
                        "{} row {}, column {} does not match episodes.json",
                        path.display(),
                        row_index + 2,
                        key
                    ),
                );
            }
        }
    }

    if row_count != rows.len() {
        report.fail(
            "csv_row_count_mismatch",
            format!(
                "{} has {row_count} data rows but episodes.json has {}",
                path.display(),
                rows.len()
            ),
        );
    } else if !report.findings.iter().any(|finding| finding.code.starts_with("csv_")) {
        report.checked("episodes_csv_reconciles");
    }
}

fn audit_aggregate(rows: &[Value], aggregate_value: Option<&Value>, report: &mut AuditReport) {
    let Some(aggregate_object) = aggregate_value.and_then(Value::as_object) else {
        return;
    };
    let Some(aggregate) = aggregate_object.get("summary").and_then(Value::as_object) else {
        report.fail(
            "aggregate_summary_missing",
            "summary.json must contain a summary object",
        );
        return;
    };
    if rows.is_empty() {
        return;
    }
    let findings_before = report.findings.len();

    let mut values_by_metric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for row in rows {
        let Some(object) = row.as_object() else {
            report.fail("episode_row_not_object", "each episodes.json row must be an object");
            continue;
        };
        for (key, value) in object {
            if !key.starts_with("final/") {
                continue;
            }
            if let Some(number) = value.as_f64() {
                if number.is_finite() {
                    values_by_metric.entry(key.clone()).or_default().push(number);
                } else {
                    report.fail(
                        "non_finite_metric",
                        format!("episodes.json contains non-finite metric {key}"),
                    );
                }
            } else if let Some(boolean) = value.as_bool() {
                // MADE's Python aggregator uses isinstance(value, (int, float));
                // Python bool is an int subclass, so preserve its 0/1 semantics.
                values_by_metric
                    .entry(key.clone())
                    .or_default()
                    .push(if boolean { 1.0 } else { 0.0 });
            }
        }
    }

    let expected_keys = values_by_metric.keys().cloned().collect::<BTreeSet<_>>();
    let actual_keys = aggregate.keys().cloned().collect::<BTreeSet<_>>();
    if expected_keys != actual_keys {
        report.fail(
            "aggregate_metric_keys_mismatch",
            format!(
                "summary metric keys differ; expected {expected_keys:?}, got {actual_keys:?}"
            ),
        );
        return;
    }

    for (key, values) in values_by_metric {
        if values.is_empty() {
            report.fail("empty_aggregate_metric", format!("no numeric values for {key}"));
            continue;
        }
        let Some(metric_summary) = aggregate.get(&key).and_then(Value::as_object) else {
            report.fail(
                "aggregate_metric_invalid",
                format!("summary for {key} must be an object"),
            );
            continue;
        };

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / n;
        let std = variance.sqrt();
        let sem = std / n.sqrt();

        for (name, expected) in [("mean", mean), ("std", std), ("sem", sem)] {
            match metric_summary.get(name).and_then(Value::as_f64) {
                Some(actual) if floats_equal(actual, expected) => {}
                Some(actual) => report.fail(
                    "aggregate_statistic_mismatch",
                    format!(
                        "summary {key}.{name} is {actual} but independently reconstructed value is {expected}"
                    ),
                ),
                None => report.fail(
                    "aggregate_statistic_missing",
                    format!("summary {key} lacks numeric field {name}"),
                ),
            }
        }
    }

    if report.findings.len() == findings_before {
        report.checked("aggregate_mean_std_sem_reconstructed");
    }
}

fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => left
            .as_f64()
            .zip(right.as_f64())
            .is_some_and(|(left, right)| floats_equal(left, right)),
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right.iter())
                    .all(|(left, right)| values_equal(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right.get(key).is_some_and(|other| values_equal(value, other))
                })
        }
        _ => left == right,
    }
}

fn floats_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs()
            <= ABS_TOLERANCE.max(REL_TOLERANCE * left.abs().max(right.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "symthaea-nanotech-audit-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(path.join(".hydra")).expect("create config dir");
            fs::create_dir_all(path.join("trajectories")).expect("create trajectories dir");
            fs::create_dir_all(path.join("summary")).expect("create summary dir");
            fs::write(
                path.join(".hydra/config.yaml"),
                "experiment:\n  num_episodes: 2\n",
            )
            .expect("write config");

            let metrics0 = json!({
                "episode_id": 0,
                "final/is_stable": true,
                "final/queries_used": 2,
                "final/score": 1.0
            });
            let metrics1 = json!({
                "episode_id": 1,
                "final/is_stable": false,
                "final/queries_used": 2,
                "final/score": 3.0
            });

            for (index, metrics) in [(0, &metrics0), (1, &metrics1)] {
                let result = json!({
                    "metrics": metrics,
                    "trajectory": [{ "step": 1 }, { "step": 2 }],
                    "final_env_state": { "phase_diagram_all_entries": [] },
                    "phase_diagram_gt": []
                });
                fs::write(
                    path.join("trajectories").join(format!("episode_{index:03}.json")),
                    serde_json::to_vec_pretty(&result).expect("serialize episode"),
                )
                .expect("write episode");
            }

            fs::write(
                path.join("summary/episodes.json"),
                serde_json::to_vec_pretty(&json!([metrics0, metrics1])).expect("serialize rows"),
            )
            .expect("write episode rows");
            fs::write(
                path.join("summary/summary.json"),
                serde_json::to_vec_pretty(&json!({
                    "episodes": 2,
                    "summary": {
                        "final/is_stable": {
                            "mean": 0.5,
                            "std": 0.5,
                            "sem": 0.35355339059327373
                        },
                        "final/queries_used": { "mean": 2.0, "std": 0.0, "sem": 0.0 },
                        "final/score": {
                            "mean": 2.0,
                            "std": 1.0,
                            "sem": 0.7071067811865475
                        }
                    }
                }))
                .expect("serialize aggregate"),
            )
            .expect("write aggregate");
            fs::write(
                path.join("summary/episodes.csv"),
                "episode_id,final/is_stable,final/queries_used,final/score\n0,True,2,1.0\n1,False,2,3.0\n",
            )
            .expect("write csv");
            Self(path)
        }

        fn write_provenance_manifest(&self) {
            fs::write(self.0.join("source-manifest.json"), b"{\"source\":\"synthetic fixture\"}")
                .expect("write source manifest");
            fs::write(self.0.join("synthetic-source.json"), b"{\"dataset\":\"fixture-only\"}")
                .expect("write synthetic source artifact");
            fs::write(self.0.join("task-contract.json"), b"{\"task\":\"synthetic fixture task\"}")
                .expect("write task contract");
            fs::write(
                self.0.join("evaluation-manifest.json"),
                b"{\"evaluation\":\"synthetic fixture evaluation\"}",
            )
            .expect("write evaluation manifest");

            let calls = [
                ("setup", None, "evaluated", 0.1_f64, false, true),
                ("setup", None, "evaluated", 0.2_f64, true, false),
                ("proposal", Some("candidate-0"), "evaluated", 0.2_f64, false, true),
                ("proposal", Some("candidate-1"), "evaluated", 0.2_f64, false, true),
                ("proposal", Some("candidate-2"), "evaluated", 0.2_f64, false, true),
                ("proposal", Some("candidate-3"), "invalid_candidate", 0.2_f64, false, false),
                ("retry", None, "evaluated", 0.4_f64, false, true),
            ];
            let mut ledger_lines = Vec::new();
            let mut receipt_paths = Vec::new();
            for (index, (stage, candidate_id, status, cost, cache_hit, external_invocation)) in
                calls.into_iter().enumerate()
            {
                let receipt_path = format!("receipts/receipt_{index:03}.json");
                fs::create_dir_all(self.0.join("receipts")).expect("create receipts directory");
                let receipt = serde_json::to_vec_pretty(&json!({
                    "sequence": index + 1,
                    "synthetic": true,
                    "stage": stage
                }))
                .expect("serialize receipt");
                fs::write(self.0.join(&receipt_path), &receipt).expect("write receipt");
                let (receipt_sha256, _) =
                    hash_artifact(&self.0, &receipt_path).expect("hash receipt");
                let record = json!({
                    "sequence": index + 1,
                    "stage": stage,
                    "status": status,
                    "candidate_id": candidate_id,
                    "receipt_path": receipt_path,
                    "receipt_sha256": receipt_sha256,
                    "cost": cost,
                    "cache_hit": cache_hit,
                    "external_invocation": external_invocation
                });
                ledger_lines.push(serde_json::to_string(&record).expect("serialize ledger row"));
                receipt_paths.push(receipt_path);
            }
            let ledger_content = format!("{}\n", ledger_lines.join("\n"));
            fs::write(self.0.join("oracle-call-ledger.jsonl"), &ledger_content)
                .expect("write oracle call ledger");

            let mut paths = vec![
                ".hydra/config.yaml".to_owned(),
                "source-manifest.json".to_owned(),
                "task-contract.json".to_owned(),
                "evaluation-manifest.json".to_owned(),
                "synthetic-source.json".to_owned(),
                "oracle-call-ledger.jsonl".to_owned(),
                "summary/episodes.json".to_owned(),
                "summary/episodes.csv".to_owned(),
                "summary/summary.json".to_owned(),
                "trajectories/episode_000.json".to_owned(),
                "trajectories/episode_001.json".to_owned(),
            ];
            paths.extend(receipt_paths);
            let mut artifacts = Vec::new();
            let mut digests = BTreeMap::new();
            for path in paths {
                let (sha256, size_bytes) =
                    hash_artifact(&self.0, &path).expect("hash fixture artifact");
                digests.insert(path.clone(), sha256.clone());
                artifacts.push(json!({
                    "path": path,
                    "sha256": sha256,
                    "size_bytes": size_bytes
                }));
            }

            let manifest = json!({
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "run_id": "synthetic-run",
                "benchmark_id": "synthetic-made",
                "upstream_repository": "https://github.com/diffractivelabs/MADE",
                "upstream_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "policy_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "task_contract_sha256": digests["task-contract.json"],
                "evaluation_manifest_sha256": digests["evaluation-manifest.json"],
                "source_manifest_sha256": digests["source-manifest.json"],
                "resolved_config_sha256": digests[".hydra/config.yaml"],
                "oracle_call_ledger_path": "oracle-call-ledger.jsonl",
                "oracle_call_ledger_sha256": digests["oracle-call-ledger.jsonl"],
                "data_sources": [{
                    "source_id": "synthetic-data",
                    "source_url": "https://example.org/synthetic",
                    "revision": "fixture-v1",
                    "license": "synthetic-only",
                    "rights_status": "not_applicable",
                    "source_artifact_path": "synthetic-source.json",
                    "source_artifact_sha256": digests["synthetic-source.json"]
                }],
                "oracle_accounting": {
                    "setup_requests": 2,
                    "proposal_requests": 4,
                    "retry_requests": 1,
                    "cache_hits": 1,
                    "local_rejections": 1,
                    "external_calls": 5,
                    "failed_external_calls": 0,
                    "total_cost": 1.5,
                    "cost_unit": "synthetic-credit"
                },
                "artifacts": artifacts
            });
            fs::write(
                self.0.join(PROVENANCE_MANIFEST_NAME),
                serde_json::to_vec_pretty(&manifest).expect("serialize provenance manifest"),
            )
            .expect("write provenance manifest");
        }

        fn write_shift_manifest(&self, inject_leakage: bool) {
            let (source_sha256, _) =
                hash_artifact(&self.0, "synthetic-source.json").expect("hash source snapshot");
            let provenance_path = self.0.join(PROVENANCE_MANIFEST_NAME);
            let mut provenance: Value = serde_json::from_str(
                &fs::read_to_string(&provenance_path).expect("read provenance manifest"),
            )
            .expect("parse provenance manifest");
            let task_contract_sha256 = provenance["task_contract_sha256"]
                .as_str()
                .expect("task-contract digest")
                .to_owned();

            let test_independence_key = if inject_leakage {
                "material-family-A"
            } else {
                "material-family-C"
            };
            let shift = json!({
                "schema_version": SHIFT_MANIFEST_SCHEMA_VERSION,
                "benchmark_id": "synthetic-made",
                "task_contract_sha256": task_contract_sha256,
                "split_policy": "independence_key_group_holdout_v1",
                "shift_groups": [
                    {
                        "group_id": "train-composition",
                        "shift_families": [],
                        "shift_axes": [],
                        "shift_kind": "in_distribution",
                        "severity": "baseline",
                        "split": "train",
                        "physical_rationale": "Synthetic baseline chemistry family for verifier tests.",
                        "severity_definition": "No shift; reference group used only to exercise schema logic.",
                        "source_artifact_path": "synthetic-source.json",
                        "source_artifact_sha256": source_sha256,
                        "independence_keys": ["material-family-A"],
                        "sample_ids": ["sample-train-001"]
                    },
                    {
                        "group_id": "calibration-temperature",
                        "shift_families": ["temperature_energy"],
                        "shift_axes": ["physical_regime"],
                        "shift_kind": "single_factor",
                        "severity": "moderate",
                        "split": "calibration",
                        "physical_rationale": "Synthetic thermal regime for verifier tests.",
                        "severity_definition": "Synthetic moderate label; no physical claim is implied by fixture data.",
                        "source_artifact_path": "synthetic-source.json",
                        "source_artifact_sha256": source_sha256,
                        "independence_keys": ["material-family-B"],
                        "sample_ids": ["sample-calibration-001"]
                    },
                    {
                        "group_id": "test-surface",
                        "shift_families": ["surface_coordination"],
                        "shift_axes": ["graph_topology"],
                        "shift_kind": "single_factor",
                        "severity": "severe",
                        "split": "test_lockbox",
                        "physical_rationale": "Synthetic surface family for verifier tests.",
                        "severity_definition": "Synthetic severe label; not derived from experimental evidence.",
                        "source_artifact_path": "synthetic-source.json",
                        "source_artifact_sha256": source_sha256,
                        "independence_keys": [test_independence_key],
                        "sample_ids": ["sample-test-001"]
                    }
                ]
            });
            fs::write(
                self.0.join("shift-manifest.json"),
                serde_json::to_vec_pretty(&shift).expect("serialize shift manifest"),
            )
            .expect("write shift manifest");
            let (shift_sha256, shift_size_bytes) =
                hash_artifact(&self.0, "shift-manifest.json").expect("hash shift manifest");

            provenance["shift_manifest_path"] = json!("shift-manifest.json");
            provenance["shift_manifest_sha256"] = json!(shift_sha256);
            provenance["artifacts"]
                .as_array_mut()
                .expect("artifact array")
                .push(json!({
                    "path": "shift-manifest.json",
                    "sha256": shift_sha256,
                    "size_bytes": shift_size_bytes
                }));
            fs::write(
                &provenance_path,
                serde_json::to_vec_pretty(&provenance).expect("serialize amended provenance"),
            )
            .expect("bind shift manifest into provenance");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn accepts_consistent_episode_trajectory_csv_and_aggregate() {
        let fixture = Fixture::new();
        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "artifact_integrity_pass", "{:?}", report.findings);
        assert!(report.findings.is_empty());
        assert!(report
            .checks_completed
            .contains(&"aggregate_mean_std_sem_reconstructed".to_owned()));
    }

    #[test]
    fn rejects_a_mismatched_aggregate_mean() {
        let fixture = Fixture::new();
        let path = fixture.0.join("summary/summary.json");
        let mut aggregate: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read summary")).unwrap();
        aggregate["summary"]["final/score"]["mean"] = json!(99.0);
        fs::write(&path, serde_json::to_vec_pretty(&aggregate).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "aggregate_statistic_mismatch"));
    }

    #[test]
    fn rejects_trajectory_query_count_mismatch() {
        let fixture = Fixture::new();
        let path = fixture.0.join("trajectories/episode_000.json");
        let mut episode: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read episode")).unwrap();
        episode["trajectory"].as_array_mut().unwrap().pop();
        fs::write(&path, serde_json::to_vec_pretty(&episode).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "query_count_mismatch"));
    }

    #[test]
    fn rejects_an_empty_episode_set() {
        let fixture = Fixture::new();
        fs::write(
            fixture.0.join("summary/episodes.json"),
            serde_json::to_vec_pretty(&json!([])).unwrap(),
        )
        .unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report.findings.iter().any(|finding| finding.code == "no_episode_rows"));
    }

    #[test]
    fn accepts_grouped_shift_manifest_with_separate_splits() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        fixture.write_shift_manifest(false);

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "artifact_integrity_pass", "{:?}", report.findings);
        assert_eq!(report.provenance_manifest_status, "manifest_consistent");
        assert!(report.checks_completed.contains(
            &"shift_manifest_provenance_and_group_split_integrity".to_owned()
        ));
    }

    #[test]
    fn rejects_independence_key_leakage_across_train_and_test() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        fixture.write_shift_manifest(true);

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert_eq!(report.provenance_manifest_status, "manifest_invalid");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "shift_manifest_independence_leakage"));
    }

    #[test]
    fn accepts_compounded_group_with_multiple_independent_axes() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        fixture.write_shift_manifest(false);

        let shift_path = fixture.0.join("shift-manifest.json");
        let mut shift: Value =
            serde_json::from_str(&fs::read_to_string(&shift_path).expect("read shift")).unwrap();
        shift["shift_groups"][1]["shift_families"] =
            json!(["temperature_energy", "composition"]);
        shift["shift_groups"][1]["shift_kind"] = json!("compounded");
        shift["shift_groups"][1]["shift_axes"] =
            json!(["physical_regime", "label_distribution"]);
        fs::write(&shift_path, serde_json::to_vec_pretty(&shift).unwrap()).unwrap();

        let (shift_sha256, shift_size_bytes) =
            hash_artifact(&fixture.0, "shift-manifest.json").expect("rehash shift manifest");
        let provenance_path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut provenance: Value =
            serde_json::from_str(&fs::read_to_string(&provenance_path).expect("read provenance"))
                .unwrap();
        provenance["shift_manifest_sha256"] = json!(shift_sha256);
        for artifact in provenance["artifacts"].as_array_mut().unwrap() {
            if artifact["path"] == "shift-manifest.json" {
                artifact["sha256"] = json!(shift_sha256);
                artifact["size_bytes"] = json!(shift_size_bytes);
            }
        }
        fs::write(&provenance_path, serde_json::to_vec_pretty(&provenance).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "artifact_integrity_pass", "{:?}", report.findings);
        assert_eq!(report.provenance_manifest_status, "manifest_consistent");
        assert!(report.checks_completed.contains(
            &"shift_manifest_provenance_and_group_split_integrity".to_owned()
        ));
    }

    #[test]
    fn rejects_single_factor_group_with_multiple_shift_axes() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        fixture.write_shift_manifest(false);

        let shift_path = fixture.0.join("shift-manifest.json");
        let mut shift: Value =
            serde_json::from_str(&fs::read_to_string(&shift_path).expect("read shift")).unwrap();
        shift["shift_groups"][1]["shift_axes"] =
            json!(["physical_regime", "label_distribution"]);
        fs::write(&shift_path, serde_json::to_vec_pretty(&shift).unwrap()).unwrap();

        let (shift_sha256, shift_size_bytes) =
            hash_artifact(&fixture.0, "shift-manifest.json").expect("rehash shift manifest");
        let provenance_path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut provenance: Value =
            serde_json::from_str(&fs::read_to_string(&provenance_path).expect("read provenance"))
                .unwrap();
        provenance["shift_manifest_sha256"] = json!(shift_sha256);
        for artifact in provenance["artifacts"].as_array_mut().unwrap() {
            if artifact["path"] == "shift-manifest.json" {
                artifact["sha256"] = json!(shift_sha256);
                artifact["size_bytes"] = json!(shift_size_bytes);
            }
        }
        fs::write(&provenance_path, serde_json::to_vec_pretty(&provenance).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "shift_manifest_axis_kind_mismatch"));
    }

    #[test]
    fn rejects_in_distribution_group_with_physical_perturbation_families() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        fixture.write_shift_manifest(false);

        let shift_path = fixture.0.join("shift-manifest.json");
        let mut shift: Value =
            serde_json::from_str(&fs::read_to_string(&shift_path).expect("read shift")).unwrap();
        shift["shift_groups"][0]["shift_families"] = json!(["composition"]);
        fs::write(&shift_path, serde_json::to_vec_pretty(&shift).unwrap()).unwrap();

        let (shift_sha256, shift_size_bytes) =
            hash_artifact(&fixture.0, "shift-manifest.json").expect("rehash shift manifest");
        let provenance_path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut provenance: Value =
            serde_json::from_str(&fs::read_to_string(&provenance_path).expect("read provenance"))
                .unwrap();
        provenance["shift_manifest_sha256"] = json!(shift_sha256);
        for artifact in provenance["artifacts"].as_array_mut().unwrap() {
            if artifact["path"] == "shift-manifest.json" {
                artifact["sha256"] = json!(shift_sha256);
                artifact["size_bytes"] = json!(shift_size_bytes);
            }
        }
        fs::write(&provenance_path, serde_json::to_vec_pretty(&provenance).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report.findings.iter().any(|finding| {
            finding.code == "shift_manifest_baseline_family_mismatch"
        }));
    }

    #[test]
    fn rejects_in_distribution_group_with_nonbaseline_severity() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        fixture.write_shift_manifest(false);

        let shift_path = fixture.0.join("shift-manifest.json");
        let mut shift: Value =
            serde_json::from_str(&fs::read_to_string(&shift_path).expect("read shift")).unwrap();
        shift["shift_groups"][1]["shift_kind"] = json!("in_distribution");
        fs::write(&shift_path, serde_json::to_vec_pretty(&shift).unwrap()).unwrap();

        let (shift_sha256, shift_size_bytes) =
            hash_artifact(&fixture.0, "shift-manifest.json").expect("rehash shift manifest");
        let provenance_path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut provenance: Value =
            serde_json::from_str(&fs::read_to_string(&provenance_path).expect("read provenance"))
                .unwrap();
        provenance["shift_manifest_sha256"] = json!(shift_sha256);
        for artifact in provenance["artifacts"].as_array_mut().unwrap() {
            if artifact["path"] == "shift-manifest.json" {
                artifact["sha256"] = json!(shift_sha256);
                artifact["size_bytes"] = json!(shift_size_bytes);
            }
        }
        fs::write(&provenance_path, serde_json::to_vec_pretty(&provenance).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report.findings.iter().any(|finding| {
            finding.code == "shift_manifest_kind_severity_mismatch"
        }));
    }

    #[test]
    fn rejects_sample_id_leakage_across_train_and_test() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        fixture.write_shift_manifest(false);

        let shift_path = fixture.0.join("shift-manifest.json");
        let mut shift: Value =
            serde_json::from_str(&fs::read_to_string(&shift_path).expect("read shift")).unwrap();
        shift["shift_groups"][2]["sample_ids"] = json!(["sample-train-001"]);
        fs::write(&shift_path, serde_json::to_vec_pretty(&shift).unwrap()).unwrap();

        let (shift_sha256, shift_size_bytes) =
            hash_artifact(&fixture.0, "shift-manifest.json").expect("rehash shift manifest");
        let provenance_path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut provenance: Value =
            serde_json::from_str(&fs::read_to_string(&provenance_path).expect("read provenance"))
                .unwrap();
        provenance["shift_manifest_sha256"] = json!(shift_sha256);
        for artifact in provenance["artifacts"].as_array_mut().unwrap() {
            if artifact["path"] == "shift-manifest.json" {
                artifact["sha256"] = json!(shift_sha256);
                artifact["size_bytes"] = json!(shift_size_bytes);
            }
        }
        fs::write(&provenance_path, serde_json::to_vec_pretty(&provenance).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "shift_manifest_sample_leakage"));
    }

    #[test]
    fn accepts_consistent_provenance_manifest_and_artifact_hashes() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "artifact_integrity_pass", "{:?}", report.findings);
        assert_eq!(report.provenance_manifest_status, "manifest_consistent");
        assert!(report
            .checks_completed
            .contains(&"provenance_manifest_artifact_hashes_and_accounting_reconcile".to_owned()));
    }

    #[test]
    fn rejects_tampered_artifact_when_provenance_manifest_is_present() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        let config = fixture.0.join(".hydra/config.yaml");
        fs::write(&config, "experiment:\\n  num_episodes: 999\\n").unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert_eq!(report.provenance_manifest_status, "manifest_invalid");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "provenance_artifact_hash_mismatch"));
    }

    #[test]
    fn rejects_local_candidate_rejection_misreported_as_external_call() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();

        let ledger_path = fixture.0.join("oracle-call-ledger.jsonl");
        let mut records = fs::read_to_string(&ledger_path)
            .expect("read oracle ledger")
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("parse ledger record"))
            .collect::<Vec<_>>();
        let rejected = records
            .iter_mut()
            .find(|record| record["status"] == "invalid_candidate")
            .expect("fixture has local rejection");
        rejected["external_invocation"] = json!(true);
        let ledger_content = records
            .iter()
            .map(|record| serde_json::to_string(record).expect("serialize ledger record"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&ledger_path, format!("{ledger_content}\n")).expect("rewrite ledger");

        let (ledger_sha256, ledger_size) =
            hash_artifact(&fixture.0, "oracle-call-ledger.jsonl").expect("hash mutated ledger");
        let manifest_path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut manifest: Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read provenance"))
                .unwrap();
        manifest["oracle_call_ledger_sha256"] = json!(ledger_sha256);
        for artifact in manifest["artifacts"].as_array_mut().unwrap() {
            if artifact["path"] == "oracle-call-ledger.jsonl" {
                artifact["sha256"] = json!(ledger_sha256);
                artifact["size_bytes"] = json!(ledger_size);
            }
        }
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap())
            .expect("update provenance hashes");

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert_eq!(report.provenance_manifest_status, "manifest_invalid");
        assert!(report.findings.iter().any(|finding| {
            finding.code == "oracle_call_ledger_local_rejection_mismatch"
        }));
    }

    #[test]
    fn rejects_omitted_call_even_when_ledger_hashes_are_updated() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();

        let ledger_path = fixture.0.join("oracle-call-ledger.jsonl");
        let mut lines = fs::read_to_string(&ledger_path)
            .expect("read oracle ledger")
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        lines.pop().expect("at least one call record");
        fs::write(&ledger_path, format!("{}\n", lines.join("\n")))
            .expect("write shortened ledger");

        let (ledger_sha256, ledger_size) =
            hash_artifact(&fixture.0, "oracle-call-ledger.jsonl").expect("hash shortened ledger");
        let manifest_path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut manifest: Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read provenance"))
                .unwrap();
        manifest["oracle_call_ledger_sha256"] = json!(ledger_sha256);
        for artifact in manifest["artifacts"].as_array_mut().unwrap() {
            if artifact["path"] == "oracle-call-ledger.jsonl" {
                artifact["sha256"] = json!(ledger_sha256);
                artifact["size_bytes"] = json!(ledger_size);
            }
        }
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap())
            .expect("update provenance hashes");

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert_eq!(report.provenance_manifest_status, "manifest_invalid");
        assert!(report.findings.iter().any(|finding| {
            finding.code == "oracle_call_ledger_counter_mismatch"
        }));
    }

    #[test]
    fn rejects_manifest_artifact_path_traversal() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        let path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut manifest: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read provenance")).unwrap();
        manifest["artifacts"][0]["path"] = json!("../outside.json");
        fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert_eq!(report.provenance_manifest_status, "manifest_invalid");
        assert!(report.findings.iter().any(|finding| {
            finding.code == "provenance_artifact_invalid_path_or_unreadable"
        }));
    }

    #[test]
    fn rejects_inconsistent_external_oracle_call_accounting() {
        let fixture = Fixture::new();
        fixture.write_provenance_manifest();
        let path = fixture.0.join(PROVENANCE_MANIFEST_NAME);
        let mut manifest: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read provenance")).unwrap();
        manifest["oracle_accounting"]["external_calls"] = json!(99);
        fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report.findings.iter().any(|finding| {
            finding.code == "oracle_accounting_request_partition_mismatch"
        }));
    }

    #[test]
    fn rejects_a_gap_in_episode_indices() {
        let fixture = Fixture::new();
        fs::rename(
            fixture.0.join("trajectories/episode_001.json"),
            fixture.0.join("trajectories/episode_002.json"),
        )
        .unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "trajectory_sequence_gap"));
    }
}
