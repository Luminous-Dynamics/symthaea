// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact-byte and semantic-contract verifier for PITB benchmark target audits.
//!
//! This does not prove that a formal statement captures its intended mathematics.
//! It enforces pinned-source/elaboration/axiom/probe/reviewer receipt joins and
//! keeps "compiled" distinct from "semantically approved".

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};

const VERSION: &str = "pitb.benchmark-integrity-receipt.v0";

#[derive(Debug, Clone)]
pub struct IntegritySummary {
    pub receipt_sha256: String,
    pub cohort_id: String,
    pub snapshot_id: String,
    pub task_id: String,
    pub source_task_id: String,
    pub statement_sha256: String,
    pub integrity_status: String,
    pub repository_remote: String,
    pub repository_branch: String,
    pub repository_commit: String,
    pub source_path: String,
    pub start_line: u64,
    pub start_column: u64,
    pub end_line: u64,
    pub end_column: u64,
    pub lean_toolchain: String,
    pub lake_manifest_sha256: String,
    pub used_axiom_count: usize,
    pub allowed_axiom_count: usize,
    pub verified_evidence_artifact_count: usize,
}

fn field<'a>(value: &'a Value, pointer: &str) -> Result<&'a Value> {
    value
        .pointer(pointer)
        .with_context(|| format!("missing integrity receipt field {pointer}"))
}

fn string<'a>(value: &'a Value, pointer: &str) -> Result<&'a str> {
    field(value, pointer)?
        .as_str()
        .with_context(|| format!("integrity receipt field {pointer} must be a string"))
}

fn integer(value: &Value, pointer: &str) -> Result<u64> {
    field(value, pointer)?
        .as_u64()
        .with_context(|| format!("integrity receipt field {pointer} must be a nonnegative integer"))
}

fn boolean(value: &Value, pointer: &str) -> Result<bool> {
    field(value, pointer)?
        .as_bool()
        .with_context(|| format!("integrity receipt field {pointer} must be boolean"))
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn check_digest(value: &str, label: &str, length: usize) -> Result<()> {
    if !lower_hex(value, length) {
        bail!("{label} must be {length} lowercase hexadecimal characters");
    }
    Ok(())
}

fn check_relative_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || value.contains('\\')
        || value.contains('\0')
        || value.contains(':')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || path.components().any(|part| {
            matches!(
                part,
                std::path::Component::ParentDir
                    | std::path::Component::CurDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        bail!("target.source_path must be a safe repository-relative path");
    }
    Ok(())
}

fn string_list(value: &Value, pointer: &str) -> Result<Vec<String>> {
    let rows = field(value, pointer)?
        .as_array()
        .with_context(|| format!("{pointer} must be an array"))?;
    let mut result = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let name = row
            .as_str()
            .with_context(|| format!("{pointer}[{index}] must be a string"))?;
        if name.trim().is_empty() {
            bail!("{pointer}[{index}] cannot be empty");
        }
        result.push(name.to_owned());
    }
    if result.windows(2).any(|pair| pair[0] >= pair[1]) {
        bail!("{pointer} must be sorted and unique");
    }
    Ok(result)
}

fn digest_string_list(rows: &[String]) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(rows)?)))
}

fn optional_receipt_digest(value: &Value, pointer: &str, required: bool) -> Result<Option<String>> {
    match field(value, pointer)? {
        Value::Null if !required => Ok(None),
        Value::String(digest) => {
            check_digest(digest, pointer, 64)?;
            Ok(Some(digest.clone()))
        }
        Value::Null => bail!("{pointer} is required for this audit status"),
        _ => bail!("{pointer} must be a SHA-256 string or null"),
    }
}

fn expected_status(value: &Value) -> Result<&'static str> {
    let elaboration_status = string(value, "/elaboration/status")?;
    if !matches!(elaboration_status, "not_run" | "passed" | "failed" | "unknown") {
        bail!("unknown target elaboration status");
    }
    let checked_statement = string(value, "/elaboration/checked_statement_sha256")?;
    let statement = string(value, "/target/statement_sha256")?;
    let metavariables = integer(value, "/elaboration/metavariable_count")?;
    let unresolved_universes = integer(value, "/elaboration/unresolved_universe_count")?;
    let contains_sorry = boolean(value, "/elaboration/contains_sorry_ax")?;
    let unexpected_axioms = integer(value, "/axioms/unexpected_axioms")?;
    let counterexample = string(value, "/counterexample_audit/status")?;
    if !matches!(counterexample, "not_run" | "not_found" | "counterexample_found" | "inconclusive" | "not_applicable") {
        bail!("unknown counterexample audit status");
    }
    let vacuity = string(value, "/vacuity_audit/status")?;
    if !matches!(vacuity, "not_run" | "no_counterexample_found" | "counterexample_found" | "inconclusive" | "not_applicable") {
        bail!("unknown vacuity audit status");
    }
    let semantic = string(value, "/semantic_review/status")?;
    if !matches!(semantic, "not_reviewed" | "approved" | "rejected" | "needs_revision") {
        bail!("unknown semantic review status");
    }
    if !matches!(
        string(value, "/semantic_review/review_scope")?,
        "statement_to_source" | "informal_to_formal" | "domain_expert_review" | "combined"
    ) {
        bail!("unknown semantic review scope");
    }

    let hard_failure = elaboration_status == "failed"
        || (elaboration_status == "passed" && checked_statement != statement)
        || metavariables > 0
        || unresolved_universes > 0
        || contains_sorry
        || unexpected_axioms > 0
        || counterexample == "counterexample_found"
        || string(value, "/vacuity_audit/status")? == "counterexample_found"
        || semantic == "rejected";
    if hard_failure {
        return Ok("ineligible");
    }

    let complete_elaboration = elaboration_status == "passed" && checked_statement == statement;
    let complete_probes = counterexample == "not_found"
        && optional_receipt_digest(value, "/counterexample_audit/receipt_sha256", true)?.is_some()
        && string(value, "/vacuity_audit/status")? == "no_counterexample_found"
        && optional_receipt_digest(value, "/vacuity_audit/receipt_sha256", true)?.is_some();
    let review_scope = string(value, "/semantic_review/review_scope")?;
    let intended_statement_present =
        !field(value, "/target/expected_informal_statement_sha256")?.is_null();
    let review_covers_intent = match review_scope {
        "informal_to_formal" | "combined" => intended_statement_present,
        "domain_expert_review" => true,
        _ => false,
    };
    let approved_review = semantic == "approved"
        && review_covers_intent
        && !string(value, "/semantic_review/reviewer_identity")?.trim().is_empty()
        && optional_receipt_digest(value, "/semantic_review/reviewer_attestation_sha256", true)?.is_some();
    let no_sorries_axioms_and_metavariables =
        !contains_sorry && metavariables == 0 && unresolved_universes == 0 && unexpected_axioms == 0;

    if complete_elaboration && complete_probes && approved_review && no_sorries_axioms_and_metavariables {
        Ok("eligible")
    } else {
        Ok("pending")
    }
}

pub fn validate_value(value: &Value, receipt_sha256: String) -> Result<IntegritySummary> {
    if string(value, "/schema_version")? != VERSION {
        bail!("unsupported benchmark integrity receipt schema version");
    }
    let evidence_class = string(value, "/evidence_class")?;
    if !matches!(evidence_class, "synthetic_contract_fixture" | "measured_audit") {
        bail!("unknown benchmark integrity evidence class");
    }

    let task_id = string(value, "/dataset/task_id")?.to_owned();
    let source_task_id = string(value, "/dataset/source_task_id")?.to_owned();
    let statement_sha256 = string(value, "/target/statement_sha256")?.to_owned();
    if source_task_id.trim().is_empty() {
        bail!("dataset.source_task_id cannot be empty");
    }
    if evidence_class == "measured_audit"
        && string(value, "/dataset/cohort_id")? != "SorryDB_2601"
    {
        bail!("measured SorryDB integrity receipts must identify the SorryDB_2601 cohort");
    }
    if evidence_class == "measured_audit"
        && hex::encode(Sha256::digest(source_task_id.as_bytes())) != task_id
    {
        bail!("dataset.task_id must be SHA-256(source_task_id) for measured receipts");
    }
    if string(value, "/dataset/cohort_id")?.trim().is_empty()
        || string(value, "/dataset/snapshot_id")?.trim().is_empty()
        || string(value, "/target/repository_remote")?.trim().is_empty()
        || string(value, "/target/repository_branch")?.trim().is_empty()
    {
        bail!("dataset cohort/snapshot and target repository remote cannot be empty");
    }
    if let Some(expected) = field(value, "/target/expected_informal_statement_sha256")?.as_str() {
        check_digest(expected, "target.expected_informal_statement_sha256", 64)?;
    } else if !field(value, "/target/expected_informal_statement_sha256")?.is_null() {
        bail!("target.expected_informal_statement_sha256 must be a SHA-256 string or null");
    }
    check_digest(&task_id, "dataset.task_id", 64)?;
    check_digest(&statement_sha256, "target.statement_sha256", 64)?;
    let task_manifest = string(value, "/dataset/task_manifest_sha256")?;
    check_digest(task_manifest, "dataset.task_manifest_sha256", 64)?;
    let repository_commit = string(value, "/target/repository_commit")?;
    check_digest(repository_commit, "target.repository_commit", 40)?;
    check_relative_path(string(value, "/target/source_path")?)?;
    for pointer in [
        "/target/source_file_sha256",
        "/target/statement_sha256",
        "/environment/lean_binary_sha256",
        "/environment/lake_manifest_sha256",
        "/environment/root_module_sha256",
        "/environment/environment_sha256",
        "/elaboration/checked_statement_sha256",
        "/axioms/used_axioms_manifest_sha256",
        "/axioms/allowlist_manifest_sha256",
    ] {
        check_digest(string(value, pointer)?, pointer, 64)?;
    }
    let start_line = integer(value, "/target/start_line")?;
    let end_line = integer(value, "/target/end_line")?;
    let start_column = integer(value, "/target/start_column")?;
    let end_column = integer(value, "/target/end_column")?;
    if start_line > end_line || (start_line == end_line && start_column > end_column) {
        bail!("target source coordinate range is reversed");
    }
    if string(value, "/environment/lean_toolchain")?.trim().is_empty() {
        bail!("Lean toolchain cannot be empty");
    }

    let used_axioms = string_list(value, "/axioms/used_axioms")?;
    let allowed_axioms = string_list(value, "/axioms/allowed_axioms")?;
    if digest_string_list(&used_axioms)? != string(value, "/axioms/used_axioms_manifest_sha256")? {
        bail!("used-axiom manifest digest does not match the sorted axiom list");
    }
    if digest_string_list(&allowed_axioms)? != string(value, "/axioms/allowlist_manifest_sha256")? {
        bail!("axiom allowlist digest does not match the sorted allowlist");
    }
    let allowlist: BTreeSet<&str> = allowed_axioms.iter().map(String::as_str).collect();
    let unexpected: Vec<&str> = used_axioms
        .iter()
        .map(String::as_str)
        .filter(|axiom| !allowlist.contains(axiom))
        .collect();
    if unexpected.len() as u64 != integer(value, "/axioms/unexpected_axioms")? {
        bail!("unexpected_axioms count does not match used-axiom/allowlist difference");
    }

    for pointer in [
        "/counterexample_audit/receipt_sha256",
        "/counterexample_audit/notes_sha256",
        "/vacuity_audit/receipt_sha256",
        "/vacuity_audit/notes_sha256",
        "/semantic_review/reviewer_attestation_sha256",
        "/elaboration/receipt_sha256",
    ] {
        optional_receipt_digest(
            value,
            pointer,
            string(value, "/evidence_class")? == "measured_audit"
                && match pointer {
                    "/semantic_review/reviewer_attestation_sha256" =>
                        string(value, "/semantic_review/status")? == "approved",
                    "/elaboration/receipt_sha256" =>
                        string(value, "/elaboration/status")? == "passed",
                    "/counterexample_audit/receipt_sha256" =>
                        matches!(string(value, "/counterexample_audit/status")?, "not_found" | "counterexample_found"),
                    "/vacuity_audit/receipt_sha256" =>
                        matches!(string(value, "/vacuity_audit/status")?, "no_counterexample_found" | "counterexample_found"),
                    _ => false,
                },
        )?;
    }

    if evidence_class == "synthetic_contract_fixture" {
        if string(value, "/integrity_status")? != "not_attempted"
            || string(value, "/elaboration/status")? != "not_run"
            || string(value, "/counterexample_audit/status")? != "not_run"
            || string(value, "/vacuity_audit/status")? != "not_run"
            || string(value, "/semantic_review/status")? != "not_reviewed"
            || optional_receipt_digest(value, "/elaboration/receipt_sha256", false)?.is_some()
            || optional_receipt_digest(value, "/semantic_review/reviewer_attestation_sha256", false)?.is_some()
        {
            bail!("synthetic benchmark integrity fixture contains attempted audit or approval claims");
        }
    } else {
        if string(value, "/counterexample_audit/method_id")?.trim().is_empty()
            || string(value, "/vacuity_audit/method_id")?.trim().is_empty()
        {
            bail!("measured counterexample/vacuity audit must identify its method");
        }
    }

    let expected = if evidence_class == "synthetic_contract_fixture" {
        "not_attempted"
    } else {
        expected_status(value)?
    };
    if string(value, "/integrity_status")? != expected {
        bail!(
            "integrity_status claim {:?} differs from independently derived status {:?}",
            string(value, "/integrity_status")?,
            expected
        );
    }

    Ok(IntegritySummary {
        receipt_sha256,
        cohort_id: string(value, "/dataset/cohort_id")?.to_owned(),
        snapshot_id: string(value, "/dataset/snapshot_id")?.to_owned(),
        task_id,
        source_task_id,
        statement_sha256,
        integrity_status: expected.to_owned(),
        repository_remote: string(value, "/target/repository_remote")?.to_owned(),
        repository_branch: string(value, "/target/repository_branch")?.to_owned(),
        repository_commit: repository_commit.to_owned(),
        source_path: string(value, "/target/source_path")?.to_owned(),
        start_line,
        start_column,
        end_line,
        end_column,
        lean_toolchain: string(value, "/environment/lean_toolchain")?.to_owned(),
        lake_manifest_sha256: string(value, "/environment/lake_manifest_sha256")?.to_owned(),
        used_axiom_count: used_axioms.len(),
        allowed_axiom_count: allowed_axioms.len(),
        verified_evidence_artifact_count: 0,
    })
}


#[derive(Debug)]
struct EvidenceArtifact {
    path: String,
    sha256: String,
}

fn evidence_artifact(value: &Value, pointer: &str) -> Result<Option<EvidenceArtifact>> {
    match field(value, pointer)? {
        Value::Null => Ok(None),
        artifact => {
            let path = artifact
                .get("path")
                .and_then(Value::as_str)
                .with_context(|| format!("{pointer}.path must be a string"))?;
            let sha256 = artifact
                .get("sha256")
                .and_then(Value::as_str)
                .with_context(|| format!("{pointer}.sha256 must be a string"))?;
            check_relative_path(path)?;
            check_digest(sha256, &format!("{pointer}.sha256"), 64)?;
            Ok(Some(EvidenceArtifact {
                path: path.to_owned(),
                sha256: sha256.to_owned(),
            }))
        }
    }
}

fn verify_artifact_bytes(
    root: &Path,
    root_canonical: &Path,
    artifact: &EvidenceArtifact,
    expected_digest: &str,
    label: &str,
) -> Result<Vec<u8>> {
    if artifact.sha256 != expected_digest {
        bail!("{label} artifact digest differs from the corresponding receipt field");
    }
    let candidate = root.join(&artifact.path);
    let canonical = fs::canonicalize(&candidate)
        .with_context(|| format!("resolve evidence artifact {}", candidate.display()))?;
    if !canonical.starts_with(root_canonical) {
        bail!("{label} artifact escapes the evidence root through a symlink or path alias");
    }
    let metadata = fs::metadata(&canonical)
        .with_context(|| format!("stat evidence artifact {}", canonical.display()))?;
    if !metadata.is_file() {
        bail!("{label} evidence artifact is not a regular file");
    }
    let bytes = fs::read(&canonical)
        .with_context(|| format!("read evidence artifact {}", canonical.display()))?;
    let actual = hex::encode(Sha256::digest(&bytes));
    if actual != artifact.sha256 {
        bail!("{label} evidence artifact bytes do not match the registered SHA-256");
    }
    Ok(bytes)
}

fn verify_axiom_artifact(bytes: &[u8], expected: &[String], label: &str) -> Result<()> {
    let parsed: Vec<String> = serde_json::from_slice(bytes)
        .with_context(|| format!("{label} must contain a JSON string array"))?;
    if parsed != expected || serde_json::to_vec(&parsed)? != bytes {
        bail!("{label} bytes must be the exact compact JSON serialization of the sorted manifest");
    }
    Ok(())
}

fn verify_evidence_artifacts(value: &Value, evidence_root: &Path) -> Result<usize> {
    let evidence_class = string(value, "/evidence_class")?;
    if evidence_class == "synthetic_contract_fixture" {
        if !field(value, "/evidence_artifacts")?.is_null() {
            bail!("synthetic integrity fixture cannot carry measured evidence artifacts");
        }
        return Ok(0);
    }

    let root_canonical = fs::canonicalize(evidence_root)
        .with_context(|| format!("resolve evidence root {}", evidence_root.display()))?;
    if !root_canonical.is_dir() {
        bail!("evidence root must be a directory");
    }

    let mut artifacts = BTreeMap::new();
    for (name, pointer) in [
        ("source_file", "/evidence_artifacts/source_file"),
        ("informal_statement", "/evidence_artifacts/informal_statement"),
        ("lean_binary", "/evidence_artifacts/lean_binary"),
        ("lake_manifest", "/evidence_artifacts/lake_manifest"),
        ("root_module", "/evidence_artifacts/root_module"),
        ("elaboration_receipt", "/evidence_artifacts/elaboration_receipt"),
        ("used_axioms_manifest", "/evidence_artifacts/used_axioms_manifest"),
        ("allowlist_manifest", "/evidence_artifacts/allowlist_manifest"),
        ("counterexample_receipt", "/evidence_artifacts/counterexample_receipt"),
        ("vacuity_receipt", "/evidence_artifacts/vacuity_receipt"),
        ("reviewer_attestation", "/evidence_artifacts/reviewer_attestation"),
    ] {
        let item = evidence_artifact(value, pointer)?;
        if let Some(item) = item {
            artifacts.insert(name, item);
        }
    }

    if string(value, "/integrity_status")? == "eligible"
        && artifacts.len() != 11
    {
        bail!("eligible measured integrity receipt must include all eleven evidence artifacts");
    }
    if artifacts.is_empty() {
        return Ok(0);
    }

    let target_source_sha = string(value, "/target/source_file_sha256")?;
    let informal_sha = string(value, "/target/expected_informal_statement_sha256")?;
    let lean_sha = string(value, "/environment/lean_binary_sha256")?;
    let lake_sha = string(value, "/environment/lake_manifest_sha256")?;
    let root_sha = string(value, "/environment/root_module_sha256")?;
    let elaboration_sha = string(value, "/elaboration/receipt_sha256")?;
    let used_axioms_sha = string(value, "/axioms/used_axioms_manifest_sha256")?;
    let allowlist_sha = string(value, "/axioms/allowlist_manifest_sha256")?;
    let counterexample_sha = string(value, "/counterexample_audit/receipt_sha256")?;
    let vacuity_sha = string(value, "/vacuity_audit/receipt_sha256")?;
    let reviewer_sha = string(value, "/semantic_review/reviewer_attestation_sha256")?;

    let expected = [
        ("source_file", target_source_sha),
        ("informal_statement", informal_sha),
        ("lean_binary", lean_sha),
        ("lake_manifest", lake_sha),
        ("root_module", root_sha),
        ("elaboration_receipt", elaboration_sha),
        ("used_axioms_manifest", used_axioms_sha),
        ("allowlist_manifest", allowlist_sha),
        ("counterexample_receipt", counterexample_sha),
        ("vacuity_receipt", vacuity_sha),
        ("reviewer_attestation", reviewer_sha),
    ];
    let used_axioms = string_list(value, "/axioms/used_axioms")?;
    let allowlist = string_list(value, "/axioms/allowed_axioms")?;
    for (name, expected_sha) in expected {
        let Some(artifact) = artifacts.get(name) else {
            if string(value, "/integrity_status")? == "eligible" {
                bail!("eligible measured receipt is missing evidence artifact {name}");
            }
            continue;
        };
        let bytes = verify_artifact_bytes(
            evidence_root,
            &root_canonical,
            artifact,
            expected_sha,
            name,
        )?;
        match name {
            "used_axioms_manifest" => verify_axiom_artifact(&bytes, &used_axioms, name)?,
            "allowlist_manifest" => verify_axiom_artifact(&bytes, &allowlist, name)?,
            _ => {}
        }
    }
    Ok(artifacts.len())
}

pub fn verify_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<IntegritySummary> {
    let actual_sha256 = hex::encode(Sha256::digest(bytes));
    if let Some(expected) = expected_sha256 {
        check_digest(expected, "expected receipt SHA-256", 64)?;
        if expected != actual_sha256 {
            bail!("benchmark integrity receipt bytes differ from the expected SHA-256");
        }
    }
    let value: Value = serde_json::from_slice(bytes).context("parse benchmark integrity receipt")?;
    validate_value(&value, actual_sha256)
}

pub fn verify_file(path: &Path, expected_sha256: Option<&str>) -> Result<IntegritySummary> {
    let bytes = fs::read(path).with_context(|| format!("read integrity receipt {}", path.display()))?;
    verify_bytes(&bytes, expected_sha256)
}

pub fn verify_bytes_with_evidence_root(
    bytes: &[u8],
    expected_sha256: Option<&str>,
    evidence_root: &Path,
) -> Result<IntegritySummary> {
    let mut summary = verify_bytes(bytes, expected_sha256)?;
    let value: Value =
        serde_json::from_slice(bytes).context("parse benchmark integrity receipt for evidence bundle")?;
    summary.verified_evidence_artifact_count = verify_evidence_artifacts(&value, evidence_root)?;
    if summary.integrity_status == "eligible" && summary.verified_evidence_artifact_count != 11 {
        bail!("eligible integrity receipt did not verify all eleven evidence artifacts");
    }
    Ok(summary)
}

pub fn verify_file_with_evidence_root(
    path: &Path,
    expected_sha256: Option<&str>,
    evidence_root: &Path,
) -> Result<IntegritySummary> {
    let bytes = fs::read(path).with_context(|| format!("read integrity receipt {}", path.display()))?;
    verify_bytes_with_evidence_root(&bytes, expected_sha256, evidence_root)
}
