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
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
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
    pub reviewer_signature_verified: bool,
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
    let _ = optional_digest_field(value, "/reviewer_key_registry_sha256")?;
    let repository_commit = string(value, "/target/repository_commit")?;
    check_digest(repository_commit, "target.repository_commit", 40)?;
    check_relative_path(string(value, "/target/source_path")?)?;
    for pointer in [
        "/target/source_file_sha256",
        "/target/source_span_sha256",
        "/reviewer_key_registry_sha256",
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
        reviewer_signature_verified: false,
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

fn optional_digest_field<'a>(value: &'a Value, pointer: &str) -> Result<Option<&'a str>> {
    match field(value, pointer)? {
        Value::Null => Ok(None),
        Value::String(digest) => {
            check_digest(digest, pointer, 64)?;
            Ok(Some(digest))
        }
        _ => bail!("{pointer} must be a SHA-256 string or null"),
    }
}

fn source_span_sha256(
    source_bytes: &[u8],
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
) -> Result<String> {
    let source = std::str::from_utf8(source_bytes).context("source file artifact must be UTF-8")?;
    if start_line == 0 || end_line < start_line {
        bail!("source span has invalid line coordinates");
    }
    let lines: Vec<&str> = source.lines().collect();
    let first_index = usize::try_from(start_line - 1).context("start line does not fit usize")?;
    let last_index = usize::try_from(end_line - 1).context("end line does not fit usize")?;
    let first = *lines.get(first_index).context("source span starts beyond end of file")?;
    let last = *lines.get(last_index).context("source span ends beyond end of file")?;
    let first_chars: Vec<char> = first.chars().collect();
    let last_chars: Vec<char> = last.chars().collect();
    let start = usize::try_from(start_column).context("start column does not fit usize")?;
    let end = usize::try_from(end_column).context("end column does not fit usize")?;
    if start > first_chars.len() || end > last_chars.len()
        || (start_line == end_line && end < start)
    {
        bail!("source span columns exceed source line bounds or are reversed");
    }

    let mut selected = String::new();
    if first_index == last_index {
        selected.extend(first_chars[start..end].iter().copied());
    } else {
        selected.extend(first_chars[start..].iter().copied());
        for line in lines.iter().take(last_index).skip(first_index + 1) {
            selected.push('\n');
            selected.push_str(line);
        }
        selected.push('\n');
        selected.extend(last_chars[..end].iter().copied());
    }
    Ok(hex::encode(Sha256::digest(selected.as_bytes())))
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

    let target_source_sha = Some(string(value, "/target/source_file_sha256")?);
    let informal_sha = optional_digest_field(value, "/target/expected_informal_statement_sha256")?;
    let lean_sha = Some(string(value, "/environment/lean_binary_sha256")?);
    let lake_sha = Some(string(value, "/environment/lake_manifest_sha256")?);
    let root_sha = Some(string(value, "/environment/root_module_sha256")?);
    let elaboration_sha = optional_digest_field(value, "/elaboration/receipt_sha256")?;
    let used_axioms_sha = Some(string(value, "/axioms/used_axioms_manifest_sha256")?);
    let allowlist_sha = Some(string(value, "/axioms/allowlist_manifest_sha256")?);
    let counterexample_sha = optional_digest_field(value, "/counterexample_audit/receipt_sha256")?;
    let vacuity_sha = optional_digest_field(value, "/vacuity_audit/receipt_sha256")?;
    let reviewer_sha = optional_digest_field(value, "/semantic_review/reviewer_attestation_sha256")?;

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
        let Some(expected_sha) = expected_sha else {
            bail!("{name} artifact is present but its corresponding receipt digest is null");
        };
        let bytes = verify_artifact_bytes(
            evidence_root,
            &root_canonical,
            artifact,
            expected_sha,
            name,
        )?;
        match name {
            "source_file" => {
                let actual_span = source_span_sha256(
                    &bytes,
                    integer(value, "/target/start_line")?,
                    integer(value, "/target/start_column")?,
                    integer(value, "/target/end_line")?,
                    integer(value, "/target/end_column")?,
                )?;
                if actual_span != string(value, "/target/source_span_sha256")? {
                    bail!("source coordinates do not select the registered source-span bytes");
                }
            }
            "used_axioms_manifest" => verify_axiom_artifact(&bytes, &used_axioms, name)?,
            "allowlist_manifest" => verify_axiom_artifact(&bytes, &allowlist, name)?,
            _ => {}
        }
    }
    Ok(artifacts.len())
}


fn read_verified_evidence_artifact(
    value: &Value,
    evidence_root: &Path,
    pointer: &str,
    expected_sha256: &str,
) -> Result<Vec<u8>> {
    let artifact = evidence_artifact(value, pointer)?
        .context("required evidence artifact is missing")?;
    let root_canonical = fs::canonicalize(evidence_root)
        .with_context(|| format!("resolve evidence root {}", evidence_root.display()))?;
    verify_artifact_bytes(
        evidence_root,
        &root_canonical,
        &artifact,
        expected_sha256,
        pointer,
    )
}

fn decode_hex_array<const N: usize>(value: &str, label: &str) -> Result<[u8; N]> {
    let decoded = hex::decode(value).with_context(|| format!("decode {label} hexadecimal bytes"))?;
    decoded
        .try_into()
        .map_err(|_| anyhow::anyhow!("{label} must decode to exactly {N} bytes"))
}

fn verify_reviewer_signature(
    attestation_bytes: &[u8],
    registry_bytes: &[u8],
    expected_registry_sha256: &str,
    receipt: &Value,
) -> Result<()> {
    check_digest(expected_registry_sha256, "expected reviewer registry SHA-256", 64)?;
    let actual_registry_sha256 = hex::encode(Sha256::digest(registry_bytes));
    if actual_registry_sha256 != expected_registry_sha256 {
        bail!("reviewer key registry bytes do not match the externally pinned SHA-256");
    }
    if string(receipt, "/reviewer_key_registry_sha256")? != expected_registry_sha256 {
        bail!("integrity receipt is bound to a different reviewer key registry");
    }

    let registry: Value =
        serde_json::from_slice(registry_bytes).context("parse reviewer key registry JSON")?;
    let registry_object = registry
        .as_object()
        .context("reviewer key registry must be a JSON object")?;
    let registry_fields: BTreeSet<&str> =
        ["schema_version", "registry_id", "keys"].into_iter().collect();
    if registry_object.len() != registry_fields.len()
        || registry_object.keys().any(|key| !registry_fields.contains(key.as_str()))
    {
        bail!("reviewer key registry contains missing or unknown fields");
    }
    if string(&registry, "/schema_version")? != "pitb.reviewer-key-registry.v0"
        || string(&registry, "/registry_id")?.trim().is_empty()
    {
        bail!("unsupported or unnamed reviewer key registry");
    }
    let registry_keys = field(&registry, "/keys")?
        .as_array()
        .context("reviewer key registry keys must be an array")?;
    if registry_keys.is_empty() {
        bail!("reviewer key registry cannot be empty");
    }

    let attestation: Value =
        serde_json::from_slice(attestation_bytes).context("parse signed reviewer attestation")?;
    let attestation_object = attestation
        .as_object()
        .context("reviewer attestation must be a JSON object")?;
    let allowed_fields: BTreeSet<&str> = [
        "schema_version",
        "reviewer_identity",
        "key_id",
        "audit_id",
        "task_id_sha256",
        "statement_sha256",
        "cohort_id",
        "snapshot_id",
        "repository_commit",
        "source_path",
        "source_file_sha256",
        "source_span_sha256",
        "expected_informal_statement_sha256",
        "review_scope",
        "decision",
        "signature_ed25519_hex",
    ]
    .into_iter()
    .collect();
    if attestation_object.len() != allowed_fields.len()
        || attestation_object.keys().any(|key| !allowed_fields.contains(key.as_str()))
    {
        bail!("reviewer attestation contains missing or unknown fields");
    }
    if string(&attestation, "/schema_version")? != "pitb.reviewer-attestation.v0"
        || string(&attestation, "/decision")? != "approved"
    {
        bail!("reviewer attestation is not an approved v0 attestation");
    }

    for (attestation_path, receipt_path) in [
        ("/audit_id", "/audit_id"),
        ("/task_id_sha256", "/dataset/task_id"),
        ("/statement_sha256", "/target/statement_sha256"),
        ("/cohort_id", "/dataset/cohort_id"),
        ("/snapshot_id", "/dataset/snapshot_id"),
        ("/repository_commit", "/target/repository_commit"),
        ("/source_path", "/target/source_path"),
        ("/source_file_sha256", "/target/source_file_sha256"),
        ("/source_span_sha256", "/target/source_span_sha256"),
        ("/review_scope", "/semantic_review/review_scope"),
    ] {
        if string(&attestation, attestation_path)? != string(receipt, receipt_path)? {
            bail!("reviewer attestation does not bind {receipt_path}");
        }
    }
    if field(&attestation, "/expected_informal_statement_sha256")?
        != field(receipt, "/target/expected_informal_statement_sha256")?
    {
        bail!("reviewer attestation does not bind the intended informal statement digest");
    }
    let reviewer_identity = string(&attestation, "/reviewer_identity")?;
    if reviewer_identity != string(receipt, "/semantic_review/reviewer_identity")? {
        bail!("reviewer attestation identity differs from the integrity receipt");
    }
    if string(receipt, "/semantic_review/status")? != "approved" {
        bail!("reviewer signature cannot approve a receipt whose semantic-review status is not approved");
    }

    let key_id = string(&attestation, "/key_id")?;
    let mut last_key_id: Option<&str> = None;
    let mut matching_key: Option<&Value> = None;
    for key in registry_keys {
        let key_object = key.as_object().context("reviewer registry key entry must be an object")?;
        let key_fields: BTreeSet<&str> =
            ["key_id", "reviewer_identity", "public_key_ed25519_hex", "status"]
                .into_iter()
                .collect();
        if key_object.len() != key_fields.len()
            || key_object.keys().any(|field| !key_fields.contains(field.as_str()))
        {
            bail!("reviewer registry key entry contains missing or unknown fields");
        }
        let current_id = string(key, "/key_id")?;
        if last_key_id.is_some_and(|previous| previous >= current_id) {
            bail!("reviewer registry keys must be sorted and unique by key_id");
        }
        last_key_id = Some(current_id);
        let status = string(key, "/status")?;
        if !matches!(status, "active" | "revoked") {
            bail!("reviewer registry has an unknown key status");
        }
        if current_id == key_id {
            matching_key = Some(key);
        }
    }
    let key = matching_key.context("reviewer key ID is not present in the pinned registry")?;
    if string(key, "/status")? != "active" {
        bail!("reviewer signing key is revoked");
    }
    if string(key, "/reviewer_identity")? != reviewer_identity {
        bail!("registry reviewer identity differs from the signed attestation");
    }

    let public_key_hex = string(key, "/public_key_ed25519_hex")?;
    check_digest(public_key_hex, "Ed25519 public key", 64)?;
    let public_key = decode_hex_array::<32>(public_key_hex, "Ed25519 public key")?;
    let signature_hex = string(&attestation, "/signature_ed25519_hex")?;
    if !lower_hex(signature_hex, 128) {
        bail!("Ed25519 reviewer signature must be 128 lowercase hexadecimal characters");
    }
    let signature_bytes = decode_hex_array::<64>(signature_hex, "Ed25519 reviewer signature")?;
    let verifying_key =
        VerifyingKey::from_bytes(&public_key).context("parse reviewer Ed25519 public key")?;
    let signature = Signature::from_bytes(&signature_bytes);
    let mut signed_payload = attestation;
    signed_payload
        .as_object_mut()
        .context("reviewer attestation must be a JSON object")?
        .remove("signature_ed25519_hex");
    // serde_json::Map uses lexicographically ordered keys here: this is the
    // canonical payload encoding that reviewers sign and verifiers reproduce.
    let canonical_payload =
        serde_json::to_vec(&signed_payload).context("canonicalize reviewer attestation payload")?;
    verifying_key
        .verify(&canonical_payload, &signature)
        .context("reviewer Ed25519 signature verification failed")?;
    Ok(())
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
    let summary = validate_value(&value, actual_sha256)?;
    if summary.integrity_status == "eligible" {
        bail!("eligible receipts require exact evidence artifacts and a reviewer signature from a pinned key registry");
    }
    Ok(summary)
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
    let actual_receipt_sha256 = hex::encode(Sha256::digest(bytes));
    if let Some(expected) = expected_sha256 {
        check_digest(expected, "expected receipt SHA-256", 64)?;
        if expected != actual_receipt_sha256 {
            bail!("benchmark integrity receipt bytes do not match the expected SHA-256");
        }
    }
    let value: Value =
        serde_json::from_slice(bytes).context("parse benchmark integrity receipt for evidence bundle")?;
    let mut summary = validate_value(&value, actual_receipt_sha256)?;
    summary.verified_evidence_artifact_count = verify_evidence_artifacts(&value, evidence_root)?;
    if summary.integrity_status == "eligible" {
        bail!("eligible receipts require an externally pinned reviewer registry and Ed25519 signature verification");
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

#[cfg(test)]
mod source_span_tests {
    use super::*;

    #[test]
    fn source_span_uses_unicode_codepoint_columns() {
        let source = "a ∃ β sorry";
        let expected = hex::encode(Sha256::digest(b"sorry"));
        assert_eq!(source_span_sha256(source.as_bytes(), 1, 6, 1, 11).unwrap(), expected);
    }

    #[test]
    fn source_span_preserves_newline_boundaries_for_multiline_targets() {
        let source = "alpha\nbeta\ngamma";
        let expected = hex::encode(Sha256::digest(b"lpha\nbeta\ngam"));
        assert_eq!(source_span_sha256(source.as_bytes(), 1, 1, 3, 3).unwrap(), expected);
    }

    #[test]
    fn source_span_rejects_out_of_range_and_reversed_coordinates() {
        assert!(source_span_sha256(b"short", 1, 0, 1, 99).is_err());
        assert!(source_span_sha256(b"short", 1, 4, 1, 2).is_err());
        assert!(source_span_sha256(b"short", 2, 0, 1, 1).is_err());
    }
}

pub fn verify_bytes_with_evidence_root_and_reviewer_registry(
    bytes: &[u8],
    expected_receipt_sha256: Option<&str>,
    evidence_root: &Path,
    registry_bytes: &[u8],
    expected_registry_sha256: &str,
) -> Result<IntegritySummary> {
    let actual_receipt_sha256 = hex::encode(Sha256::digest(bytes));
    if let Some(expected) = expected_receipt_sha256 {
        check_digest(expected, "expected receipt SHA-256", 64)?;
        if expected != actual_receipt_sha256 {
            bail!("benchmark integrity receipt bytes do not match the expected SHA-256");
        }
    }
    let value: Value =
        serde_json::from_slice(bytes).context("parse benchmark integrity receipt for signed verification")?;
    let mut summary = validate_value(&value, actual_receipt_sha256)?;
    summary.verified_evidence_artifact_count = verify_evidence_artifacts(&value, evidence_root)?;
    if summary.integrity_status == "eligible" {
        if summary.verified_evidence_artifact_count != 11 {
            bail!("eligible integrity receipt did not verify all eleven evidence artifacts");
        }
        let reviewer_digest = string(&value, "/semantic_review/reviewer_attestation_sha256")?;
        let reviewer_bytes = read_verified_evidence_artifact(
            &value,
            evidence_root,
            "/evidence_artifacts/reviewer_attestation",
            reviewer_digest,
        )?;
        verify_reviewer_signature(
            &reviewer_bytes,
            registry_bytes,
            expected_registry_sha256,
            &value,
        )?;
        summary.reviewer_signature_verified = true;
    }
    Ok(summary)
}

pub fn verify_file_with_evidence_root_and_reviewer_registry(
    receipt_path: &Path,
    expected_receipt_sha256: Option<&str>,
    evidence_root: &Path,
    registry_path: &Path,
    expected_registry_sha256: &str,
) -> Result<IntegritySummary> {
    let receipt_bytes =
        fs::read(receipt_path).with_context(|| format!("read integrity receipt {}", receipt_path.display()))?;
    let registry_bytes =
        fs::read(registry_path).with_context(|| format!("read reviewer registry {}", registry_path.display()))?;
    verify_bytes_with_evidence_root_and_reviewer_registry(
        &receipt_bytes,
        expected_receipt_sha256,
        evidence_root,
        &registry_bytes,
        expected_registry_sha256,
    )
}
