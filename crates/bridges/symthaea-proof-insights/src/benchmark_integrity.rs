// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact-byte and semantic-contract verifier for PITB benchmark target audits.
//!
//! This does not prove that a formal statement captures its intended mathematics.
//! It enforces pinned-source/elaboration/axiom/probe/reviewer receipt joins and
//! keeps "compiled" distinct from "semantically approved".

use std::{
    collections::BTreeSet,
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
    pub task_id: String,
    pub statement_sha256: String,
    pub integrity_status: String,
    pub used_axiom_count: usize,
    pub allowed_axiom_count: usize,
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
    let statement_sha256 = string(value, "/target/statement_sha256")?.to_owned();
    if string(value, "/dataset/cohort_id")?.trim().is_empty()
        || string(value, "/dataset/snapshot_id")?.trim().is_empty()
        || string(value, "/target/repository_remote")?.trim().is_empty()
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
        task_id,
        statement_sha256,
        integrity_status: expected.to_owned(),
        used_axiom_count: used_axioms.len(),
        allowed_axiom_count: allowed_axioms.len(),
    })
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
