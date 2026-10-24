// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic AtomFab PCB manufacturing-package integrity evaluator.
//!
//! This module checks whether a package's native inputs, DRC evidence and
//! manufacturing outputs are internally bound to one board revision. It does
//! not prove manufacturability, authorize execution, or qualify a physical PCB.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PACKAGE_CORPUS_SCHEMA: &str = "atomfab-pcb-package-corpus-v0.1";
pub const PACKAGE_CORPUS_VERSION: &str = "0.1.0";
pub const PACKAGE_CASE_COUNT: usize = 12;
pub const PACKAGE_AUTHORITY: &str =
    "synthetic_package_integrity_only_no_fabrication_or_execution_authority";
pub const EMBEDDED_PACKAGE_CORPUS: &str =
    include_str!("../../../../docs/research/fixtures/ATOMFAB_PCB_PACKAGE_CORPUS_V0_1.json");

const REQUIRED_SOURCE_ROLES: &[&str] =
    &["board_native", "schematic_native", "netlist", "bom_source"];
const REQUIRED_OUTPUT_ROLES: &[&str] = &[
    "gerber_front_copper",
    "gerber_back_copper",
    "gerber_outline",
    "drill_file",
    "bom_export",
    "placement_csv",
];
const PACKAGE_SCHEMA: &str = "atomfab-pcb-manufacturing-package-v0.1";
const PACKAGE_VERSION: &str = "0.1.0";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageCorpus {
    schema: String,
    version: String,
    authority: String,
    case_count: usize,
    evidence_boundary: PackageEvidenceBoundary,
    required_source_roles: Vec<String>,
    required_output_roles: Vec<String>,
    cases: Vec<PackageCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageEvidenceBoundary {
    inputs_are_synthetic: bool,
    ready_for_review_does_not_mean_manufacturable: bool,
    ready_for_review_does_not_mean_physical_qualification: bool,
    package_validation_does_not_authorize_machine_execution: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageCase {
    id: String,
    description: String,
    package: PackageManifest,
    expected: PackageGateResult,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageManifest {
    schema: String,
    version: String,
    package_id: String,
    board_revision: String,
    target_drc_profile_sha256: String,
    source_artifacts: Vec<PackageArtifact>,
    output_artifacts: Vec<PackageArtifact>,
    drc: DrcEvidence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageArtifact {
    role: String,
    path: String,
    sha256: String,
    board_revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DrcEvidence {
    status: String,
    source_board_sha256: String,
    board_revision: String,
    profile_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PackageGateResult {
    pub disposition: String,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackageCaseResult {
    pub case_id: String,
    pub actual: PackageGateResult,
    pub expected: PackageGateResult,
    pub matches: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackageCorpusReceipt {
    pub schema: String,
    pub corpus_schema: String,
    pub corpus_version: String,
    pub corpus_sha256: String,
    pub git_head: String,
    pub exact_head_bound: bool,
    pub authority: String,
    pub case_count: usize,
    pub matched_cases: usize,
    pub disposition_counts: BTreeMap<String, usize>,
    pub cases: Vec<PackageCaseResult>,
    pub status: String,
}

impl PackageCorpusReceipt {
    pub fn passed(&self) -> bool {
        self.status == "passed" && self.exact_head_bound && self.matched_cases == self.case_count
    }
}

/// Recompute every package disposition from inputs and verify its expected
/// result. The caller must provide the declared SHA and the actual checkout
/// SHA separately; a mismatch is a corpus-evaluation error.
pub fn evaluate_package_corpus_bytes(
    bytes: &[u8],
    declared_git_head: &str,
    actual_checkout_head: &str,
) -> Result<PackageCorpusReceipt, String> {
    if !is_commit_sha(declared_git_head) {
        return Err("ATOMFAB_GIT_HEAD must be a 40- or 64-character hexadecimal commit SHA".into());
    }
    if !is_commit_sha(actual_checkout_head) {
        return Err(
            "actual checkout HEAD must be a 40- or 64-character hexadecimal commit SHA".into(),
        );
    }
    if !declared_git_head.eq_ignore_ascii_case(actual_checkout_head) {
        return Err("ATOMFAB_GIT_HEAD does not match the actual checkout HEAD".into());
    }

    let corpus: PackageCorpus = serde_json::from_slice(bytes)
        .map_err(|error| format!("package corpus JSON/schema invalid: {error}"))?;
    if corpus.schema != PACKAGE_CORPUS_SCHEMA || corpus.version != PACKAGE_CORPUS_VERSION {
        return Err(format!(
            "unsupported package corpus identity: schema={} version={}",
            corpus.schema, corpus.version
        ));
    }
    if corpus.authority != PACKAGE_AUTHORITY {
        return Err("package corpus authority marker mismatch".into());
    }
    if corpus.case_count != PACKAGE_CASE_COUNT || corpus.cases.len() != PACKAGE_CASE_COUNT {
        return Err(format!(
            "package case count mismatch: declared={}, actual={}, required={}",
            corpus.case_count,
            corpus.cases.len(),
            PACKAGE_CASE_COUNT
        ));
    }
    let boundary = &corpus.evidence_boundary;
    if !(boundary.inputs_are_synthetic
        && boundary.ready_for_review_does_not_mean_manufacturable
        && boundary.ready_for_review_does_not_mean_physical_qualification
        && boundary.package_validation_does_not_authorize_machine_execution)
    {
        return Err("package evidence-boundary declaration weakened or missing".into());
    }
    require_exact_roles(
        &corpus.required_source_roles,
        REQUIRED_SOURCE_ROLES,
        "source",
    )?;
    require_exact_roles(
        &corpus.required_output_roles,
        REQUIRED_OUTPUT_ROLES,
        "output",
    )?;

    let mut case_ids = BTreeSet::new();
    for case in &corpus.cases {
        if case.id.trim().is_empty() || case.description.trim().is_empty() {
            return Err("package case ID and description must be non-empty".into());
        }
        if !case_ids.insert(case.id.clone()) {
            return Err(format!("duplicate package case ID: {}", case.id));
        }
        if !matches!(
            case.expected.disposition.as_str(),
            "ready_for_review" | "incomplete" | "rejected"
        ) || case.expected.reason_codes.is_empty()
        {
            return Err(format!("{}: invalid expected package disposition", case.id));
        }
    }

    let mut results = Vec::with_capacity(corpus.cases.len());
    for case in &corpus.cases {
        let actual = evaluate_package(&case.package);
        results.push(PackageCaseResult {
            case_id: case.id.clone(),
            matches: actual == case.expected,
            actual,
            expected: case.expected.clone(),
        });
    }
    // Input case order is presentation-only; receipt case order is canonical.
    results.sort_by(|a, b| a.case_id.cmp(&b.case_id));

    let mut disposition_counts = BTreeMap::new();
    for result in &results {
        *disposition_counts
            .entry(result.actual.disposition.clone())
            .or_insert(0) += 1;
    }
    let matched_cases = results.iter().filter(|result| result.matches).count();
    let status = if matched_cases == PACKAGE_CASE_COUNT {
        "passed"
    } else {
        "failed"
    };

    Ok(PackageCorpusReceipt {
        schema: "atomfab-pcb-package-receipt-v0.1".into(),
        corpus_schema: corpus.schema,
        corpus_version: corpus.version,
        corpus_sha256: sha256_hex(bytes),
        git_head: actual_checkout_head.to_ascii_lowercase(),
        exact_head_bound: true,
        authority: PACKAGE_AUTHORITY.into(),
        case_count: corpus.cases.len(),
        matched_cases,
        disposition_counts,
        cases: results,
        status: status.into(),
    })
}

fn evaluate_package(package: &PackageManifest) -> PackageGateResult {
    let mut rejected = Vec::new();
    let mut incomplete = Vec::new();

    if package.schema != PACKAGE_SCHEMA || package.version != PACKAGE_VERSION {
        rejected.push("UNSUPPORTED_PACKAGE_SCHEMA_OR_VERSION".to_string());
    }
    if package.package_id.trim().is_empty() || package.board_revision.trim().is_empty() {
        rejected.push("PACKAGE_ID_OR_REVISION_MISSING".to_string());
    }
    if !is_hex_digest(&package.target_drc_profile_sha256) {
        rejected.push("INVALID_TARGET_DRC_PROFILE_DIGEST".to_string());
    }

    let all_artifacts = package
        .source_artifacts
        .iter()
        .map(|artifact| ("source", artifact))
        .chain(
            package
                .output_artifacts
                .iter()
                .map(|artifact| ("output", artifact)),
        );
    let mut seen_paths = BTreeSet::new();
    let mut seen_source_roles = BTreeSet::new();
    let mut seen_output_roles = BTreeSet::new();

    for (kind, artifact) in all_artifacts {
        if artifact.role.trim().is_empty() {
            rejected.push(format!("EMPTY_ARTIFACT_ROLE:{kind}"));
        }
        if !is_safe_relative_path(&artifact.path) {
            rejected.push(format!("UNSAFE_ARTIFACT_PATH:{}", artifact.path));
        }
        if !seen_paths.insert(artifact.path.clone()) {
            rejected.push(format!("DUPLICATE_ARTIFACT_PATH:{}", artifact.path));
        }
        if !is_hex_digest(&artifact.sha256) {
            rejected.push(format!("INVALID_ARTIFACT_DIGEST:{}", artifact.role));
        }
        if artifact.board_revision != package.board_revision {
            rejected.push(format!("ARTIFACT_REVISION_MISMATCH:{}", artifact.role));
        }
        let seen_roles = if kind == "source" {
            &mut seen_source_roles
        } else {
            &mut seen_output_roles
        };
        if !seen_roles.insert(artifact.role.clone()) {
            rejected.push(format!("DUPLICATE_ARTIFACT_ROLE:{kind}:{}", artifact.role));
        }
    }

    for role in REQUIRED_SOURCE_ROLES {
        if !seen_source_roles.contains(*role) {
            incomplete.push(format!("MISSING_REQUIRED_SOURCE:{role}"));
        }
    }
    for role in REQUIRED_OUTPUT_ROLES {
        if !seen_output_roles.contains(*role) {
            incomplete.push(format!("MISSING_REQUIRED_OUTPUT:{role}"));
        }
    }

    if !matches!(
        package.drc.status.as_str(),
        "clean" | "violations" | "unknown"
    ) {
        rejected.push("INVALID_DRC_STATUS".to_string());
    }
    if !is_hex_digest(&package.drc.source_board_sha256) {
        rejected.push("INVALID_DRC_BOARD_DIGEST".to_string());
    }
    if !is_hex_digest(&package.drc.profile_sha256) {
        rejected.push("INVALID_DRC_PROFILE_DIGEST".to_string());
    }
    if package.drc.board_revision != package.board_revision {
        rejected.push("DRC_REVISION_MISMATCH".to_string());
    }
    if package.drc.profile_sha256 != package.target_drc_profile_sha256 {
        rejected.push("DRC_PROFILE_DIGEST_MISMATCH".to_string());
    }
    if let Some(board) = package
        .source_artifacts
        .iter()
        .find(|artifact| artifact.role == "board_native")
    {
        if package.drc.source_board_sha256 != board.sha256 {
            rejected.push("DRC_BOARD_DIGEST_MISMATCH".to_string());
        }
    }
    match package.drc.status.as_str() {
        "violations" => rejected.push("DRC_HAS_VIOLATIONS".to_string()),
        "unknown" => incomplete.push("DRC_NOT_CLEAN:unknown".to_string()),
        "clean" => {}
        _ => {}
    }

    rejected.sort();
    rejected.dedup();
    incomplete.sort();
    incomplete.dedup();
    if !rejected.is_empty() {
        gate("rejected", rejected)
    } else if !incomplete.is_empty() {
        gate("incomplete", incomplete)
    } else {
        gate(
            "ready_for_review",
            vec!["PACKAGE_AND_DRC_BINDINGS_VALID".into()],
        )
    }
}

fn is_safe_relative_path(value: &str) -> bool {
    if value.is_empty() || value.starts_with('/') || value.contains('\\') || value.contains(':') {
        return false;
    }
    value
        .split('/')
        .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

fn is_commit_sha(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_exact_roles(actual: &[String], expected: &[&str], label: &str) -> Result<(), String> {
    let actual_set: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    let expected_set: BTreeSet<&str> = expected.iter().copied().collect();
    if actual.len() != actual_set.len() || actual_set != expected_set {
        return Err(format!("required package {label} role list mismatch"));
    }
    Ok(())
}

fn gate(disposition: &str, reason_codes: Vec<String>) -> PackageGateResult {
    PackageGateResult {
        disposition: disposition.into(),
        reason_codes,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn valid_head() -> String {
        "a".repeat(40)
    }

    fn evaluate_valid(bytes: &[u8]) -> Result<PackageCorpusReceipt, String> {
        let head = valid_head();
        evaluate_package_corpus_bytes(bytes, &head, &head)
    }

    fn corpus_value() -> Value {
        serde_json::from_str(EMBEDDED_PACKAGE_CORPUS).expect("embedded package corpus parses")
    }

    #[test]
    fn frozen_package_corpus_matches_independent_oracle() {
        let receipt =
            evaluate_valid(EMBEDDED_PACKAGE_CORPUS.as_bytes()).expect("valid package corpus");
        assert!(receipt.passed(), "receipt: {receipt:#?}");
        assert_eq!(receipt.case_count, PACKAGE_CASE_COUNT);
        assert_eq!(receipt.matched_cases, PACKAGE_CASE_COUNT);
        assert_eq!(
            receipt.disposition_counts,
            BTreeMap::from([
                ("incomplete".into(), 3),
                ("ready_for_review".into(), 1),
                ("rejected".into(), 8),
            ])
        );
        assert_eq!(
            receipt.corpus_sha256,
            sha256_hex(EMBEDDED_PACKAGE_CORPUS.as_bytes())
        );
    }

    #[test]
    fn package_ready_for_review_does_not_grant_execution_or_quality_release() {
        let receipt =
            evaluate_valid(EMBEDDED_PACKAGE_CORPUS.as_bytes()).expect("valid package corpus");
        let case = receipt
            .cases
            .iter()
            .find(|case| case.case_id == "PCB-PKG-001")
            .expect("valid package positive control");
        assert_eq!(case.actual.disposition, "ready_for_review");
        assert!(
            !case
                .actual
                .reason_codes
                .iter()
                .any(|reason| reason.contains("qualified"))
        );
    }

    #[test]
    fn stale_drc_source_digest_is_rejected() {
        let receipt =
            evaluate_valid(EMBEDDED_PACKAGE_CORPUS.as_bytes()).expect("valid package corpus");
        let case = receipt
            .cases
            .iter()
            .find(|case| case.case_id == "PCB-PKG-004")
            .expect("stale DRC digest negative control");
        assert_eq!(case.actual.disposition, "rejected");
        assert!(
            case.actual
                .reason_codes
                .contains(&"DRC_BOARD_DIGEST_MISMATCH".to_string())
        );
    }

    #[test]
    fn path_traversal_is_rejected() {
        let receipt =
            evaluate_valid(EMBEDDED_PACKAGE_CORPUS.as_bytes()).expect("valid package corpus");
        let case = receipt
            .cases
            .iter()
            .find(|case| case.case_id == "PCB-PKG-009")
            .expect("path traversal negative control");
        assert_eq!(case.actual.disposition, "rejected");
        assert!(
            case.actual
                .reason_codes
                .contains(&"UNSAFE_ARTIFACT_PATH:../unsafe.gbr".to_string())
        );
    }

    #[test]
    fn duplicate_case_ids_and_declared_count_are_rejected() {
        let mut value = corpus_value();
        value["cases"][1]["id"] = value["cases"][0]["id"].clone();
        let duplicated = serde_json::to_vec(&value).expect("serialize duplicate corpus");
        assert!(
            evaluate_valid(&duplicated)
                .expect_err("duplicate package case must reject")
                .contains("duplicate package case ID")
        );

        let mut value = corpus_value();
        value["case_count"] = serde_json::json!(11);
        let altered_count = serde_json::to_vec(&value).expect("serialize altered package corpus");
        assert!(
            evaluate_valid(&altered_count)
                .expect_err("package count tampering must reject")
                .contains("package case count mismatch")
        );
    }

    #[test]
    fn altered_expectation_fails_receipt() {
        let mut value = corpus_value();
        value["cases"][0]["expected"]["disposition"] = serde_json::Value::String("rejected".into());
        let mutated = serde_json::to_vec(&value).expect("serialize altered expected output");
        let receipt = evaluate_valid(&mutated).expect("altered expectation remains parseable");
        assert!(!receipt.passed());
        assert_eq!(receipt.matched_cases, PACKAGE_CASE_COUNT - 1);
    }

    #[test]
    fn case_order_changes_digest_but_not_normalized_outcomes() {
        let original =
            evaluate_valid(EMBEDDED_PACKAGE_CORPUS.as_bytes()).expect("valid package corpus");
        let mut value = corpus_value();
        value["cases"]
            .as_array_mut()
            .expect("cases array")
            .reverse();
        let reverse = serde_json::to_vec(&value).expect("serialize reordered cases");
        let reordered = evaluate_valid(&reverse).expect("reordered package corpus");
        assert!(reordered.passed());
        assert_eq!(original.cases, reordered.cases);
        assert_ne!(original.corpus_sha256, reordered.corpus_sha256);
    }

    #[test]
    fn declared_head_must_match_actual_checkout() {
        let declared = "a".repeat(40);
        let actual = "b".repeat(40);
        assert!(
            evaluate_package_corpus_bytes(EMBEDDED_PACKAGE_CORPUS.as_bytes(), &declared, &actual)
                .expect_err("Git SHA mismatch must reject")
                .contains("does not match the actual checkout HEAD")
        );
    }

    #[test]
    fn checkout_head_validation_accepts_sha1_and_sha256_ids() {
        let bytes = EMBEDDED_PACKAGE_CORPUS.as_bytes();
        for head in ["a".repeat(40), "b".repeat(64)] {
            let receipt = evaluate_package_corpus_bytes(bytes, &head, &head)
                .expect("valid SHA-1 or SHA-256 Git object ID");
            assert!(receipt.passed(), "receipt: {receipt:#?}");
            assert_eq!(receipt.git_head, head);
        }
        assert!(
            evaluate_package_corpus_bytes(bytes, &"a".repeat(39), &"a".repeat(39))
                .expect_err("short object ID must reject")
                .contains("40- or 64-character hexadecimal commit SHA")
        );
        assert!(
            evaluate_package_corpus_bytes(bytes, &"g".repeat(40), &"g".repeat(40))
                .expect_err("non-hex object ID must reject")
                .contains("40- or 64-character hexadecimal commit SHA")
        );
    }
}
