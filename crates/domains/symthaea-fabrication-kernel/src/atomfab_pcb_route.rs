// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent synthetic AtomFab PCB route/evidence corpus evaluator.
//!
//! This evaluator checks a frozen corpus and independently derives route
//! feasibility and evidence completeness. It does not call machine-control
//! APIs, perform physical qualification, or grant fabrication authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS_SCHEMA: &str = "atomfab-pcb-route-corpus-v0.3";
pub const CORPUS_VERSION: &str = "0.3.0";
pub const EXPECTED_CASE_COUNT: usize = 16;
pub const EMBEDDED_CORPUS: &str =
    include_str!("../../../../docs/research/fixtures/ATOMFAB_PCB_ROUTE_CORPUS_V0_3.json");

const CORPUS_AUTHORITY: &str =
    "synthetic_route_and_evidence_gate_only_no_fabrication_or_execution_authority";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema: String,
    version: String,
    authority: String,
    case_count: usize,
    evidence_boundary: EvidenceBoundary,
    route_dispositions: Vec<String>,
    evidence_dispositions: Vec<String>,
    cases: Vec<RouteCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceBoundary {
    all_case_inputs_are_synthetic: bool,
    route_feasibility_does_not_mean_physical_part_conforms: bool,
    evidence_completeness_does_not_mean_physical_qualification: bool,
    route_or_evidence_result_does_not_authorize_execution: bool,
    synthetic_evidence_cannot_establish_physical_qualification: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteCase {
    id: String,
    description: String,
    route: RouteProfile,
    requirements: RouteRequirements,
    evidence: EvidenceProfile,
    expected: ExpectedResult,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteProfile {
    profile: String,
    capability_state: String,
    capabilities: Vec<String>,
    minimum_trace_space_um: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteRequirements {
    capabilities: Vec<String>,
    minimum_trace_space_um: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceProfile {
    drc: String,
    gerber: String,
    drill: String,
    bom_substitutions: String,
    bare_board_inspection: String,
    calibration: String,
    assembly_package: String,
    assembly_inspection: String,
    functional_test_revision_match: bool,
    provenance: String,
    evidence_source_class: String,
    qualification_claim: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedResult {
    route: GateResult,
    evidence: GateResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GateResult {
    pub disposition: String,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CaseResult {
    pub case_id: String,
    pub actual_route: GateResult,
    pub expected_route: GateResult,
    pub route_match: bool,
    pub actual_evidence: GateResult,
    pub expected_evidence: GateResult,
    pub evidence_match: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusReceipt {
    pub schema: String,
    pub corpus_schema: String,
    pub corpus_version: String,
    pub corpus_sha256: String,
    pub git_head: String,
    pub exact_head_bound: bool,
    pub authority: String,
    pub case_count: usize,
    pub matched_route_cases: usize,
    pub matched_evidence_cases: usize,
    pub route_disposition_counts: BTreeMap<String, usize>,
    pub evidence_disposition_counts: BTreeMap<String, usize>,
    pub cases: Vec<CaseResult>,
    pub status: String,
}

impl CorpusReceipt {
    pub fn passed(&self) -> bool {
        self.status == "passed"
            && self.exact_head_bound
            && self.matched_route_cases == self.case_count
            && self.matched_evidence_cases == self.case_count
    }
}

/// Evaluate corpus bytes using a caller-supplied exact Git HEAD.
///
/// Structural/corpus-integrity failures return Err. Fixture expectation
/// mismatches are captured in a failed receipt, allowing negative controls to
/// be observed and reported rather than silently disappearing.
pub fn evaluate_corpus_bytes(
    bytes: &[u8],
    declared_git_head: &str,
    actual_checkout_head: &str,
) -> Result<CorpusReceipt, String> {
    if !is_git_sha(declared_git_head) {
        return Err("ATOMFAB_GIT_HEAD must be a 40- or 64-character hexadecimal commit SHA".into());
    }
    if !is_git_sha(actual_checkout_head) {
        return Err(
            "actual checkout HEAD must be a 40- or 64-character hexadecimal commit SHA".into(),
        );
    }
    if !declared_git_head.eq_ignore_ascii_case(actual_checkout_head) {
        return Err("ATOMFAB_GIT_HEAD does not match the actual checkout HEAD".into());
    }

    let corpus: Corpus = serde_json::from_slice(bytes)
        .map_err(|error| format!("corpus JSON/schema invalid: {error}"))?;

    if corpus.schema != CORPUS_SCHEMA || corpus.version != CORPUS_VERSION {
        return Err(format!(
            "unsupported corpus identity: schema={} version={}",
            corpus.schema, corpus.version
        ));
    }
    if corpus.authority != CORPUS_AUTHORITY {
        return Err("corpus authority marker mismatch".into());
    }
    if corpus.case_count != EXPECTED_CASE_COUNT || corpus.cases.len() != EXPECTED_CASE_COUNT {
        return Err(format!(
            "case count mismatch: declared={}, actual={}, required={}",
            corpus.case_count,
            corpus.cases.len(),
            EXPECTED_CASE_COUNT
        ));
    }

    let boundary = &corpus.evidence_boundary;
    if !(boundary.all_case_inputs_are_synthetic
        && boundary.route_feasibility_does_not_mean_physical_part_conforms
        && boundary.evidence_completeness_does_not_mean_physical_qualification
        && boundary.route_or_evidence_result_does_not_authorize_execution
        && boundary.synthetic_evidence_cannot_establish_physical_qualification)
    {
        return Err("evidence-boundary declaration weakened or missing".into());
    }

    require_exact_vocabulary(
        &corpus.route_dispositions,
        &["feasible", "infeasible", "unresolved"],
        "route dispositions",
    )?;
    require_exact_vocabulary(
        &corpus.evidence_dispositions,
        &["complete", "incomplete", "rejected"],
        "evidence dispositions",
    )?;

    let mut ids = BTreeSet::new();
    for case in &corpus.cases {
        if case.id.trim().is_empty() || case.description.trim().is_empty() {
            return Err("case identity and description must be non-empty".into());
        }
        if !ids.insert(case.id.clone()) {
            return Err(format!("duplicate case ID: {}", case.id));
        }
        validate_case_vocabulary(case)?;
    }

    let mut cases = Vec::with_capacity(corpus.cases.len());
    for case in &corpus.cases {
        let actual_route = derive_route_result(case);
        let actual_evidence = derive_evidence_result(case);
        let route_match = actual_route == case.expected.route;
        let evidence_match = actual_evidence == case.expected.evidence;
        cases.push(CaseResult {
            case_id: case.id.clone(),
            actual_route,
            expected_route: case.expected.route.clone(),
            route_match,
            actual_evidence,
            expected_evidence: case.expected.evidence.clone(),
            evidence_match,
        });
    }

    // Stable receipt order makes result comparison invariant to corpus case order.
    cases.sort_by(|a, b| a.case_id.cmp(&b.case_id));

    let mut route_disposition_counts = BTreeMap::new();
    let mut evidence_disposition_counts = BTreeMap::new();
    for result in &cases {
        *route_disposition_counts
            .entry(result.actual_route.disposition.clone())
            .or_insert(0) += 1;
        *evidence_disposition_counts
            .entry(result.actual_evidence.disposition.clone())
            .or_insert(0) += 1;
    }

    let matched_route_cases = cases.iter().filter(|case| case.route_match).count();
    let matched_evidence_cases = cases.iter().filter(|case| case.evidence_match).count();
    let status = if matched_route_cases == EXPECTED_CASE_COUNT
        && matched_evidence_cases == EXPECTED_CASE_COUNT
    {
        "passed"
    } else {
        "failed"
    };

    Ok(CorpusReceipt {
        schema: "atomfab-pcb-route-receipt-v0.3".into(),
        corpus_schema: corpus.schema,
        corpus_version: corpus.version,
        corpus_sha256: sha256_hex(bytes),
        git_head: actual_checkout_head.to_ascii_lowercase(),
        exact_head_bound: true,
        authority: CORPUS_AUTHORITY.into(),
        case_count: corpus.cases.len(),
        matched_route_cases,
        matched_evidence_cases,
        route_disposition_counts,
        evidence_disposition_counts,
        cases,
        status: status.into(),
    })
}

fn validate_case_vocabulary(case: &RouteCase) -> Result<(), String> {
    if !matches!(case.route.profile.as_str(), "P0" | "P1") {
        return Err(format!("{}: unsupported v0.2 route profile", case.id));
    }
    if !matches!(
        case.route.capability_state.as_str(),
        "unknown" | "declared_for_synthetic_fixture"
    ) {
        return Err(format!("{}: unsupported capability state", case.id));
    }
    if !matches!(
        case.evidence.evidence_source_class.as_str(),
        "synthetic" | "measured" | "vendor_declared" | "assumed"
    ) {
        return Err(format!("{}: unsupported evidence-source class", case.id));
    }
    if !matches!(
        case.evidence.qualification_claim.as_str(),
        "route_admissibility_only" | "independent_physical_qualification"
    ) {
        return Err(format!(
            "{}: unsupported qualification-claim class",
            case.id
        ));
    }
    for (label, result, allowed) in [
        (
            "expected route disposition",
            &case.expected.route,
            &["feasible", "infeasible", "unresolved"][..],
        ),
        (
            "expected evidence disposition",
            &case.expected.evidence,
            &["complete", "incomplete", "rejected"][..],
        ),
    ] {
        if !allowed.contains(&result.disposition.as_str()) || result.reason_codes.is_empty() {
            return Err(format!("{}: invalid {label}", case.id));
        }
    }
    Ok(())
}

fn derive_route_result(case: &RouteCase) -> GateResult {
    if case.route.capability_state == "unknown" {
        return gate("unresolved", vec!["CAPABILITY_UNKNOWN".into()]);
    }

    let mut reasons = Vec::new();
    let route_caps = &case.route.capabilities;
    let required_caps = &case.requirements.capabilities;

    if has_duplicates(route_caps) {
        reasons.push("DUPLICATE_ROUTE_CAPABILITY".into());
    }
    if has_duplicates(required_caps) {
        reasons.push("DUPLICATE_REQUIREMENT_CAPABILITY".into());
    }

    let route_set: BTreeSet<String> = route_caps.iter().cloned().collect();
    let mut unique_requirements = required_caps.clone();
    unique_requirements.sort();
    unique_requirements.dedup();
    for capability in unique_requirements {
        if !route_set.contains(&capability) {
            reasons.push(format!("CAPABILITY_MISSING:{capability}"));
        }
    }

    let mut unresolved = false;
    let mut infeasible = reasons.iter().any(|reason| {
        reason == "DUPLICATE_ROUTE_CAPABILITY"
            || reason == "DUPLICATE_REQUIREMENT_CAPABILITY"
            || reason.starts_with("CAPABILITY_MISSING:")
    });

    if case.requirements.minimum_trace_space_um == Some(0) {
        reasons.push("INVALID_TRACE_SPACE_REQUIREMENT".into());
        infeasible = true;
    }
    if case.route.minimum_trace_space_um == Some(0) {
        reasons.push("INVALID_ROUTE_TRACE_SPACE_PROFILE".into());
        infeasible = true;
    }
    match (
        case.requirements.minimum_trace_space_um,
        case.route.minimum_trace_space_um,
    ) {
        (Some(_), None) => {
            reasons.push("TRACE_SPACE_CAPABILITY_UNKNOWN".into());
            unresolved = true;
        }
        (Some(required), Some(route_minimum)) if required < route_minimum => {
            reasons.push("TRACE_SPACE_OUTSIDE_ROUTE_ENVELOPE".into());
            infeasible = true;
        }
        _ => {}
    }

    reasons.sort();
    reasons.dedup();
    if infeasible {
        gate("infeasible", reasons)
    } else if unresolved {
        gate("unresolved", reasons)
    } else {
        gate(
            "feasible",
            vec!["ALL_DECLARED_ROUTE_CAPABILITIES_SATISFIED".into()],
        )
    }
}

fn derive_evidence_result(case: &RouteCase) -> GateResult {
    let evidence = &case.evidence;
    let mut rejected = Vec::new();
    let mut incomplete = Vec::new();

    if evidence.drc != "current" {
        incomplete.push("DRC_NOT_CURRENT".to_string());
    }
    if evidence.gerber != "current" {
        incomplete.push("GERBER_NOT_CURRENT".to_string());
    }
    if evidence.drill != "current" {
        incomplete.push("DRILL_EXPORT_NOT_CURRENT".to_string());
    }
    if evidence.bom_substitutions != "approved" {
        rejected.push("UNAPPROVED_BOM_SUBSTITUTION".to_string());
    }
    if evidence.bare_board_inspection != "complete" {
        incomplete.push("BARE_BOARD_INSPECTION_INCOMPLETE".to_string());
    }
    if evidence.calibration != "current" {
        incomplete.push("CALIBRATION_NOT_CURRENT".to_string());
    }
    if evidence.assembly_package != "complete" {
        incomplete.push("ASSEMBLY_PACKAGE_INCOMPLETE".to_string());
    }
    if evidence.assembly_inspection != "complete" {
        incomplete.push("ASSEMBLY_INSPECTION_INCOMPLETE".to_string());
    }
    if !evidence.functional_test_revision_match {
        incomplete.push("FUNCTIONAL_TEST_REVISION_MISMATCH".to_string());
    }
    if evidence.provenance != "complete" {
        incomplete.push("PROVENANCE_INCOMPLETE".to_string());
    }
    if evidence.qualification_claim == "independent_physical_qualification" {
        // This synthetic v0.3 corpus has no independently bound physical
        // measurement objects, calibration certificates, or uncertainty
        // analysis. A string label such as "measured" cannot qualify a part.
        rejected.push("PHYSICAL_QUALIFICATION_OUT_OF_SCOPE".to_string());
    }

    // Show every declared issue. Rejection takes precedence, but never hides
    // incomplete evidence from the receipt.
    let mut reasons = rejected.clone();
    reasons.extend(incomplete);
    if !rejected.is_empty() {
        gate("rejected", reasons)
    } else if !reasons.is_empty() {
        gate("incomplete", reasons)
    } else {
        gate("complete", vec!["DECLARED_EVIDENCE_GATES_SATISFIED".into()])
    }
}

fn gate(disposition: &str, reason_codes: Vec<String>) -> GateResult {
    GateResult {
        disposition: disposition.into(),
        reason_codes,
    }
}

fn has_duplicates(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().any(|value| !seen.insert(value.as_str()))
}

fn require_exact_vocabulary(
    actual: &[String],
    expected: &[&str],
    label: &str,
) -> Result<(), String> {
    let actual_set: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    let expected_set: BTreeSet<&str> = expected.iter().copied().collect();
    if actual.len() != actual_set.len() || actual_set != expected_set {
        return Err(format!("{label} vocabulary mismatch"));
    }
    Ok(())
}

fn is_git_sha(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn valid_head() -> String {
        "a".repeat(40)
    }

    fn evaluate_valid(bytes: &[u8]) -> Result<CorpusReceipt, String> {
        let head = valid_head();
        evaluate_corpus_bytes(bytes, &head, &head)
    }

    fn corpus_value() -> Value {
        serde_json::from_str(EMBEDDED_CORPUS).expect("embedded corpus parses")
    }

    #[test]
    fn frozen_corpus_matches_independent_route_and_evidence_oracle() {
        let receipt = evaluate_valid(EMBEDDED_CORPUS.as_bytes()).expect("valid corpus");
        assert!(receipt.passed(), "receipt: {receipt:#?}");
        assert_eq!(receipt.case_count, EXPECTED_CASE_COUNT);
        assert_eq!(receipt.matched_route_cases, EXPECTED_CASE_COUNT);
        assert_eq!(receipt.matched_evidence_cases, EXPECTED_CASE_COUNT);
        assert_eq!(
            receipt.route_disposition_counts,
            BTreeMap::from([
                ("feasible".into(), 12),
                ("infeasible".into(), 3),
                ("unresolved".into(), 1),
            ])
        );
        assert_eq!(
            receipt.evidence_disposition_counts,
            BTreeMap::from([
                ("complete".into(), 6),
                ("incomplete".into(), 8),
                ("rejected".into(), 2),
            ])
        );
        assert_eq!(
            receipt.corpus_sha256,
            sha256_hex(EMBEDDED_CORPUS.as_bytes())
        );
    }

    #[test]
    fn route_feasibility_is_separate_from_evidence_completeness() {
        let receipt = evaluate_valid(EMBEDDED_CORPUS.as_bytes()).expect("valid corpus");
        let via_case = receipt
            .cases
            .iter()
            .find(|case| case.case_id == "PCB-ROUTE-003")
            .expect("via negative control");
        assert_eq!(via_case.actual_route.disposition, "infeasible");
        assert_eq!(via_case.actual_evidence.disposition, "complete");

        let inspection_case = receipt
            .cases
            .iter()
            .find(|case| case.case_id == "PCB-ROUTE-010")
            .expect("inspection negative control");
        assert_eq!(inspection_case.actual_route.disposition, "feasible");
        assert_eq!(inspection_case.actual_evidence.disposition, "incomplete");
    }

    #[test]
    fn duplicate_case_ids_are_rejected_before_evaluation() {
        let mut value = corpus_value();
        let first = value["cases"][0].clone();
        value["cases"][1]["id"] = first["id"].clone();
        let bytes = serde_json::to_vec(&value).expect("serialize changed corpus");
        let error = evaluate_valid(&bytes).expect_err("duplicate ID rejects");
        assert!(error.contains("duplicate case ID"));
    }

    #[test]
    fn altered_declared_case_count_fails_closed() {
        let mut value = corpus_value();
        value["case_count"] = serde_json::json!(15);
        let bytes = serde_json::to_vec(&value).expect("serialize changed corpus");
        let error = evaluate_valid(&bytes).expect_err("count mismatch rejects");
        assert!(error.contains("case count mismatch"));
    }

    #[test]
    fn expectation_mutation_produces_failed_receipt() {
        let mut value = corpus_value();
        value["cases"][0]["expected"]["route"]["disposition"] =
            serde_json::Value::String("infeasible".into());
        let bytes = serde_json::to_vec(&value).expect("serialize changed corpus");
        let receipt = evaluate_valid(&bytes).expect("parse valid schema");
        assert!(!receipt.passed());
        assert_eq!(receipt.matched_route_cases, EXPECTED_CASE_COUNT - 1);
    }

    #[test]
    fn reordering_cases_preserves_normalized_results() {
        let original = evaluate_valid(EMBEDDED_CORPUS.as_bytes()).expect("valid corpus");
        let mut value = corpus_value();
        let cases = value["cases"].as_array_mut().expect("cases array");
        cases.reverse();
        let reordered_bytes = serde_json::to_vec(&value).expect("serialize reordered corpus");
        let reordered = evaluate_valid(&reordered_bytes).expect("reordered corpus");
        assert!(reordered.passed());
        assert_eq!(original.cases, reordered.cases);
        assert_ne!(original.corpus_sha256, reordered.corpus_sha256);
    }

    #[test]
    fn physical_qualification_claim_cannot_be_supported_by_synthetic_evidence() {
        let receipt = evaluate_valid(EMBEDDED_CORPUS.as_bytes()).expect("valid corpus");
        let case = receipt
            .cases
            .iter()
            .find(|case| case.case_id == "PCB-ROUTE-016")
            .expect("qualification negative control");
        assert_eq!(case.actual_evidence.disposition, "rejected");
        assert!(
            case.actual_evidence
                .reason_codes
                .contains(&"PHYSICAL_QUALIFICATION_OUT_OF_SCOPE".to_string())
        );
    }

    #[test]
    fn declared_head_must_match_checkout_head() {
        let declared = "a".repeat(40);
        let actual = "b".repeat(40);
        let error = evaluate_corpus_bytes(EMBEDDED_CORPUS.as_bytes(), &declared, &actual)
            .expect_err("head mismatch rejects");
        assert!(error.contains("does not match the actual checkout HEAD"));
    }

    #[test]
    fn measured_label_alone_still_cannot_qualify_physical_output() {
        let mut value = corpus_value();
        value["cases"][15]["evidence"]["evidence_source_class"] =
            serde_json::Value::String("measured".into());
        let bytes = serde_json::to_vec(&value).expect("serialize changed corpus");
        let receipt = evaluate_valid(&bytes).expect("valid schema with altered label");
        let case = receipt
            .cases
            .iter()
            .find(|case| case.case_id == "PCB-ROUTE-016")
            .expect("physical qualification control");
        assert_eq!(case.actual_evidence.disposition, "rejected");
        assert!(
            case.actual_evidence
                .reason_codes
                .contains(&"PHYSICAL_QUALIFICATION_OUT_OF_SCOPE".to_string())
        );
    }

    #[test]
    fn invalid_git_head_is_rejected() {
        let actual_head = valid_head();
        let error = evaluate_corpus_bytes(EMBEDDED_CORPUS.as_bytes(), "not-a-commit", &actual_head)
            .expect_err("invalid declared head");
        assert!(error.contains("ATOMFAB_GIT_HEAD"));
    }
}
