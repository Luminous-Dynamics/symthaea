// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent structural verification for selector request/response receipts.
//!
//! This module checks that a response is bound to the exact request and that
//! the selection trace is structurally valid. It does not rerun the policy,
//! validate feature semantics, or prove scientific utility.

use crate::{MAX_BUDGET, MAX_CANDIDATES, SCHEMA_VERSION, SelectionRequest, SelectionResponse};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// One independently evaluated receipt invariant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiptCheck {
    /// Stable check identifier.
    pub check: String,
    /// Whether the invariant passed.
    pub passed: bool,
    /// Human-readable diagnostic bounded to 300 characters.
    pub detail: String,
}

/// Result of checking a request/response pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiptAudit {
    /// Current JSONL contract schema version.
    pub schema_version: String,
    /// `passed` only when all required structural checks passed.
    pub status: String,
    /// Digest of exact request-line bytes, excluding a line terminator.
    pub request_sha256: String,
    /// Digest of exact response-line bytes, excluding a line terminator.
    pub response_sha256: String,
    /// Individual checks; every failed check is preserved.
    pub checks: Vec<ReceiptCheck>,
    /// Limits of what structural verification proves.
    pub limitations: Vec<String>,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn euclidean_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn distance_trace_is_valid(
    request: &SelectionRequest,
    response: &SelectionResponse,
    candidates: &BTreeMap<&str, &crate::Candidate>,
) -> bool {
    let mut prior = Vec::with_capacity(response.selected.len());
    for (index, decision) in response.selected.iter().enumerate() {
        let Some(candidate) = candidates.get(decision.candidate_id.as_str()) else {
            return false;
        };
        if index == 0 {
            if decision.min_distance.is_some() {
                return false;
            }
        } else if request.policy == crate::Policy::RandomSeeded {
            if decision.min_distance.is_some() {
                return false;
            }
        } else {
            let expected = prior
                .iter()
                .map(|previous: &&crate::Candidate| {
                    euclidean_distance(
                        &candidate.normalized_features,
                        &previous.normalized_features,
                    )
                })
                .fold(f64::INFINITY, f64::min);
            let Some(reported) = decision.min_distance else {
                return false;
            };
            let tolerance = 1e-12 * expected.max(1.0);
            if !reported.is_finite() || (reported - expected).abs() > tolerance {
                return false;
            }
        }
        prior.push(*candidate);
    }
    true
}

fn push_check(
    checks: &mut Vec<ReceiptCheck>,
    check: &str,
    passed: bool,
    detail: impl Into<String>,
) {
    let detail = detail.into();
    checks.push(ReceiptCheck {
        check: check.to_owned(),
        passed,
        detail: detail.chars().take(300).collect(),
    });
}

fn failed_early(
    request_sha256: &str,
    response_sha256: &str,
    check: &str,
    detail: String,
) -> ReceiptAudit {
    ReceiptAudit {
        schema_version: SCHEMA_VERSION.to_owned(),
        status: "failed".to_owned(),
        request_sha256: request_sha256.to_owned(),
        response_sha256: response_sha256.to_owned(),
        checks: vec![ReceiptCheck {
            check: check.to_owned(),
            passed: false,
            detail: detail.chars().take(300).collect(),
        }],
        limitations: vec![
            "Receipt checks are structural; they do not prove scientific accuracy or utility."
                .to_owned(),
        ],
    }
}

/// Verify a raw JSON request line against its raw JSON response line.
///
/// The request and response must be the exact line contents (without LF/CRLF).
/// The checker does not invoke the selector. It independently checks digest
/// binding, request identity, budget/count invariants, selected ID membership
/// and uniqueness, rank ordering, family/cost lineage, and numeric diagnostics.
///
/// Requests containing unknown target/label fields are rejected by the typed
/// request contract before the receipt can be accepted.
pub fn verify_receipt(request_line: &str, response_line: &str) -> ReceiptAudit {
    let request_sha256 = digest(request_line.as_bytes());
    let response_sha256 = digest(response_line.as_bytes());
    if request_line.len() > crate::MAX_LINE_BYTES {
        return failed_early(
            &request_sha256,
            &response_sha256,
            "request_size",
            format!("Request line exceeds {} bytes.", crate::MAX_LINE_BYTES),
        );
    }
    if response_line.len() > crate::MAX_LINE_BYTES {
        return failed_early(
            &request_sha256,
            &response_sha256,
            "response_size",
            format!("Response line exceeds {} bytes.", crate::MAX_LINE_BYTES),
        );
    }

    let request: SelectionRequest = match serde_json::from_str(request_line) {
        Ok(request) => request,
        Err(error) => {
            return failed_early(
                &request_sha256,
                &response_sha256,
                "request_parse",
                format!("Request is invalid or includes forbidden fields: {error}"),
            );
        }
    };
    let response: SelectionResponse = match serde_json::from_str(response_line) {
        Ok(response) => response,
        Err(error) => {
            return failed_early(
                &request_sha256,
                &response_sha256,
                "response_parse",
                format!("Response is not a valid successful selection receipt: {error}"),
            );
        }
    };

    let mut checks = Vec::new();
    push_check(
        &mut checks,
        "schema_version",
        request.schema_version == SCHEMA_VERSION && response.schema_version == SCHEMA_VERSION,
        "Request and response use the supported schema version.",
    );
    push_check(
        &mut checks,
        "response_status",
        response.status == "completed",
        "Only a completed selector response can be accepted as a successful receipt.",
    );
    push_check(
        &mut checks,
        "request_digest",
        response.request_sha256 == request_sha256,
        "Response digest must equal SHA-256 of the exact request line.",
    );
    push_check(
        &mut checks,
        "feature_contract",
        response.feature_contract_sha256 == request.feature_contract_sha256,
        "Feature/preprocessing contract digest matches the request.",
    );
    let feature_digest_valid = request.feature_contract_sha256.len() == 64
        && request
            .feature_contract_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    push_check(
        &mut checks,
        "feature_contract_digest_syntax",
        feature_digest_valid,
        "Feature-contract identity is a lowercase 64-character SHA-256 hex digest.",
    );
    push_check(
        &mut checks,
        "policy_identity",
        response.policy == request.policy && response.seed == request.seed,
        "Policy and seed match the request.",
    );
    push_check(
        &mut checks,
        "request_bounds",
        request.budget > 0
            && request.budget <= MAX_BUDGET
            && !request.candidates.is_empty()
            && request.candidates.len() <= MAX_CANDIDATES,
        "Request budget and candidate-pool size are within supported bounds.",
    );
    push_check(
        &mut checks,
        "candidate_count",
        response.candidate_count == request.candidates.len(),
        "Response candidate count matches the request.",
    );
    push_check(
        &mut checks,
        "budget_accounting",
        response.requested_budget == request.budget
            && response.selected_count == response.selected.len()
            && response.selected_count == request.budget.min(request.candidates.len())
            && response.selected_count <= request.budget,
        "Selected count, request budget and trace length agree.",
    );
    push_check(
        &mut checks,
        "cost_unit",
        response.cost_unit == request.cost_unit,
        "Declared cost unit is preserved in the response.",
    );

    let mut candidates = BTreeMap::new();
    let mut candidate_ids_unique = true;
    for candidate in &request.candidates {
        if candidates.insert(candidate.candidate_id.as_str(), candidate).is_some() {
            candidate_ids_unique = false;
        }
    }
    push_check(
        &mut checks,
        "input_candidate_ids_unique",
        candidate_ids_unique,
        "Input candidate identifiers are unique.",
    );
    let feature_dimension = request
        .candidates
        .first()
        .map(|candidate| candidate.normalized_features.len())
        .unwrap_or(0);
    let input_values_valid = feature_dimension > 0
        && feature_dimension <= crate::MAX_FEATURE_DIMENSIONS
        && request.candidates.iter().all(|candidate| {
            !candidate.candidate_id.trim().is_empty()
                && candidate.candidate_id.len() <= crate::MAX_ID_BYTES
                && !candidate.family_id.trim().is_empty()
                && candidate.family_id.len() <= crate::MAX_ID_BYTES
                && candidate.normalized_features.len() == feature_dimension
                && candidate.normalized_features.iter().all(|value| {
                    value.is_finite() && (0.0..=1.0).contains(value)
                })
                && candidate
                    .estimated_cost
                    .is_none_or(|cost| cost.is_finite() && cost > 0.0)
        });
    push_check(
        &mut checks,
        "input_values_valid",
        input_values_valid,
        "Input identifiers, family IDs, descriptor dimensions/values, and any declared costs are valid.",
    );
    let cost_policy_valid = request.policy != crate::Policy::CostAwareFarthestFirst
        || (request
            .cost_unit
            .as_deref()
            .is_some_and(|unit| !unit.trim().is_empty() && unit.len() <= 64)
            && request.candidates.iter().all(|candidate| {
                candidate
                    .estimated_cost
                    .is_some_and(|cost| cost.is_finite() && cost > 0.0)
            }));
    push_check(
        &mut checks,
        "cost_policy_contract",
        cost_policy_valid,
        "Cost-aware policy has an explicit shared unit and positive costs for every candidate.",
    );

    let mut selected_ids = BTreeSet::new();
    let mut selected_membership = true;
    let mut lineage_matches = true;
    let mut ranks_ordered = true;
    let mut diagnostics_finite = true;
    for (index, decision) in response.selected.iter().enumerate() {
        if decision.rank != index + 1 {
            ranks_ordered = false;
        }
        if !selected_ids.insert(decision.candidate_id.as_str()) {
            selected_membership = false;
        }
        match candidates.get(decision.candidate_id.as_str()) {
            Some(candidate) => {
                if candidate.family_id != decision.family_id
                    || candidate.estimated_cost != decision.estimated_cost
                {
                    lineage_matches = false;
                }
            }
            None => selected_membership = false,
        }
        if decision.min_distance.is_some_and(|value| !value.is_finite() || value < 0.0) {
            diagnostics_finite = false;
        }
        if decision
            .estimated_cost
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            diagnostics_finite = false;
        }
    }
    push_check(
        &mut checks,
        "selected_ids_unique_and_present",
        selected_membership && selected_ids.len() == response.selected_count,
        "Every selected ID is unique and exists in the frozen input candidate pool.",
    );
    push_check(
        &mut checks,
        "selection_ranks",
        ranks_ordered,
        "Selection ranks are contiguous and ordered from one.",
    );
    push_check(
        &mut checks,
        "candidate_lineage",
        lineage_matches,
        "Selected family and declared cost match the request candidate.",
    );
    push_check(
        &mut checks,
        "numeric_diagnostics",
        diagnostics_finite,
        "Distances and present estimated costs are finite and in their allowed ranges.",
    );
    push_check(
        &mut checks,
        "distance_trace",
        distance_trace_is_valid(&request, &response, &candidates),
        "Reported nearest-selected distances are independently recomputed from the frozen descriptors.",
    );

    let passed = checks.iter().all(|check| check.passed);
    ReceiptAudit {
        schema_version: SCHEMA_VERSION.to_owned(),
        status: if passed { "passed" } else { "failed" }.to_owned(),
        request_sha256,
        response_sha256,
        checks,
        limitations: vec![
            "This verifies structural consistency, not whether the selector chose the best candidates."
                .to_owned(),
            "Descriptor coverage is not a physical property, calibrated uncertainty, or proof of novelty."
                .to_owned(),
            "Source rights, feature-transform provenance, leakage controls, and scientific metrics require separate checks."
                .to_owned(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Candidate, Policy, SelectionRequest, handle_line};

    fn request() -> SelectionRequest {
        SelectionRequest {
            schema_version: SCHEMA_VERSION.to_owned(),
            policy: Policy::FarthestFirst,
            budget: 2,
            seed: 7,
            feature_contract_sha256: "a".repeat(64),
            cost_unit: None,
            candidates: vec![
                Candidate {
                    candidate_id: "a".to_owned(),
                    estimated_cost: Some(1.0),
                    family_id: "f1".to_owned(),
                    normalized_features: vec![0.0, 0.0],
                },
                Candidate {
                    candidate_id: "b".to_owned(),
                    estimated_cost: Some(2.0),
                    family_id: "f2".to_owned(),
                    normalized_features: vec![1.0, 1.0],
                },
                Candidate {
                    candidate_id: "c".to_owned(),
                    estimated_cost: None,
                    family_id: "f3".to_owned(),
                    normalized_features: vec![0.5, 0.5],
                },
            ],
        }
    }

    #[test]
    fn accepts_valid_structural_receipt() {
        let request_line = serde_json::to_string(&request()).unwrap();
        let response_line = handle_line(&request_line);
        let report = verify_receipt(&request_line, &response_line);
        assert_eq!(report.status, "passed", "{report:?}");
        assert_eq!(report.request_sha256, digest(request_line.as_bytes()));
        assert_eq!(report.response_sha256, digest(response_line.as_bytes()));
        assert!(report.checks.iter().all(|check| check.passed));
    }

    #[test]
    fn detects_request_line_mutation() {
        let request_line = serde_json::to_string(&request()).unwrap();
        let response_line = handle_line(&request_line);
        let changed_request = request_line.replace("\"seed\":7", "\"seed\":8");
        let report = verify_receipt(&changed_request, &response_line);
        assert_eq!(report.status, "failed");
        assert!(report.checks.iter().any(|check| check.check == "request_digest" && !check.passed));
    }

    #[test]
    fn detects_selected_id_outside_candidate_pool() {
        let request_line = serde_json::to_string(&request()).unwrap();
        let response_value: serde_json::Value =
            serde_json::from_str(&handle_line(&request_line)).unwrap();
        let mut value = response_value;
        value["selected"][0]["candidate_id"] = serde_json::Value::String("not-in-pool".to_owned());
        let response_line = serde_json::to_string(&value).unwrap();
        let report = verify_receipt(&request_line, &response_line);
        assert_eq!(report.status, "failed");
        assert!(report.checks.iter().any(|check| {
            check.check == "selected_ids_unique_and_present" && !check.passed
        }));
    }

    #[test]
    fn accepts_cost_aware_receipt_and_preserves_declared_unit() {
        let mut request = request();
        request.policy = Policy::CostAwareFarthestFirst;
        request.cost_unit = Some("simulation_seconds".to_owned());
        let request_line = serde_json::to_string(&request).unwrap();
        let response_line = handle_line(&request_line);
        let report = verify_receipt(&request_line, &response_line);
        assert_eq!(report.status, "passed", "{report:?}");
    }

    #[test]
    fn rejects_response_with_unknown_fields() {
        let request_line = serde_json::to_string(&request()).unwrap();
        let response: serde_json::Value =
            serde_json::from_str(&handle_line(&request_line)).unwrap();
        let mut response_with_unknown = response;
        response_with_unknown["target_energy_eV"] = serde_json::Value::from(-5.0);
        let report = verify_receipt(
            &request_line,
            &serde_json::to_string(&response_with_unknown).unwrap(),
        );
        assert_eq!(report.status, "failed");
        assert_eq!(report.checks[0].check, "response_parse");
    }

    #[test]
    fn detects_tampered_distance_diagnostic() {
        let request_line = serde_json::to_string(&request()).unwrap();
        let response: serde_json::Value =
            serde_json::from_str(&handle_line(&request_line)).unwrap();
        let mut tampered = response;
        tampered["selected"][1]["min_distance"] = serde_json::Value::from(123.0);
        let report = verify_receipt(
            &request_line,
            &serde_json::to_string(&tampered).unwrap(),
        );
        assert_eq!(report.status, "failed");
        assert!(report.checks.iter().any(|check| {
            check.check == "distance_trace" && !check.passed
        }));
    }

    #[test]
    fn receipt_audit_schema_is_parseable_and_closed() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/receipt-audit.schema.json")).unwrap();
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["status"]["enum"][0], "passed");
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "response_sha256"));
    }

    #[test]
    fn rejects_request_with_hidden_target_field() {
        let request_line = r#"{"schema_version":"nano-policy-request-v1","policy":"farthest_first","budget":1,"seed":1,"feature_contract_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","candidates":[{"candidate_id":"x","family_id":"f","normalized_features":[0.5],"target_energy_eV":-1.0}]}"#;
        let report = verify_receipt(request_line, "{}");
        assert_eq!(report.status, "failed");
        assert_eq!(report.checks[0].check, "request_parse");
    }
}
