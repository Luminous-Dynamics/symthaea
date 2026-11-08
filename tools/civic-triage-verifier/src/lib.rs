use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EXPECTED_SCHEMA: &str = "luminous.civic_triage_cases.v0.1";
pub const EXPECTED_STATUS: &str = "synthetic_non_identifying_effects_disabled";
pub const EXPECTED_AUTHORITY_CEILING: &str = "advisory_only_human_review_required";
pub const EXPECTED_FIXTURE_SHA256: &str = "32f8c68268d8874d567a41c857dd69adaa70a01961b4978ebb625b4f0b84f206";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub status: String,
    pub authority_ceiling: String,
    pub cases: Vec<Case>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub name: String,
    pub class: String,
    pub expected_state: String,
    pub assertions: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct VerificationReport {
    pub report_schema: &'static str,
    pub verifier_version: &'static str,
    pub status: &'static str,
    pub manifest_sha256: String,
    pub manifest_bytes: usize,
    pub case_count: usize,
    pub checked_case_ids: Vec<String>,
    pub errors: Vec<String>,
    pub claim_ceiling: &'static str,
}

const REQUIRED_CASES: [(&str, &str, &[&str]); 11] = [
    ("CT-001", "benign_or_unknown_pending_evidence", &["maintenance_context_visible", "no_identity_inference", "no_threat_label_from_anomaly"]),
    ("CT-002", "sensor_fault_or_unknown", &["sensor_health_explicit", "fault_not_promoted_to_intrusion"]),
    ("CT-003", "urgent_human_review_with_sensor_uncertainty", &["life_safety_not_suppressed", "calibration_state_preserved"]),
    ("CT-004", "not_independent_corroboration", &["dependency_graph_preserved", "duplicate_evidence_not_double_counted"]),
    ("CT-005", "disputed_or_unknown", &["both_claims_retained", "no_newest_timestamp_truth_rule"]),
    ("CT-006", "no_threat_inference", &["no_identity_or_association_feature", "benign_alternative_visible"]),
    ("CT-007", "freshness_unknown", &["event_time_separate_from_receive_time", "clock_quality_explicit"]),
    ("CT-008", "degraded_unknown_no_automatic_escalation", &["loss_of_data_not_evidence_of_hostility", "no_automatic_action"]),
    ("CT-009", "rejected", &["decision_frontier_frozen", "future_information_not_ancestor"]),
    ("CT-010", "rejected", &["recommendation_not_permission", "human_authorization_separate"]),
    ("CT-011", "rejected", &["feature_contract_blocks_proxy", "no_individual_risk_score"]),
];

pub fn verify_manifest(bytes: &[u8]) -> VerificationReport {
    let digest = format!("{:x}", Sha256::digest(bytes));
    let mut errors = Vec::new();
    if digest != EXPECTED_FIXTURE_SHA256 {
        errors.push(format!(
            "fixture_digest_mismatch: expected {}, got {}",
            EXPECTED_FIXTURE_SHA256,
            digest
        ));
    }

    let manifest: Option<Manifest> = match serde_json::from_slice(bytes) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(format!("strict_json_decode: {error}"));
            None
        }
    };

    let mut checked_case_ids = Vec::new();
    let mut case_count = 0;

    if let Some(manifest) = manifest {
        case_count = manifest.cases.len();

        if manifest.schema != EXPECTED_SCHEMA {
            errors.push(format!("schema_mismatch: expected {EXPECTED_SCHEMA}"));
        }
        if manifest.status != EXPECTED_STATUS {
            errors.push(format!("status_mismatch: expected {EXPECTED_STATUS}"));
        }
        if manifest.authority_ceiling != EXPECTED_AUTHORITY_CEILING {
            errors.push(format!(
                "authority_ceiling_mismatch: expected {EXPECTED_AUTHORITY_CEILING}"
            ));
        }
        if manifest.cases.len() != REQUIRED_CASES.len() {
            errors.push(format!(
                "case_count_mismatch: expected {}, got {}",
                REQUIRED_CASES.len(),
                manifest.cases.len()
            ));
        }

        let required: BTreeMap<&str, (&str, &[&str])> = REQUIRED_CASES
            .iter()
            .map(|(id, state, assertions)| (*id, (*state, *assertions)))
            .collect();
        let mut seen = BTreeSet::new();

        for case in &manifest.cases {
            checked_case_ids.push(case.id.clone());

            if case.id.trim().is_empty() || !seen.insert(case.id.as_str()) {
                errors.push(format!("empty_or_duplicate_case_id: {}", case.id));
            }
            if case.name.trim().is_empty() || case.class.trim().is_empty() {
                errors.push(format!("empty_case_metadata: {}", case.id));
            }
            if case.assertions.is_empty()
                || case.assertions.iter().any(|value| value.trim().is_empty())
            {
                errors.push(format!("empty_assertion_set: {}", case.id));
            }
            let assertion_set: BTreeSet<&str> =
                case.assertions.iter().map(String::as_str).collect();
            if assertion_set.len() != case.assertions.len() {
                errors.push(format!("duplicate_assertion: {}", case.id));
            }

            match required.get(case.id.as_str()) {
                None => errors.push(format!("unexpected_case_id: {}", case.id)),
                Some((expected_state, required_assertions)) => {
                    if case.expected_state != *expected_state {
                        errors.push(format!(
                            "expected_state_mismatch: {} expected {}, got {}",
                            case.id, expected_state, case.expected_state
                        ));
                    }
                    for assertion in *required_assertions {
                        if !assertion_set.contains(assertion) {
                            errors.push(format!(
                                "missing_required_assertion: {} -> {}",
                                case.id, assertion
                            ));
                        }
                    }
                }
            }
        }

        for (required_id, _, _) in REQUIRED_CASES {
            if !seen.contains(required_id) {
                errors.push(format!("missing_case_id: {required_id}"));
            }
        }
    }

    checked_case_ids.sort();
    errors.sort();

    VerificationReport {
        report_schema: "luminous.civic_triage_verification.v0.1",
        verifier_version: env!("CARGO_PKG_VERSION"),
        status: if errors.is_empty() { "CONFORMS" } else { "REJECTED" },
        manifest_sha256: digest,
        manifest_bytes: bytes.len(),
        case_count,
        checked_case_ids,
        errors,
        claim_ceiling: "synthetic_fixture_semantic_conformance_only",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../../docs/research/civic-triage/CIVIC_TRIAGE_CASES_V0_1.json");

    #[test]
    fn frozen_corpus_conforms() {
        let report = verify_manifest(FIXTURE.as_bytes());
        assert_eq!(report.status, "CONFORMS", "{:?}", report.errors);
        assert_eq!(report.case_count, REQUIRED_CASES.len());
        assert_eq!(report.checked_case_ids.len(), REQUIRED_CASES.len());
        assert_eq!(report.manifest_sha256, EXPECTED_FIXTURE_SHA256);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mutated = FIXTURE.replace(
            "\"id\": \"CT-001\"",
            "\"id\": \"CT-001\", \"unrecognized\": true",
        );
        assert_ne!(mutated, FIXTURE);
        let report = verify_manifest(mutated.as_bytes());
        assert_eq!(report.status, "REJECTED");
        assert!(report.errors.iter().any(|error| error.starts_with("strict_json_decode:")));
    }

    #[test]
    fn duplicate_and_missing_case_ids_are_rejected() {
        let mutated = FIXTURE.replace("\"id\": \"CT-011\"", "\"id\": \"CT-010\"");
        assert_ne!(mutated, FIXTURE);
        let report = verify_manifest(mutated.as_bytes());
        assert_eq!(report.status, "REJECTED");
        assert!(report.errors.iter().any(|error| error.starts_with("empty_or_duplicate_case_id:")));
        assert!(report.errors.iter().any(|error| error == "missing_case_id: CT-011"));
    }

    #[test]
    fn recommendation_cannot_become_authority() {
        let mutated = FIXTURE.replace(
            EXPECTED_AUTHORITY_CEILING,
            "autonomous_dispatch_authorized",
        );
        assert_ne!(mutated, FIXTURE);
        let report = verify_manifest(mutated.as_bytes());
        assert_eq!(report.status, "REJECTED");
        assert!(report.errors.iter().any(|error| error.starts_with("authority_ceiling_mismatch:")));
    }

    #[test]
    fn unknown_is_not_silently_promoted_to_intrusion() {
        let mutated = FIXTURE.replace(
            "degraded_unknown_no_automatic_escalation",
            "confirmed_hostile_intrusion",
        );
        assert_ne!(mutated, FIXTURE);
        let report = verify_manifest(mutated.as_bytes());
        assert_eq!(report.status, "REJECTED");
        assert!(report.errors.iter().any(|error| error.starts_with("expected_state_mismatch: CT-008")));
    }
    #[test]
    fn fixture_byte_change_is_rejected_by_digest() {
        let changed = FIXTURE.replace("benign_maintenance_access", "benign_maintenance_access ");
        assert_ne!(changed, FIXTURE);
        let report = verify_manifest(changed.as_bytes());
        assert_eq!(report.status, "REJECTED");
        assert!(report.errors.iter().any(|e| e.starts_with("fixture_digest_mismatch:")));
    }

}
