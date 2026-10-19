// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-layer audit bundle for the synthetic HABITAT-VERTICAL-001 benchmark.
//!
//! SHA-256 digests bind the manifest to exact input bytes. They do not
//! authenticate authors or sources and do not prove that any underlying claim
//! is true. Every component is rerun and each result is compared independently.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

pub const EMBEDDED_AUDIT_BUNDLE: &str =
    include_str!("../../../../docs/research/fixtures/habitat-audit-bundle-v0.1.json");

const DISPOSITIONS: [&str; 5] = [
    "Conforming", "Nonconforming", "Indeterminate", "Blocked", "NotEvaluable",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditBundleIssue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditBundleReport {
    pub issues: Vec<AuditBundleIssue>,
    pub artifact_count: usize,
    pub event_count: usize,
    pub material_transaction_count: usize,
    pub measurement_case_count: usize,
    pub measurement_dispositions: BTreeMap<String, usize>,
    pub recall_affected_terminal_count: usize,
    pub recall_unresolved_affected_count: usize,
    pub recall_closure_status: String,
    pub quantity_genealogy_operation_count: usize,
    pub quantity_genealogy_initial_mass_mg: u64,
    pub quantity_genealogy_ending_mass_mg: u64,
    pub quantity_genealogy_affected_terminal_count: usize,
    pub verified_sha256_by_artifact: BTreeMap<String, String>,
}

impl AuditBundleReport {
    /// True means exact fingerprints and all synthetic component known answers agree.
    /// It is not a physical-evidence or source-authentication judgment.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

pub fn validate_audit_bundle_str(
    manifest_input: &str,
    base_input: &str,
    event_input: &str,
    ledger_input: &str,
    measurement_input: &str,
    recall_input: &str,
    quantity_genealogy_input: &str,
) -> Result<AuditBundleReport, serde_json::Error> {
    let manifest: Value = serde_json::from_str(manifest_input)?;
    let base: Value = serde_json::from_str(base_input)?;
    let events: Value = serde_json::from_str(event_input)?;
    let ledger: Value = serde_json::from_str(ledger_input)?;
    let measurements: Value = serde_json::from_str(measurement_input)?;
    let recall: Value = serde_json::from_str(recall_input)?;
    let quantity_genealogy: Value = serde_json::from_str(quantity_genealogy_input)?;
    Ok(validate_audit_bundle(
        &manifest,
        &base,
        &events,
        &ledger,
        &measurements,
        &recall,
        &quantity_genealogy,
        base_input,
        event_input,
        ledger_input,
        measurement_input,
        recall_input,
        quantity_genealogy_input,
    ))
}

pub fn validate_embedded_audit_bundle() -> Result<AuditBundleReport, serde_json::Error> {
    validate_audit_bundle_str(
        EMBEDDED_AUDIT_BUNDLE,
        crate::EMBEDDED_FIXTURE,
        crate::event_replay::EMBEDDED_EVENT_LOG,
        crate::material_ledger::EMBEDDED_MATERIAL_LEDGER,
        crate::measurement_gate::EMBEDDED_MEASUREMENT_GATE,
        crate::recall_drill::EMBEDDED_RECALL_DRILL,
        crate::quantity_genealogy::EMBEDDED_QUANTITY_GENEALOGY,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn validate_audit_bundle(
    manifest: &Value,
    base: &Value,
    events: &Value,
    ledger: &Value,
    measurements: &Value,
    recall: &Value,
    quantity_genealogy: &Value,
    base_bytes: &str,
    event_bytes: &str,
    ledger_bytes: &str,
    measurement_bytes: &str,
    recall_bytes: &str,
    quantity_genealogy_bytes: &str,
) -> AuditBundleReport {
    let mut issues = Vec::new();
    expect_string(manifest, "schema_version", "0.1.0", &mut issues);
    expect_string(manifest, "bundle_id", "HABITAT-AUDIT-BUNDLE-001", &mut issues);
    expect_string(manifest, "program_id", "HABITAT-VERTICAL-001", &mut issues);
    expect_string(manifest, "evidence_status", "synthetic_audit_manifest", &mut issues);

    if !manifest.get("claim_ceiling").and_then(Value::as_str)
        .is_some_and(|text| text.contains("do not authenticate authors"))
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "the manifest must explicitly separate byte identity from source authenticity",
        );
    }
    if manifest.pointer("/fingerprint_policy/algorithm").and_then(Value::as_str)
        != Some("SHA-256")
        || manifest.pointer("/fingerprint_policy/canonicalization").and_then(Value::as_str)
            != Some("none; fingerprints cover exact bytes including JSON whitespace and final newline")
    {
        push(
            &mut issues,
            "FINGERPRINT_POLICY_UNSUPPORTED",
            "fingerprint_policy",
            "only SHA-256 over exact UTF-8 bytes without canonicalization is supported",
        );
    }
    if manifest.pointer("/fingerprint_policy/interpretation").and_then(Value::as_str)
        .is_none_or(|text| !text.contains("not a signature"))
    {
        push(
            &mut issues,
            "FINGERPRINT_CLAIM_OVERREACH",
            "fingerprint_policy.interpretation",
            "digest interpretation must disclaim signatures and source authentication",
        );
    }

    let artifacts = [
        ("product_process_fixture", "docs/research/fixtures/habitat-vertical-v0.1.json", base_bytes, base),
        ("event_replay_fixture", "docs/research/fixtures/habitat-event-replay-v0.1.json", event_bytes, events),
        ("material_ledger_fixture", "docs/research/fixtures/habitat-material-ledger-v0.1.json", ledger_bytes, ledger),
        ("measurement_gate_fixture", "docs/research/fixtures/habitat-measurement-gate-v0.1.json", measurement_bytes, measurements),
        ("recall_response_fixture", "docs/research/fixtures/habitat-recall-closure-v0.1.json", recall_bytes, recall),
        ("quantity_genealogy_fixture", "docs/research/fixtures/habitat-quantity-genealogy-v0.1.json", quantity_genealogy_bytes, quantity_genealogy),
    ];
    let known_ids: HashSet<&str> = artifacts.iter().map(|(id, _, _, _)| *id).collect();
    let Some(rows) = manifest.get("artifacts").and_then(Value::as_array) else {
        push(&mut issues, "ARTIFACTS_ARRAY_MISSING", "artifacts", "artifacts must be a JSON array");
        return report(issues, BTreeMap::new(), 0, 0, 0, BTreeMap::new(), 0, 0, "NOT_EVALUATED".to_owned(), 0, 0, 0, 0);
    };

    let mut manifest_by_id: HashMap<String, &Value> = HashMap::new();
    for (index, row) in rows.iter().enumerate() {
        let path = format!("artifacts[{index}]");
        let Some(id) = row.get("artifact_id").and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
        else {
            push(&mut issues, "ARTIFACT_ID_MISSING", format!("{path}.artifact_id"), "artifact ID must be a non-empty string");
            continue;
        };
        if !known_ids.contains(id) {
            push(&mut issues, "UNEXPECTED_ARTIFACT", format!("{path}.artifact_id"), format!("artifact {id:?} is not part of this six-fixture bundle"));
            continue;
        }
        if manifest_by_id.insert(id.to_owned(), row).is_some() {
            push(&mut issues, "DUPLICATE_ARTIFACT_ID", format!("{path}.artifact_id"), format!("artifact {id:?} appears more than once"));
        }
    }

    let mut verified_sha256_by_artifact = BTreeMap::new();
    for (id, expected_path, bytes, value) in artifacts {
        let Some(row) = manifest_by_id.get(id).copied() else {
            push(&mut issues, "ARTIFACT_SET_MISMATCH", "artifacts", format!("required artifact {id:?} is missing"));
            continue;
        };
        if row.get("path").and_then(Value::as_str) != Some(expected_path) {
            push(&mut issues, "ARTIFACT_PATH_MISMATCH", format!("artifacts[{id}].path"), format!("expected exact path {expected_path:?}"));
        }
        if row.get("byte_length").and_then(Value::as_u64) != Some(bytes.len() as u64) {
            push(&mut issues, "ARTIFACT_LENGTH_MISMATCH", format!("artifacts[{id}].byte_length"), format!("manifest length differs from actual UTF-8 byte length {}", bytes.len()));
        }
        let digest = sha256_hex(bytes.as_bytes());
        if row.get("sha256").and_then(Value::as_str) != Some(digest.as_str()) {
            push(&mut issues, "ARTIFACT_DIGEST_MISMATCH", format!("artifacts[{id}].sha256"), format!("SHA-256 does not match exact fixture bytes; actual digest is {digest}"));
        } else {
            verified_sha256_by_artifact.insert(id.to_owned(), digest);
        }
        let actual_schema = value.get("schema_version").and_then(Value::as_str);
        if row.get("schema_version").and_then(Value::as_str) != actual_schema {
            push(&mut issues, "ARTIFACT_SCHEMA_MISMATCH", format!("artifacts[{id}].schema_version"), "manifest schema version differs from the source fixture");
        }
        if row.get("evidence_status").and_then(Value::as_str)
            != value.get("evidence_status").and_then(Value::as_str)
        {
            push(
                &mut issues,
                "ARTIFACT_EVIDENCE_STATUS_MISMATCH",
                format!("artifacts[{id}].evidence_status"),
                "manifest evidence status must exactly match the source fixture status",
            );
        }
    }
    if rows.len() != artifacts.len() {
        push(
            &mut issues,
            "ARTIFACT_SET_MISMATCH",
            "artifacts",
            format!("exactly {} unique required artifact rows are expected, found {}", artifacts.len(), rows.len()),
        );
    }

    // Re-run all component validators; the manifest cannot override their findings.
    let structural = crate::validate_fixture(base);
    if !structural.is_valid() {
        for finding in &structural.issues {
            push(&mut issues, "STRUCTURAL_COMPONENT_INVALID", format!("base.{}", finding.path), format!("{}: {}", finding.code, finding.message));
        }
    }

    let event_report = crate::event_replay::validate_event_log(events, base);
    if !event_report.is_valid() {
        for finding in &event_report.issues {
            push(&mut issues, "EVENT_COMPONENT_INVALID", format!("events.{}", finding.path), format!("{}: {}", finding.code, finding.message));
        }
    }

    let ledger_report = crate::material_ledger::validate_material_ledger(ledger, base, events);
    if !ledger_report.is_valid() {
        for finding in &ledger_report.issues {
            push(&mut issues, "LEDGER_COMPONENT_INVALID", format!("ledger.{}", finding.path), format!("{}: {}", finding.code, finding.message));
        }
    }

    let measurement_report = crate::measurement_gate::validate_measurement_gate(measurements);
    if !measurement_report.is_valid() {
        for finding in &measurement_report.issues {
            push(&mut issues, "MEASUREMENT_COMPONENT_INVALID", format!("measurements.{}", finding.path), format!("{}: {}", finding.code, finding.message));
        }
    }

    let recall_report = crate::recall_drill::validate_recall_drill(recall, base, events);
    if !recall_report.is_valid() {
        for finding in &recall_report.issues {
            push(&mut issues, "RECALL_COMPONENT_INVALID", format!("recall.{}", finding.path), format!("{}: {}", finding.code, finding.message));
        }
    }
    let quantity_report = crate::quantity_genealogy::validate_quantity_genealogy(quantity_genealogy);
    if !quantity_report.is_valid() {
        for finding in &quantity_report.issues {
            push(&mut issues, "QUANTITY_GENEALOGY_COMPONENT_INVALID", format!("quantity_genealogy.{}", finding.path), format!("{}: {}", finding.code, finding.message));
        }
    }

    // Compare component outputs with the manifest's independent known answers.
    compare_u64(
        manifest,
        "/expected_results/event_replay/unique_event_count",
        event_report.unique_event_count as u64,
        &mut issues,
    );
    compare_u64(
        manifest,
        "/expected_results/event_replay/duplicate_delivery_count",
        event_report.duplicate_delivery_count as u64,
        &mut issues,
    );
    compare_u64(
        manifest,
        "/expected_results/material_ledger/account_count",
        ledger_report.account_count as u64,
        &mut issues,
    );
    compare_u64(
        manifest,
        "/expected_results/material_ledger/transaction_count",
        ledger_report.transaction_count as u64,
        &mut issues,
    );
    compare_u64(
        manifest,
        "/expected_results/material_ledger/opening_mass_mg",
        ledger_report.opening_mass_mg,
        &mut issues,
    );
    compare_u64(
        manifest,
        "/expected_results/material_ledger/closing_mass_mg",
        ledger_report.closing_mass_mg,
        &mut issues,
    );
    compare_u64(
        manifest,
        "/expected_results/material_ledger/nonzero_closing_account_count",
        ledger_report.nonzero_closing_account_count as u64,
        &mut issues,
    );
    compare_u64(
        manifest,
        "/expected_results/measurement_gate/case_count",
        measurement_report.case_count as u64,
        &mut issues,
    );

    let mut measurement_dispositions = BTreeMap::new();
    for disposition in DISPOSITIONS {
        let count = measurement_report.results.iter()
            .filter(|result| result.disposition == disposition)
            .count();
        measurement_dispositions.insert(disposition.to_owned(), count);
        let expected = manifest
            .pointer(&format!("/expected_results/measurement_gate/disposition_counts/{disposition}"))
            .and_then(Value::as_u64);
        if expected != Some(count as u64) {
            push(
                &mut issues,
                "EXPECTED_RESULT_MISMATCH",
                format!("expected_results.measurement_gate.disposition_counts.{disposition}"),
                format!("manifest expected {expected:?}; re-evaluated component result is {count}"),
            );
        }
    }

    compare_u64(manifest, "/expected_results/recall_closure/affected_terminal_count", recall_report.affected_terminal_ids.len() as u64, &mut issues);
    compare_u64(manifest, "/expected_results/recall_closure/target_count", recall_report.target_count as u64, &mut issues);
    compare_u64(manifest, "/expected_results/recall_closure/response_record_count", recall_report.response_record_count as u64, &mut issues);
    compare_u64(manifest, "/expected_results/recall_closure/confirmed_contained_count", recall_report.confirmed_contained_count as u64, &mut issues);
    compare_u64(manifest, "/expected_results/recall_closure/unresolved_affected_count", recall_report.unresolved_affected_count as u64, &mut issues);
    compare_u64(manifest, "/expected_results/recall_closure/false_positive_control_count", recall_report.false_positive_control_count as u64, &mut issues);
    if manifest.pointer("/expected_results/recall_closure/closure_status").and_then(Value::as_str) != Some(recall_report.closure_status.as_str()) {
        push(&mut issues, "EXPECTED_RESULT_MISMATCH", "expected_results.recall_closure.closure_status", format!("manifest expected {:?}; re-evaluated recall status is {}", manifest.pointer("/expected_results/recall_closure/closure_status").and_then(Value::as_str), recall_report.closure_status));
    }

    compare_u64(manifest, "/expected_results/quantity_genealogy/operation_count", quantity_report.operation_count as u64, &mut issues);
    compare_u64(manifest, "/expected_results/quantity_genealogy/initial_mass_mg", quantity_report.initial_mass_mg, &mut issues);
    compare_u64(manifest, "/expected_results/quantity_genealogy/ending_mass_mg", quantity_report.ending_mass_mg, &mut issues);
    compare_u64(manifest, "/expected_results/quantity_genealogy/terminal_count", quantity_report.terminal_count as u64, &mut issues);
    compare_u64(manifest, "/expected_results/quantity_genealogy/affected_terminal_count", quantity_report.affected_terminal_ids.len() as u64, &mut issues);
    let declared_qg: BTreeSet<String> = manifest.pointer("/expected_results/quantity_genealogy/affected_terminal_ids").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(ToOwned::to_owned).collect();
    let actual_qg: BTreeSet<String> = quantity_report.affected_terminal_ids.iter().cloned().collect();
    if declared_qg != actual_qg {
        push(&mut issues, "EXPECTED_RESULT_MISMATCH", "expected_results.quantity_genealogy.affected_terminal_ids", format!("manifest expected {declared_qg:?}; component derived {actual_qg:?}"));
    }

    // Cross-layer invariants that make this a bundle rather than six independent passes.
    let actual_measurement_evidence = measurements.get("evidence_status").and_then(Value::as_str);
    if actual_measurement_evidence != Some("synthetic_decision_rule_challenge_only") {
        push(&mut issues, "CROSS_LAYER_EVIDENCE_STATUS_MISMATCH", "measurements.evidence_status", "measurement fixture evidence status is outside the declared synthetic boundary");
    }
    if base.get("evidence_status").and_then(Value::as_str) != Some("synthetic_only") {
        push(&mut issues, "CROSS_LAYER_EVIDENCE_STATUS_MISMATCH", "base.evidence_status", "product/process fixture is not synthetic-only");
    }
    if events.get("evidence_status").and_then(Value::as_str) != Some("synthetic_contract_only_not_executed") {
        push(&mut issues, "CROSS_LAYER_EVIDENCE_STATUS_MISMATCH", "events.evidence_status", "event replay fixture is not a not-executed synthetic contract");
    }
    if ledger.get("evidence_status").and_then(Value::as_str) != Some("synthetic_contract_only_not_executed") {
        push(&mut issues, "CROSS_LAYER_EVIDENCE_STATUS_MISMATCH", "ledger.evidence_status", "material ledger fixture is not a not-executed synthetic contract");
    }
    if recall.get("evidence_status").and_then(Value::as_str) != Some("synthetic_rehearsal_contract_only_not_executed") {
        push(&mut issues, "CROSS_LAYER_EVIDENCE_STATUS_MISMATCH", "recall.evidence_status", "recall-response fixture is not explicitly synthetic and not executed");
    }
    if quantity_genealogy.get("evidence_status").and_then(Value::as_str) != Some("synthetic_quantity_genealogy_challenge_only_not_executed") {
        push(&mut issues, "CROSS_LAYER_EVIDENCE_STATUS_MISMATCH", "quantity_genealogy.evidence_status", "quantity-genealogy fixture is not explicitly synthetic and not executed");
    }
    if manifest.pointer("/expected_results/structural_fixture/valid").and_then(Value::as_bool) != Some(structural.is_valid()) {
        push(&mut issues, "EXPECTED_RESULT_MISMATCH", "expected_results.structural_fixture.valid", "structural validation result differs from the manifest");
    }

    report(
        issues,
        verified_sha256_by_artifact,
        event_report.unique_event_count,
        ledger_report.transaction_count,
        measurement_report.case_count,
        measurement_dispositions,
        recall_report.affected_terminal_ids.len(),
        recall_report.unresolved_affected_count,
        recall_report.closure_status.clone(),
        quantity_report.operation_count,
        quantity_report.initial_mass_mg,
        quantity_report.ending_mass_mg,
        quantity_report.affected_terminal_ids.len(),
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn compare_u64(manifest: &Value, pointer: &str, actual: u64, issues: &mut Vec<AuditBundleIssue>) {
    if manifest.pointer(pointer).and_then(Value::as_u64) != Some(actual) {
        push(
            issues,
            "EXPECTED_RESULT_MISMATCH",
            pointer.trim_start_matches('/').replace('/', "."),
            format!("manifest expected {:?}; re-evaluated component result is {actual}", manifest.pointer(pointer).and_then(Value::as_u64)),
        );
    }
}

fn expect_string(root: &Value, field: &str, expected: &str, issues: &mut Vec<AuditBundleIssue>) {
    if root.get(field).and_then(Value::as_str) != Some(expected) {
        push(issues, "EXPECTED_STRING_MISMATCH", field, format!("required value is {expected:?}"));
    }
}

fn push(
    issues: &mut Vec<AuditBundleIssue>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(AuditBundleIssue { code, path: path.into(), message: message.into() });
}

fn report(
    mut issues: Vec<AuditBundleIssue>,
    verified_sha256_by_artifact: BTreeMap<String, String>,
    event_count: usize,
    material_transaction_count: usize,
    measurement_case_count: usize,
    measurement_dispositions: BTreeMap<String, usize>,
    recall_affected_terminal_count: usize,
    recall_unresolved_affected_count: usize,
    recall_closure_status: String,
    quantity_genealogy_operation_count: usize,
    quantity_genealogy_initial_mass_mg: u64,
    quantity_genealogy_ending_mass_mg: u64,
    quantity_genealogy_affected_terminal_count: usize,
) -> AuditBundleReport {
    issues.sort_by(|a, b| a.code.cmp(b.code).then_with(|| a.path.cmp(&b.path)).then_with(|| a.message.cmp(&b.message)));
    AuditBundleReport {
        artifact_count: verified_sha256_by_artifact.len(),
        issues,
        event_count,
        material_transaction_count,
        measurement_case_count,
        measurement_dispositions,
        recall_affected_terminal_count,
        recall_unresolved_affected_count,
        recall_closure_status,
        quantity_genealogy_operation_count,
        quantity_genealogy_initial_mass_mg,
        quantity_genealogy_ending_mass_mg,
        quantity_genealogy_affected_terminal_count,
        verified_sha256_by_artifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(
        manifest: &str,
        base: &str,
        events: &str,
        ledger: &str,
        measurements: &str,
        recall: &str,
        quantity_genealogy: &str,
    ) -> AuditBundleReport {
        validate_audit_bundle_str(manifest, base, events, ledger, measurements, recall, quantity_genealogy).expect("all input fixtures are JSON")
    }

    fn texts() -> (&'static str, &'static str, &'static str, &'static str, &'static str, &'static str, &'static str) {
        (
            EMBEDDED_AUDIT_BUNDLE,
            crate::EMBEDDED_FIXTURE,
            crate::event_replay::EMBEDDED_EVENT_LOG,
            crate::material_ledger::EMBEDDED_MATERIAL_LEDGER,
            crate::measurement_gate::EMBEDDED_MEASUREMENT_GATE,
            crate::recall_drill::EMBEDDED_RECALL_DRILL,
            crate::quantity_genealogy::EMBEDDED_QUANTITY_GENEALOGY,
        )
    }

    fn has_code(report: &AuditBundleReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_bundle_replays_every_layer_and_hash() {
        let (m, b, e, l, q, r, g) = texts();
        let report = validate(m, b, e, l, q, r, g);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.artifact_count, 6);
        assert_eq!(report.event_count, 22);
        assert_eq!(report.material_transaction_count, 9);
        assert_eq!(report.measurement_case_count, 13);
        assert_eq!(report.recall_affected_terminal_count, 2);
        assert_eq!(report.recall_unresolved_affected_count, 1);
        assert_eq!(report.recall_closure_status, "OPEN_UNRESOLVED");
        assert_eq!(report.quantity_genealogy_operation_count, 8);
        assert_eq!(report.quantity_genealogy_initial_mass_mg, 2700);
        assert_eq!(report.quantity_genealogy_ending_mass_mg, 2700);
        assert_eq!(report.quantity_genealogy_affected_terminal_count, 2);
    }

    #[test]
    fn stale_event_fingerprint_fails_closed() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = e.replacen("EV-22-UNRELATED-DISPATCH", "EV-99-ALTERED-DISPATCH", 1);
        let report = validate(m, b, &altered, l, q, r, g);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
    }

    #[test]
    fn incorrect_byte_length_fails_closed() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["artifacts"][0]["byte_length"] = Value::from(1);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "ARTIFACT_LENGTH_MISMATCH"));
    }

    #[test]
    fn omitted_artifact_is_rejected() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["artifacts"].as_array_mut().expect("artifacts").remove(0);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "ARTIFACT_SET_MISMATCH"));
    }

    #[test]
    fn duplicated_artifact_identity_is_rejected() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        let first = manifest["artifacts"][0].clone();
        manifest["artifacts"].as_array_mut().expect("artifacts").push(first);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "DUPLICATE_ARTIFACT_ID"));
    }

    #[test]
    fn unsupported_hash_policy_is_rejected() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["fingerprint_policy"]["algorithm"] = Value::String("MD5".to_owned());
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "FINGERPRINT_POLICY_UNSUPPORTED"));
    }

    #[test]
    fn component_known_answer_cannot_be_overridden() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["expected_results"]["event_replay"]["unique_event_count"] = Value::from(21);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "EXPECTED_RESULT_MISMATCH"));
    }

    #[test]
    fn ledger_mutation_is_not_hidden_by_manifest() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = l.replacen("\"quantity_mg\": 480000", "\"quantity_mg\": 479999", 1);
        let report = validate(m, b, e, &altered, q, r, g);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "LEDGER_COMPONENT_INVALID"));
    }

    #[test]
    fn measurement_mutation_is_not_hidden_by_manifest() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = q.replacen("\"expected_disposition\": \"Conforming\"", "\"expected_disposition\": \"Indeterminate\"", 1);
        let report = validate(m, b, e, l, &altered, r, g);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "MEASUREMENT_COMPONENT_INVALID"));
    }

    #[test]
    fn artifact_order_does_not_change_audit_outcome() {
        let (m, b, e, l, q, r, g) = texts();
        let baseline = validate(m, b, e, l, q, r, g);
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["artifacts"].as_array_mut().expect("artifacts").reverse();
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let reordered = validate(&serialized, b, e, l, q, r, g);
        assert!(baseline.is_valid(), "{:?}", baseline.issues);
        assert!(reordered.is_valid(), "{:?}", reordered.issues);
        assert_eq!(baseline.verified_sha256_by_artifact, reordered.verified_sha256_by_artifact);
    }

    #[test]
    fn source_evidence_status_remains_synthetic_only() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = b.replacen("\"evidence_status\": \"synthetic_only\"", "\"evidence_status\": \"observed_physical\"", 1);
        let report = validate(m, &altered, e, l, q, r, g);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "STRUCTURAL_COMPONENT_INVALID"));
    }

    #[test]
    fn manifest_cannot_claim_authenticity_from_fingerprints() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["fingerprint_policy"]["interpretation"] = Value::String("a signature and proof that the records are true".to_owned());
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "FINGERPRINT_CLAIM_OVERREACH"));
    }

    #[test]
    fn measurement_disposition_counts_remain_separate() {
        let (m, b, e, l, q, r, g) = texts();
        let report = validate(m, b, e, l, q, r, g);
        assert_eq!(report.measurement_dispositions["Conforming"], 1);
        assert_eq!(report.measurement_dispositions["Indeterminate"], 1);
        assert_eq!(report.measurement_dispositions["NotEvaluable"], 5);
    }


    #[test]
    fn stale_recall_fixture_fingerprint_fails_closed() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = r.replacen("SYN-RESPONSE-RECORD-A", "SYN-RESPONSE-RECORD-X", 1);
        let report = validate(m, b, e, l, q, &altered, g);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
    }

    #[test]
    fn recall_known_answer_cannot_be_overridden_by_manifest() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["expected_results"]["recall_closure"]["unresolved_affected_count"] = Value::from(0);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "EXPECTED_RESULT_MISMATCH"));
    }

    #[test]
    fn recall_component_failure_is_not_hidden_by_a_valid_manifest() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = r.replacen("\"evidence_status\": \"synthetic_only\"", "\"evidence_status\": \"observed_physical\"", 1);
        let report = validate(m, b, e, l, q, &altered, g);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "RECALL_COMPONENT_INVALID"));
    }


    #[test]
    fn stale_quantity_genealogy_fingerprint_fails_closed() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = g.replacen("LOT-SUSPECT-FIBER", "LOT-SUSPECT-X", 1);
        let report = validate(m, b, e, l, q, r, &altered);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
    }

    #[test]
    fn quantity_genealogy_known_answer_cannot_be_overridden_by_manifest() {
        let (m, b, e, l, q, r, g) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["expected_results"]["quantity_genealogy"]["ending_mass_mg"] = Value::from(2600);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q, r, g);
        assert!(has_code(&report, "EXPECTED_RESULT_MISMATCH"));
    }

    #[test]
    fn quantity_genealogy_component_failure_is_not_hidden_by_manifest() {
        let (m, b, e, l, q, r, g) = texts();
        let altered = g.replacen("\"mass_unit\": \"mg\"", "\"mass_unit\": \"kg\"", 1);
        let report = validate(m, b, e, l, q, r, &altered);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "QUANTITY_GENEALOGY_COMPONENT_INVALID"));
    }
}
