// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-layer audit bundle for the synthetic HABITAT-VERTICAL-001 benchmark.
//!
//! SHA-256 digests bind the manifest to exact input bytes. They do not
//! authenticate authors or sources and do not prove that any underlying claim
//! is true. Every component is rerun and each result is compared independently.

use std::collections::{BTreeMap, HashMap, HashSet};

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
) -> Result<AuditBundleReport, serde_json::Error> {
    let manifest: Value = serde_json::from_str(manifest_input)?;
    let base: Value = serde_json::from_str(base_input)?;
    let events: Value = serde_json::from_str(event_input)?;
    let ledger: Value = serde_json::from_str(ledger_input)?;
    let measurements: Value = serde_json::from_str(measurement_input)?;
    Ok(validate_audit_bundle(
        &manifest,
        &base,
        &events,
        &ledger,
        &measurements,
        base_input,
        event_input,
        ledger_input,
        measurement_input,
    ))
}

pub fn validate_embedded_audit_bundle() -> Result<AuditBundleReport, serde_json::Error> {
    validate_audit_bundle_str(
        EMBEDDED_AUDIT_BUNDLE,
        crate::EMBEDDED_FIXTURE,
        crate::event_replay::EMBEDDED_EVENT_LOG,
        crate::material_ledger::EMBEDDED_MATERIAL_LEDGER,
        crate::measurement_gate::EMBEDDED_MEASUREMENT_GATE,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn validate_audit_bundle(
    manifest: &Value,
    base: &Value,
    events: &Value,
    ledger: &Value,
    measurements: &Value,
    base_bytes: &str,
    event_bytes: &str,
    ledger_bytes: &str,
    measurement_bytes: &str,
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
    ];
    let known_ids: HashSet<&str> = artifacts.iter().map(|(id, _, _, _)| *id).collect();
    let Some(rows) = manifest.get("artifacts").and_then(Value::as_array) else {
        push(&mut issues, "ARTIFACTS_ARRAY_MISSING", "artifacts", "artifacts must be a JSON array");
        return report(issues, BTreeMap::new(), 0, 0, 0, BTreeMap::new());
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
            push(&mut issues, "UNEXPECTED_ARTIFACT", format!("{path}.artifact_id"), format!("artifact {id:?} is not part of this four-fixture bundle"));
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

    // Cross-layer invariants that make this a bundle rather than four independent passes.
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
) -> AuditBundleReport {
    issues.sort_by(|a, b| a.code.cmp(b.code).then_with(|| a.path.cmp(&b.path)).then_with(|| a.message.cmp(&b.message)));
    AuditBundleReport {
        artifact_count: verified_sha256_by_artifact.len(),
        issues,
        event_count,
        material_transaction_count,
        measurement_case_count,
        measurement_dispositions,
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
    ) -> AuditBundleReport {
        validate_audit_bundle_str(manifest, base, events, ledger, measurements).expect("all input fixtures are JSON")
    }

    fn texts() -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
        (
            EMBEDDED_AUDIT_BUNDLE,
            crate::EMBEDDED_FIXTURE,
            crate::event_replay::EMBEDDED_EVENT_LOG,
            crate::material_ledger::EMBEDDED_MATERIAL_LEDGER,
            crate::measurement_gate::EMBEDDED_MEASUREMENT_GATE,
        )
    }

    fn has_code(report: &AuditBundleReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_bundle_replays_every_layer_and_hash() {
        let (m, b, e, l, q) = texts();
        let report = validate(m, b, e, l, q);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.artifact_count, 4);
        assert_eq!(report.event_count, 22);
        assert_eq!(report.material_transaction_count, 9);
        assert_eq!(report.measurement_case_count, 13);
    }

    #[test]
    fn stale_event_fingerprint_fails_closed() {
        let (m, b, e, l, q) = texts();
        let altered = e.replacen("EV-22-UNRELATED-DISPATCH", "EV-99-ALTERED-DISPATCH", 1);
        let report = validate(m, b, &altered, l, q);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
    }

    #[test]
    fn incorrect_byte_length_fails_closed() {
        let (m, b, e, l, q) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["artifacts"][0]["byte_length"] = Value::from(1);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q);
        assert!(has_code(&report, "ARTIFACT_LENGTH_MISMATCH"));
    }

    #[test]
    fn omitted_artifact_is_rejected() {
        let (m, b, e, l, q) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["artifacts"].as_array_mut().expect("artifacts").remove(0);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q);
        assert!(has_code(&report, "ARTIFACT_SET_MISMATCH"));
    }

    #[test]
    fn duplicated_artifact_identity_is_rejected() {
        let (m, b, e, l, q) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        let first = manifest["artifacts"][0].clone();
        manifest["artifacts"].as_array_mut().expect("artifacts").push(first);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q);
        assert!(has_code(&report, "DUPLICATE_ARTIFACT_ID"));
    }

    #[test]
    fn unsupported_hash_policy_is_rejected() {
        let (m, b, e, l, q) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["fingerprint_policy"]["algorithm"] = Value::String("MD5".to_owned());
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q);
        assert!(has_code(&report, "FINGERPRINT_POLICY_UNSUPPORTED"));
    }

    #[test]
    fn component_known_answer_cannot_be_overridden() {
        let (m, b, e, l, q) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["expected_results"]["event_replay"]["unique_event_count"] = Value::from(21);
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q);
        assert!(has_code(&report, "EXPECTED_RESULT_MISMATCH"));
    }

    #[test]
    fn ledger_mutation_is_not_hidden_by_manifest() {
        let (m, b, e, l, q) = texts();
        let altered = l.replacen("\"quantity_mg\": 480000", "\"quantity_mg\": 479999", 1);
        let report = validate(m, b, e, &altered, q);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "LEDGER_COMPONENT_INVALID"));
    }

    #[test]
    fn measurement_mutation_is_not_hidden_by_manifest() {
        let (m, b, e, l, q) = texts();
        let altered = q.replacen("\"expected_disposition\": \"Conforming\"", "\"expected_disposition\": \"Indeterminate\"", 1);
        let report = validate(m, b, e, l, &altered);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "MEASUREMENT_COMPONENT_INVALID"));
    }

    #[test]
    fn artifact_order_does_not_change_audit_outcome() {
        let (m, b, e, l, q) = texts();
        let baseline = validate(m, b, e, l, q);
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["artifacts"].as_array_mut().expect("artifacts").reverse();
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let reordered = validate(&serialized, b, e, l, q);
        assert!(baseline.is_valid(), "{:?}", baseline.issues);
        assert!(reordered.is_valid(), "{:?}", reordered.issues);
        assert_eq!(baseline.verified_sha256_by_artifact, reordered.verified_sha256_by_artifact);
    }

    #[test]
    fn source_evidence_status_remains_synthetic_only() {
        let (m, b, e, l, q) = texts();
        let altered = b.replacen("\"evidence_status\": \"synthetic_only\"", "\"evidence_status\": \"observed_physical\"", 1);
        let report = validate(m, &altered, e, l, q);
        assert!(has_code(&report, "ARTIFACT_DIGEST_MISMATCH"));
        assert!(has_code(&report, "STRUCTURAL_COMPONENT_INVALID"));
    }

    #[test]
    fn manifest_cannot_claim_authenticity_from_fingerprints() {
        let (m, b, e, l, q) = texts();
        let mut manifest: Value = serde_json::from_str(m).expect("manifest JSON");
        manifest["fingerprint_policy"]["interpretation"] = Value::String("a signature and proof that the records are true".to_owned());
        let serialized = serde_json::to_string(&manifest).expect("manifest serialization");
        let report = validate(&serialized, b, e, l, q);
        assert!(has_code(&report, "FINGERPRINT_CLAIM_OVERREACH"));
    }

    #[test]
    fn measurement_disposition_counts_remain_separate() {
        let (m, b, e, l, q) = texts();
        let report = validate(m, b, e, l, q);
        assert_eq!(report.measurement_dispositions["Conforming"], 1);
        assert_eq!(report.measurement_dispositions["Indeterminate"], 1);
        assert_eq!(report.measurement_dispositions["NotEvaluable"], 5);
    }
}
