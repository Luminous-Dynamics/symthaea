// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Separately compiled oracle for the evidence-bundle byte contract.
//! This intentionally reimplements canonical encodings and digest computation
//! instead of calling the production canonicalization helpers to create receipts.

use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};
use symthaea_operations_research::aviation_evidence_bundle::{
    verify_evidence_bundle, EvidenceBundleStatus, EvidenceManifest, EvidenceVerificationPolicy, ExpectedSourceIdentity, TrustAnchor,
    VerifierReceipt, EVIDENCE_MANIFEST_SCHEMA_VERSION, VERIFIER_RECEIPT_SCHEMA_VERSION,
};

const NOW: i64 = 1_800_000_000;


fn default_policy() -> EvidenceVerificationPolicy {
    EvidenceVerificationPolicy {
        max_manifest_age_seconds: 3_600,
        max_decision_age_seconds: 900,
    }
}

fn verify_fixture_with_policy(
    manifest: &EvidenceManifest,
    receipts: &[VerifierReceipt],
    anchors: &[TrustAnchor],
    policy: &EvidenceVerificationPolicy,
    now: i64,
) -> symthaea_operations_research::aviation_evidence_bundle::EvidenceBundleReport {
    let expected = ExpectedSourceIdentity {
        commit_sha: &manifest.source_commit_sha,
        tree_sha256: &manifest.source_tree_sha256,
    };
    verify_evidence_bundle(manifest, &expected, policy, receipts, anchors, now)
}

fn verify_fixture(
    manifest: &EvidenceManifest,
    receipts: &[VerifierReceipt],
    anchors: &[TrustAnchor],
    now: i64,
) -> symthaea_operations_research::aviation_evidence_bundle::EvidenceBundleReport {
    let expected = ExpectedSourceIdentity {
        commit_sha: &manifest.source_commit_sha,
        tree_sha256: &manifest.source_tree_sha256,
    };
    let policy = default_policy();
    verify_evidence_bundle(manifest, &expected, &policy, receipts, anchors, now)
}

fn independent_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut text, "{byte:02x}").unwrap();
    }
    text
}

fn independent_field(out: &mut Vec<u8>, key: &str, value: &str) {
    out.extend_from_slice(key.as_bytes());
    out.push(b':');
    out.extend_from_slice(value.as_bytes().len().to_string().as_bytes());
    out.push(b':');
    out.extend_from_slice(value.as_bytes());
    out.push(b'\n');
}

fn independent_manifest_bytes(m: &EvidenceManifest) -> Vec<u8> {
    let mut out = Vec::new();
    independent_field(&mut out, "domain", "luminous-dynamics.aviation-evidence-manifest.v1");
    independent_field(&mut out, "schema_version", &m.schema_version.to_string());
    independent_field(&mut out, "run_id", &m.run_id);
    independent_field(&mut out, "created_at_epoch_seconds", &m.created_at_epoch_seconds.to_string());
    independent_field(&mut out, "decision_time_epoch_seconds", &m.decision_time_epoch_seconds.to_string());
    independent_field(&mut out, "source_commit_sha", &m.source_commit_sha);
    independent_field(&mut out, "source_tree_sha256", &m.source_tree_sha256);
    independent_field(&mut out, "data_snapshot_sha256", &m.data_snapshot_sha256);
    independent_field(&mut out, "scenario_sha256", &m.scenario_sha256);
    independent_field(&mut out, "model_sha256", &m.model_sha256);
    independent_field(&mut out, "policy_config_sha256", &m.policy_config_sha256);
    independent_field(&mut out, "candidate_plan_sha256", &m.candidate_plan_sha256);
    independent_field(&mut out, "calibration_artifact_sha256", &m.calibration_artifact_sha256);
    independent_field(&mut out, "drift_report_sha256", &m.drift_report_sha256);
    independent_field(&mut out, "plan_verification_report_sha256", &m.plan_verification_report_sha256);
    independent_field(&mut out, "distributional_assessment_sha256", &m.distributional_assessment_sha256);
    independent_field(&mut out, "network_assessment_sha256", &m.network_assessment_sha256);
    out
}

fn independent_receipt_bytes(r: &VerifierReceipt) -> Vec<u8> {
    let mut out = Vec::new();
    independent_field(&mut out, "domain", "luminous-dynamics.aviation-verifier-receipt.v1");
    independent_field(&mut out, "schema_version", &r.schema_version.to_string());
    independent_field(&mut out, "receipt_id", &r.receipt_id);
    independent_field(&mut out, "role", &r.role);
    independent_field(&mut out, "manifest_sha256", &r.manifest_sha256);
    independent_field(&mut out, "subject_sha256", &r.subject_sha256);
    independent_field(&mut out, "verifier_id", &r.verifier_id);
    independent_field(&mut out, "verifier_binary_sha256", &r.verifier_binary_sha256);
    independent_field(&mut out, "key_id", &r.key_id);
    independent_field(&mut out, "issued_at_epoch_seconds", &r.issued_at_epoch_seconds.to_string());
    independent_field(&mut out, "expires_at_epoch_seconds", &r.expires_at_epoch_seconds.to_string());
    out
}

fn independent_hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut text, "{byte:02x}").unwrap();
    }
    text
}

fn digest_label(label: &str) -> String {
    independent_sha256(label.as_bytes())
}

fn subject(m: &EvidenceManifest, role: &str) -> &str {
    match role {
        "calibration" => &m.calibration_artifact_sha256,
        "drift_monitor" => &m.drift_report_sha256,
        "plan_feasibility" => &m.plan_verification_report_sha256,
        "distributional_risk" => &m.distributional_assessment_sha256,
        "network_harm" => &m.network_assessment_sha256,
        _ => unreachable!("fixture uses only declared roles"),
    }
}

fn fixture_manifest() -> EvidenceManifest {
    EvidenceManifest {
        schema_version: EVIDENCE_MANIFEST_SCHEMA_VERSION,
        run_id: "independent-oracle-run".into(),
        created_at_epoch_seconds: NOW - 60,
        decision_time_epoch_seconds: NOW - 30,
        source_commit_sha: "b".repeat(40),
        source_tree_sha256: digest_label("tree"),
        data_snapshot_sha256: digest_label("data"),
        scenario_sha256: digest_label("scenario"),
        model_sha256: digest_label("model"),
        policy_config_sha256: digest_label("policy"),
        candidate_plan_sha256: digest_label("plan"),
        calibration_artifact_sha256: digest_label("calibration"),
        drift_report_sha256: digest_label("drift"),
        plan_verification_report_sha256: digest_label("plan-verification"),
        distributional_assessment_sha256: digest_label("distributional"),
        network_assessment_sha256: digest_label("network"),
    }
}

fn signed_bundle(m: &EvidenceManifest) -> (Vec<VerifierReceipt>, Vec<TrustAnchor>) {
    let manifest_digest = independent_sha256(&independent_manifest_bytes(m));
    let roles = [
        "calibration",
        "drift_monitor",
        "plan_feasibility",
        "distributional_risk",
        "network_harm",
    ];
    let mut receipts = Vec::new();
    let mut anchors = Vec::new();
    for (i, role) in roles.into_iter().enumerate() {
        let seed = [u8::try_from(i + 41).unwrap(); 32];
        let signing_key = SigningKey::from_bytes(&seed);
        let key_id = format!("oracle-key-{role}");
        let verifier_id = format!("oracle-verifier-{role}");
        let verifier_binary_sha256 = digest_label(&format!("verifier-binary-{role}"));
        let mut receipt = VerifierReceipt {
            schema_version: VERIFIER_RECEIPT_SCHEMA_VERSION,
            receipt_id: format!("oracle-receipt-{role}"),
            role: role.into(),
            manifest_sha256: manifest_digest.clone(),
            subject_sha256: subject(m, role).into(),
            verifier_id: verifier_id.clone(),
            verifier_binary_sha256: verifier_binary_sha256.clone(),
            key_id: key_id.clone(),
            issued_at_epoch_seconds: NOW - 5,
            expires_at_epoch_seconds: NOW + 120,
            signature_hex: String::new(),
        };
        receipt.signature_hex = independent_hex(
            &signing_key.sign(&independent_receipt_bytes(&receipt)).to_bytes(),
        );
        receipts.push(receipt);
        anchors.push(TrustAnchor {
            key_id,
            role: role.into(),
            verifier_id,
            verifier_binary_sha256,
            public_key_hex: independent_hex(&signing_key.verifying_key().to_bytes()),
            active_from_epoch_seconds: NOW - 1000,
            active_until_epoch_seconds: NOW + 1000,
            max_receipt_lifetime_seconds: 600,
            revoked: false,
        });
    }
    (receipts, anchors)
}

#[test]
fn independent_byte_contract_matches_canonical_api_and_accepts_signed_bundle() {
    let manifest = fixture_manifest();
    assert_eq!(
        independent_manifest_bytes(&manifest),
        symthaea_operations_research::aviation_evidence_bundle::canonical_manifest_bytes(&manifest),
    );
    assert_eq!(
        independent_sha256(&independent_manifest_bytes(&manifest)),
        symthaea_operations_research::aviation_evidence_bundle::sha256_hex(
            &symthaea_operations_research::aviation_evidence_bundle::canonical_manifest_bytes(&manifest),
        ),
    );
    let (receipts, anchors) = signed_bundle(&manifest);
    for receipt in &receipts {
        assert_eq!(
            independent_receipt_bytes(receipt),
            symthaea_operations_research::aviation_evidence_bundle::canonical_receipt_signing_bytes(receipt),
        );
    }
    let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
    assert_eq!(report.status, EvidenceBundleStatus::Verified);
    assert!(report.blockers.is_empty());
}

#[test]
fn independent_oracle_detects_mutated_snapshot_and_permutation_is_irrelevant() {
    let manifest = fixture_manifest();
    let (receipts, anchors) = signed_bundle(&manifest);
    let first = verify_fixture(&manifest, &receipts, &anchors, NOW);
    let mut reversed = receipts.clone();
    reversed.reverse();
    let second = verify_fixture(&manifest, &reversed, &anchors, NOW);
    assert_eq!(first, second);

    let mut mutated = manifest;
    mutated.data_snapshot_sha256 = digest_label("mutated-data");
    let blocked = verify_fixture(&mutated, &receipts, &anchors, NOW);
    assert_eq!(blocked.status, EvidenceBundleStatus::Blocked);
    assert!(blocked.blockers.iter().any(|b| {
        matches!(b, symthaea_operations_research::aviation_evidence_bundle::EvidenceBundleBlocker::ReceiptManifestMismatch { .. })
    }));
}


#[test]
fn unrecognized_wire_fields_are_rejected_on_security_boundaries() {
    let manifest = fixture_manifest();
    let manifest_json = serde_json::to_value(&manifest).unwrap();
    let mut manifest_object = manifest_json.as_object().unwrap().clone();
    manifest_object.insert("unexpected_override".into(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<EvidenceManifest>(
        serde_json::Value::Object(manifest_object)
    ).is_err());

    let (receipts, anchors) = signed_bundle(&manifest);
    let receipt_json = serde_json::to_value(&receipts[0]).unwrap();
    let mut receipt_object = receipt_json.as_object().unwrap().clone();
    receipt_object.insert("skip_signature_check".into(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<VerifierReceipt>(
        serde_json::Value::Object(receipt_object)
    ).is_err());

    let anchor_json = serde_json::to_value(&anchors[0]).unwrap();
    let mut anchor_object = anchor_json.as_object().unwrap().clone();
    anchor_object.insert("implicitly_trusted".into(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<TrustAnchor>(
        serde_json::Value::Object(anchor_object)
    ).is_err());
}


#[test]
fn externally_pinned_source_identity_mismatch_blocks_verification() {
    let manifest = fixture_manifest();
    let (receipts, anchors) = signed_bundle(&manifest);
    let policy = default_policy();
    let wrong_commit = "c".repeat(40);
    let expected = ExpectedSourceIdentity {
        commit_sha: &wrong_commit,
        tree_sha256: &manifest.source_tree_sha256,
    };
    let report = verify_evidence_bundle(&manifest, &expected, &policy, &receipts, &anchors, NOW);
    assert_eq!(report.status, EvidenceBundleStatus::Blocked);
    assert!(report.blockers.iter().any(|blocker| {
        matches!(
            blocker,
            symthaea_operations_research::aviation_evidence_bundle::EvidenceBundleBlocker::SourceIdentityMismatch { field }
                if field == "source_commit_sha"
        )
    }));
}

#[test]
fn every_manifest_identity_and_artifact_digest_is_receipt_bound() {
    let manifest = fixture_manifest();
    let (receipts, anchors) = signed_bundle(&manifest);
    let policy = default_policy();
    let mutations: [fn(&mut EvidenceManifest); 12] = [
        |m| m.source_commit_sha = "c".repeat(40),
        |m| m.source_tree_sha256 = digest_label("changed-tree"),
        |m| m.data_snapshot_sha256 = digest_label("changed-data"),
        |m| m.scenario_sha256 = digest_label("changed-scenario"),
        |m| m.model_sha256 = digest_label("changed-model"),
        |m| m.policy_config_sha256 = digest_label("changed-policy"),
        |m| m.candidate_plan_sha256 = digest_label("changed-plan"),
        |m| m.calibration_artifact_sha256 = digest_label("changed-calibration"),
        |m| m.drift_report_sha256 = digest_label("changed-drift"),
        |m| m.plan_verification_report_sha256 = digest_label("changed-verification"),
        |m| m.distributional_assessment_sha256 = digest_label("changed-distributional"),
        |m| m.network_assessment_sha256 = digest_label("changed-network"),
    ];

    for mutate in mutations {
        let mut changed = manifest.clone();
        mutate(&mut changed);
        let expected = ExpectedSourceIdentity {
            commit_sha: &changed.source_commit_sha,
            tree_sha256: &changed.source_tree_sha256,
        };
        let report = verify_evidence_bundle(&changed, &expected, &policy, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(report.blockers.iter().any(|blocker| {
            matches!(
                blocker,
                symthaea_operations_research::aviation_evidence_bundle::EvidenceBundleBlocker::ReceiptManifestMismatch { .. }
            )
        }));
    }
}
