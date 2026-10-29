// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cryptographically bound evidence bundles for aviation decision readiness.
//!
//! A passing report means only that required receipts are internally valid and
//! anchored to a caller-supplied trust store. Trust-store provisioning, signer
//! governance, source truth, and domain-verifier correctness remain external.
//! This module grants no dispatch, maintenance-release, airworthiness, or
//! aircraft-control authority.

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EVIDENCE_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const VERIFIER_RECEIPT_SCHEMA_VERSION: u32 = 1;
pub const EVIDENCE_BUNDLE_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRole {
    Calibration,
    DriftMonitor,
    PlanFeasibility,
    DistributionalRisk,
    NetworkHarm,
}

impl EvidenceRole {
    pub const REQUIRED: [Self; 5] = [
        Self::Calibration,
        Self::DriftMonitor,
        Self::PlanFeasibility,
        Self::DistributionalRisk,
        Self::NetworkHarm,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::DriftMonitor => "drift_monitor",
            Self::PlanFeasibility => "plan_feasibility",
            Self::DistributionalRisk => "distributional_risk",
            Self::NetworkHarm => "network_harm",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "calibration" => Some(Self::Calibration),
            "drift_monitor" => Some(Self::DriftMonitor),
            "plan_feasibility" => Some(Self::PlanFeasibility),
            "distributional_risk" => Some(Self::DistributionalRisk),
            "network_harm" => Some(Self::NetworkHarm),
            _ => None,
        }
    }
}

/// Immutable snapshot of the exact artifacts relevant to a decision.
/// Digest fields are lowercase SHA-256 hex except source_commit_sha, which
/// accepts a 40-character SHA-1 Git object ID or a 64-character SHA-256 Git ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub created_at_epoch_seconds: i64,
    pub decision_time_epoch_seconds: i64,
    pub source_commit_sha: String,
    pub source_tree_sha256: String,
    pub data_snapshot_sha256: String,
    pub scenario_sha256: String,
    pub model_sha256: String,
    pub policy_config_sha256: String,
    pub candidate_plan_sha256: String,
    pub calibration_artifact_sha256: String,
    pub drift_report_sha256: String,
    pub plan_verification_report_sha256: String,
    pub distributional_assessment_sha256: String,
    pub network_assessment_sha256: String,
}

/// Source identity obtained outside the evidence bundle, e.g. from pinned build metadata.
/// Do not populate these values by merely copying the untrusted manifest fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedSourceIdentity<'a> {
    pub commit_sha: &'a str,
    pub tree_sha256: &'a str,
}

/// Freshness limits must come from the caller's explicit policy, not the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceVerificationPolicy {
    /// Maximum age of the manifest itself at verification time.
    pub max_manifest_age_seconds: u64,
    /// Maximum age of the represented decision snapshot at verification time.
    pub max_decision_age_seconds: u64,
}

/// Signed role-specific verifier evidence. The signature covers every field
/// except signature_hex; see canonical_receipt_signing_bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierReceipt {
    pub schema_version: u32,
    pub receipt_id: String,
    /// Raw string at the untrusted boundary so unknown roles fail closed.
    pub role: String,
    pub manifest_sha256: String,
    pub subject_sha256: String,
    pub verifier_id: String,
    pub verifier_binary_sha256: String,
    pub key_id: String,
    pub issued_at_epoch_seconds: i64,
    pub expires_at_epoch_seconds: i64,
    pub signature_hex: String,
}

/// Trust entry supplied independently of the evidence bundle. Production trust
/// stores should be versioned, controlled, reviewed, and provisioned out-of-band.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustAnchor {
    pub key_id: String,
    pub role: String,
    pub verifier_id: String,
    pub verifier_binary_sha256: String,
    pub public_key_hex: String,
    pub active_from_epoch_seconds: i64,
    pub active_until_epoch_seconds: i64,
    pub max_receipt_lifetime_seconds: u64,
    pub revoked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceBundleStatus {
    Blocked,
    Verified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceBundleBlocker {
    UnsupportedManifestSchema { actual: u32 },
    InvalidManifestField { field: String },
    SourceIdentityMismatch { field: String },
    InvalidManifestTimestamp { field: String },
    InvalidFreshnessPolicy { field: String },
    StaleManifest { age_seconds: u64, max_age_seconds: u64 },
    StaleDecisionSnapshot { age_seconds: u64, max_age_seconds: u64 },
    FutureManifestTimestamp { field: String },
    UnsupportedReceiptSchema { receipt_id: String, actual: u32 },
    UnknownReceiptRole { receipt_id: String, role: String },
    MissingReceipt { role: String },
    DuplicateReceiptRole { role: String },
    DuplicateReceiptId { receipt_id: String },
    MalformedReceiptField { receipt_id: String, field: String },
    ReceiptManifestMismatch { receipt_id: String },
    ReceiptSubjectMismatch { receipt_id: String, role: String },
    ReceiptInvalidTimeRange { receipt_id: String },
    ReceiptIssuedBeforeManifest { receipt_id: String },
    ReceiptIssuedInFuture { receipt_id: String },
    ReceiptExpired { receipt_id: String },
    ReceiptLifetimeTooLong { receipt_id: String },
    UnknownTrustAnchor { receipt_id: String, key_id: String },
    DuplicateTrustAnchor { key_id: String },
    RevokedTrustAnchor { key_id: String },
    TrustAnchorOutsideValidity { key_id: String },
    InvalidTrustAnchorPolicy { key_id: String },
    TrustIdentityMismatch { receipt_id: String, field: String },
    InvalidSignature { receipt_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceBundleReport {
    pub schema_version: u32,
    pub status: EvidenceBundleStatus,
    pub manifest_sha256: String,
    pub blockers: Vec<EvidenceBundleBlocker>,
    /// Receipts passing local cryptographic and binding checks. Overall status
    /// can still be Blocked because another required role is missing or invalid.
    pub verified_roles: Vec<EvidenceRole>,
    pub receipt_count: usize,
    pub decision_time_epoch_seconds: i64,
    pub verification_time_epoch_seconds: i64,
    pub expected_source_commit_sha: String,
    pub expected_source_tree_sha256: String,
    pub trust_store_sha256: String,
    pub verification_policy_sha256: String,
    pub max_manifest_age_seconds: u64,
    pub max_decision_age_seconds: u64,
}

/// Deterministic length-prefixed encoding used for the manifest digest.
/// Each record is field-name:UTF-8-byte-length:value followed by a newline.
/// Field order is fixed and independent of JSON serialization or key ordering.
pub fn canonical_manifest_bytes(manifest: &EvidenceManifest) -> Vec<u8> {
    let mut out = Vec::with_capacity(1_400);
    field(&mut out, "domain", "luminous-dynamics.aviation-evidence-manifest.v1");
    field(&mut out, "schema_version", &manifest.schema_version.to_string());
    field(&mut out, "run_id", &manifest.run_id);
    field(&mut out, "created_at_epoch_seconds", &manifest.created_at_epoch_seconds.to_string());
    field(&mut out, "decision_time_epoch_seconds", &manifest.decision_time_epoch_seconds.to_string());
    field(&mut out, "source_commit_sha", &manifest.source_commit_sha);
    field(&mut out, "source_tree_sha256", &manifest.source_tree_sha256);
    field(&mut out, "data_snapshot_sha256", &manifest.data_snapshot_sha256);
    field(&mut out, "scenario_sha256", &manifest.scenario_sha256);
    field(&mut out, "model_sha256", &manifest.model_sha256);
    field(&mut out, "policy_config_sha256", &manifest.policy_config_sha256);
    field(&mut out, "candidate_plan_sha256", &manifest.candidate_plan_sha256);
    field(&mut out, "calibration_artifact_sha256", &manifest.calibration_artifact_sha256);
    field(&mut out, "drift_report_sha256", &manifest.drift_report_sha256);
    field(&mut out, "plan_verification_report_sha256", &manifest.plan_verification_report_sha256);
    field(&mut out, "distributional_assessment_sha256", &manifest.distributional_assessment_sha256);
    field(&mut out, "network_assessment_sha256", &manifest.network_assessment_sha256);
    out
}

/// SHA-256 digest of bytes as 64 lowercase hexadecimal characters.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

/// Canonical Ed25519 receipt payload; signature_hex is intentionally excluded.
pub fn canonical_receipt_signing_bytes(receipt: &VerifierReceipt) -> Vec<u8> {
    let mut out = Vec::with_capacity(700);
    field(&mut out, "domain", "luminous-dynamics.aviation-verifier-receipt.v1");
    field(&mut out, "schema_version", &receipt.schema_version.to_string());
    field(&mut out, "receipt_id", &receipt.receipt_id);
    field(&mut out, "role", &receipt.role);
    field(&mut out, "manifest_sha256", &receipt.manifest_sha256);
    field(&mut out, "subject_sha256", &receipt.subject_sha256);
    field(&mut out, "verifier_id", &receipt.verifier_id);
    field(&mut out, "verifier_binary_sha256", &receipt.verifier_binary_sha256);
    field(&mut out, "key_id", &receipt.key_id);
    field(&mut out, "issued_at_epoch_seconds", &receipt.issued_at_epoch_seconds.to_string());
    field(&mut out, "expires_at_epoch_seconds", &receipt.expires_at_epoch_seconds.to_string());
    out
}

/// Canonical representation of the externally provisioned trust store.
/// Entries are sorted by all fields so input order cannot change its digest.
/// Duplicate entries are preserved so duplication remains visible in the digest.
pub fn canonical_trust_store_bytes(anchors: &[TrustAnchor]) -> Vec<u8> {
    let mut ordered: Vec<&TrustAnchor> = anchors.iter().collect();
    ordered.sort_by(|a, b| {
        a.key_id.cmp(&b.key_id)
            .then_with(|| a.role.cmp(&b.role))
            .then_with(|| a.verifier_id.cmp(&b.verifier_id))
            .then_with(|| a.verifier_binary_sha256.cmp(&b.verifier_binary_sha256))
            .then_with(|| a.public_key_hex.cmp(&b.public_key_hex))
            .then_with(|| a.active_from_epoch_seconds.cmp(&b.active_from_epoch_seconds))
            .then_with(|| a.active_until_epoch_seconds.cmp(&b.active_until_epoch_seconds))
            .then_with(|| a.max_receipt_lifetime_seconds.cmp(&b.max_receipt_lifetime_seconds))
            .then_with(|| a.revoked.cmp(&b.revoked))
    });
    let mut out = Vec::new();
    field(&mut out, "domain", "luminous-dynamics.aviation-trust-store.v1");
    field(&mut out, "anchor_count", &ordered.len().to_string());
    for (index, anchor) in ordered.iter().enumerate() {
        indexed_field(&mut out, index, "key_id", &anchor.key_id);
        indexed_field(&mut out, index, "role", &anchor.role);
        indexed_field(&mut out, index, "verifier_id", &anchor.verifier_id);
        indexed_field(&mut out, index, "verifier_binary_sha256", &anchor.verifier_binary_sha256);
        indexed_field(&mut out, index, "public_key_hex", &anchor.public_key_hex);
        indexed_field(&mut out, index, "active_from_epoch_seconds", &anchor.active_from_epoch_seconds.to_string());
        indexed_field(&mut out, index, "active_until_epoch_seconds", &anchor.active_until_epoch_seconds.to_string());
        indexed_field(&mut out, index, "max_receipt_lifetime_seconds", &anchor.max_receipt_lifetime_seconds.to_string());
        indexed_field(&mut out, index, "revoked", if anchor.revoked { "true" } else { "false" });
    }
    out
}

/// Canonical representation of caller-supplied freshness policy. The returned
/// digest makes the exact policy auditable; trust policy still must be provisioned
/// and selected independently of the untrusted evidence bundle.
pub fn canonical_verification_policy_bytes(policy: &EvidenceVerificationPolicy) -> Vec<u8> {
    let mut out = Vec::new();
    field(&mut out, "domain", "luminous-dynamics.aviation-verification-policy.v1");
    field(&mut out, "max_manifest_age_seconds", &policy.max_manifest_age_seconds.to_string());
    field(&mut out, "max_decision_age_seconds", &policy.max_decision_age_seconds.to_string());
    out
}

fn indexed_field(out: &mut Vec<u8>, index: usize, name: &str, value: &str) {
    let key = format!("anchor.{index}.{name}");
    field(out, &key, value);
}

pub fn subject_digest_for_role(manifest: &EvidenceManifest, role: EvidenceRole) -> &str {
    match role {
        EvidenceRole::Calibration => &manifest.calibration_artifact_sha256,
        EvidenceRole::DriftMonitor => &manifest.drift_report_sha256,
        EvidenceRole::PlanFeasibility => &manifest.plan_verification_report_sha256,
        EvidenceRole::DistributionalRisk => &manifest.distributional_assessment_sha256,
        EvidenceRole::NetworkHarm => &manifest.network_assessment_sha256,
    }
}

/// Verify the manifest and role receipts against an independently supplied trust store.
/// Verification checks byte identity and signatures, not the domain conclusion itself.
pub fn verify_evidence_bundle(
    manifest: &EvidenceManifest,
    expected_source: &ExpectedSourceIdentity<'_>,
    freshness_policy: &EvidenceVerificationPolicy,
    receipts: &[VerifierReceipt],
    trust_anchors: &[TrustAnchor],
    verification_time_epoch_seconds: i64,
) -> EvidenceBundleReport {
    let manifest_digest = sha256_hex(&canonical_manifest_bytes(manifest));
    let trust_store_digest = sha256_hex(&canonical_trust_store_bytes(trust_anchors));
    let verification_policy_digest = sha256_hex(&canonical_verification_policy_bytes(freshness_policy));
    let mut blockers = validate_manifest(
        manifest,
        freshness_policy,
        verification_time_epoch_seconds,
    );
    if !is_lower_hex_length(expected_source.commit_sha, &[40, 64]) {
        blockers.push(EvidenceBundleBlocker::InvalidManifestField { field: "expected_source_commit_sha".into() });
    }
    if !is_lower_hex_length(expected_source.tree_sha256, &[64]) {
        blockers.push(EvidenceBundleBlocker::InvalidManifestField { field: "expected_source_tree_sha256".into() });
    }
    if expected_source.commit_sha != manifest.source_commit_sha.as_str() {
        blockers.push(EvidenceBundleBlocker::SourceIdentityMismatch { field: "source_commit_sha".into() });
    }
    if expected_source.tree_sha256 != manifest.source_tree_sha256.as_str() {
        blockers.push(EvidenceBundleBlocker::SourceIdentityMismatch { field: "source_tree_sha256".into() });
    }
    if verification_time_epoch_seconds < 0 {
        blockers.push(EvidenceBundleBlocker::InvalidManifestTimestamp {
            field: "verification_time_epoch_seconds".into(),
        });
    }

    let mut anchor_counts = BTreeMap::<String, usize>::new();
    for anchor in trust_anchors {
        *anchor_counts.entry(anchor.key_id.clone()).or_default() += 1;
    }
    for (key_id, count) in &anchor_counts {
        if *count > 1 {
            blockers.push(EvidenceBundleBlocker::DuplicateTrustAnchor { key_id: key_id.clone() });
        }
    }

    let mut ordered: Vec<&VerifierReceipt> = receipts.iter().collect();
    ordered.sort_by(|a, b| {
        a.role.cmp(&b.role)
            .then_with(|| a.receipt_id.cmp(&b.receipt_id))
            .then_with(|| a.key_id.cmp(&b.key_id))
    });

    let mut receipt_id_counts = BTreeMap::<String, usize>::new();
    let mut role_counts = BTreeMap::<EvidenceRole, usize>::new();
    for receipt in &ordered {
        if !receipt.receipt_id.trim().is_empty() {
            *receipt_id_counts.entry(receipt.receipt_id.clone()).or_default() += 1;
        }
        if let Some(role) = EvidenceRole::parse(&receipt.role) {
            *role_counts.entry(role).or_default() += 1;
        }
    }
    for (receipt_id, count) in &receipt_id_counts {
        if *count > 1 {
            blockers.push(EvidenceBundleBlocker::DuplicateReceiptId { receipt_id: receipt_id.clone() });
        }
    }
    for (role, count) in &role_counts {
        if *count > 1 {
            blockers.push(EvidenceBundleBlocker::DuplicateReceiptRole { role: role.as_str().into() });
        }
    }

    let mut locally_verified = BTreeSet::<EvidenceRole>::new();
    for receipt in &ordered {
        let before = blockers.len();
        verify_one_receipt(
            receipt,
            manifest,
            &manifest_digest,
            trust_anchors,
            &anchor_counts,
            verification_time_epoch_seconds,
            &mut blockers,
        );
        if blockers.len() == before {
            if let Some(role) = EvidenceRole::parse(&receipt.role) {
                let unique_receipt_id =
                    receipt_id_counts.get(&receipt.receipt_id).copied() == Some(1);
                let unique_role = role_counts.get(&role).copied() == Some(1);
                let unique_anchor = anchor_counts.get(&receipt.key_id).copied() == Some(1);
                if unique_receipt_id && unique_role && unique_anchor {
                    locally_verified.insert(role);
                }
            }
        }
    }
    for role in EvidenceRole::REQUIRED {
        if role_counts.get(&role).copied().unwrap_or(0) == 0 {
            blockers.push(EvidenceBundleBlocker::MissingReceipt { role: role.as_str().into() });
        }
    }

    blockers.sort_by_key(|blocker| format!("{blocker:?}"));
    blockers.dedup();
    let verified_roles: Vec<EvidenceRole> = ordered
        .iter()
        .filter_map(|receipt| {
            let role = EvidenceRole::parse(&receipt.role)?;
            let unique_receipt_id = receipt_id_counts.get(&receipt.receipt_id).copied() == Some(1);
            let unique_role = role_counts.get(&role).copied() == Some(1);
            (unique_receipt_id && unique_role && locally_verified.contains(&role)).then_some(role)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    EvidenceBundleReport {
        schema_version: EVIDENCE_BUNDLE_REPORT_SCHEMA_VERSION,
        status: if blockers.is_empty() {
            EvidenceBundleStatus::Verified
        } else {
            EvidenceBundleStatus::Blocked
        },
        manifest_sha256: manifest_digest,
        blockers,
        verified_roles,
        receipt_count: receipts.len(),
        decision_time_epoch_seconds: manifest.decision_time_epoch_seconds,
        verification_time_epoch_seconds,
        expected_source_commit_sha: expected_source.commit_sha.into(),
        expected_source_tree_sha256: expected_source.tree_sha256.into(),
        trust_store_sha256: trust_store_digest,
        verification_policy_sha256: verification_policy_digest,
        max_manifest_age_seconds: freshness_policy.max_manifest_age_seconds,
        max_decision_age_seconds: freshness_policy.max_decision_age_seconds,
    }
}

fn validate_manifest(
    manifest: &EvidenceManifest,
    policy: &EvidenceVerificationPolicy,
    now: i64,
) -> Vec<EvidenceBundleBlocker> {
    let mut blockers = Vec::new();
    if manifest.schema_version != EVIDENCE_MANIFEST_SCHEMA_VERSION {
        blockers.push(EvidenceBundleBlocker::UnsupportedManifestSchema { actual: manifest.schema_version });
    }
    if policy.max_manifest_age_seconds == 0 {
        blockers.push(EvidenceBundleBlocker::InvalidFreshnessPolicy {
            field: "max_manifest_age_seconds".into(),
        });
    }
    if policy.max_decision_age_seconds == 0 {
        blockers.push(EvidenceBundleBlocker::InvalidFreshnessPolicy {
            field: "max_decision_age_seconds".into(),
        });
    }
    if manifest.run_id.trim().is_empty() {
        blockers.push(EvidenceBundleBlocker::InvalidManifestField { field: "run_id".into() });
    }
    if !is_lower_hex_length(&manifest.source_commit_sha, &[40, 64]) {
        blockers.push(EvidenceBundleBlocker::InvalidManifestField { field: "source_commit_sha".into() });
    }
    let digests = [
        ("source_tree_sha256", manifest.source_tree_sha256.as_str()),
        ("data_snapshot_sha256", manifest.data_snapshot_sha256.as_str()),
        ("scenario_sha256", manifest.scenario_sha256.as_str()),
        ("model_sha256", manifest.model_sha256.as_str()),
        ("policy_config_sha256", manifest.policy_config_sha256.as_str()),
        ("candidate_plan_sha256", manifest.candidate_plan_sha256.as_str()),
        ("calibration_artifact_sha256", manifest.calibration_artifact_sha256.as_str()),
        ("drift_report_sha256", manifest.drift_report_sha256.as_str()),
        ("plan_verification_report_sha256", manifest.plan_verification_report_sha256.as_str()),
        ("distributional_assessment_sha256", manifest.distributional_assessment_sha256.as_str()),
        ("network_assessment_sha256", manifest.network_assessment_sha256.as_str()),
    ];
    for (name, digest) in digests {
        if !is_lower_hex_length(digest, &[64]) {
            blockers.push(EvidenceBundleBlocker::InvalidManifestField { field: name.into() });
        }
    }
    for (name, timestamp) in [
        ("created_at_epoch_seconds", manifest.created_at_epoch_seconds),
        ("decision_time_epoch_seconds", manifest.decision_time_epoch_seconds),
    ] {
        if timestamp < 0 {
            blockers.push(EvidenceBundleBlocker::InvalidManifestTimestamp { field: name.into() });
        } else if now >= 0 && timestamp > now {
            blockers.push(EvidenceBundleBlocker::FutureManifestTimestamp { field: name.into() });
        }
    }
    if now >= 0 {
        if manifest.created_at_epoch_seconds >= 0
            && now >= manifest.created_at_epoch_seconds
            && policy.max_manifest_age_seconds > 0
        {
            let age = (now - manifest.created_at_epoch_seconds) as u64;
            if age > policy.max_manifest_age_seconds {
                blockers.push(EvidenceBundleBlocker::StaleManifest {
                    age_seconds: age,
                    max_age_seconds: policy.max_manifest_age_seconds,
                });
            }
        }
        if manifest.decision_time_epoch_seconds >= 0
            && now >= manifest.decision_time_epoch_seconds
            && policy.max_decision_age_seconds > 0
        {
            let age = (now - manifest.decision_time_epoch_seconds) as u64;
            if age > policy.max_decision_age_seconds {
                blockers.push(EvidenceBundleBlocker::StaleDecisionSnapshot {
                    age_seconds: age,
                    max_age_seconds: policy.max_decision_age_seconds,
                });
            }
        }
    }
    blockers
}

fn verify_one_receipt(
    receipt: &VerifierReceipt,
    manifest: &EvidenceManifest,
    manifest_digest: &str,
    trust_anchors: &[TrustAnchor],
    anchor_counts: &BTreeMap<String, usize>,
    now: i64,
    blockers: &mut Vec<EvidenceBundleBlocker>,
) {
    let id = receipt.receipt_id.clone();
    let before = blockers.len();

    if receipt.schema_version != VERIFIER_RECEIPT_SCHEMA_VERSION {
        blockers.push(EvidenceBundleBlocker::UnsupportedReceiptSchema {
            receipt_id: id.clone(),
            actual: receipt.schema_version,
        });
    }
    if id.trim().is_empty() {
        blockers.push(EvidenceBundleBlocker::MalformedReceiptField {
            receipt_id: id.clone(), field: "receipt_id".into(),
        });
    }
    let role = match EvidenceRole::parse(&receipt.role) {
        Some(role) => Some(role),
        None => {
            blockers.push(EvidenceBundleBlocker::UnknownReceiptRole {
                receipt_id: id.clone(), role: receipt.role.clone(),
            });
            None
        }
    };
    for (name, value, lengths) in [
        ("manifest_sha256", receipt.manifest_sha256.as_str(), &[64][..]),
        ("subject_sha256", receipt.subject_sha256.as_str(), &[64][..]),
        ("verifier_binary_sha256", receipt.verifier_binary_sha256.as_str(), &[64][..]),
        ("signature_hex", receipt.signature_hex.as_str(), &[128][..]),
    ] {
        if !is_lower_hex_length(value, lengths) {
            blockers.push(EvidenceBundleBlocker::MalformedReceiptField {
                receipt_id: id.clone(), field: name.into(),
            });
        }
    }
    if receipt.verifier_id.trim().is_empty() {
        blockers.push(EvidenceBundleBlocker::MalformedReceiptField {
            receipt_id: id.clone(), field: "verifier_id".into(),
        });
    }
    if receipt.key_id.trim().is_empty() {
        blockers.push(EvidenceBundleBlocker::MalformedReceiptField {
            receipt_id: id.clone(), field: "key_id".into(),
        });
    }

    if receipt.manifest_sha256 != manifest_digest {
        blockers.push(EvidenceBundleBlocker::ReceiptManifestMismatch { receipt_id: id.clone() });
    }
    if let Some(role) = role {
        if receipt.subject_sha256 != subject_digest_for_role(manifest, role) {
            blockers.push(EvidenceBundleBlocker::ReceiptSubjectMismatch {
                receipt_id: id.clone(), role: role.as_str().into(),
            });
        }
    }

    if receipt.issued_at_epoch_seconds < 0
        || receipt.expires_at_epoch_seconds < 0
        || receipt.expires_at_epoch_seconds < receipt.issued_at_epoch_seconds
    {
        blockers.push(EvidenceBundleBlocker::ReceiptInvalidTimeRange { receipt_id: id.clone() });
    } else {
        if receipt.issued_at_epoch_seconds < manifest.created_at_epoch_seconds {
            blockers.push(EvidenceBundleBlocker::ReceiptIssuedBeforeManifest { receipt_id: id.clone() });
        }
        if receipt.issued_at_epoch_seconds > now {
            blockers.push(EvidenceBundleBlocker::ReceiptIssuedInFuture { receipt_id: id.clone() });
        }
        if now > receipt.expires_at_epoch_seconds {
            blockers.push(EvidenceBundleBlocker::ReceiptExpired { receipt_id: id.clone() });
        }
    }

    let count = anchor_counts.get(&receipt.key_id).copied().unwrap_or(0);
    let anchor = if count == 1 {
        trust_anchors.iter().find(|item| item.key_id == receipt.key_id)
    } else {
        None
    };
    if count == 0 {
        blockers.push(EvidenceBundleBlocker::UnknownTrustAnchor {
            receipt_id: id.clone(), key_id: receipt.key_id.clone(),
        });
    }
    if let Some(anchor) = anchor {
        if anchor.revoked {
            blockers.push(EvidenceBundleBlocker::RevokedTrustAnchor { key_id: anchor.key_id.clone() });
        }
        if anchor.max_receipt_lifetime_seconds == 0
            || anchor.active_from_epoch_seconds < 0
            || anchor.active_until_epoch_seconds < anchor.active_from_epoch_seconds
        {
            blockers.push(EvidenceBundleBlocker::InvalidTrustAnchorPolicy { key_id: anchor.key_id.clone() });
        } else if now < anchor.active_from_epoch_seconds
            || now > anchor.active_until_epoch_seconds
            || receipt.issued_at_epoch_seconds < anchor.active_from_epoch_seconds
            || receipt.issued_at_epoch_seconds > anchor.active_until_epoch_seconds
        {
            blockers.push(EvidenceBundleBlocker::TrustAnchorOutsideValidity { key_id: anchor.key_id.clone() });
        }
        if let Some(role) = role {
            if anchor.role != role.as_str() {
                blockers.push(EvidenceBundleBlocker::TrustIdentityMismatch {
                    receipt_id: id.clone(), field: "role".into(),
                });
            }
        }
        if anchor.verifier_id != receipt.verifier_id {
            blockers.push(EvidenceBundleBlocker::TrustIdentityMismatch {
                receipt_id: id.clone(), field: "verifier_id".into(),
            });
        }
        if anchor.verifier_binary_sha256 != receipt.verifier_binary_sha256 {
            blockers.push(EvidenceBundleBlocker::TrustIdentityMismatch {
                receipt_id: id.clone(), field: "verifier_binary_sha256".into(),
            });
        }
        if let Some(lifetime) = receipt.expires_at_epoch_seconds.checked_sub(receipt.issued_at_epoch_seconds) {
            if lifetime >= 0 && anchor.max_receipt_lifetime_seconds < lifetime as u64 {
                blockers.push(EvidenceBundleBlocker::ReceiptLifetimeTooLong { receipt_id: id.clone() });
            }
        }

        match (decode_hex::<32>(&anchor.public_key_hex), decode_hex::<64>(&receipt.signature_hex)) {
            (Some(key_bytes), Some(signature_bytes)) => match VerifyingKey::from_bytes(&key_bytes) {
                Ok(key) => {
                    let signature = Signature::from_bytes(&signature_bytes);
                    if key.verify_strict(&canonical_receipt_signing_bytes(receipt), &signature).is_err() {
                        blockers.push(EvidenceBundleBlocker::InvalidSignature { receipt_id: id.clone() });
                    }
                }
                Err(_) => blockers.push(EvidenceBundleBlocker::MalformedReceiptField {
                    receipt_id: id.clone(), field: "trust_anchor_public_key_hex".into(),
                }),
            },
            _ => blockers.push(EvidenceBundleBlocker::MalformedReceiptField {
                receipt_id: id.clone(), field: "signature_or_public_key_hex".into(),
            }),
        }
    }

    // The caller uses the change in blocker count to determine local validity.
    let _receipt_added_blockers = blockers.len().saturating_sub(before);
}

fn field(out: &mut Vec<u8>, name: &str, value: &str) {
    out.extend_from_slice(name.as_bytes());
    out.push(b':');
    out.extend_from_slice(value.as_bytes().len().to_string().as_bytes());
    out.push(b':');
    out.extend_from_slice(value.as_bytes());
    out.push(b'\n');
}

fn is_lower_hex_length(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len())
        && value.as_bytes().iter().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}

fn decode_hex<const N: usize>(value: &str) -> Option<[u8; N]> {
    if value.len() != N * 2 || !is_lower_hex_length(value, &[N * 2]) {
        return None;
    }
    let mut out = [0_u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        out[index] = (decode_nibble(pair[0])? << 4) | decode_nibble(pair[1])?;
    }
    Some(out)
}

fn decode_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const NOW: i64 = 1_800_000_000;

    fn digest(value: &str) -> String { sha256_hex(value.as_bytes()) }

    fn manifest() -> EvidenceManifest {
        EvidenceManifest {
            schema_version: EVIDENCE_MANIFEST_SCHEMA_VERSION,
            run_id: "aviation-fixture-run-001".into(),
            created_at_epoch_seconds: NOW - 20,
            decision_time_epoch_seconds: NOW - 10,
            source_commit_sha: "a".repeat(40),
            source_tree_sha256: digest("tree"),
            data_snapshot_sha256: digest("data"),
            scenario_sha256: digest("scenario"),
            model_sha256: digest("model"),
            policy_config_sha256: digest("config"),
            candidate_plan_sha256: digest("plan"),
            calibration_artifact_sha256: digest("calibration"),
            drift_report_sha256: digest("drift"),
            plan_verification_report_sha256: digest("verification"),
            distributional_assessment_sha256: digest("distributional"),
            network_assessment_sha256: digest("network"),
        }
    }

    fn signed_receipts(manifest: &EvidenceManifest) -> (Vec<VerifierReceipt>, Vec<TrustAnchor>) {
        let manifest_sha256 = sha256_hex(&canonical_manifest_bytes(manifest));
        let mut receipts = Vec::new();
        let mut anchors = Vec::new();
        for (index, role) in EvidenceRole::REQUIRED.into_iter().enumerate() {
            let seed = [u8::try_from(index + 1).unwrap(); 32];
            let signing_key = SigningKey::from_bytes(&seed);
            let key_id = format!("fixture-key-{}", role.as_str());
            let verifier_id = format!("fixture-verifier-{}", role.as_str());
            let binary = digest(&format!("binary:{}", role.as_str()));
            let mut receipt = VerifierReceipt {
                schema_version: VERIFIER_RECEIPT_SCHEMA_VERSION,
                receipt_id: format!("receipt-{}", role.as_str()),
                role: role.as_str().into(),
                manifest_sha256: manifest_sha256.clone(),
                subject_sha256: subject_digest_for_role(manifest, role).into(),
                verifier_id: verifier_id.clone(),
                verifier_binary_sha256: binary.clone(),
                key_id: key_id.clone(),
                issued_at_epoch_seconds: NOW - 5,
                expires_at_epoch_seconds: NOW + 300,
                signature_hex: String::new(),
            };
            receipt.signature_hex =
                hex_encode(&signing_key.sign(&canonical_receipt_signing_bytes(&receipt)).to_bytes());
            receipts.push(receipt);
            anchors.push(TrustAnchor {
                key_id,
                role: role.as_str().into(),
                verifier_id,
                verifier_binary_sha256: binary,
                public_key_hex: hex_encode(&signing_key.verifying_key().to_bytes()),
                active_from_epoch_seconds: NOW - 10_000,
                active_until_epoch_seconds: NOW + 10_000,
                max_receipt_lifetime_seconds: 600,
                revoked: false,
            });
        }
        (receipts, anchors)
    }


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
    ) -> EvidenceBundleReport {
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
    ) -> EvidenceBundleReport {
        let expected = ExpectedSourceIdentity {
            commit_sha: &manifest.source_commit_sha,
            tree_sha256: &manifest.source_tree_sha256,
        };
        let policy = default_policy();
        verify_evidence_bundle(manifest, &expected, &policy, receipts, anchors, now)
    }

    fn has(report: &EvidenceBundleReport, predicate: impl Fn(&EvidenceBundleBlocker) -> bool) -> bool {
        report.blockers.iter().any(predicate)
    }

    #[test]
    fn valid_bundle_is_verified_without_authorizing_a_decision() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Verified);
        assert!(report.blockers.is_empty());
        assert_eq!(report.verified_roles, EvidenceRole::REQUIRED.to_vec());
        assert_eq!(report.receipt_count, EvidenceRole::REQUIRED.len());
        assert_eq!(report.expected_source_commit_sha, manifest.source_commit_sha);
        assert_eq!(report.expected_source_tree_sha256, manifest.source_tree_sha256);
        assert_eq!(report.trust_store_sha256, sha256_hex(&canonical_trust_store_bytes(&anchors)));
        assert_eq!(report.verification_policy_sha256, sha256_hex(&canonical_verification_policy_bytes(&default_policy())));
    }

    #[test]
    fn canonical_manifest_encoding_is_repeatable() {
        let manifest = manifest();
        assert_eq!(canonical_manifest_bytes(&manifest), canonical_manifest_bytes(&manifest));
        assert_eq!(sha256_hex(&canonical_manifest_bytes(&manifest)).len(), 64);
    }

    #[test]
    fn trust_store_digest_is_independent_of_anchor_order() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let mut reversed = anchors.clone();
        reversed.reverse();
        assert_eq!(
            canonical_trust_store_bytes(&anchors),
            canonical_trust_store_bytes(&reversed)
        );
        let policy = default_policy();
        assert_eq!(
            canonical_verification_policy_bytes(&policy),
            canonical_verification_policy_bytes(&policy)
        );
        let first = verify_fixture(&manifest, &receipts, &anchors, NOW);
        let second = verify_fixture(&manifest, &receipts, &reversed, NOW);
        assert_eq!(first, second);
    }

    #[test]
    fn changing_manifest_invalidates_prior_receipt_bindings() {
        let original = manifest();
        let (receipts, anchors) = signed_receipts(&original);
        let mut changed = original;
        changed.data_snapshot_sha256 = digest("changed-data");
        let report = verify_fixture(&changed, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptManifestMismatch { .. })));
    }

    #[test]
    fn receipt_order_does_not_change_report() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let forward = verify_fixture(&manifest, &receipts, &anchors, NOW);
        let mut reversed = receipts.clone();
        reversed.reverse();
        assert_eq!(forward, verify_fixture(&manifest, &reversed, &anchors, NOW));
    }

    #[test]
    fn duplicate_role_and_missing_role_block_promotion() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts.pop();
        let duplicate = receipts[0].clone();
        receipts.push(duplicate);
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::DuplicateReceiptRole { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::MissingReceipt { .. })));
    }

    #[test]
    fn expired_receipt_blocks_promotion() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].expires_at_epoch_seconds = NOW - 1;
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptExpired { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::InvalidSignature { .. })));
    }

    #[test]
    fn changed_verifier_binary_is_not_trusted() {
        let manifest = manifest();
        let (receipts, mut anchors) = signed_receipts(&manifest);
        anchors[0].verifier_binary_sha256 = digest("other-binary");
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::TrustIdentityMismatch { field, .. } if field == "verifier_binary_sha256")));
    }

    #[test]
    fn unknown_key_fails_closed() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].key_id = "unknown-key".into();
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::UnknownTrustAnchor { .. })));
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        let replacement = if receipts[0].signature_hex.starts_with("00") { "01" } else { "00" };
        receipts[0].signature_hex.replace_range(0..2, replacement);
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::InvalidSignature { .. })));
    }

    #[test]
    fn future_dated_manifest_and_receipt_are_blocked() {
        let mut manifest = manifest();
        manifest.decision_time_epoch_seconds = NOW + 1;
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].issued_at_epoch_seconds = NOW + 1;
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::FutureManifestTimestamp { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptIssuedInFuture { .. })));
    }

    #[test]
    fn receipt_must_not_predate_manifest() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].issued_at_epoch_seconds = manifest.created_at_epoch_seconds - 1;
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptIssuedBeforeManifest { .. })));
    }

    #[test]
    fn stale_manifest_and_decision_snapshot_block_promotion() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let policy = EvidenceVerificationPolicy {
            max_manifest_age_seconds: 10,
            max_decision_age_seconds: 5,
        };
        let report = verify_fixture_with_policy(&manifest, &receipts, &anchors, &policy, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::StaleManifest { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::StaleDecisionSnapshot { .. })));
    }

    #[test]
    fn zero_freshness_limits_block_promotion() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let policy = EvidenceVerificationPolicy {
            max_manifest_age_seconds: 0,
            max_decision_age_seconds: 900,
        };
        let report = verify_fixture_with_policy(&manifest, &receipts, &anchors, &policy, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::InvalidFreshnessPolicy { field } if field == "max_manifest_age_seconds")));
    }

    #[test]
    fn source_identity_must_match_external_build_metadata() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let wrong_commit = "c".repeat(40);
        let expected = ExpectedSourceIdentity { commit_sha: &wrong_commit, tree_sha256: &manifest.source_tree_sha256 };
        let policy = default_policy();
        let report = verify_evidence_bundle(&manifest, &expected, &policy, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::SourceIdentityMismatch { field } if field == "source_commit_sha")));
    }

    #[test]
    fn revoked_trust_anchor_blocks_promotion() {
        let manifest = manifest();
        let (receipts, mut anchors) = signed_receipts(&manifest);
        anchors[0].revoked = true;
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::RevokedTrustAnchor { .. })));
    }

    #[test]
    fn subject_digest_must_match_manifest() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].subject_sha256 = digest("different-subject");
        let report = verify_fixture(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptSubjectMismatch { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::InvalidSignature { .. })));
    }
}
