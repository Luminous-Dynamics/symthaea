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

/// Signed role-specific verifier evidence. The signature covers every field
/// except signature_hex; see canonical_receipt_signing_bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    EligibleForHumanReview,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceBundleBlocker {
    UnsupportedManifestSchema { actual: u32 },
    InvalidManifestField { field: String },
    InvalidManifestTimestamp { field: String },
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
    pub status: EvidenceBundleStatus,
    pub manifest_sha256: String,
    pub blockers: Vec<EvidenceBundleBlocker>,
    /// Receipts passing local cryptographic and binding checks. Overall status
    /// can still be Blocked because another required role is missing or invalid.
    pub verified_roles: Vec<EvidenceRole>,
    pub receipt_count: usize,
    pub decision_time_epoch_seconds: i64,
    pub verification_time_epoch_seconds: i64,
}

/// Deterministic length-prefixed encoding used for the manifest digest.
/// Each record is field-name:UTF-8-byte-length:value followed by a newline.
/// Field order is fixed and independent of JSON serialization or key ordering.
pub fn canonical_manifest_bytes(manifest: &EvidenceManifest) -> Vec<u8> {
    let mut out = Vec::with_capacity(1_400);
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
    receipts: &[VerifierReceipt],
    trust_anchors: &[TrustAnchor],
    verification_time_epoch_seconds: i64,
) -> EvidenceBundleReport {
    let manifest_digest = sha256_hex(&canonical_manifest_bytes(manifest));
    let mut blockers = validate_manifest(manifest, verification_time_epoch_seconds);
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
                locally_verified.insert(role);
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
    let verified_roles: Vec<EvidenceRole> = locally_verified
        .into_iter()
        .filter(|role| role_counts.get(role).copied() == Some(1))
        .collect();

    EvidenceBundleReport {
        status: if blockers.is_empty() {
            EvidenceBundleStatus::EligibleForHumanReview
        } else {
            EvidenceBundleStatus::Blocked
        },
        manifest_sha256: manifest_digest,
        blockers,
        verified_roles,
        receipt_count: receipts.len(),
        decision_time_epoch_seconds: manifest.decision_time_epoch_seconds,
        verification_time_epoch_seconds,
    }
}

fn validate_manifest(manifest: &EvidenceManifest, now: i64) -> Vec<EvidenceBundleBlocker> {
    let mut blockers = Vec::new();
    if manifest.schema_version != EVIDENCE_MANIFEST_SCHEMA_VERSION {
        blockers.push(EvidenceBundleBlocker::UnsupportedManifestSchema { actual: manifest.schema_version });
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

    fn has(report: &EvidenceBundleReport, predicate: impl Fn(&EvidenceBundleBlocker) -> bool) -> bool {
        report.blockers.iter().any(predicate)
    }

    #[test]
    fn valid_bundle_is_eligible_only_for_human_review() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::EligibleForHumanReview);
        assert!(report.blockers.is_empty());
        assert_eq!(report.verified_roles, EvidenceRole::REQUIRED.to_vec());
        assert_eq!(report.receipt_count, EvidenceRole::REQUIRED.len());
    }

    #[test]
    fn canonical_manifest_encoding_is_repeatable() {
        let manifest = manifest();
        assert_eq!(canonical_manifest_bytes(&manifest), canonical_manifest_bytes(&manifest));
        assert_eq!(sha256_hex(&canonical_manifest_bytes(&manifest)).len(), 64);
    }

    #[test]
    fn changing_manifest_invalidates_prior_receipt_bindings() {
        let original = manifest();
        let (receipts, anchors) = signed_receipts(&original);
        let mut changed = original;
        changed.data_snapshot_sha256 = digest("changed-data");
        let report = verify_evidence_bundle(&changed, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptManifestMismatch { .. })));
    }

    #[test]
    fn receipt_order_does_not_change_report() {
        let manifest = manifest();
        let (receipts, anchors) = signed_receipts(&manifest);
        let forward = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        let mut reversed = receipts.clone();
        reversed.reverse();
        assert_eq!(forward, verify_evidence_bundle(&manifest, &reversed, &anchors, NOW));
    }

    #[test]
    fn duplicate_role_and_missing_role_block_promotion() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts.pop();
        let duplicate = receipts[0].clone();
        receipts.push(duplicate);
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::DuplicateReceiptRole { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::MissingReceipt { .. })));
    }

    #[test]
    fn expired_receipt_blocks_promotion() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].expires_at_epoch_seconds = NOW - 1;
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptExpired { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::InvalidSignature { .. })));
    }

    #[test]
    fn changed_verifier_binary_is_not_trusted() {
        let manifest = manifest();
        let (receipts, mut anchors) = signed_receipts(&manifest);
        anchors[0].verifier_binary_sha256 = digest("other-binary");
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::TrustIdentityMismatch { field, .. } if field == "verifier_binary_sha256")));
    }

    #[test]
    fn unknown_key_fails_closed() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].key_id = "unknown-key".into();
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::UnknownTrustAnchor { .. })));
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].signature_hex.replace_range(0..2, "ff");
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::InvalidSignature { .. })));
    }

    #[test]
    fn future_dated_manifest_and_receipt_are_blocked() {
        let mut manifest = manifest();
        manifest.decision_time_epoch_seconds = NOW + 1;
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].issued_at_epoch_seconds = NOW + 1;
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::FutureManifestTimestamp { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptIssuedInFuture { .. })));
    }

    #[test]
    fn revoked_trust_anchor_blocks_promotion() {
        let manifest = manifest();
        let (receipts, mut anchors) = signed_receipts(&manifest);
        anchors[0].revoked = true;
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::RevokedTrustAnchor { .. })));
    }

    #[test]
    fn subject_digest_must_match_manifest() {
        let manifest = manifest();
        let (mut receipts, anchors) = signed_receipts(&manifest);
        receipts[0].subject_sha256 = digest("different-subject");
        let report = verify_evidence_bundle(&manifest, &receipts, &anchors, NOW);
        assert_eq!(report.status, EvidenceBundleStatus::Blocked);
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::ReceiptSubjectMismatch { .. })));
        assert!(has(&report, |b| matches!(b, EvidenceBundleBlocker::InvalidSignature { .. })));
    }
}
