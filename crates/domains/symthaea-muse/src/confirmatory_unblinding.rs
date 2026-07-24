// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Controlled confirmatory unblinding after irreversible collection closure.
//!
//! The receipt proves that the private codebook matches the frozen public
//! schedule and that the revealed 256-bit key opens the preregistered
//! randomization commitment. Two distinct authorized people must approve the
//! reveal after the collection-close receipt exists.

use crate::blinded_study::{BlindedSchedule, BlindingCodebook, validate_blinded_schedule};
use crate::confirmatory_collection_close::{
    ConfirmatoryCollectionCloseReceipt, confirmatory_collection_close_commitment,
};
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::experiment_manifest::FrozenStudyManifest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONFIRMATORY_UNBLINDING_VERSION: &str = "symthaea-muse-confirmatory-unblinding-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfirmatoryUnblindingRole {
    EvidenceCustodian,
    IndependentWitness,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryUnblindingAuthorization {
    pub role: ConfirmatoryUnblindingRole,
    pub signer_id: String,
    pub authorization_sha256: String,
    pub signed_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryUnblindingReceipt {
    pub receipt_version: String,
    pub study_id: String,
    pub collection_close_sha256: String,
    pub manifest_sha256: String,
    pub schedule_sha256: String,
    pub codebook_sha256: String,
    pub key_reveal_file_sha256: String,
    pub randomization_commitment_sha256: String,
    pub revealed_key_commitment_sha256: String,
    pub unblinded_at_utc: String,
    pub authorizations: Vec<ConfirmatoryUnblindingAuthorization>,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct UnblindingCommitment<'a> {
    receipt_version: &'a str,
    study_id: &'a str,
    collection_close_sha256: &'a str,
    manifest_sha256: &'a str,
    schedule_sha256: &'a str,
    codebook_sha256: &'a str,
    key_reveal_file_sha256: &'a str,
    randomization_commitment_sha256: &'a str,
    revealed_key_commitment_sha256: &'a str,
    unblinded_at_utc: &'a str,
    authorizations: &'a [ConfirmatoryUnblindingAuthorization],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryUnblindingIssue {
    InvalidCollectionClose,
    InvalidBlindedSchedule,
    WrongVersion {
        found: String,
    },
    EmptyField {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    SecretCommitmentMismatch,
    RandomizationAuthorityMismatch,
    MissingAuthorization {
        role: ConfirmatoryUnblindingRole,
    },
    DuplicateAuthorization {
        role: ConfirmatoryUnblindingRole,
    },
    DuplicateSigner,
    InvalidAuthorization {
        role: ConfirmatoryUnblindingRole,
        field: String,
    },
    SerializationFailed,
    ReceiptDigestMismatch,
}

pub fn confirmatory_unblinding_commitment(
    receipt: &ConfirmatoryUnblindingReceipt,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&UnblindingCommitment {
        receipt_version: &receipt.receipt_version,
        study_id: &receipt.study_id,
        collection_close_sha256: &receipt.collection_close_sha256,
        manifest_sha256: &receipt.manifest_sha256,
        schedule_sha256: &receipt.schedule_sha256,
        codebook_sha256: &receipt.codebook_sha256,
        key_reveal_file_sha256: &receipt.key_reveal_file_sha256,
        randomization_commitment_sha256: &receipt.randomization_commitment_sha256,
        revealed_key_commitment_sha256: &receipt.revealed_key_commitment_sha256,
        unblinded_at_utc: &receipt.unblinded_at_utc,
        authorizations: &receipt.authorizations,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_confirmatory_unblinding_receipt(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    revealed_key: [u8; 32],
    key_reveal_file_sha256: String,
    collection_close: &ConfirmatoryCollectionCloseReceipt,
    unblinded_at_utc: String,
    mut authorizations: Vec<ConfirmatoryUnblindingAuthorization>,
) -> Result<ConfirmatoryUnblindingReceipt, Vec<ConfirmatoryUnblindingIssue>> {
    authorizations.sort_by_key(|authorization| authorization.role);
    let collection_close_sha256 = confirmatory_collection_close_commitment(collection_close)
        .map_err(|_| vec![ConfirmatoryUnblindingIssue::SerializationFailed])?;
    let manifest_sha256 = canonical_json_sha256(manifest)
        .map_err(|_| vec![ConfirmatoryUnblindingIssue::SerializationFailed])?;
    let schedule_sha256 = canonical_json_sha256(schedule)
        .map_err(|_| vec![ConfirmatoryUnblindingIssue::SerializationFailed])?;
    let codebook_sha256 = canonical_json_sha256(codebook)
        .map_err(|_| vec![ConfirmatoryUnblindingIssue::SerializationFailed])?;
    let revealed_key_commitment_sha256 = sha256_hex(&revealed_key);
    let mut receipt = ConfirmatoryUnblindingReceipt {
        receipt_version: CONFIRMATORY_UNBLINDING_VERSION.into(),
        study_id: collection_close.study_id.clone(),
        collection_close_sha256,
        manifest_sha256,
        schedule_sha256,
        codebook_sha256,
        key_reveal_file_sha256,
        randomization_commitment_sha256: schedule.randomization_commitment_sha256.clone(),
        revealed_key_commitment_sha256,
        unblinded_at_utc,
        authorizations,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = confirmatory_unblinding_commitment(&receipt)
        .map_err(|_| vec![ConfirmatoryUnblindingIssue::SerializationFailed])?;
    let issues = validate_confirmatory_unblinding_receipt(
        manifest,
        schedule,
        codebook,
        revealed_key,
        collection_close,
        &receipt,
    );
    if issues.is_empty() {
        Ok(receipt)
    } else {
        Err(issues)
    }
}

pub fn validate_confirmatory_unblinding_receipt(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    revealed_key: [u8; 32],
    collection_close: &ConfirmatoryCollectionCloseReceipt,
    receipt: &ConfirmatoryUnblindingReceipt,
) -> Vec<ConfirmatoryUnblindingIssue> {
    let mut issues = Vec::new();
    match confirmatory_collection_close_commitment(collection_close) {
        Ok(found)
            if found == collection_close.receipt_sha256
                && found == receipt.collection_close_sha256
                && collection_close.collection_irreversibly_closed => {}
        _ => issues.push(ConfirmatoryUnblindingIssue::InvalidCollectionClose),
    }
    if !validate_blinded_schedule(manifest, schedule, Some(codebook)).is_empty() {
        issues.push(ConfirmatoryUnblindingIssue::InvalidBlindedSchedule);
    }
    if receipt.receipt_version != CONFIRMATORY_UNBLINDING_VERSION {
        issues.push(ConfirmatoryUnblindingIssue::WrongVersion {
            found: receipt.receipt_version.clone(),
        });
    }
    for (field, value) in [
        ("study_id", receipt.study_id.as_str()),
        ("unblinded_at_utc", receipt.unblinded_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryUnblindingIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        (
            "collection_close_sha256",
            receipt.collection_close_sha256.as_str(),
        ),
        ("manifest_sha256", receipt.manifest_sha256.as_str()),
        ("schedule_sha256", receipt.schedule_sha256.as_str()),
        ("codebook_sha256", receipt.codebook_sha256.as_str()),
        (
            "key_reveal_file_sha256",
            receipt.key_reveal_file_sha256.as_str(),
        ),
        (
            "randomization_commitment_sha256",
            receipt.randomization_commitment_sha256.as_str(),
        ),
        (
            "revealed_key_commitment_sha256",
            receipt.revealed_key_commitment_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            issues.push(ConfirmatoryUnblindingIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    verify_digest(
        "manifest_sha256",
        canonical_json_sha256(manifest),
        &receipt.manifest_sha256,
        &mut issues,
    );
    verify_digest(
        "schedule_sha256",
        canonical_json_sha256(schedule),
        &receipt.schedule_sha256,
        &mut issues,
    );
    verify_digest(
        "codebook_sha256",
        canonical_json_sha256(codebook),
        &receipt.codebook_sha256,
        &mut issues,
    );
    let key_commitment = sha256_hex(&revealed_key);
    if key_commitment != receipt.revealed_key_commitment_sha256 {
        issues.push(ConfirmatoryUnblindingIssue::SecretCommitmentMismatch);
    }
    if key_commitment != schedule.randomization_commitment_sha256
        || key_commitment != codebook.randomization_commitment_sha256
        || receipt.randomization_commitment_sha256 != schedule.randomization_commitment_sha256
    {
        issues.push(ConfirmatoryUnblindingIssue::RandomizationAuthorityMismatch);
    }
    validate_authorizations(receipt, &mut issues);
    match confirmatory_unblinding_commitment(receipt) {
        Ok(found) if found == receipt.receipt_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryUnblindingIssue::ReceiptDigestMismatch),
        Err(_) => issues.push(ConfirmatoryUnblindingIssue::SerializationFailed),
    }
    issues
}

fn validate_authorizations(
    receipt: &ConfirmatoryUnblindingReceipt,
    issues: &mut Vec<ConfirmatoryUnblindingIssue>,
) {
    let mut roles = BTreeSet::new();
    let mut signers = BTreeSet::new();
    for authorization in &receipt.authorizations {
        if !roles.insert(authorization.role) {
            issues.push(ConfirmatoryUnblindingIssue::DuplicateAuthorization {
                role: authorization.role,
            });
        }
        if !signers.insert(authorization.signer_id.as_str()) {
            issues.push(ConfirmatoryUnblindingIssue::DuplicateSigner);
        }
        for (field, valid) in [
            ("signer_id", !authorization.signer_id.trim().is_empty()),
            (
                "authorization_sha256",
                is_sha256(&authorization.authorization_sha256),
            ),
            (
                "signed_at_utc",
                !authorization.signed_at_utc.trim().is_empty(),
            ),
        ] {
            if !valid {
                issues.push(ConfirmatoryUnblindingIssue::InvalidAuthorization {
                    role: authorization.role,
                    field: field.into(),
                });
            }
        }
    }
    for role in [
        ConfirmatoryUnblindingRole::EvidenceCustodian,
        ConfirmatoryUnblindingRole::IndependentWitness,
    ] {
        if !roles.contains(&role) {
            issues.push(ConfirmatoryUnblindingIssue::MissingAuthorization { role });
        }
    }
}

fn verify_digest(
    field: &str,
    expected: Result<String, serde_json::Error>,
    found: &str,
    issues: &mut Vec<ConfirmatoryUnblindingIssue>,
) {
    match expected {
        Ok(value) if value == found => {}
        Ok(_) => issues.push(ConfirmatoryUnblindingIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ConfirmatoryUnblindingIssue::SerializationFailed),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_signer_cannot_authorize_both_roles() {
        let receipt = ConfirmatoryUnblindingReceipt {
            receipt_version: CONFIRMATORY_UNBLINDING_VERSION.into(),
            study_id: "study".into(),
            collection_close_sha256: "1".repeat(64),
            manifest_sha256: "2".repeat(64),
            schedule_sha256: "3".repeat(64),
            codebook_sha256: "4".repeat(64),
            key_reveal_file_sha256: "5".repeat(64),
            randomization_commitment_sha256: "6".repeat(64),
            revealed_key_commitment_sha256: "6".repeat(64),
            unblinded_at_utc: "now".into(),
            authorizations: vec![
                ConfirmatoryUnblindingAuthorization {
                    role: ConfirmatoryUnblindingRole::EvidenceCustodian,
                    signer_id: "same".into(),
                    authorization_sha256: "a".repeat(64),
                    signed_at_utc: "now".into(),
                },
                ConfirmatoryUnblindingAuthorization {
                    role: ConfirmatoryUnblindingRole::IndependentWitness,
                    signer_id: "same".into(),
                    authorization_sha256: "b".repeat(64),
                    signed_at_utc: "now".into(),
                },
            ],
            receipt_sha256: String::new(),
        };
        let mut issues = Vec::new();
        validate_authorizations(&receipt, &mut issues);
        assert!(issues.contains(&ConfirmatoryUnblindingIssue::DuplicateSigner));
    }
}
