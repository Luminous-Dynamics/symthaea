// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold-authorized containment of a compromised signing identity.
//!
//! A trust snapshot rotation may eventually revoke a key, but incident response
//! cannot wait for a scheduled rotation ceremony. This module creates a short,
//! explicit authority record that identifies the compromised key, the affected
//! authority domains, the supporting evidence, and the exact time at which the
//! key must cease to count.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::threshold::VerifiedThresholdCeremony;
use crate::trust::{KeyLifecycleStatus, KeyUsage, TrustSnapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SIGNER_COMPROMISE_SCHEMA: &str = "symthaea.fabrication.signer-compromise.v1";
pub const MAX_COMPROMISE_REASON_BYTES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompromisedSignerIdentity {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignerCompromiseNotice {
    pub schema_version: String,
    pub sequence: u64,
    pub signer: CompromisedSignerIdentity,
    pub affected_usages: BTreeSet<KeyUsage>,
    pub discovered_at_unix_s: u64,
    pub effective_at_unix_s: u64,
    pub source_trust_snapshot_digest: Sha256Digest,
    pub evidence_digest: Sha256Digest,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignerCompromisePolicy {
    pub maximum_activation_delay_s: u64,
    pub require_nonzero_evidence: bool,
}

impl Default for SignerCompromisePolicy {
    fn default() -> Self {
        Self {
            maximum_activation_delay_s: 300,
            require_nonzero_evidence: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignerCompromiseError {
    UnsupportedSchema,
    InvalidPolicy,
    SequenceZero,
    InvalidAlgorithm,
    InvalidKeyId,
    InvalidWindow,
    EmptyUsages,
    InvalidReason,
    EmptyEvidenceDigest,
    TrustSnapshotInvalid,
    TrustSnapshotMismatch,
    SignerUnknown,
    SignerAlreadyRevoked,
    UsageNotHeld(KeyUsage),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedSignerCompromise {
    notice: SignerCompromiseNotice,
    notice_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedSignerCompromise {
    pub fn notice(&self) -> &SignerCompromiseNotice {
        &self.notice
    }
    pub fn notice_digest(&self) -> Sha256Digest {
        self.notice_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_signer_compromise_notice(
    sequence: u64,
    algorithm: SignatureAlgorithm,
    key_id: impl Into<String>,
    affected_usages: BTreeSet<KeyUsage>,
    discovered_at_unix_s: u64,
    effective_at_unix_s: u64,
    evidence_digest: Sha256Digest,
    reason: impl Into<String>,
    trust_snapshot: &TrustSnapshot,
    policy: &SignerCompromisePolicy,
) -> Result<SignerCompromiseNotice, SignerCompromiseError> {
    validate_policy(policy)?;
    trust_snapshot
        .validate()
        .map_err(|_| SignerCompromiseError::TrustSnapshotInvalid)?;
    if sequence == 0 {
        return Err(SignerCompromiseError::SequenceZero);
    }
    let signer = CompromisedSignerIdentity {
        algorithm,
        key_id: key_id.into(),
    };
    validate_identity(&signer)?;
    if affected_usages.is_empty() {
        return Err(SignerCompromiseError::EmptyUsages);
    }
    if discovered_at_unix_s > effective_at_unix_s
        || effective_at_unix_s.saturating_sub(discovered_at_unix_s)
            > policy.maximum_activation_delay_s
    {
        return Err(SignerCompromiseError::InvalidWindow);
    }
    if policy.require_nonzero_evidence && evidence_digest == Sha256Digest([0; 32]) {
        return Err(SignerCompromiseError::EmptyEvidenceDigest);
    }
    let reason = reason.into();
    validate_reason(&reason)?;
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| record.algorithm == signer.algorithm && record.key_id == signer.key_id)
    else {
        return Err(SignerCompromiseError::SignerUnknown);
    };
    if record.status == KeyLifecycleStatus::Revoked {
        return Err(SignerCompromiseError::SignerAlreadyRevoked);
    }
    for usage in &affected_usages {
        if !record.usages.contains(usage) {
            return Err(SignerCompromiseError::UsageNotHeld(*usage));
        }
    }
    Ok(SignerCompromiseNotice {
        schema_version: SIGNER_COMPROMISE_SCHEMA.into(),
        sequence,
        signer,
        affected_usages,
        discovered_at_unix_s,
        effective_at_unix_s,
        source_trust_snapshot_digest: crate::trust::digest_trust_snapshot(trust_snapshot)
            .map_err(|_| SignerCompromiseError::TrustSnapshotInvalid)?,
        evidence_digest,
        reason,
    })
}

pub fn digest_signer_compromise_notice(
    notice: &SignerCompromiseNotice,
) -> Result<Sha256Digest, SignerCompromiseError> {
    validate_notice(notice)?;
    let bytes = serde_json::to_vec(notice)
        .map_err(|error| SignerCompromiseError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.signer-compromise-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_signer_compromise(
    notice: SignerCompromiseNotice,
    trust_snapshot: &TrustSnapshot,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedSignerCompromise, SignerCompromiseError> {
    validate_notice(&notice)?;
    let trust_digest = crate::trust::digest_trust_snapshot(trust_snapshot)
        .map_err(|_| SignerCompromiseError::TrustSnapshotInvalid)?;
    if notice.source_trust_snapshot_digest != trust_digest {
        return Err(SignerCompromiseError::TrustSnapshotMismatch);
    }
    let notice_digest = digest_signer_compromise_notice(&notice)?;
    if ceremony.purpose() != "signer-compromise-containment" {
        return Err(SignerCompromiseError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != notice_digest {
        return Err(SignerCompromiseError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedSignerCompromise {
        notice,
        notice_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_policy(policy: &SignerCompromisePolicy) -> Result<(), SignerCompromiseError> {
    if policy.maximum_activation_delay_s == 0 {
        return Err(SignerCompromiseError::InvalidPolicy);
    }
    Ok(())
}

fn validate_notice(notice: &SignerCompromiseNotice) -> Result<(), SignerCompromiseError> {
    if notice.schema_version != SIGNER_COMPROMISE_SCHEMA {
        return Err(SignerCompromiseError::UnsupportedSchema);
    }
    if notice.sequence == 0 {
        return Err(SignerCompromiseError::SequenceZero);
    }
    validate_identity(&notice.signer)?;
    if notice.affected_usages.is_empty() {
        return Err(SignerCompromiseError::EmptyUsages);
    }
    if notice.discovered_at_unix_s > notice.effective_at_unix_s {
        return Err(SignerCompromiseError::InvalidWindow);
    }
    validate_reason(&notice.reason)
}

fn validate_identity(identity: &CompromisedSignerIdentity) -> Result<(), SignerCompromiseError> {
    if !identity.algorithm.is_canonical() {
        return Err(SignerCompromiseError::InvalidAlgorithm);
    }
    if identity.key_id.trim().is_empty()
        || identity.key_id != identity.key_id.trim()
        || identity.key_id.len() > 256
        || identity.key_id.chars().any(char::is_control)
    {
        return Err(SignerCompromiseError::InvalidKeyId);
    }
    Ok(())
}

fn validate_reason(reason: &str) -> Result<(), SignerCompromiseError> {
    if reason.trim().is_empty()
        || reason != reason.trim()
        || reason.len() > MAX_COMPROMISE_REASON_BYTES
        || reason.chars().any(char::is_control)
    {
        return Err(SignerCompromiseError::InvalidReason);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_rejects_empty_usage_scope() {
        let notice = SignerCompromiseNotice {
            schema_version: SIGNER_COMPROMISE_SCHEMA.into(),
            sequence: 1,
            signer: CompromisedSignerIdentity {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "operator-a".into(),
            },
            affected_usages: BTreeSet::new(),
            discovered_at_unix_s: 10,
            effective_at_unix_s: 10,
            source_trust_snapshot_digest: Sha256Digest([1; 32]),
            evidence_digest: Sha256Digest([2; 32]),
            reason: "key copied from secured host".into(),
        };
        assert_eq!(
            digest_signer_compromise_notice(&notice),
            Err(SignerCompromiseError::EmptyUsages)
        );
    }
}
